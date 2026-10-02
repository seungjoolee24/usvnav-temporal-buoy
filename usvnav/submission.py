"""A submission's packaging and the local validator (T1-RES-16, T1-RES-17).

**The form is a directory** (6-R10, owner's decision 2026-09-10), and nothing here may
assume anything a container image would add. What is in it is 6-R9's list, as files:

    my-submission/
      submission.json     the manifest (below)
      agent.py            exposes `Agent(config_path)` -- 6-R6's interface
      config.json         anything; its absolute path is what `Agent()` receives
      SUBMISSION.md       required, read by nobody who scores (6-R9)
      weights/ ...        optional; there is no network at scoring time, so anything the
                          agent loads has to be in here (6-R1)

The manifest, `submission.json`:

    {"format":  "usvnav-submission/1",
     "team":    "herons",             letters, digits, `-` and `_`; at most 32
     "entry":   "agent:Agent",           module:Class, the module a file inside this directory
     "config":  "config.json",           a path inside this directory
     "runtime": "usvnav-runtime/1"}     the runtime definition it expects (`usvnav.runtime`)

**Every requirement is mechanically checkable, and the validator checks it** -- that is
6-R9's own condition on the list, and the reason `SUBMISSION.md` is checked for existence
and nothing else. `validate(dir)` runs the static checks here and then a **smoke run in a
child process**: the agent is constructed with its config path, `reset` with a real
course's metadata, and driven for a few hundred ticks under each condition through the
same `sim.run_episode` the scorer uses (7-A2), so an interface mistake cannot pass here
and fail there. The child process is the shape 6-R5 gives scoring -- the agent in a
process of its own, the runner in another -- and it is what lets a segfault in a team's
extension module show up as a report line rather than as the validator dying.

What the smoke run checks, each naming its decision in the report:

* the entry point imports and lives inside the directory (6-R9);
* `Agent(config_path)` constructs (6-R6);
* `act` returns two finite numbers on every tick of every condition (1-W6 -- an
  out-of-range but finite value is clipped and legal, and is reported as information);
* the same episode driven twice, on the same instance with `reset` between and on a
  fresh instance, produces the same actions (6-R8: no state carries across episodes, and
  no unseeded randomness -- a re-run has to reproduce the trace, 3-K6);
* no network attempt (6-R1) -- the child disables sockets and records any call;
* the wall clock per `act`, reported against a **provisional** level (6-R3 is `MEASURE`).

What it does not check: whether the agent is any good. The tutorial agent passes.

**Loading a submission for scoring** is `load_agent`, which the leaderboard uses. It puts
the directory on `sys.path` and imports the entry module by name, so two submissions whose
modules share a name (they will: everyone's is `agent.py`) cannot both be loaded in one
interpreter without `unload` between them. The CLI loads one submission per process, which
is also 6-R7b's shape; `unload` exists for the tests and for an operator's batch loop.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time

from .runtime import RUNTIME_ID
from .sim import TICK_CAP_MS, TICK_CAP_MULTIPLE, TICK_FLOOR_MS

FORMAT = "usvnav-submission/1"
MANIFEST = "submission.json"
DOC = "SUBMISSION.md"
REQUIRED = ("format", "team", "entry", "config", "runtime")
TEAM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")

#: 6-R9 says the cap is "set to accommodate a small vision model" and leaves the number
#: open; the value is 6-R9a in `DECISIONS.md` (signed 2026-09-10) and the validator applies it
#: so the check exists. A ResNet-18 is 45 MB and a MobileNet 15 MB; 200 MB is generous
#: for those and still rules out shipping a framework inside the directory.
SIZE_CAP_BYTES = 200 * 1024 * 1024

#: 6-R3's cap is `sim.TICK_CAP_MS` (X3's 8.8 ms floor times 25, signed 2026-09-10). The
#: validator *reports* the per-`act` wall clock and warns above a
#: quarter of the cap so a team sees the number while it still has room; nothing fails on
#: it here -- the scorer's runner applies the cap (6-R2), and it measures at the process
#: boundary, which adds under a millisecond to what is measured here.
TICK_MS_WARN = TICK_CAP_MS / 4

CONDITIONS = ("1-1", "1-2", "1-3", "1-4")
ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Every check the validator can report, with the decision it enforces. `docs/PACKAGING.md`
#: is generated from this list, and `tests/test_submission.py` asserts that no check id
#: appears in a report without appearing here -- so the document and the validator cannot
#: name different things.
CHECKS = (
    ("directory", "the path is a directory", "6-R10"),
    ("manifest", f"`{MANIFEST}` exists, is a JSON object, has {', '.join(REQUIRED)}, and its "
                 f"format is `{FORMAT}`", "6-R9"),
    ("team", "`team` matches `[A-Za-z0-9][A-Za-z0-9_-]{0,31}`; it names the leaderboard row "
             "and the team's page file", "3-K8"),
    ("entry", "`entry` is `module:Class` and the module is a file inside the directory", "6-R6, 6-R9"),
    ("config", "`config` names a file inside the directory; its absolute path is what "
               "`Agent()` receives", "6-R6"),
    ("runtime", "`runtime` names the runtime the scorer provides", "T1-RES-15"),
    ("doc", f"`{DOC}` exists and is not empty -- its content is neither scored nor read", "6-R9"),
    ("size", f"the directory is under {SIZE_CAP_BYTES // 2**20} MB, `__pycache__/` and "
             f"`.git/` not counted", "6-R9, 6-R9a"),
    ("import", "the entry module imports and the class is defined inside the directory", "6-R9"),
    ("construct", "`Agent(config_path)` constructs without raising", "6-R6"),
    ("act", "`act` returns two finite numbers on every tick of a short episode under each "
            "condition, and raises nothing; an out-of-range finite value is clipped and "
            "reported as information", "1-W6, 5-S11"),
    ("determinism", "the same episode driven twice -- on one instance with `reset()` between, "
                    "and on a fresh instance -- gives the same actions", "6-R8, 3-K6"),
    ("tick_ms", f"the wall clock per `act` is reported; a warning above {TICK_MS_WARN:.0f} ms "
                f"(a quarter of the {TICK_CAP_MS:.0f} ms cap), never a failure -- the "
                f"scorer applies the cap, not the validator", "6-R3, 6-R2"),
    ("network", "no socket connection or name lookup during the smoke run", "6-R1"),
    ("crash", "the smoke run itself ended abnormally (the child process died or hung)", "5-S11"),
)
DEFAULT_COURSE = ROOT / "tools" / "demo-course.json"

_SKIP_DIRS = ("__pycache__", ".git")


class SubmissionError(Exception):
    """The directory is not a submission in the sense of the spec above."""


# --------------------------------------------------------------------------- static

def load_manifest(root) -> dict:
    root = pathlib.Path(root)
    path = root / MANIFEST
    if not path.is_file():
        raise SubmissionError(f"{MANIFEST} is missing (6-R9: the manifest is required)")
    try:
        m = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SubmissionError(f"{MANIFEST} is not JSON: {exc}") from None
    if not isinstance(m, dict):
        raise SubmissionError(f"{MANIFEST} must be a JSON object")
    missing = [k for k in REQUIRED if k not in m]
    if missing:
        raise SubmissionError(f"{MANIFEST} lacks {', '.join(missing)}")
    if m["format"] != FORMAT:
        raise SubmissionError(f"format is {m['format']!r}, this validator reads {FORMAT!r}")
    return m


def size_bytes(root) -> int:
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for f in filenames:
            total += (pathlib.Path(dirpath) / f).stat().st_size
    return total


def _module_file(root: pathlib.Path, module: str) -> pathlib.Path | None:
    stem = root.joinpath(*module.split("."))
    for cand in (stem.with_suffix(".py"), stem / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _inside(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def check_static(root) -> tuple[list[dict], dict | None]:
    """The checks that need no interpreter: manifest, entry file, config, doc, size."""
    root = pathlib.Path(root)
    out: list[dict] = []

    def add(severity, check, message):
        out.append({"severity": severity, "check": check, "message": message})

    if not root.is_dir():
        add("error", "directory", f"{root} is not a directory (6-R10: a submission is one)")
        return out, None
    try:
        m = load_manifest(root)
    except SubmissionError as exc:
        add("error", "manifest", str(exc))
        return out, None
    add("ok", "manifest", f"{MANIFEST} parses, format {FORMAT}")

    team = m["team"]
    if not isinstance(team, str) or not TEAM_RE.match(team):
        add("error", "team", f"team {team!r} must match {TEAM_RE.pattern} -- it names "
                             f"the leaderboard row and the team's page file")
    else:
        add("ok", "team", repr(team))

    entry = m["entry"]
    if not isinstance(entry, str) or entry.count(":") != 1 or not all(entry.split(":")):
        add("error", "entry", f"entry {entry!r} must be 'module:Class' (6-R6)")
    else:
        module, _ = entry.split(":")
        f = _module_file(root, module)
        if f is None:
            add("error", "entry", f"entry module {module!r} is not a file inside the "
                                  f"submission (6-R9: the agent code ships in the directory)")
        else:
            add("ok", "entry", f"{entry} -> {f.relative_to(root)}")

    cfg = m["config"]
    if not isinstance(cfg, str) or not cfg:
        add("error", "config", "config must name a file inside the submission (6-R6)")
    else:
        cpath = root / cfg
        if not cpath.is_file() or not _inside(cpath, root):
            add("error", "config", f"config {cfg!r} is not a file inside the submission "
                                   f"(6-R6: config_path points inside the bundle)")
        else:
            add("ok", "config", f"{cfg} ({cpath.stat().st_size} bytes)")

    if m["runtime"] != RUNTIME_ID:
        add("error", "runtime", f"runtime {m['runtime']!r} is not one the scorer provides; "
                                f"the runtime is {RUNTIME_ID!r} (T1-RES-15)")
    else:
        add("ok", "runtime", RUNTIME_ID)

    doc = root / DOC
    if not doc.is_file() or not doc.read_text().strip():
        add("error", "doc", f"{DOC} is required and must not be empty (6-R9: required, "
                            f"neither scored nor verified -- this check reads its size only)")
    else:
        add("ok", "doc", f"{DOC} present ({len(doc.read_text().splitlines())} lines)")

    n = size_bytes(root)
    if n > SIZE_CAP_BYTES:
        add("error", "size", f"{n / 2**20:.1f} MB exceeds the cap of "
                             f"{SIZE_CAP_BYTES / 2**20:.0f} MB (6-R9, 6-R9a)")
    else:
        add("ok", "size", f"{n / 2**20:.2f} MB of {SIZE_CAP_BYTES / 2**20:.0f} MB")

    for skip in _SKIP_DIRS:
        if any(p.name == skip for p in root.rglob(skip)):
            add("info", "size", f"{skip}/ directories are not counted and should not "
                                f"be submitted")
    return out, m


# --------------------------------------------------------------------------- probe

class _Probe:
    """Wraps the team's agent so the smoke run can see what it returned and how long it
    took, before `sim.sanitize` clips or rejects it (1-W6)."""

    def __init__(self, agent):
        self.agent = agent
        self.ms: list[list[float]] = []            # per episode, per tick
        self.actions: list[list[tuple[float, float]]] = []
        self.bad: list[str] = []
        self.clipped = 0
        self.n_acts = 0

    def reset(self, meta):
        self.actions.append([])
        self.ms.append([])
        return self.agent.reset(meta)

    def act(self, obs):
        import numpy as np
        from .plant import V_MAX, V_MIN, W_MAX

        t0 = time.perf_counter()
        a = self.agent.act(obs)
        self.ms[-1].append(1000.0 * (time.perf_counter() - t0))
        self.n_acts += 1
        try:
            arr = np.asarray(a, dtype=float).reshape(-1)
        except Exception:
            self.bad.append(f"t={int(obs['t'])}: {a!r}"[:120])
            return a
        if arr.size != 2 or not np.all(np.isfinite(arr)):
            self.bad.append(f"t={int(obs['t'])}: {a!r}"[:120])
        else:
            if not (V_MIN - 1e-9 <= arr[0] <= V_MAX + 1e-9) or abs(arr[1]) > W_MAX + 1e-9:
                self.clipped += 1
            self.actions[-1].append((float(arr[0]), float(arr[1])))
        return a


def _disable_network(attempts: list):
    """Make any outbound attempt fail and be recorded (6-R1). A courtesy check: the
    scoring machine has no network at all, so this tells a team early what would fail
    late. It cannot see a C extension opening sockets on its own, and does not try to."""
    import socket

    def guard(name):
        def f(*args, **kwargs):
            attempts.append(name)
            raise OSError(f"network is disabled at scoring time (6-R1): {name}")
        return f

    for name in ("connect", "connect_ex", "sendto", "sendmsg"):
        setattr(socket.socket, name, guard(f"socket.{name}"))
    socket.getaddrinfo = guard("socket.getaddrinfo")
    socket.create_connection = guard("socket.create_connection")
    socket.create_server = guard("socket.create_server")


def _probe_main(root: str, course_path: str, ticks: int, conditions: list[str], out: str):
    """The child process. Writes a JSON report to `out`; never raises past this frame."""
    report: dict = {"problems": [], "runs": {}, "network_attempts": []}

    def add(severity, check, message):
        report["problems"].append({"severity": severity, "check": check, "message": message})

    def dump():
        pathlib.Path(out).write_text(json.dumps(report, indent=1))

    try:
        _disable_network(report["network_attempts"])
        rootp = pathlib.Path(root).resolve()
        m = load_manifest(rootp)
        module, cls_name = m["entry"].split(":")
        sys.path.insert(0, str(rootp))
        try:
            mod = importlib.import_module(module)
        except Exception as exc:
            add("error", "import", f"importing {module!r} failed: {type(exc).__name__}: {exc}")
            return dump()
        cls = getattr(mod, cls_name, None)
        if cls is None:
            add("error", "import", f"{module!r} has no attribute {cls_name!r} (6-R6)")
            return dump()
        try:
            where = pathlib.Path(inspect.getfile(cls))
        except TypeError:
            where = None
        if where is None or not _inside(where, rootp):
            add("error", "import", f"{cls_name} is defined in {where}, outside the "
                                   f"submission (6-R9: the agent code ships in the directory)")
            return dump()
        add("ok", "import", f"{m['entry']} imports from {where.relative_to(rootp)}")

        config = str(rootp / m["config"])
        try:
            t0 = time.perf_counter()
            agent = cls(config)
            add("ok", "construct", f"Agent(config_path) constructed in "
                                   f"{1000 * (time.perf_counter() - t0):.0f} ms (6-R6)")
        except Exception as exc:
            add("error", "construct", f"Agent(config_path) raised {type(exc).__name__}: "
                                      f"{exc} (6-R6: the only construction signature)")
            return dump()

        from . import coursefile
        from .sim import run_episode

        course = coursefile.load(course_path)
        probe = _Probe(agent)
        for cond in conditions:
            res = run_episode(course, probe, condition=cond, seed=0, tick_limit=ticks)
            ms = probe.ms[-1] or [0.0]
            report["runs"][cond] = {"outcome": res.outcome, "tier": res.tier,
                                    "ticks": res.ticks, "waypoints": res.waypoints_reached,
                                    "ms_mean": round(sum(ms) / len(ms), 3),
                                    "ms_max": round(max(ms), 3)}
            if res.tier == "submission_fault":
                detail = probe.bad[-1] if probe.bad else ""
                add("error", "act", f"{cond}: episode ended as {res.outcome} at tick "
                                    f"{res.ticks} {detail} (1-W6 / 5-S11: two finite "
                                    f"numbers every tick, and no exception)")
        if not any(p["check"] == "act" for p in report["problems"]):
            add("ok", "act", f"two finite numbers on every tick under "
                             f"{', '.join(conditions)} (1-W6)")
        if probe.clipped:
            add("info", "act", f"{probe.clipped} of {probe.n_acts} actions were outside "
                               f"[{-0.5}, 2.0] m/s x [-0.6, 0.6] rad/s and would be clipped "
                               f"(1-W6: legal; prev_action reports the clipped value)")

        # 6-R8: the same episode twice on the same instance, then on a fresh one.
        if not any(p["severity"] == "error" for p in report["problems"]):
            cond = conditions[0]
            probe.actions.clear(); probe.ms.clear()
            run_episode(course, probe, condition=cond, seed=0, tick_limit=ticks)
            run_episode(course, probe, condition=cond, seed=0, tick_limit=ticks)
            same = probe.actions[0] == probe.actions[1]
            fresh = _Probe(cls(config))
            run_episode(course, fresh, condition=cond, seed=0, tick_limit=ticks)
            fresh_same = probe.actions[0] == fresh.actions[0]
            if same and fresh_same:
                add("ok", "determinism", f"the same episode twice gives the same "
                                        f"{len(probe.actions[0])} actions, on one instance "
                                        f"with reset() between and on a fresh one (6-R8)")
            elif not same:
                add("error", "determinism", "the same episode driven twice on one instance "
                    "with reset() between gives different actions: state carries across "
                    "episodes, or the agent draws unseeded randomness (6-R8; a re-run must "
                    "reproduce the trace, 3-K6)")
            else:
                add("error", "determinism", "a fresh instance gives different actions from "
                    "a reset one: reset() does not restore what __init__ set up (6-R8)")

        worst = max((r["ms_mean"] for r in report["runs"].values()), default=0.0)
        if worst > TICK_MS_WARN:
            add("warning", "tick_ms", f"{worst:.1f} ms per act() on average -- within 4x of the "
                                      f"{TICK_CAP_MS:.0f} ms cap (6-R3, {TICK_CAP_MULTIPLE}x "
                                      f"X3's {TICK_FLOOR_MS} ms floor); a late tick applies the "
                                      f"previous action, persistent lateness fails the episode (6-R2)")
        else:
            add("ok", "tick_ms", f"{worst:.2f} ms per act() at worst over the conditions "
                                 f"(the cap is {TICK_CAP_MS:.0f} ms, 6-R3)")
        if report["network_attempts"]:
            add("error", "network", f"{len(report['network_attempts'])} network attempt(s) "
                                    f"during the smoke run: {sorted(set(report['network_attempts']))} "
                                    f"(6-R1: no network at scoring time)")
        else:
            add("ok", "network", "no network attempt during the smoke run (6-R1)")
    except BaseException as exc:  # noqa: BLE001 - the child reports everything
        add("error", "crash", f"the smoke run died: {type(exc).__name__}: {exc}")
    dump()


def probe(root, *, course=DEFAULT_COURSE, conditions=CONDITIONS, ticks: int = 200,
          timeout_s: float = 600.0) -> dict:
    """Run `_probe_main` in a child interpreter and return its report."""
    with tempfile.TemporaryDirectory(prefix="usvnav-probe-") as tmp:
        out = pathlib.Path(tmp) / "report.json"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
        # The child mirrors the parent's interpreter flags, so a validator run under
        # `python -S -P` (a fresh interpreter reading only the bundle, as the bundle test
        # does) probes under the same rules.
        flags = (["-S"] if sys.flags.no_site else []) + (["-P"] if sys.flags.safe_path else [])
        cmd = [sys.executable, *flags, "-m", "usvnav.submission", "--probe", str(root),
               str(course), str(ticks), ",".join(conditions), str(out)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                                  timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return {"problems": [{"severity": "error", "check": "crash",
                                  "message": f"the smoke run did not finish in {timeout_s:.0f} s"}],
                    "runs": {}, "network_attempts": [], "stdout": "", "stderr": ""}
        if out.is_file():
            report = json.loads(out.read_text())
        else:
            report = {"problems": [{"severity": "error", "check": "crash",
                                    "message": f"the child process exited with {proc.returncode} "
                                               f"and wrote no report"}],
                      "runs": {}, "network_attempts": []}
        report["stdout"] = proc.stdout[-2000:]
        report["stderr"] = proc.stderr[-2000:]
        return report


# --------------------------------------------------------------------------- validate

def validate(root, *, course=DEFAULT_COURSE, conditions=CONDITIONS, ticks: int = 200) -> dict:
    """Static checks, then the smoke run if they pass. `ok` is "no error"."""
    root = pathlib.Path(root)
    problems, manifest = check_static(root)
    report = {"format": "usvnav-validation/1", "submission": str(root),
              "team": manifest["team"] if manifest else None,
              "course": str(course), "ticks": ticks, "conditions": list(conditions),
              "problems": problems, "runs": {}, "ok": False}
    if any(p["severity"] == "error" for p in problems):
        return report
    child = probe(root, course=course, conditions=conditions, ticks=ticks)
    report["problems"] += child["problems"]
    report["runs"] = child["runs"]
    report["stdout"], report["stderr"] = child.get("stdout", ""), child.get("stderr", "")
    report["ok"] = not any(p["severity"] == "error" for p in report["problems"])
    return report


def render(report: dict) -> str:
    lines = [f"submission {report['team'] or '?'}  ({report['submission']})"]
    for p in report["problems"]:
        lines.append(f"  [{p['severity']:<7}] {p['check']:<12} {p['message']}")
    if report["runs"]:
        lines.append(f"  smoke run: {pathlib.Path(report['course']).name}, "
                     f"{report['ticks']} ticks per condition")
        for cond, r in report["runs"].items():
            lines.append(f"    {cond}  {r['outcome']:<18} tick {r['ticks']:>4}  "
                         f"waypoints {r['waypoints']}  act {r['ms_mean']:.2f} ms mean, "
                         f"{r['ms_max']:.1f} max")
    if report.get("stderr", "").strip():
        lines.append("  child stderr (tail):")
        lines += ["    " + l for l in report["stderr"].strip().splitlines()[-12:]]
    n_err = sum(p["severity"] == "error" for p in report["problems"])
    lines.append(f"  result: {'ACCEPTED' if report['ok'] else f'REJECTED ({n_err} error(s))'}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- loading

def load_agent(root):
    """`(manifest, factory)` -- the factory builds a fresh `Agent(config_path)` each call.

    Imports the entry module by name with the directory first on `sys.path`; see the
    module note on why that is one submission per interpreter unless `unload` is called.
    """
    root = pathlib.Path(root).resolve()
    m = load_manifest(root)
    module, cls_name = m["entry"].split(":")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    mod = importlib.import_module(module)
    cls = getattr(mod, cls_name)
    config = str(root / m["config"])
    return m, (lambda: cls(config))


def unload(root):
    """Forget every module imported from `root`, and take `root` off `sys.path`."""
    root = pathlib.Path(root).resolve()
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and _inside(pathlib.Path(f), root):
            del sys.modules[name]
    while str(root) in sys.path:
        sys.path.remove(str(root))


def cmd_validate_submission(args) -> int:
    conditions = tuple(args.conditions.split(",")) if args.conditions else CONDITIONS
    report = validate(args.dir, course=args.course or DEFAULT_COURSE,
                      conditions=conditions, ticks=args.ticks)
    print(render(report))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
        print("wrote", args.out)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    if len(sys.argv) == 7 and sys.argv[1] == "--probe":
        _, _, root, course, ticks, conds, out = sys.argv
        _probe_main(root, course, int(ticks), conds.split(","), out)
    else:
        raise SystemExit("internal: python -m usvnav.submission --probe ROOT COURSE TICKS "
                         "CONDS OUT (use `usvnav validate-submission DIR`)")
