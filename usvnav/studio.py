"""The studio: the participant's one test surface, as a page (T1-RES-21; 7-A2 amended, 7-A8).

    usvnav studio [DIR] [--port N] [--no-open]

**Why it exists.** The participant model is the coding-test form (7-A8): no generator, no
situation taxonomy, no published difficulty. A team imagines the hard cases for the
capability list in `docs/RULES.md`, builds them, drives its agent through them and looks at
what happened -- and that loop is the *whole* of its testing, so the owner made its
usability a requirement (7-A2 as amended). Before this page the loop was four shell
commands and a browser download; now it is: pick a course, edit it, pick an agent, press
play, watch.

**What it is.** One HTML page (`tools/studio.html`) served by one stdlib HTTP server (this
module) on the loopback interface. The page embeds the course editor (`tools/editor.html`,
unchanged in what it edits) and adds the three things an editor cannot do alone: a file
list it can save to, a play button, and playback. A run is a **child process** running
`sim.run_episode` -- the same loop, the same code path as `usvnav run` and the scorer
(7-A2) -- so an edit to the agent's code is picked up by the next run without a restart,
and a crash or an infinite loop in a team's agent cannot take the studio down with it.
What the child records is every tick's pose, velocity, action and clearance, plus where
the traffic was, so the page can scrub through the run on the very map the author drew it
on, and can ask "what did the agent see at this tick" (`/api/run/<id>/observe/<tick>`),
which is rendered from the recorded state by the same perception functions the run used.

**What it is not.** Not the scorer's process isolation (6-R5) -- the child is for reloading
and robustness, and it can see the filesystem -- and not a leaderboard. It writes nothing
outside the work folder.

The work folder:

    courses/*.json      course files (`usvnav-course/2`), the editor's documents
    agents/<name>/      submission directories (`docs/PACKAGING.md`), each with `submission.json`
    runs/<id>.json      one record per run: the course as it was run, the result, every tick
    runs/<id>.meta.json the record's summary, for the list
    runs/<id>.log       what the agent process printed

An empty `courses/` is seeded with the practice courses, an empty `agents/` with the worked
example, so the first thing a team sees works.

The API the page uses, all JSON unless said otherwise:

    GET    /api/state                       courses, agents, runs, constants
    GET    /api/course/<name>               the course file
    PUT    /api/course/<name>               save; body is the course; returns the full rule check
    DELETE /api/course/<name>
    POST   /api/validate                    the full rule check without saving
    POST   /api/agent/<name>                new submission directory, copied from the example
    POST   /api/agent/<name>/check          the submission validator, quick settings
    POST   /api/run                         {course, agent, condition, seed} -> {id}
    GET    /api/run/<id>                    status: running (with the tick), done, failed, cancelled
    GET    /api/run/<id>/record             the full record
    GET    /api/run/<id>/observe/<tick>     the observation at that tick; ?view=1-2 renders another
                                            condition's perception from the same state (PNG for the
                                            raster conditions)
    POST   /api/run/<id>/cancel
    DELETE /api/run/<id>
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from . import coursefile
from . import record as _rec
from . import submission as _sub
from .plant import DT, HULL_LENGTH, HULL_WIDTH, Vessel
from .record import FRAME_FIELDS, RUN_FORMAT

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGE = ROOT / "tools" / "studio.html"
EDITOR = ROOT / "tools" / "editor.html"
EXAMPLE = ROOT / "examples" / "submission"
SEED_COURSES = (ROOT / "sets" / "practice" / "courses", ROOT / "tools")
CONDITIONS = ("1-1", "1-2", "1-3", "1-4")
PROGRESS_EVERY = 20
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class StudioError(Exception):
    """A request the studio refuses, with a message for the page."""


def _name(s: str) -> str:
    if not isinstance(s, str) or not NAME_RE.match(s) or s in (".", ".."):
        raise StudioError(f"{s!r} is not a usable name: letters, digits, '.', '_' and '-' only")
    return s


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _write_json(path: pathlib.Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj) + "\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- the work folder

NOISE_TEMPLATE = '''"""Renderer noise for this work folder -- yours to write.

The scorer renders the 1-2 / 1-4 picture with **no noise at all**. Noise is something you
add here to test your agent against: pick a function below in the studio's *noise* selector
(or `usvnav run --noise noise:my_noise`) and every run you start renders with it, as does
"What the agent saw".

A noise function takes the class name of the pixels being painted ("water", "bank", "pier",
"vessel", "dock", "buoy", "ego") and two float arrays of the picture's shape -- the world x
and y of every pixel centre, in metres -- and returns integer offsets of that shape, added
to the class's base colour before the shoreline, shadows and outlines are drawn. Anchor it
to the world coordinates and it moves with the scene; keep it a pure function and runs stay
reproducible. The kit's own example is `usvnav.look.grain`.
"""
import numpy as np

from usvnav import look


def grain_x2(class_name, x, y):
    """The kit's example grain, twice as strong."""
    return look.grain(class_name, x, y) * 2


def my_noise(class_name, x, y):
    """Start here."""
    return np.zeros(x.shape, dtype=np.int16)
'''


class Workdir:
    def __init__(self, path):
        self.path = pathlib.Path(path).resolve()
        self.courses = self.path / "courses"
        self.agents = self.path / "agents"
        self.runs = self.path / "runs"

    def init(self) -> list[str]:
        """Create the layout; seed an empty courses/ and an empty agents/."""
        notes = []
        for d in (self.courses, self.agents, self.runs):
            d.mkdir(parents=True, exist_ok=True)
        if not any(self.courses.glob("*.json")):
            for src_dir in SEED_COURSES:
                for src in sorted(src_dir.glob("*.json")) if src_dir.is_dir() else []:
                    shutil.copy(src, self.courses / src.name)
                    notes.append(f"courses/{src.name} (copied from the kit)")
        if not any(p.is_dir() for p in self.agents.iterdir()):
            shutil.copytree(EXAMPLE, self.agents / "example")
            notes.append("agents/example (the worked example submission)")
        if not (self.path / "noise.py").is_file():
            (self.path / "noise.py").write_text(NOISE_TEMPLATE, encoding="utf-8")
            notes.append("noise.py (renderer noise functions -- yours to write)")
        return notes

    def noises(self) -> list[str]:
        """What the *noise* selector offers: the kit's names, then every module-level
        function in this folder's `noise.py` as `noise:<name>` (found by parsing, not by
        importing -- the file is the participant's code and runs in the worker)."""
        import ast
        from . import look

        out = list(look.NOISES)
        src = self.path / "noise.py"
        if src.is_file():
            try:
                tree = ast.parse(src.read_text(encoding="utf-8"))
            except SyntaxError:
                return out
            out += [f"noise:{n.name}" for n in tree.body
                    if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
        return out

    # courses
    def course_file(self, name: str) -> pathlib.Path:
        return self.courses / f"{_name(name)}.json"

    def list_courses(self) -> list[dict]:
        out = []
        for p in sorted(self.courses.glob("*.json")):
            st = p.stat()
            out.append({"name": p.stem, "modified": _dt.datetime.fromtimestamp(st.st_mtime)
                        .isoformat(timespec="seconds"), "bytes": st.st_size})
        return out

    def read_course(self, name: str) -> dict:
        p = self.course_file(name)
        if not p.is_file():
            raise StudioError(f"no course named {name!r}")
        return json.loads(p.read_text())

    def save_course(self, name: str, data: dict) -> list[dict]:
        """Write the course as the editor sent it, after checking it parses; return the rules."""
        if not isinstance(data, dict):
            raise StudioError("the course must be a JSON object")
        try:
            course = coursefile.from_dict(data)
        except (coursefile.CourseFileError, ValueError, KeyError, TypeError) as exc:
            raise StudioError(f"not a course file: {exc}") from None
        self.course_file(name).write_text(json.dumps(data, indent=1) + "\n")
        return coursefile.validate(course)

    def delete_course(self, name: str) -> None:
        p = self.course_file(name)
        if not p.is_file():
            raise StudioError(f"no course named {name!r}")
        p.unlink()

    # agents
    def agent_dir(self, name: str) -> pathlib.Path:
        return self.agents / _name(name)

    def list_agents(self) -> list[dict]:
        out = []
        for d in sorted(p for p in self.agents.iterdir() if p.is_dir()):
            row = {"name": d.name, "path": str(d)}
            try:
                m = _sub.load_manifest(d)
                row.update(team=m.get("team"), entry=m.get("entry"), ok=True)
            except Exception as exc:
                row.update(ok=False, problem=str(exc))
            out.append(row)
        return out

    def new_agent(self, name: str, source: str | None = None) -> pathlib.Path:
        dest = self.agent_dir(name)
        if dest.exists():
            raise StudioError(f"agents/{name} exists already")
        src = self.agent_dir(source) if source else EXAMPLE
        if not (src / _sub.MANIFEST).is_file():
            raise StudioError(f"{src} is not a submission directory")
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__"))
        m = json.loads((dest / _sub.MANIFEST).read_text())
        m["team"] = re.sub(r"[^A-Za-z0-9_-]", "-", name)[:32] or "team"
        (dest / _sub.MANIFEST).write_text(json.dumps(m, indent=1) + "\n")
        return dest

    # runs
    def run_paths(self, run_id: str) -> dict[str, pathlib.Path]:
        _name(run_id)
        base = self.runs / run_id
        return {"record": base.with_suffix(".json"), "meta": self.runs / f"{run_id}.meta.json",
                "log": base.with_suffix(".log"), "spec": self.runs / f"{run_id}.spec.json",
                "progress": self.runs / f"{run_id}.progress"}

    def list_runs(self) -> list[dict]:
        out = []
        for p in self.runs.glob("*.meta.json"):
            try:
                out.append(json.loads(p.read_text()))
            except (OSError, ValueError):
                continue
        out.sort(key=lambda m: (m.get("started", ""), m.get("started_epoch", 0.0)), reverse=True)
        return out

    def read_meta(self, run_id: str) -> dict:
        p = self.run_paths(run_id)["meta"]
        if not p.is_file():
            raise StudioError(f"no run {run_id!r}")
        return json.loads(p.read_text())

    def read_record(self, run_id: str) -> dict:
        p = self.run_paths(run_id)["record"]
        if not p.is_file():
            meta = self.read_meta(run_id)
            raise StudioError(f"run {run_id} has no record ({meta.get('status')})")
        return json.loads(p.read_text())


# --------------------------------------------------------------------------- runs

class Runner:
    """Starts run workers, watches the live ones, and writes each run's summary at the end."""

    def __init__(self, wd: Workdir):
        self.wd = wd
        self.live: dict[str, dict] = {}
        self.lock = threading.Lock()

    def start(self, course: str, agent: str, condition: str, seed, noise: str = "none") -> str:
        if condition not in CONDITIONS:
            raise StudioError(f"condition must be one of {', '.join(CONDITIONS)}")
        noise = (noise or "none").strip()
        if noise not in self.wd.noises():
            raise StudioError(f"noise must be one of {', '.join(self.wd.noises())}")
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            raise StudioError("seed must be an integer") from None
        data = self.wd.read_course(course)
        try:
            coursefile.from_dict(data)
        except Exception as exc:
            raise StudioError(f"course {course!r} does not parse: {exc}") from None
        agent_dir = self.wd.agent_dir(agent)
        if not (agent_dir / _sub.MANIFEST).is_file():
            raise StudioError(f"agents/{agent} has no {_sub.MANIFEST}")
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        paths = self.wd.run_paths(run_id)
        spec = {"format": RUN_FORMAT, "id": run_id, "course": course, "agent": agent,
                "condition": condition, "seed": seed, "noise": noise, "started": _now(),
                "course_snapshot": data, "agent_dir": str(agent_dir), "workdir": str(self.wd.path),
                "n_waypoints": len(data.get("waypoints", []))}
        _write_json(paths["spec"], spec)
        meta = {k: spec[k] for k in ("id", "course", "agent", "condition", "seed", "noise", "started")}
        # `started` is to the second; two runs started inside one second would otherwise
        # list in file-system order (a flake the integration run caught, 2026-09-11).
        meta.update(status="running", tick=0, started_epoch=time.time())
        _write_json(paths["meta"], meta)

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
        env["PYTHONUNBUFFERED"] = "1"
        flags = (["-S"] if sys.flags.no_site else []) + (["-P"] if sys.flags.safe_path else [])
        cmd = [sys.executable, *flags, "-m", "usvnav.studio", "--worker",
               str(paths["spec"]), str(paths["record"])]
        log = open(paths["log"], "w")
        proc = subprocess.Popen(cmd, cwd=agent_dir, stdout=log, stderr=subprocess.STDOUT, env=env)
        with self.lock:
            self.live[run_id] = {"proc": proc, "log": log, "cancelled": False, "meta": meta}
        threading.Thread(target=self._watch, args=(run_id,), daemon=True).start()
        return run_id

    def _watch(self, run_id: str) -> None:
        live = self.live[run_id]
        proc, paths = live["proc"], self.wd.run_paths(run_id)
        proc.wait()
        live["log"].close()
        meta = dict(live["meta"])
        record = None
        if paths["record"].is_file():
            try:
                record = json.loads(paths["record"].read_text())
            except ValueError:
                record = None
        if record is not None and record.get("status") == "done":
            meta.update(status="done", result=record["result"], finished=record["finished"])
        else:
            meta.update(status="cancelled" if live["cancelled"] else "failed",
                        finished=_now(), exit_code=proc.returncode,
                        error=(self._log_tail(run_id, 4000) or
                               f"the run process exited with {proc.returncode} "
                               f"and wrote no record"))
        meta.pop("tick", None)
        _write_json(paths["meta"], meta)
        for k in ("spec", "progress"):
            paths[k].unlink(missing_ok=True)
        with self.lock:
            self.live.pop(run_id, None)

    def _log_tail(self, run_id: str, n: int) -> str:
        p = self.wd.run_paths(run_id)["log"]
        try:
            return p.read_text(errors="replace")[-n:]
        except OSError:
            return ""

    def status(self, run_id: str) -> dict:
        meta = self.wd.read_meta(run_id)
        if meta.get("status") == "running":
            p = self.wd.run_paths(run_id)["progress"]
            try:
                meta["tick"] = int(p.read_text().strip() or 0)
            except (OSError, ValueError):
                pass
            with self.lock:
                if run_id not in self.live:
                    # A server restart orphaned it: the process is gone and nobody will
                    # write the summary. Say so rather than showing a spinner forever.
                    meta.update(status="failed", error="the studio was restarted while this "
                                                       "run was in progress")
                    _write_json(self.wd.run_paths(run_id)["meta"], meta)
        meta["log"] = self._log_tail(run_id, 2000)
        return meta

    def cancel(self, run_id: str) -> bool:
        with self.lock:
            live = self.live.get(run_id)
            if live is None:
                return False
            live["cancelled"] = True
            live["proc"].terminate()
        return True

    def delete(self, run_id: str) -> None:
        self.cancel(run_id)
        paths = self.wd.run_paths(run_id)
        if not paths["meta"].is_file():
            raise StudioError(f"no run {run_id!r}")
        for _ in range(50):            # let a cancelled worker finish writing
            with self.lock:
                if run_id not in self.live:
                    break
            time.sleep(0.05)
        for p in paths.values():
            p.unlink(missing_ok=True)

    def shutdown(self) -> None:
        with self.lock:
            ids = list(self.live)
        for run_id in ids:
            self.cancel(run_id)


# --------------------------------------------------------------------------- the worker

def _worker_main(spec_path: str, record_path: str) -> int:
    """One run, in its own interpreter: load the agent, run the episode, write the record."""
    from .sim import run_episode

    spec = json.loads(pathlib.Path(spec_path).read_text())
    record_path = pathlib.Path(record_path)
    progress_path = record_path.parent / f"{spec['id']}.progress"
    course = coursefile.from_dict(spec["course_snapshot"])

    def progress(tick):
        try:
            progress_path.write_text(str(tick))
        except OSError:
            pass

    record = {k: v for k, v in spec.items() if k != "agent_dir"}
    record.update(_rec.header())
    t0 = time.perf_counter()
    try:
        manifest, factory = _sub.load_agent(spec["agent_dir"])
        record["agent_team"], record["agent_entry"] = manifest.get("team"), manifest.get("entry")
        agent = factory()
    except Exception:
        record.update(_rec.failed_record(course, "crash", traceback.format_exc()))
        _write_json(record_path, record)
        return 0

    recorder = _rec.Recorder(progress, PROGRESS_EVERY)
    res = run_episode(course, agent, condition=spec["condition"], seed=spec["seed"],
                      record_trace=True, on_observation=recorder.on_observation,
                      raster=_raster_for(spec.get("noise"), spec.get("workdir")))
    record.update(_rec.episode_record(course, res, recorder, time.perf_counter() - t0))
    _write_json(record_path, record)
    return 0


def _raster_for(noise_spec, workdir):
    """`render.TRACK1` with the run's noise. `noise:<fn>` lives in `<workdir>/noise.py`."""
    from dataclasses import replace

    from . import look
    from .render import TRACK1
    if not noise_spec or noise_spec == "none":
        return TRACK1
    if workdir and str(workdir) not in sys.path:
        sys.path.insert(0, str(workdir))
    return replace(TRACK1, noise=look.resolve_noise(noise_spec))


# --------------------------------------------------------------------------- observe

def observe(record: dict, tick: int, view: str | None = None):
    """What the agent saw at `tick`, rebuilt from the recorded state.

    Perception is a pure function of the hull's pose, the course and the time (4-V15,
    4-V4, 4-V6, 4-V7), so it is *recomputed* from the frame rather than stored -- a raster
    per tick would make a record 6000 x 120 KB. `view` renders another condition's
    perception from the same state: what the raster would have shown where the object
    list was read. Returns `(content_type, bytes)`.
    """
    from .png import encode_png
    from .sim import observation

    frames = record["frames"]
    if not 0 <= tick < len(frames):
        raise StudioError(f"tick must be in [0, {len(frames) - 1}]")
    view = view or record["condition"]
    if view not in CONDITIONS:
        raise StudioError(f"view must be one of {', '.join(CONDITIONS)}")
    course = coursefile.from_dict(record["course_snapshot"])
    f = dict(zip(FRAME_FIELDS, frames[tick]))
    vessel = Vessel(f["x"], f["y"], f["psi"], u=f["u"] or 0.0, v=f["v"] or 0.0, r=f["r"] or 0.0)
    prev = frames[tick - 1] if tick > 0 else None
    prev_action = np.array([prev[8] or 0.0, prev[9] or 0.0]) if prev else np.zeros(2)
    t = tick * DT
    common = {k: np.asarray(v).tolist() for k, v in
              observation(vessel, course, tick, int(f["wp_index"]), prev_action).items()}
    if view in ("1-2", "1-4"):
        from .render import top_view
        raster = _raster_for(record.get("noise"), record.get("workdir"))
        return "image/png", encode_png(top_view(course, vessel, t, raster), scale=3)
    if view == "1-3":
        from .lidar import MAX_RANGE, scan
        body = {"condition": view, "tick": tick, "common": common,
                "ranges": [round(float(v), 3) for v in scan(vessel, course, t)],
                "max_range": MAX_RANGE}
    else:
        from .objects import CLASSES, object_list
        ol = object_list(course, vessel, t)
        n = int(ol["valid"].sum())
        rows = [{"cls": CLASSES[int(ol["cls"][i])], "pos": ol["pos"][i].tolist(),
                 "heading": float(ol["heading"][i]), "vel": ol["vel"][i].tolist(),
                 "extent": ol["extent"][i].tolist(),
                 "primitive": int(ol["primitive"][i]) if "primitive" in ol else None}
                for i in range(n)]
        body = {"condition": view, "tick": tick, "common": common, "objects": rows,
                "capacity": len(ol["valid"])}
    return "application/json", (json.dumps(body) + "\n").encode()


# --------------------------------------------------------------------------- HTTP

class _Handler(BaseHTTPRequestHandler):
    server: "StudioServer"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):       # quiet; the terminal shows the URL and the folder
        if self.server.verbose:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- plumbing
    def _send(self, status: int, ctype: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, "application/json", (json.dumps(obj) + "\n").encode())

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            raise StudioError("the request body is not JSON") from None

    def _route(self, method: str):
        url = urllib.parse.urlsplit(self.path)
        parts = [urllib.parse.unquote(p) for p in url.path.split("/") if p]
        query = dict(urllib.parse.parse_qsl(url.query))
        try:
            handled = self._dispatch(method, parts, query)
        except StudioError as exc:
            self._json({"error": str(exc)}, 400)
            return
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            self._json({"error": traceback.format_exc()}, 500)
            return
        if not handled:
            self._json({"error": f"no such route: {method} {url.path}"}, 404)

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def do_PUT(self):
        self._route("PUT")

    def do_DELETE(self):
        self._route("DELETE")

    # -- routes
    def _dispatch(self, method: str, parts: list[str], query: dict) -> bool:
        wd, runner = self.server.wd, self.server.runner
        if method == "GET" and parts in ([], ["index.html"]):
            self._send(200, "text/html; charset=utf-8", PAGE.read_bytes())
            return True
        if method == "GET" and parts == ["editor.html"]:
            self._send(200, "text/html; charset=utf-8", EDITOR.read_bytes())
            return True
        if method == "GET" and parts == ["favicon.ico"]:
            self._send(204, "image/x-icon", b"")
            return True
        if not parts or parts[0] != "api":
            return False
        api = parts[1:]

        if api == ["state"] and method == "GET":
            from .sim import TICK_LIMIT
            self._json({"workdir": str(wd.path), "courses": wd.list_courses(), "noises": wd.noises(),
                        "agents": wd.list_agents(), "runs": wd.list_runs(),
                        "conditions": CONDITIONS, "dt": DT, "tick_limit": TICK_LIMIT,
                        "hull": [HULL_LENGTH, HULL_WIDTH],
                        "frame_fields": FRAME_FIELDS})
            return True

        if len(api) == 2 and api[0] == "course":
            name = api[1]
            if method == "GET":
                self._json(wd.read_course(name))
            elif method == "PUT":
                self._json({"saved": name, "problems": wd.save_course(name, self._body())})
            elif method == "DELETE":
                wd.delete_course(name)
                self._json({"deleted": name})
            else:
                return False
            return True

        if api == ["validate"] and method == "POST":
            data = self._body()
            try:
                course = coursefile.from_dict(data)
            except Exception as exc:
                raise StudioError(f"not a course file: {exc}") from None
            self._json({"problems": coursefile.validate(course)})
            return True

        if len(api) >= 2 and api[0] == "agent" and method == "POST":
            name = api[1]
            if len(api) == 2:
                body = self._body() or {}
                dest = wd.new_agent(name, body.get("from"))
                self._json({"created": name, "path": str(dest)})
                return True
            if api[2:] == ["check"]:
                report = _sub.validate(wd.agent_dir(name), conditions=CONDITIONS, ticks=100)
                self._json({"ok": report["ok"], "text": _sub.render(report),
                            "problems": report["problems"]})
                return True
            return False

        if api == ["run"] and method == "POST":
            body = self._body() or {}
            for key in ("course", "agent", "condition"):
                if key not in body:
                    raise StudioError(f"run needs {key}")
            run_id = runner.start(body["course"], body["agent"], body["condition"],
                                  body.get("seed", 0), body.get("noise", "none"))
            self._json({"id": run_id}, 201)
            return True

        if len(api) >= 2 and api[0] == "run":
            run_id, rest = api[1], api[2:]
            if method == "GET" and not rest:
                self._json(runner.status(run_id))
            elif method == "GET" and rest == ["record"]:
                self._json(wd.read_record(run_id))
            elif method == "GET" and len(rest) == 2 and rest[0] == "observe":
                try:
                    tick = int(rest[1])
                except ValueError:
                    raise StudioError("tick must be an integer") from None
                ctype, data = observe(wd.read_record(run_id), tick, query.get("view"))
                self._send(200, ctype, data)
            elif method == "POST" and rest == ["cancel"]:
                self._json({"cancelled": runner.cancel(run_id)})
            elif method == "DELETE" and not rest:
                runner.delete(run_id)
                self._json({"deleted": run_id})
            else:
                return False
            return True
        return False


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, workdir: Workdir, port: int = 0, host: str = "127.0.0.1", verbose=False):
        super().__init__((host, port), _Handler)
        self.wd = workdir
        self.runner = Runner(workdir)
        self.verbose = verbose

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/"

    def close(self) -> None:
        self.runner.shutdown()
        self.server_close()


def serve(workdir, port: int = 8765, open_browser: bool = True) -> int:
    if not PAGE.is_file() or not EDITOR.is_file():
        raise SystemExit(f"{PAGE} or {EDITOR} is missing")
    wd = Workdir(workdir)
    notes = wd.init()
    try:
        srv = StudioServer(wd, port)
    except OSError as exc:
        raise SystemExit(f"cannot listen on port {port}: {exc}; try --port 0") from None
    print(f"studio: {srv.url}")
    print(f"  work folder   {wd.path}")
    for n in notes:
        print(f"  seeded        {n}")
    print("  Ctrl-C stops it; runs in progress are cancelled.")
    sys.stdout.flush()
    if open_browser:
        import webbrowser
        webbrowser.open(srv.url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
    return 0


def cmd_studio(args) -> int:
    return serve(args.dir, args.port, not args.no_open)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--worker"]:
        return _worker_main(argv[1], argv[2])
    import argparse
    ap = argparse.ArgumentParser(prog="python -m usvnav.studio", description=__doc__)
    ap.add_argument("dir", nargs="?", default=".")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true")
    return cmd_studio(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
