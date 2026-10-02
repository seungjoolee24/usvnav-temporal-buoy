"""`python -m usvnav <command>` -- everything an author or a team needs, no Python.

The commands, in the order someone actually uses them:

    validate   check a course against the rules, each problem naming its decision
    view       draw the course as a PNG, optionally with a run's trajectory on it
    run        drive a course with an agent and report the outcome
    observe    write the observation for one tick, as the agent receives it
    studio     the studio: courses, agents, a play button and playback in one page (7-A2)
    replay     feed a run record's actions back into the simulator and check it reproduces (3-K6)
    rerun      run a submission again on a record's episode and attribute any divergence (6-R2, 6-R8)
    editor     open the map editor alone
    score      run one or more submissions over a set and score the field (5-S1)
    validate-submission   check a submission directory against the packaging rules (6-R9)
    leaderboard           a persistent leaderboard: init / submit / show (3-K8, 5-S4, 5-S5);
                          submit runs the agent in child processes with the per-tick cap (6-R5, 6-R2)
    docs       regenerate the schema, rules and runtime documents from the code
    bundle     audit what ships to participants, and optionally build it (organizer-side)

Courses come from the editor (7-A2) or from a set you were given; the generator that makes
the hidden courses is an organiser tool and is not in this package (7-A8).

`run --agent mymodule:MyAgent` imports a class with `reset(meta)` and `act(obs)`, which
is the whole submission interface (6-R6). `score` drives the same class through the same
`run_episode`, so "it worked locally" and "it scored" cannot diverge for interface reasons
(7-A2: one code path for the numbers a team reads and the numbers that rank it).

5-S1 is a *relative* procedure -- an item value divided by the best among the submissions
that completed the same episode -- so `score` with a single `--agent` gives 1.0 on every
episode that agent completes and 0 on the rest, i.e. its completion rate; the item
ratios only mean something against a field. Weights are 5-S6's 0.3 / 0.4 / 0.3.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys
import time

import numpy as np

from . import coursefile
from .plant import DT, Vessel


def _load_agent(spec: str, config_path=None):
    """`module:Class`, imported from the current directory and constructed per 6-R6.

    6-R6 is `Agent(config_path)`, where the path points inside the submitted bundle. It
    is passed here too, and this is deliberately the *only* construction signature: the
    previous version fell back to `cls(course)` on a `TypeError` so the internal reference
    agent could be driven from the CLI. That is a submission-facing code path shaped
    around an agent 7-A1 does not ship, and it also swallowed a genuine `TypeError` raised
    inside a team's own `__init__` and re-raised it as a confusing second failure. Driving
    a privileged agent is `tools/`' job; the CLI runs what a team can run.
    """
    if ":" not in spec:
        raise SystemExit(f"--agent wants 'module:Class', got {spec!r}")
    mod_name, cls_name = spec.split(":", 1)
    sys.path.insert(0, str(pathlib.Path.cwd()))
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    try:
        return cls(config_path)
    except TypeError as exc:
        raise SystemExit(
            f"{spec} could not be constructed as {cls_name}(config_path): {exc}\n"
            f"6-R6 is `Agent(config_path)`; give __init__ a config_path parameter "
            f"(defaulting to None is fine).") from None


def cmd_validate(args):
    course = coursefile.load(args.course)
    problems = coursefile.validate(course, strict_sizes=not args.allow_any_size)
    errors = [p for p in problems if p["severity"] == "error"]
    for p in problems:
        print(f"  [{p['severity']:<7}] {p['rule']:<7} {p['message']}")
    if not problems:
        print("  no problems")
    return 1 if errors else 0


NOISE_HELP = ("renderer noise for the 1-2/1-4 picture: none (default, and what scoring uses), "
              "grain (the kit's example), or module:function -- your own "
              "(class_name, x_world, y_world) -> offsets")


def _raster(noise_spec):
    """`render.TRACK1` with the `--noise` choice applied; `None` spec -> scoring's own."""
    from dataclasses import replace

    from . import look
    from .render import TRACK1
    return replace(TRACK1, noise=look.resolve_noise(noise_spec))


def cmd_run(args):
    from .sim import run_episode

    course = coursefile.load(args.course)
    agent = _load_agent(args.agent, args.config)
    t0 = time.perf_counter()
    res = run_episode(course, agent, condition=args.condition, seed=args.seed,
                      record_trace=bool(args.trace), raster=_raster(args.noise))
    wall = time.perf_counter() - t0

    print(f"outcome        {res.outcome}   ({res.tier})")
    print(f"waypoints      {res.waypoints_reached} of {course.n_waypoints}")
    print(f"ticks          {res.ticks}   ({res.elapsed:.1f} s simulated)")
    print(f"distance       {res.distance:.1f} m")
    print(f"clearance      {res.clearance:.2f} m   (the 5-S13 statistic)")
    print(f"wall clock     {wall:.1f} s   ({1000 * wall / max(res.ticks, 1):.2f} ms/tick)")
    if args.trace:
        rows = [{"t": t, "x": round(x, 4), "y": round(y, 4), "psi": round(p, 6),
                 "v_cmd": round(v, 4), "w_cmd": round(w, 4)}
                for t, x, y, p, v, w in res.trace]
        pathlib.Path(args.trace).write_text(json.dumps(rows) + "\n")
        print(f"trace          {args.trace}   ({len(rows)} ticks)")
    return 0 if res.completed else 1


def cmd_replay(args):
    from . import record as _rec

    rec = json.loads(pathlib.Path(args.record).read_text())
    if rec.get("format") not in _rec.READ_RUN_FORMATS:
        raise SystemExit(f"{args.record}: format {rec.get('format')!r} is not one of "
                         f"{list(_rec.READ_RUN_FORMATS)}")
    res = _rec.replay(rec)
    diff = _rec.replay_matches(rec, res)
    print(f"record   {rec['id']}  {rec['condition']} seed {rec['seed']}  "
          f"{rec['result']['outcome']} at tick {rec['result']['ticks']}")
    print(f"replay   {res.outcome} at tick {res.ticks}; {len(res.trace)} actions fed back")
    late = rec.get("overrun_ticks")
    if late:
        print(f"late     {len(late)} tick(s) applied the previous action (6-R2): "
              f"{late[:12]}{' ...' if len(late) > 12 else ''}")
    if diff:
        print("differs:")
        for d in diff:
            print("  " + d)
        return 1
    print("the replay reproduces the record tick for tick (3-K6)")
    return 0


def cmd_rerun(args):
    from . import record as _rec
    from .runner import rerun

    rec = json.loads(pathlib.Path(args.record).read_text())
    if rec.get("format") not in _rec.READ_RUN_FORMATS:
        raise SystemExit(f"{args.record}: format {rec.get('format')!r} is not one of "
                         f"{list(_rec.READ_RUN_FORMATS)}")
    out = pathlib.Path(args.out) if args.out else pathlib.Path(args.record).parent / f"rerun-{rec['id']}"
    kw = {"cap_ms": float(args.cap_ms)} if args.cap_ms is not None else {}
    report = rerun(rec, args.submission, out, **kw)
    print(f"record   {rec['id']}  {rec['condition']} seed {rec['seed']}  "
          f"{report['outcome'][0]} at tick {report['ticks'][0]}")
    print(f"rerun    {report['outcome'][1]} at tick {report['ticks'][1]}  ({report['rerun_record']})")
    print(f"verdict  {report['verdict']}")
    return 0 if report["attribution"] == "reproduces" else 1


def cmd_observe(args):
    from .png import write_png
    from .render import top_view
    from .sim import observation

    course = coursefile.load(args.course)
    vessel = Vessel(*course.start)
    obs = observation(vessel, course, args.tick, 0, np.zeros(2))
    print("common fields (4-V15, identical in all four conditions):")
    for k, v in obs.items():
        print(f"  {k:<12} {np.asarray(v).tolist()}")
    if args.condition in ("1-2", "1-4"):
        img = top_view(course, vessel, args.tick * DT, _raster(args.noise))
        out = args.out or "observation.png"
        write_png(out, img, scale=args.scale)
        print(f"\nthe {args.condition} raster: {out}  "
              f"({img.shape[1]}x{img.shape[0]}, upscaled {args.scale}x for viewing)")
    elif args.condition == "1-3":
        from .lidar import scan
        r = scan(vessel, course, args.tick * DT)
        print(f"\nthe 1-3 range array: {len(r)} beams, "
              f"min {r.min():.2f} m, max {r.max():.2f} m")
        print("  " + " ".join(f"{v:.1f}" for v in r[:24]) + " ...")
    else:
        from .objects import CLASSES, object_list
        ol = object_list(course, vessel, args.tick * DT)
        n = int(ol["valid"].sum())
        print(f"\nthe 1-1 object list (4-V4): {len(ol['valid'])} rows, {n} valid, "
              f"nearest first")
        print(f"  classes {dict(enumerate(CLASSES))}, primitive 0=circle 1=rect")
        print(f"  {'cls':<15}{'pos (x,y) body':>20}{'head':>8}{'vel':>16}{'extent':>16}")
        for i in range(min(n, 12)):
            x, y = ol["pos"][i]
            vx, vy = ol["vel"][i]
            L, W = ol["extent"][i]
            print(f"  {CLASSES[ol['cls'][i]]:<15}{f'({x:7.1f},{y:7.1f})':>20}"
                  f"{ol['heading'][i]:>8.2f}{f'({vx:5.2f},{vy:5.2f})':>16}"
                  f"{f'{L:.1f} x {W:.1f}':>16}")
        if n > 12:
            print(f"  ... and {n - 12} more")
    return 0


def cmd_score(args):
    from .contest import leaderboard, load_set, score_set

    manifest, root = load_set(args.set)
    agents = {}
    for spec in args.agent:
        if "=" not in spec:
            raise SystemExit(f"--agent wants NAME=module:Class[:config], got {spec!r}")
        name, rest = spec.split("=", 1)
        parts = rest.split(":")
        module_class = ":".join(parts[:2])
        config = parts[2] if len(parts) > 2 else None
        agents[name] = (lambda mc=module_class, cf=config: _load_agent(mc, cf))
    conditions = tuple(args.conditions.split(",")) if args.conditions else None
    scores = score_set(manifest, root, agents, conditions=conditions, log=print)
    print()
    print(leaderboard(scores))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(scores, indent=1) + "\n")
        print("\nwrote", args.out)
    return 0


def cmd_editor(args):
    import webbrowser

    editor = pathlib.Path(__file__).resolve().parent.parent / "tools" / "editor.html"
    if not editor.exists():
        raise SystemExit(f"{editor} is missing")
    url = editor.as_uri()
    print(f"editor: {url}")
    print("  Open a course with the file picker, edit, then Save to download the JSON.")
    print("  The editor is a single file with no server and no dependencies, so it")
    print("  cannot write to disk on its own -- the browser downloads instead.")
    if not args.no_open:
        webbrowser.open(url)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m usvnav", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="check a course against the rules")
    v.add_argument("course")
    v.add_argument("--allow-any-size", action="store_true",
                   help="skip 2-E5's per-class size ranges")
    v.set_defaults(func=cmd_validate)

    w = sub.add_parser("view", help="draw the course as a PNG")
    w.add_argument("course")
    w.add_argument("--out", default="course.png")
    w.add_argument("--trace", help="a trace written by `run --trace`, drawn on top")
    w.add_argument("--span", type=float, default=340.0, help="metres across the view")
    w.add_argument("--noise", default=None, help=NOISE_HELP)
    w.set_defaults(func=_cmd_view)

    r = sub.add_parser("run", help="drive a course with an agent")
    r.add_argument("course")
    r.add_argument("--agent", default="usvnav.agent:TutorialAgent",
                   help="module:Class with reset(meta) and act(obs)")
    r.add_argument("--condition", default="1-1", choices=("1-1", "1-2", "1-3", "1-4"))
    r.add_argument("--seed", type=int, default=0,
                   help="drives the 1-4 disturbance only; the course is already fixed")
    r.add_argument("--trace", help="write the per-tick trace here")
    r.add_argument("--config", help="passed to Agent(config_path) per 6-R6")
    r.add_argument("--noise", default=None, help=NOISE_HELP)
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("replay", help="feed a run record's actions back into the simulator "
                                       "and check that it reproduces the record (3-K6)")
    rp.add_argument("record", help="a `usvnav-run/2` (or `/1`) record: a studio run or a scored episode")
    rp.set_defaults(func=cmd_replay)

    rr = sub.add_parser("rerun", help="run a submission again on a record's episode through the "
                                      "isolating runner and attribute any divergence (T1-SCO-12)")
    rr.add_argument("record", help="a `usvnav-run/2` (or `/1`) record")
    rr.add_argument("submission", help="the submission directory to run again")
    rr.add_argument("--out", help="where the rerun's set and record go (default: beside the record)")
    rr.add_argument("--cap-ms", type=float, default=None, help="the per-tick cap for the rerun")
    rr.set_defaults(func=cmd_rerun)

    o = sub.add_parser("observe", help="write one tick's observation")
    o.add_argument("course")
    o.add_argument("--condition", default="1-2", choices=("1-1", "1-2", "1-3", "1-4"))
    o.add_argument("--tick", type=int, default=0)
    o.add_argument("--scale", type=int, default=3)
    o.add_argument("--out")
    o.add_argument("--noise", default=None, help=NOISE_HELP)
    o.set_defaults(func=cmd_observe)

    st = sub.add_parser("studio", help="the studio: edit courses, run an agent on one, watch "
                                       "the run, manage your test cases and agents (T1-RES-21)")
    st.add_argument("dir", nargs="?", default=".",
                    help="the work folder (courses/, agents/, runs/); created if missing, "
                         "seeded with the practice courses and the example submission")
    st.add_argument("--port", type=int, default=8765, help="0 picks a free port")
    st.add_argument("--no-open", action="store_true", help="do not open a browser")
    st.set_defaults(func=_cmd_studio)

    e = sub.add_parser("editor", help="open the map editor alone (the studio contains it)")
    e.add_argument("--no-open", action="store_true")
    e.set_defaults(func=cmd_editor)

    b = sub.add_parser("bundle", help="audit and build what ships to participants")
    b.add_argument("dest", nargs="?", help="write the bundle here")
    b.set_defaults(func=_cmd_bundle)

    sc = sub.add_parser("score", help="run submissions over a set and score the field (5-S1)")
    sc.add_argument("set", help="a set's manifest.json or its directory")
    sc.add_argument("--agent", action="append", required=True, metavar="NAME=module:Class[:config]",
                    help="a submission; repeat for a field. One alone scores 1.0 on what it "
                         "completes, which is the self-assessment 7-A2 describes")
    sc.add_argument("--conditions", default=None, help="restrict to these, e.g. 1-1,1-3")
    sc.add_argument("--out", help="write the full scores as JSON here")
    sc.set_defaults(func=cmd_score)

    vs = sub.add_parser("validate-submission",
                        help="check a submission directory: manifest, files, size, and a "
                             "smoke run in a child process (6-R9, T1-RES-17)")
    vs.add_argument("dir", help="the submission directory (6-R10)")
    vs.add_argument("--course", help="course for the smoke run (default: tools/demo-course.json)")
    vs.add_argument("--ticks", type=int, default=200, help="ticks per condition in the smoke run")
    vs.add_argument("--conditions", default=None, help="restrict to these, e.g. 1-1,1-3")
    vs.add_argument("--out", help="write the report as JSON here")
    vs.set_defaults(func=_cmd_validate_submission)

    lb = sub.add_parser("leaderboard", help="a persistent leaderboard over one set")
    lbs = lb.add_subparsers(dest="board_cmd", required=True)
    li = lbs.add_parser("init", help="create a board bound to a set")
    li.add_argument("board", help="directory to create")
    li.add_argument("--set", required=True, help="the set's manifest.json or directory")
    li.add_argument("--conditions", default=None, help="restrict the board to these conditions")
    li.add_argument("--intake", default=None,
                    help="where accepted submissions are copied (default: $USVNAV_HOME/intake/<board>, "
                         "the data disk on the scoring machine, else beside the board; 6-R13)")
    li.add_argument("--runs", default=None,
                    help="where run records go (default: $USVNAV_HOME/runs/<board>, as above)")
    li.set_defaults(func=_cmd_board_init)
    lsub = lbs.add_parser("submit", help="validate a submission, copy it into the intake, run it "
                                          "over the board's set in child processes, store its raw "
                                          "items and re-score the field")
    lsub.add_argument("board")
    lsub.add_argument("dir", help="the submission directory")
    lsub.add_argument("--no-validate", action="store_true",
                      help="skip the validator (it is run first by default)")
    lsub.add_argument("--jobs", type=int, default=None,
                      help="process pairs run at once (default 2: two 8 GB submissions fit the card, 6-R1)")
    lsub.add_argument("--cap-ms", type=float, default=None,
                      help="the per-tick wall-clock cap (default: 220 ms, 6-R3)")
    lsub.set_defaults(func=_cmd_board_submit)
    lsh = lbs.add_parser("show", help="print the ranking, or one team's page")
    lsh.add_argument("board")
    lsh.add_argument("--team", help="print this team's own page instead of the ranking")
    lsh.set_defaults(func=_cmd_board_show)

    dc = sub.add_parser("docs", help="regenerate docs/ from the code (schema, dumps, rules, runtime)")
    dc.add_argument("--out", default=None, help="directory (default: the package's docs/)")
    dc.add_argument("--check", action="store_true",
                    help="exit 1 if the committed documents differ from what the code generates")
    dc.set_defaults(func=_cmd_docs)

    args = ap.parse_args(argv)
    return args.func(args)


def _cmd_view(args):
    from .figure import figure_course

    course = coursefile.load(args.course)
    result = None
    if args.trace:
        rows = json.loads(pathlib.Path(args.trace).read_text())
        result = type("Trace", (), {"trace": [(r["t"], r["x"], r["y"], r["psi"],
                                               r["v_cmd"], r["w_cmd"]) for r in rows]})()
    from . import look
    print("wrote", figure_course(course, args.out, result, span_m=args.span,
                                 noise=look.resolve_noise(args.noise)))
    return 0


def _cmd_studio(args):
    from .studio import cmd_studio
    return cmd_studio(args)


def _cmd_bundle(args):
    from .bundle import cmd_bundle
    return cmd_bundle(args)


def _cmd_validate_submission(args):
    from .submission import cmd_validate_submission
    return cmd_validate_submission(args)


def _cmd_board_init(args):
    from .board import cmd_init
    return cmd_init(args)


def _cmd_board_submit(args):
    from .board import cmd_submit
    return cmd_submit(args)


def _cmd_board_show(args):
    from .board import cmd_show
    return cmd_show(args)


def _cmd_docs(args):
    from .docs import cmd_docs
    return cmd_docs(args)


if __name__ == "__main__":
    raise SystemExit(main())
