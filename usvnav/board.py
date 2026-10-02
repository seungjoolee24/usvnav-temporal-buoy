"""A persistent leaderboard over one set (3-K8, 5-S4, 5-S5, 5-S8, 5-S9, 6-R11).

`usvnav score` ranks one field in one call and stores nothing. A contest needs the other
thing: submissions arrive one at a time, and 5-S1's "best completing submission on this
episode" moves with every arrival, so a stored *score* is stale the moment the next team
submits. What can be stored is each submission's **raw items per episode** -- outcome, fuel,
clearance, time, waypoints -- and the field is re-scored from those whenever it changes.
That is what a board is: raw items per submission, plus the two documents rendered from them.

    board/
      board.json          the store: which set, which conditions, every submission's raw items
      leaderboard.md      the ranking -- final score (a 20/20/30/30 weighted mean, 5-S8) and
                          the four condition scores
      teams/<team>.md     that team's own page: per-episode scores (3-K8), the failure-cause
                          distribution (5-S4), submission faults on their own (5-S11), the
                          1-2 versus 1-4 drift diagnostic, and every submission it made

`submit` validates the directory first (T1-RES-17; a rejected submission is not run and
not stored), **copies it into the intake** as a read-only directory (6-R5's read-only bundle;
the copy is what runs and what the record names, so a team editing its directory afterwards
changes nothing), runs it over the board's set through the scoring runner -- the agent in a
child process, one process pair per condition, the per-tick cap at the boundary
(`usvnav.runner`; 6-R5, 6-R7b, 6-R2), the simulator loop being the same `sim.run_episode`
as `usvnav score` (7-A2) -- stores the raw items as one more submission of the manifest's
team, re-scores the field and rewrites both documents. Every episode's run record is under
the board's `runs` directory (T1-SCO-11, T1-SCO-17).

**Where things go (6-R13).** A board names its `intake` and `runs` directories at `init`. The
defaults are `<home>/intake/<board name>` and `<home>/runs/<board name>`, where `<home>` is
`$USVNAV_HOME` or, on the scoring machine, `/data/hackaton/track_1`; with neither, beside the
board. Submissions and run records are the large things and must not land on the internal disk.

**A team's best accepted submission is the one that counts (6-R11, owner 2026-09-10).**
Because 5-S1 is relative, a submission has no score of its own, so "best" is defined against
a field: the counting submission is the one with the highest final score when the other
teams' counting submissions form the field. `select_counting` finds that by iteration --
start from every team's latest, re-pick each team's best against the others' picks, repeat
until nothing moves. A team's own submissions keep their order under almost any field, so
this settles in a pass or two; among a team's equal scores the earlier one counts. 5-S9's
tie-break timestamp is the counting submission's.

The ranking is 5-S9's chain: full-precision final score, then total waypoints reached
across every episode of every condition, then the earlier submission. Ties happen at the
bottom -- every team that completed nothing scores exactly 0 -- and the chain orders them.

What the board does not do, on purpose: run two submissions at once, or limit submissions
per day. Two `submit`s racing on one board would both rewrite `board.json`; an operator runs
them one at a time, and the rate limit is an operations value (6-R11).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import shutil
import stat
import time

from . import runner as _runner
from . import runtime as _runtime
from . import submission as _sub
from .contest import CONDITION_WEIGHTS, WEIGHTS, load_set, score_raw, set_digest
from .score import ITEMS, tie_break_key
from .sim import NAVIGATIONAL, SUBMISSION_FAULT

BOARD_FORMAT = "usvnav-board/3"
BOARD_FILE = "board.json"
MAX_SELECT_PASSES = 10


class BoardError(Exception):
    pass


# --------------------------------------------------------------------------- store

def _now():
    t = time.time()
    return _dt.datetime.fromtimestamp(t, _dt.timezone.utc).isoformat(timespec="seconds"), t


def _home_for(board_dir: pathlib.Path, kind: str) -> pathlib.Path:
    home = _runner.data_home()
    return (home / kind / board_dir.name) if home else (board_dir / kind)


def init(board_dir, set_path, *, conditions=None, intake_dir=None, runs_dir=None) -> dict:
    board_dir = pathlib.Path(board_dir).resolve()
    if (board_dir / BOARD_FILE).exists():
        raise BoardError(f"{board_dir / BOARD_FILE} exists already")
    manifest, root = load_set(set_path)
    conds = list(conditions) if conditions else None
    if conds:
        known = {c for ep in manifest["episodes"] for c in ep["conditions"]}
        bad = [c for c in conds if c not in known]
        if bad:
            raise BoardError(f"the set has no episodes under {bad}")
    stamp, _ = _now()
    intake_dir = pathlib.Path(intake_dir).resolve() if intake_dir else _home_for(board_dir, "intake")
    runs_dir = pathlib.Path(runs_dir).resolve() if runs_dir else _home_for(board_dir, "runs")
    board = {"format": BOARD_FORMAT, "created_at": stamp,
             "set": str((root / "manifest.json").resolve()), "set_kind": manifest["kind"],
             "set_digest": set_digest(manifest, root)["digest"],
             "set_frozen": bool(manifest.get("frozen")),
             "conditions": conds, "weights": dict(WEIGHTS),
             "condition_weights": dict(CONDITION_WEIGHTS),
             "intake": str(intake_dir), "runs": str(runs_dir), "teams": {}}
    intake_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    board_dir.mkdir(parents=True, exist_ok=True)
    (board_dir / "teams").mkdir(exist_ok=True)
    save(board_dir, board)
    _render_all(board_dir, board)
    return board


def load(board_dir) -> dict:
    p = pathlib.Path(board_dir) / BOARD_FILE
    if not p.is_file():
        raise BoardError(f"{p} is not a board (run `usvnav leaderboard init` first)")
    board = json.loads(p.read_text())
    if board.get("format") != BOARD_FORMAT:
        raise BoardError(f"{p}: format {board.get('format')!r} is not {BOARD_FORMAT!r}")
    return board


def save(board_dir, board: dict):
    (pathlib.Path(board_dir) / BOARD_FILE).write_text(json.dumps(board, indent=1) + "\n")


def check_set_unchanged(board: dict) -> None:
    """3-K8, T1-SCO-05/06: one set, the same set, for every submission the board ranks. The
    digest taken at `init` must still be the set's; otherwise two teams were scored on
    different episodes and the board is not a ranking."""
    manifest, root = load_set(board["set"])
    now = set_digest(manifest, root)["digest"]
    if now != board["set_digest"]:
        raise BoardError(f"the set at {board['set']} has changed since the board was created "
                         f"(digest {board['set_digest'][:12]}... then, {now[:12]}... now); a board "
                         f"ranks one set -- make a new board for a new set")


def counting(board: dict, team: str) -> dict:
    """The team's counting submission record."""
    rec = board["teams"][team]
    by_id = {s["id"]: s for s in rec["submissions"]}
    return by_id[rec["counting"]]


# --------------------------------------------------------------------------- scoring

def _merge(raw_by_team: dict) -> dict:
    """`{team: raw}` -> `run_field`'s shape `raw[cond][ep][team]`."""
    raw: dict = {}
    for team, own in raw_by_team.items():
        for cond, eps in own.items():
            for ep_id, entry in eps.items():
                raw.setdefault(cond, {}).setdefault(ep_id, {})[team] = entry
    return raw


def field_raw(board: dict, picks: dict | None = None) -> dict:
    """The field's raw items under `picks` (`{team: submission id}`; default: the stored
    counting picks)."""
    by_team = {}
    for team, rec in board["teams"].items():
        sid = (picks or {}).get(team, rec["counting"])
        by_team[team] = next(s for s in rec["submissions"] if s["id"] == sid)["raw"]
    return _merge(by_team)


def submission_score(board: dict, team: str, sid: str, picks: dict) -> float:
    """`team`'s final score if its submission `sid` stood against the others' picks."""
    by_team = {}
    for other, rec in board["teams"].items():
        chosen = sid if other == team else picks.get(other, rec["counting"])
        by_team[other] = next(s for s in rec["submissions"] if s["id"] == chosen)["raw"]
    return score_raw(_merge(by_team), board.get("weights", WEIGHTS),
                     board.get("condition_weights", CONDITION_WEIGHTS))["final"].get(team, 0.0)


def select_counting(board: dict) -> dict:
    """6-R11: each team's best submission against the others' best, to a fixed point."""
    picks = {t: rec["counting"] for t, rec in board["teams"].items()}
    for _ in range(MAX_SELECT_PASSES):
        changed = False
        for team, rec in board["teams"].items():
            best = None
            for s in rec["submissions"]:
                score = submission_score(board, team, s["id"], picks)
                key = (-score, s["submitted_epoch"])          # highest score, then earlier
                if best is None or key < best[0]:
                    best = (key, s["id"])
            if best[1] != picks[team]:
                picks[team] = best[1]
                changed = True
        if not changed:
            break
    return picks


def rescore(board: dict) -> dict:
    """Pick the counting submissions (writing the picks into `board`) and score the field."""
    picks = select_counting(board)
    for team, sid in picks.items():
        board["teams"][team]["counting"] = sid
    return score_raw(field_raw(board, picks), board.get("weights", WEIGHTS),
                     board.get("condition_weights", CONDITION_WEIGHTS))


def _waypoints(raw: dict) -> int:
    return sum(e["waypoints"] for eps in raw.values() for e in eps.values())


def _completed(raw: dict) -> int:
    return sum(e["outcome"] == "goal" for eps in raw.values() for e in eps.values())


def ranking(board: dict, scores: dict) -> list[dict]:
    rows = []
    for team, rec in board["teams"].items():
        c = counting(board, team)
        final = scores["final"].get(team, 0.0)
        rows.append({"team": team, "final": final, "waypoints": _waypoints(c["raw"]),
                     "submitted_at": c["submitted_at"], "epoch": c["submitted_epoch"],
                     "counting": c["id"], "submissions": len(rec["submissions"]),
                     "condition": {k: scores["condition"][k].get(team, {}).get("total", 0.0)
                                   for k in sorted(scores["condition"])},
                     "completed": _completed(c["raw"]),
                     "episodes": sum(len(eps) for eps in c["raw"].values())})
    rows.sort(key=lambda r: tie_break_key(r["team"], r["final"], r["waypoints"], r["epoch"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


# --------------------------------------------------------------------------- intake

def intake(src, dst) -> pathlib.Path:
    """Copy a submission into the intake as the frozen, read-only directory the runner
    executes (6-R5: a read-only bundle; 6-R13: on the data disk). `__pycache__/`, compiled
    files and `.git/` are left behind: the size cap leaves them uncounted (6-R9a), the hub's
    submission hash skips them, and a stale `.pyc` beside a `.py` would otherwise be the code
    that actually runs. A leftover at
    `dst` from an interrupted submit is replaced -- `board.json` is the record of what
    counts, and a directory it does not name is nothing."""
    src, dst = pathlib.Path(src).resolve(), pathlib.Path(dst).resolve()
    if dst.exists():
        release(dst)
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", ".git", "*.pyc", "*.pyo"))
    for dirpath, dirnames, filenames in os.walk(dst, topdown=False):
        for f in filenames:
            os.chmod(pathlib.Path(dirpath) / f, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.chmod(dirpath, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    return dst


def release(path) -> None:
    """Make an intake copy writable again (to delete it)."""
    for dirpath, dirnames, filenames in os.walk(path):
        os.chmod(dirpath, stat.S_IRWXU)
        for f in filenames:
            os.chmod(pathlib.Path(dirpath) / f, stat.S_IRUSR | stat.S_IWUSR)


# --------------------------------------------------------------------------- submit

def submit(board_dir, submission_dir, *, validate_first: bool = True, log=None,
           validate_kwargs=None, jobs: int = _runner.DEFAULT_JOBS, runner_kwargs=None) -> dict:
    """Validate, take into the intake, run through the scoring runner, store as one more
    submission, re-score, render. Returns `{"accepted", "team", "submission", "validation",
    "scores", "rank", "of", "counting", "runs"}`; a rejected submission changes nothing on
    disk. `runner_kwargs` reach `runner.score_submission` (`cap_ms` and the rest)."""
    board_dir = pathlib.Path(board_dir)
    board = load(board_dir)
    sub_dir = pathlib.Path(submission_dir).resolve()
    check_set_unchanged(board)

    validation = None
    if validate_first:
        validation = _sub.validate(sub_dir, **(validate_kwargs or {}))
        if log:
            log(_sub.render(validation))
        if not validation["ok"]:
            return {"accepted": False, "team": validation["team"], "submission": None,
                    "validation": validation, "scores": None, "rank": None, "of": None,
                    "counting": None, "runs": None}

    team = _sub.load_manifest(sub_dir)["team"]
    rec = board["teams"].get(team, {"submissions": [], "counting": None})
    sid = f"s{len(rec['submissions']) + 1:02d}"
    frozen = intake(sub_dir, pathlib.Path(board["intake"]) / team / sid)
    runs = pathlib.Path(board["runs"]) / team / sid
    if log:
        log(f"intake {frozen}\nruns   {runs}")
    out = _runner.score_submission(frozen, board["set"], runs, conditions=board["conditions"],
                                   jobs=jobs, log=log, **(runner_kwargs or {}))
    own = {cond: {ep_id: entries[team] for ep_id, entries in eps.items()}
           for cond, eps in out["raw"].items()}
    stamp, epoch = _now()
    rec = board["teams"].setdefault(team, rec)
    rec["submissions"].append({
        "id": sid, "submitted_at": stamp, "submitted_epoch": epoch, "dir": str(frozen),
        "source": str(sub_dir), "runs": str(runs), "raw": own,
        "restarts": sum(p["restarts"] for p in out["pairs"].values()),
        "validation": None if validation is None else
        {"ok": validation["ok"],
         "problems": [p for p in validation["problems"] if p["severity"] != "ok"]}})
    if rec["counting"] is None:
        rec["counting"] = sid
    board["stale"] = [t for t in board.get("stale", []) if t != team]     # scored here now
    scores = _render_all(board_dir, board)          # picks the counting ones and saves
    rows = ranking(board, scores)
    me = next(r for r in rows if r["team"] == team)
    return {"accepted": True, "team": team, "submission": sid, "validation": validation,
            "scores": scores, "rank": me["rank"], "of": len(rows), "counting": me["counting"],
            "runs": str(runs)}


# --------------------------------------------------------------------------- render

def _render_all(board_dir: pathlib.Path, board: dict) -> dict:
    scores = rescore(board)
    save(board_dir, board)
    (board_dir / "leaderboard.md").write_text(render_ranking(board, scores) + "\n")
    (board_dir / "teams").mkdir(exist_ok=True)
    for team in board["teams"]:
        (board_dir / "teams" / f"{team}.md").write_text(render_team(board, scores, team) + "\n")
    return scores


def render_ranking(board: dict, scores: dict) -> str:
    rows = ranking(board, scores)
    conds = sorted(scores["condition"]) if scores["condition"] else (board["conditions"] or [])
    # The board carries no rule text (owner, 2026-09-13, all tracks; 5-S14): the rules live on
    # the problem page, and a second copy here would only drift from it. The head is the same
    # on every track's board: the title alone, one line with the time of posting (ISO with the
    # offset; the hub renders it as KST), one line pointing at the statement's evaluation
    # section, then the table; a rule and one footnote line carry the set, the conditions, the
    # team count, the x100 note and the scoring environment.
    posted = _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")
    out = ["# Track 1 리더보드", "",
           f"게시 {posted}.", "",
           "점수·순위·동점 규칙은 문제 페이지의 '평가 기준: 완주와 상대점수' 절에 있습니다.", ""]
    head = ("| rank | team | final | " + " | ".join(conds)
            + " | completed | counting | submitted |")
    out.append(head)
    out.append("|" + "---|" * (head.count("|") - 1))
    for r in rows:
        out.append(f"| {r['rank']} | {r['team']} | {100 * r['final']:.1f} | "
                   + " | ".join(f"{100 * r['condition'].get(c, 0.0):.1f}" for c in conds)
                   + f" | {r['completed']}/{r['episodes']} | {r['counting']} of "
                     f"{r['submissions']} | {r['submitted_at']} |")
    if not rows:
        out.append("| - | *(no submission yet)* | | " + " | ".join("" for _ in conds) + " | | | |")
    # Teams carried over from a previous set without a submission scored on this one (owner,
    # 2026-09-26: the old scores are not shown; a team is listed here until it submits again).
    stale = [t for t in board.get("stale", []) if t not in board["teams"]]
    if stale:
        out += ["", "**다시 제출이 필요한 팀**: " + ", ".join(f"`{t}`" for t in sorted(stale))
                + " — 이전 채점 셋에 낸 제출물은 이 셋에서 채점되지 않았습니다. 새로 제출하면 채점됩니다."]
    R = _runtime.RUNTIME
    env = ", ".join([f"Python {R['python']}"] + [f"{k} {v}" for k, v in R["packages"].items()])
    out += ["", "---", "",
            f"세트 `{pathlib.Path(board['set']).parent.name}` ({board['set_kind']}"
            f"{', frozen' if board.get('set_frozen') else ''}), 조건 "
            f"{'·'.join(conds) if conds else '세트의 전체'}, {len(rows)}팀. 표의 점수는 ×100. "
            f"채점 환경 `{R['id']}` ({env})."]
    return "\n".join(out)


def render_team(board: dict, scores: dict, team: str) -> str:
    rec = board["teams"][team]
    c = counting(board, team)
    raw = c["raw"]
    rows = ranking(board, scores)
    me = next(r for r in rows if r["team"] == team)
    conds = sorted(raw)
    w = board.get("weights", WEIGHTS)
    picks = {t: r["counting"] for t, r in board["teams"].items()}
    out = [f"# {team}", "",
           f"Rank {me['rank']} of {len(rows)}; final {100 * me['final']:.1f}; counting "
           f"submission `{c['id']}` of {len(rec['submissions'])}, submitted {c['submitted_at']}; "
           f"{me['completed']} of {me['episodes']} episodes completed, {me['waypoints']} "
           f"waypoints reached in all.", ""]

    out += ["## Submissions", "",
            "Each row's score is what that submission would score against the field as it stands now; "
            "the counting one is marked.", "",
            "| id | submitted | score | completed | counting |", "|---|---|---|---|---|"]
    for s in rec["submissions"]:
        sc = submission_score(board, team, s["id"], picks)
        out.append(f"| {s['id']} | {s['submitted_at']} | {100 * sc:.1f} | "
                   f"{_completed(s['raw'])}/{sum(len(e) for e in s['raw'].values())} | "
                   f"{'**yes**' if s['id'] == c['id'] else ''} |")

    out += ["", "## Condition scores", "",
            "| condition | score | path | clearance | time | completed |", "|---|---|---|---|---|---|"]
    for k in conds:
        cs = scores["condition"].get(k, {}).get(team, {i: 0.0 for i in ITEMS} | {"total": 0.0})
        done = sum(e["outcome"] == "goal" for e in raw[k].values())
        out.append(f"| {k} | {100 * cs['total']:.1f} | "
                   + " | ".join(f"{100 * cs[i]:.1f}" for i in ITEMS)
                   + f" | {done}/{len(raw[k])} |")

    out += ["", "## Per-episode scores (3-K8)", "",
            "Each item is your value against the best completing submission on that "
            "episode: path and time are best/yours, clearance is yours/best; 1.000 is best "
            "in the field, 0 is a non-completion.", "",
            "| condition | episode | score | path | clearance | time | outcome | ticks | waypoints | late ticks |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for k in conds:
        for ep_id, e in raw[k].items():
            r = scores["per_episode"][k][ep_id].get(team, {i: 0.0 for i in ITEMS})
            total = sum(w[i] * r[i] for i in ITEMS)
            out.append(f"| {k} | {ep_id} | {100 * total:.1f} | "
                       + " | ".join(f"{r[i]:.3f}" for i in ITEMS)
                       + f" | {e['outcome']} | {e['ticks']} | {e['waypoints']}/{e['of']} | "
                         f"{e.get('overruns', 0)} |")
    out += ["", "A late tick (past the per-tick cap, 6-R3) applied your previous action and is "
                "counted here; it never changes a score (6-R2)."]

    out += ["", "## Failure causes (5-S4)", "",
            "Navigational outcomes only; a submission fault is a bug, not a weakness, and is "
            "listed separately below (5-S11).", "",
            "| condition | " + " | ".join(NAVIGATIONAL) + " |",
            "|---|" + "---|" * len(NAVIGATIONAL)]
    for k in conds:
        counts = {o: 0 for o in NAVIGATIONAL}
        for e in raw[k].values():
            if e["outcome"] in counts:
                counts[e["outcome"]] += 1
        out.append(f"| {k} | " + " | ".join(str(counts[o]) for o in NAVIGATIONAL) + " |")

    faults = [(k, ep_id, e["outcome"]) for k in conds for ep_id, e in raw[k].items()
              if e["outcome"] in SUBMISSION_FAULT]
    out += ["", "## Submission faults (5-S11)", ""]
    out += [f"- {k} {ep_id}: `{o}`" for k, ep_id, o in faults] if faults else ["None."]

    if "1-2" in conds and "1-4" in conds:
        d12 = sum(e["outcome"] == "goal" for e in raw["1-2"].values())
        d14 = sum(e["outcome"] == "goal" for e in raw["1-4"].values())
        out += ["", "## Drift degradation (5-S4)", "",
                f"Completed {d12} of {len(raw['1-2'])} under 1-2 and {d14} of "
                f"{len(raw['1-4'])} under 1-4 on the same courses; the difference is "
                f"what the disturbance costs your policy."]
    return "\n".join(out)


# --------------------------------------------------------------------------- CLI

def cmd_init(args) -> int:
    conds = tuple(args.conditions.split(",")) if args.conditions else None
    board = init(args.board, args.set, conditions=conds, intake_dir=args.intake, runs_dir=args.runs)
    print(f"board {args.board}: set {board['set']} ({board['set_kind']}"
          f"{', frozen' if board['set_frozen'] else ', NOT frozen -- freeze it before the contest'}), "
          f"conditions {board['conditions'] or 'all'}\n  intake {board['intake']}\n  runs   {board['runs']}")
    return 0


def cmd_submit(args) -> int:
    kw = {}
    if getattr(args, "cap_ms", None) is not None:
        kw["cap_ms"] = float(args.cap_ms)
    result = submit(args.board, args.dir, validate_first=not args.no_validate, log=print,
                    jobs=getattr(args, "jobs", None) or _runner.DEFAULT_JOBS, runner_kwargs=kw)
    if not result["accepted"]:
        print("\nrejected: not run, not stored")
        return 1
    board = load(args.board)
    print()
    print(render_ranking(board, result["scores"]))
    print(f"\n{result['team']}: stored as {result['submission']}, counting {result['counting']}; "
          f"rank {result['rank']} of {result['of']}; "
          f"page {pathlib.Path(args.board) / 'teams' / (result['team'] + '.md')}")
    return 0


def cmd_show(args) -> int:
    board = load(args.board)
    scores = rescore(board)
    if args.team:
        if args.team not in board["teams"]:
            raise SystemExit(f"no submission from {args.team!r}")
        print(render_team(board, scores, args.team))
    else:
        print(render_ranking(board, scores))
    return 0
