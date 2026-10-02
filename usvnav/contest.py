"""Sets of episodes, and scoring a field of submissions over one -- the end-to-end path.

Two things the rest of the package did not have until 2026-09-10 (`DECISIONS.md` §9.5:
"finish the whole system before tuning it further"):

* **A set.** A directory of course files plus a `manifest.json` naming the episodes --
  which course, which seed for the dynamic parts, which conditions. `write_set` writes one
  from courses in hand and ships; `usvnav.internal.sets.make_set` generates the courses
  first and does not (7-A8). The **public practice set** is a few composite courses of
  both families (single pass and out-and-back) with anonymised ids and independent
  disturbance seeds (3-K3 as amended 2026-09-10), so nothing in it says what the course is
  made of. A **hidden-style** set's manifest also
  carries the measured axes and the generator seed; it is never published, so follow-up
  item 6 does not bind it. Course files themselves carry no hidden field either way.

* **Scoring a field.** `score_set` runs every submission on every episode of a set under
  its conditions and then applies 5-S1 exactly as `score.py` states it: per episode, each
  item divided by the best value among the submissions that *completed* that episode; a
  non-completion is 0 on all three; the per-episode ratios are averaged over the
  condition's episodes and weighted with 5-S6's `w_P`, `w_C`, `w_T`; the final score is
  the weighted mean of the condition scores, 20/20/30/30 (5-S8 as decided by the owner on
  2026-09-11). The same function is the leaderboard a
  team reads and the scorer that ranks them (7-A2), and a team running it alone gets its
  own raw items and a score of 1.0 on everything it completed -- which is what "no
  organiser reference figures" (5-S5) means in practice.

The runner is deliberately plain: a loop over episodes, conditions and agents, calling
`sim.run_episode` with the manifest's seed. Parallelism, per-tick wall-clock caps (6-R2,
6-R3) and containment (6-R1) are the scoring *infrastructure*, which is contest operations
and not decided; nothing here should be read as a statement about them.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import pathlib
import stat
import time

from . import coursefile
from .score import ITEMS, condition_score, episode_ratios, final_score

SET_FORMAT = "usvnav-set/1"
#: Read-side tags, as `coursefile.READ_FORMATS`: the frozen sets predate the 2026-09-15 rename.
READ_SET_FORMATS = (SET_FORMAT, "mallard-set/1")
CONDITIONS = ("1-1", "1-2", "1-3", "1-4")

#: 5-S6, decided by the owner 2026-09-10: the statement's original figures.
WEIGHTS = {"fuel": 0.3, "clearance": 0.4, "time": 0.3}   # 5-S6 (owner, 2026-09-25): control effort replaces path length

#: 5-S8, decided by the owner 2026-09-11: the final score weights the conditions -- the
#: two perception conditions that carry the harder capabilities (sparse range, and image
#: plus adaptation) at 30% each, the object list and the plain image at 20% each. Replaces
#: the plain mean of 2026-09-07. `score.final_score` renormalises over the conditions a
#: board actually scores.
CONDITION_WEIGHTS = {"1-1": 0.2, "1-2": 0.2, "1-3": 0.3, "1-4": 0.3}

#: 5-S13a, signed 2026-09-10: the lowest 5% of ticks, and a cap that is a guard rather than
#: the mechanism -- it cancels in 5-S1's ratio (§6.3). Passed to `run_episode`, which applies them.
CLEARANCE_PERCENTILE = 5.0
D_CAP = 10.0


# --------------------------------------------------------------------------- sets

def write_set(out_dir, episodes, *, kind: str = "public") -> dict:
    """Write a set from courses already in hand and return its manifest.

    `episodes` is a sequence of `(id, course, seed, conditions[, extra])`: the id names
    the course file (`courses/<id>.json`) and the episode; `seed` drives the dynamic parts
    (the 1-4 disturbance); `extra`, if given, is merged into the episode's manifest entry.
    Course files are written by `coursefile.save`, which carries no hidden field. This is
    the shipped half of set-making -- a team packages its own courses into a set with it,
    the studio does the same, and the organisers' `usvnav.internal.sets.make_set` calls it
    after generating the courses (7-A8: the generator itself does not ship).
    """
    out = pathlib.Path(out_dir)
    (out / "courses").mkdir(parents=True, exist_ok=True)
    entries = []
    for ep in episodes:
        ep_id, course, seed, conditions = ep[:4]
        extra = ep[4] if len(ep) > 4 else {}
        coursefile.save(course, out / "courses" / f"{ep_id}.json")
        entries.append({"id": ep_id, "course": f"courses/{ep_id}.json", "seed": int(seed),
                        "conditions": list(conditions), **extra})
    manifest = {"format": SET_FORMAT, "kind": kind, "episodes": entries}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def load_set(path) -> tuple[dict, pathlib.Path]:
    """`(manifest, root)`; `path` may be the manifest or the set's directory."""
    p = pathlib.Path(path)
    if p.is_dir():
        p = p / "manifest.json"
    manifest = json.loads(p.read_text())
    if manifest.get("format") not in READ_SET_FORMATS:
        raise ValueError(f"not a set manifest: format {manifest.get('format')!r}")
    return manifest, p.parent


# --------------------------------------------------------------------------- freezing

def set_digest(manifest: dict, root) -> dict:
    """`{"digest", "courses"}`: one SHA-256 over every episode's identity (id, course path,
    seed, conditions) and every course file's bytes, in manifest order, plus each course
    file's own hash. Whatever else the manifest carries -- measured axes, the gate's report,
    the freeze block itself -- is outside the digest, so annotating a frozen set does not
    unfreeze it and only a change to what is *scored* does."""
    root = pathlib.Path(root)
    h = hashlib.sha256()
    courses = {}
    for ep in manifest["episodes"]:
        data = (root / ep["course"]).read_bytes()
        courses[ep["id"]] = hashlib.sha256(data).hexdigest()
        h.update(json.dumps({k: ep[k] for k in ("id", "course", "seed", "conditions")},
                            sort_keys=True).encode())
        h.update(data)
    return {"digest": h.hexdigest(), "courses": courses}


def freeze_set(path) -> dict:
    """Stamp the set as frozen (T1-SCO-05: frozen, versioned, checksummed): write the digest
    into the manifest and make the course files read-only. Returns the `frozen` block.
    Freezing twice is a no-op if nothing changed and an error if something did."""
    manifest, root = load_set(path)
    d = set_digest(manifest, root)
    prior = manifest.get("frozen")
    if prior and prior["digest"] != d["digest"]:
        raise ValueError(f"the set was frozen at {prior['at']} with digest {prior['digest'][:12]}... "
                         f"and its episodes have changed since ({d['digest'][:12]}...)")
    manifest["frozen"] = prior or {
        "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "digest": d["digest"], "courses": d["courses"], "episodes": len(manifest["episodes"])}
    mpath = root / "manifest.json"
    if mpath.exists():
        os.chmod(mpath, stat.S_IRUSR | stat.S_IWUSR)
    mpath.write_text(json.dumps(manifest, indent=1) + "\n")
    ro = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    for ep in manifest["episodes"]:
        os.chmod(root / ep["course"], ro)
    os.chmod(mpath, ro)
    return manifest["frozen"]


def check_set(path) -> list[str]:
    """Problems with a set's integrity; empty means it is frozen and unchanged."""
    manifest, root = load_set(path)
    frozen = manifest.get("frozen")
    if not frozen:
        return ["the set is not frozen (no `frozen` block in the manifest)"]
    d = set_digest(manifest, root)
    out = []
    for ep_id, want in frozen.get("courses", {}).items():
        got = d["courses"].get(ep_id)
        if got is None:
            out.append(f"{ep_id}: episode missing since the freeze")
        elif got != want:
            out.append(f"{ep_id}: course file changed since the freeze")
    for ep_id in d["courses"]:
        if ep_id not in frozen.get("courses", {}):
            out.append(f"{ep_id}: episode added since the freeze")
    if d["digest"] != frozen["digest"] and not out:
        out.append("an episode's id, seed or conditions changed since the freeze")
    return out


# --------------------------------------------------------------------------- scoring

def run_field(manifest, root, agents: dict, *, conditions=None, log=None) -> dict:
    """Run every agent on every episode of the set; return the **raw items** only.

    `agents` maps a submission name to a zero-argument factory returning a fresh agent
    (6-R8: no state across episodes -- a fresh instance per episode is the simplest way to
    make that true). `conditions` restricts the manifest's conditions.

    Returns `{condition: {episode_id: {team: {outcome, tier, path, clearance, time, ticks,
    waypoints, of, wall_s}}}}`. Scoring is `score_raw`, kept apart so a persistent
    leaderboard (`usvnav.board`) can store raw items per team and re-score the whole
    field whenever a submission arrives -- 5-S1's "best completing submission" changes
    with every new entry, so the ratios cannot be stored, only the items can.
    """
    from .sim import run_episode

    raw: dict = {}
    for ep in manifest["episodes"]:
        course = coursefile.load(root / ep["course"])
        for cond in ep["conditions"]:
            if conditions and cond not in conditions:
                continue
            entries = {}
            for team, make in agents.items():
                t0 = time.perf_counter()
                res = run_episode(course, make(), condition=cond, seed=int(ep["seed"]),
                                  d_cap=D_CAP, clearance_percentile=CLEARANCE_PERCENTILE)
                entries[team] = {"outcome": res.outcome, "tier": res.tier,
                                 "path": float(res.distance), "fuel": float(res.fuel),
                                 "clearance": float(res.clearance), "time": float(res.ticks),
                                 "ticks": res.ticks, "waypoints": res.waypoints_reached,
                                 "of": course.n_waypoints,
                                 "wall_s": round(time.perf_counter() - t0, 2)}
                if log:
                    log(f"{cond} {ep['id']:<28} {team:<18} {res.outcome:<18} "
                        f"{res.ticks:>5} ticks  {res.waypoints_reached}/{course.n_waypoints}")
            raw.setdefault(cond, {})[ep["id"]] = entries
    return raw


def score_raw(raw: dict, weights=WEIGHTS, condition_weights=CONDITION_WEIGHTS) -> dict:
    """5-S1 / 5-S6 / 5-S8 over a field's raw items, exactly as `score.py` states them.

    `raw` is `run_field`'s shape. A team absent from an episode's entries (it was not run
    on it) scores 0 there, the same as a non-completion: the average in `condition_score`
    runs over all of the condition's episodes.
    """
    per_episode = {cond: {ep_id: episode_ratios(entries, weights) for ep_id, entries in eps.items()}
                   for cond, eps in raw.items()}
    condition = {cond: condition_score(list(eps.values()), weights)
                 for cond, eps in per_episode.items()}
    final = final_score(condition, condition_weights)
    return {"format": "usvnav-scores/1", "weights": dict(weights),
            "condition_weights": dict(condition_weights),
            "clearance_percentile": CLEARANCE_PERCENTILE, "d_cap": D_CAP,
            "raw": raw, "per_episode": per_episode, "condition": condition, "final": final}


def score_set(manifest, root, agents: dict, *, conditions=None, weights=WEIGHTS,
              condition_weights=CONDITION_WEIGHTS, log=None) -> dict:
    """Run every agent on every episode and score the field: `run_field` then `score_raw`.

    Returns `{"raw", "per_episode", "condition", "final", "weights", ...}` where `raw` is
    `run_field`'s items, `per_episode` the 5-S1 ratios in the same shape, `condition` the
    per-condition item means and totals per team, and `final` the 5-S8 weighted mean over conditions (`condition_weights`).
    """
    raw = run_field(manifest, root, agents, conditions=conditions, log=log)
    return score_raw(raw, weights, condition_weights)


def leaderboard(scores: dict) -> str:
    """The table a team reads: final score, per-condition totals, and -- for each team --
    its own per-episode scores (3-K8 discloses those to the team that earned them)."""
    conds = sorted(scores["condition"])
    teams = sorted(scores["final"], key=lambda t: -scores["final"][t])
    w = scores["weights"]
    cw = scores.get("condition_weights", {})
    lines = [f"weights  fuel {w['fuel']:.2f}  clearance {w['clearance']:.2f}  "
             f"time {w['time']:.2f}   (5-S6);  conditions "
             + "  ".join(f"{c} {cw.get(c, 1.0):.2f}" for c in conds) + "   (5-S8)",
             f"{'team':<20}{'final':>8}" + "".join(f"{c:>9}" for c in conds)]
    for t in teams:
        row = f"{t:<20}{100 * scores['final'][t]:>8.1f}"
        row += "".join(f"{100 * scores['condition'][c].get(t, {}).get('total', 0.0):>9.1f}"
                       for c in conds)
        lines.append(row)
    lines.append("")
    lines.append("per-episode scores, per team (0 = not completed; each item is the ratio "
                 "against the best completing submission on that episode)")
    for t in teams:
        lines.append(f"  {t}")
        for c in conds:
            for ep_id, ratios in scores["per_episode"][c].items():
                r = ratios.get(t, {i: 0.0 for i in ITEMS})
                total = sum(w[i] * r[i] for i in ITEMS)
                lines.append(f"    {c} {ep_id:<28} {100 * total:>6.1f}   "
                             + "  ".join(f"{i} {r[i]:.3f}" for i in ITEMS))
    return "\n".join(lines)
