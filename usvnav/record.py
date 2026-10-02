"""One run record format for the studio and the scorer (`usvnav-run/2`; T1-SCO-11, T1-SCO-17).

A run record holds what a replay needs and what a team wants to look at afterwards: the
course as it was run, the condition and the seed, every tick's pose, velocity, waypoint
index, *applied* action and clearance, the traffic's poses at every tick, and the result
with its cause. 3-K6's first tier -- feeding the recorded actions back into the simulator
reproduces the run -- rests on the course, the seed and the action column; everything else
is there so a person can see the run without replaying it (`studio.observe` rebuilds any
tick's observation from a frame, because perception is a pure function of state, 4-V15).

The studio wrote this format first (§9.9) and the scoring runner (`usvnav.runner`) writes
the same one, so a scored episode can be dropped into a studio's `runs/` and played back.
That is the whole of the audit replay tool for now: the record, and the studio reading it.
The scorer's records add a few fields of their own (`overrun_ticks`, `tick_ms`, `cap_ms`;
see `runner.py`), which the studio ignores.

Since 2026-09-21 each tick's `traffic` row lists the vessels that exist at that tick with
their sizes, because lanes add and remove vessels.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np

from .plant import DT

RUN_FORMAT = "usvnav-run/2"
#: Records a reader accepts. `/1` (before 2026-09-21) holds one pose per loop vessel per tick
#: in `traffic` and their sizes once in `traffic_dims`; `/2` holds, per tick, the vessels that
#: exist then -- lanes add and remove vessels -- each with its own size.
READ_RUN_FORMATS = (RUN_FORMAT, "usvnav-run/1")
#: One row per tick of a record's `frames`. `v_cmd`/`w_cmd` are the action *applied* on
#: that tick (after 1-W6's clipping, and the previous action on a 6-R2 overrun); the
#: terminal frame -- the pose after the step that ended the episode -- has none.
FRAME_FIELDS = ("tick", "x", "y", "psi", "u", "v", "r", "wp_index", "v_cmd", "w_cmd", "clearance")


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def traffic_at(course, t: float) -> list:
    """`[x, y, heading, length, width]` for every moving vessel that exists at `t`."""
    return [[round(r.x, 3), round(r.y, 3), round(float(r.heading), 4), r.length, r.width]
            for r in course.traffic_rects(t)]


class Recorder:
    """Collects the per-tick state through `run_episode`'s `on_observation` hook.

    The observation is the one source: what the agent saw is what gets recorded, taken
    from the same dict the agent receives, so the record cannot disagree with the run.
    `progress`, if given, is called with the tick every `progress_every` ticks (the
    studio's live counter)."""

    def __init__(self, progress=None, progress_every: int = 20):
        self.frames: list[list] = []
        self._progress = progress
        self._every = progress_every

    def on_observation(self, tick, meta, obs):
        self.frames.append([int(tick), *map(float, obs["pose"]), *map(float, obs["vel"]),
                            int(obs["wp_index"])])
        if self._progress is not None and tick % self._every == 0:
            self._progress(int(tick))


def result_dict(res, wall_s: float) -> dict:
    """The `result` block of a record from a `sim.Result`."""
    clear = res.per_tick_clearance
    finite = [(c, i) for i, c in enumerate(clear) if c is not None]
    cmin = min(finite) if finite else (None, None)
    return {"outcome": res.outcome, "tier": res.tier, "ticks": res.ticks,
            "elapsed": res.elapsed, "distance": res.distance, "clearance": res.clearance, "fuel": res.fuel,
            "waypoints_reached": res.waypoints_reached, "completed": res.completed,
            "detail": res.detail, "wall_s": wall_s,
            "ms_per_tick": 1000.0 * wall_s / max(res.ticks, 1),
            "clearance_min": cmin[0], "clearance_min_tick": cmin[1]}


def frames_of(res, recorder: Recorder) -> list[list]:
    """`FRAME_FIELDS` rows: the recorder's per-observation state joined with the trace's
    applied actions and the per-tick clearance, plus the terminal pose (which the trace
    does not hold -- its last row is the tick *before* the end)."""
    actions = {t: (v, w) for t, _, _, _, v, w in res.trace}
    clear = res.per_tick_clearance
    out = []
    for tick, x, y, psi, u, v, r, wp in recorder.frames:
        vc, wc = actions.get(tick, (None, None))
        c = float(clear[tick]) if tick < len(clear) else None
        out.append([tick, x, y, psi, u, v, r, wp, vc, wc, c])
    if res.pose and (not out or out[-1][0] < res.ticks):
        c = float(clear[res.ticks]) if res.ticks < len(clear) else None
        out.append([res.ticks, *map(float, res.pose), None, None, None,
                    res.waypoints_reached, None, None, c])
    return out


def episode_record(course, res, recorder: Recorder, wall_s: float) -> dict:
    """The run-dependent half of a record -- `result`, `frames`, `traffic`, `status`,
    `finished`. The caller adds what identifies the run (`id`, `course`, `agent`,
    `condition`, `seed`, `started`, `course_snapshot`, `n_waypoints`, `agent_team`,
    `agent_entry`) and `header()`'s fields."""
    frames = frames_of(res, recorder)
    return {"status": "done", "finished": now(), "result": result_dict(res, wall_s),
            "frames": frames, "traffic": [traffic_at(course, f[0] * DT) for f in frames]}


def failed_record(course, outcome: str, detail: str, wall_s: float = 0.0) -> dict:
    """A record for an episode that never started -- the agent could not be constructed
    (`crash`) or did not construct in time (`time_overrun`). One frame, at the start."""
    x0, y0, psi0 = course.start
    result = {"outcome": outcome, "tier": "submission_fault", "ticks": 0, "elapsed": 0.0,
              "distance": 0.0, "clearance": 0.0, "fuel": 0.0, "waypoints_reached": 0, "completed": False,
              "detail": detail, "wall_s": wall_s, "ms_per_tick": 0.0,
              "clearance_min": None, "clearance_min_tick": None}
    return {"status": "done", "finished": now(), "result": result,
            "frames": [[0, x0, y0, psi0, 0.0, 0.0, 0.0, 0, None, None, None]],
            "traffic": [traffic_at(course, 0.0)]}


def header() -> dict:
    return {"format": RUN_FORMAT, "frame_fields": list(FRAME_FIELDS)}


# --------------------------------------------------------------------------- replay

class _Replay:
    """An agent that answers every tick with the action the record says was applied."""

    def __init__(self, actions: dict):
        self.actions = actions

    def reset(self, meta):
        pass

    def act(self, obs):
        return np.array(self.actions[int(obs["t"])], dtype=float)


def replay(record: dict):
    """3-K6's first tier: feed the recorded actions back into the simulator and return the
    `sim.Result`. On the same code the trajectory is the record's, tick for tick --
    `replay_matches` checks that -- and that is what dispute resolution rests on
    (T1-SCO-17). The agent is not involved: a re-run of the *agent* may differ when a
    tick was late (6-R2), and the record says which ticks those were (`overrun_ticks`)."""
    from . import coursefile
    from .contest import CLEARANCE_PERCENTILE, D_CAP
    from .sim import run_episode

    course = coursefile.from_dict(record["course_snapshot"])
    actions = {f[0]: (f[8], f[9]) for f in record["frames"] if f[8] is not None}
    return run_episode(course, _Replay(actions), condition=record["condition"],
                       seed=int(record["seed"]), d_cap=D_CAP,
                       clearance_percentile=CLEARANCE_PERCENTILE, record_trace=True)


def replay_matches(record: dict, res) -> list[str]:
    """Where a replay's result differs from the record; empty means it reproduced it."""
    out = []
    r = record["result"]
    if res.outcome != r["outcome"]:
        out.append(f"outcome {res.outcome} != {r['outcome']}")
    if res.ticks != r["ticks"]:
        out.append(f"ticks {res.ticks} != {r['ticks']}")
    # A frame's pose is the observation's, which is float32 (4-V15); the replayed pose is
    # the plant's float64. Deterministic replay gives the same float64, hence the same
    # float32, so the comparison is exact at the record's precision.
    poses = [(x, y, psi) for _, x, y, psi, _, _ in res.trace]
    if res.pose:
        poses.append(tuple(map(float, res.pose)))
    recorded = [tuple(f[1:4]) for f in record["frames"]]
    if len(poses) != len(recorded):
        out.append(f"{len(poses)} poses replayed, {len(recorded)} recorded")
    for i, (a, b) in enumerate(zip(poses, recorded)):
        if tuple(np.float32(a)) != tuple(np.float32(b)):
            out.append(f"tick {i}: pose {a} != {b}")
            break
    return out
