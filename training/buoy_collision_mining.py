"""Add training buoys to a learner's saved successful physical trajectory.

The mined obstacle lies on the route the boat actually sailed, including its
lateral bypass. Replaying the original saved commands must finish the original
course and collide on the mined one. This is offline curriculum construction;
neither poses nor obstacle ground truth become policy observation features.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from usvnav.coursefile import from_dict, load, to_dict, validate
from usvnav.geometry import Circle
from usvnav.world import BUOY, StaticBody
from training.collect_perception import course_identity


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_trace(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a nonempty saved decision trace: {path}")
    actions, previous_tick = [], 0
    for row in rows:
        group = np.asarray(row.get("actions"), dtype=float)
        pose = np.asarray(row.get("pose"), dtype=float)
        tick = row.get("tick")
        if (group.ndim != 2 or group.shape[1] != 2 or not len(group)
                or not np.isfinite(group).all() or pose.shape != (3,)
                or not np.isfinite(pose).all() or not isinstance(tick, int)
                or tick != previous_tick + len(group)):
            raise ValueError(f"Trace must contain finite poses and contiguous physical tick commands: {path}")
        actions.extend(group.tolist())
        previous_tick = tick
    return rows, actions


def replay_saved_commands(course, actions, *, condition="1-2", seed=101, tick_limit=6000):
    """Replay exact saved commands through official dynamics, without RGB/PPO."""
    from training.navigation_env import NavigationEpisode

    episode = NavigationEpisode(course, condition=condition, seed=seed, tick_limit=tick_limit)
    for action in actions:
        if episode.outcome is not None:
            break
        episode.advance(action)
    return dict(outcome=episode.outcome, ticks=episode.tick,
                elapsed_s=episode.tick * .1, waypoints_reached=episode.wp_index,
                min_clearance_m=float(min(episode.per_tick_clearance)))


def _candidates(course, trace):
    start = np.asarray(course.start[:2], dtype=float)
    first_leg = course.waypoints[0] - start
    first_direction = first_leg / np.linalg.norm(first_leg)
    existing = [np.array([body.shape.x, body.shape.y], dtype=float)
                for body in course.bodies if body.cls == BUOY]
    first_obstacles = [float(np.dot(center - start, first_direction)) for center in existing
                       if 0. < np.dot(center - start, first_direction) < np.linalg.norm(first_leg)]
    candidates = []
    for row in trace:
        point = np.asarray(row["pose"][:2], dtype=float)
        # A millimetre of slack preserves the exclusion distances after the
        # official serializer rounds new buoy centres to four decimal places.
        if np.linalg.norm(point - start) < 12.001:
            continue
        if any(np.linalg.norm(point - target) <= radius + 3.001
               for target, radius in zip(course.waypoints, course.arrival_radii)):
            continue
        if any(np.linalg.norm(point - center) < 7.001 for center in existing):
            continue
        first = int(row.get("waypoints", 0)) == 0
        along = float(np.dot(point - start, first_direction))
        after_existing = first and (not first_obstacles or along > max(first_obstacles) + 3.)
        priority = 0 if after_existing else 1 if first else 2
        candidates.append(dict(tick=int(row["tick"]), point=point,
                               priority=priority, first_leg=first,
                               after_existing=bool(after_existing)))
    return candidates


def _select_points(candidates, add_count, variant, trace_hash):
    # Distinct timings on the same public route cannot be inferred from route
    # coordinates alone. A trace-dependent random index breaks fixed selection
    # tied to a route ID; variant 0/1 prefer earlier/later safe encounter times.
    rng = np.random.default_rng(int(trace_hash[:16], 16) ^ variant)
    selected = []
    while len(selected) < add_count:
        available = [candidate for candidate in candidates
                     if all(np.linalg.norm(candidate["point"] - old["point"]) >= 7.001
                            for old in selected)]
        if not available:
            return None
        priority = min(candidate["priority"] for candidate in available)
        group = sorted((candidate for candidate in available if candidate["priority"] == priority),
                       key=lambda candidate: candidate["tick"])
        fraction = .2 if variant == 0 else .8
        index = int(np.clip(round((len(group) - 1) * fraction) + rng.integers(-1, 2), 0, len(group) - 1))
        selected.append(group[index])
    return selected


def mine_buoy_courses(previous_run, output, *, add_count=1, max_courses=4):
    """Write fresh training-only courses mined from ``post-suite`` successes.

    Up to two distinct timings/layouts are mined per successful source route.
    Each saved course retains the original public geometry and static objects.
    The manifest records source hashes and genuine before/after command replay
    proof. Previous validation courses used for mining become training sources;
    future held-out evaluation must use fresh layouts.
    """
    if not isinstance(add_count, int) or not 1 <= add_count <= 4:
        raise ValueError("add_count must be an integer in [1, 4]")
    if not isinstance(max_courses, int) or max_courses < 1:
        raise ValueError("max_courses must be a positive integer")
    previous_run, output = Path(previous_run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Mining output must be a fresh directory: {output}")
    suite_path = previous_run / "post-suite.json"
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    config_path = previous_run / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    condition = config.get("condition", "1-2")
    tick_limit = int(config.get("tick_limit", 6000))
    course_directory = (previous_run / "courses").resolve()
    mined, seen = [], set()
    for suite_index, summary in enumerate(suite.get("rows", [])):
        if summary.get("official_outcome", summary.get("outcome")) != "goal":
            continue
        source_name = summary.get("course")
        if not isinstance(source_name, str):
            raise ValueError("Successful suite row is missing its source course filename")
        source_path = (course_directory / source_name).resolve()
        if not source_path.is_relative_to(course_directory):
            raise ValueError("Source course must be inside the previous run's courses directory")
        trace_path = previous_run / f"post-{suite_index:02d}-trace.json"
        source = load(source_path)
        trace, actions = _read_trace(trace_path)
        seed = int(summary.get("seed", 101))
        replay_before = replay_saved_commands(source, actions, condition=condition, seed=seed, tick_limit=tick_limit)
        if replay_before["outcome"] != "goal" or replay_before["ticks"] != len(actions):
            raise ValueError(f"Saved commands do not exactly replay the original goal: {source_name}")
        source_hash, trace_hash = course_identity(source), _sha256(trace_path)
        candidates = _candidates(source, trace)
        for variant in range(2):
            selected = _select_points(candidates, add_count, variant, trace_hash)
            if selected is None:
                continue
            added = [StaticBody(BUOY, Circle(float(point["point"][0]), float(point["point"][1]), .3))
                     for point in selected]
            course = from_dict(to_dict(replace(source, bodies=[*source.bodies, *added])))
            problems = validate(course)
            if problems:
                raise ValueError(f"Mined course violates official geometry rules: {problems}")
            identity = course_identity(course)
            if identity in seen:
                continue
            replay_after = replay_saved_commands(course, actions, condition=condition, seed=seed, tick_limit=tick_limit)
            if replay_after["outcome"] != "static_collision" or replay_after["ticks"] >= replay_before["ticks"]:
                raise ValueError("Added buoy did not cause a genuine earlier saved-command collision")
            seen.add(identity)
            name = f"mined-source{suite_index:02d}-variant{variant}-buoy{sum(body.cls == BUOY for body in course.bodies)}.json"
            evidence = dict(file=name, sha256=identity, count=sum(body.cls == BUOY for body in course.bodies),
                            added_count=add_count, kind="mined", stage=3,
                            source_run=str(previous_run), source_course=source_name,
                            source_phase="post", source_hash=source_hash,
                            source_course_file_sha256=_sha256(source_path),
                            source_trace=trace_path.name, source_trace_sha256=trace_hash,
                            source_suite_sha256=_sha256(suite_path), source_seed=seed,
                            candidate_ticks=[point["tick"] for point in selected],
                            candidate_points=[point["point"].tolist() for point in selected],
                            first_leg=[point["first_leg"] for point in selected],
                            after_existing_buoy=[point["after_existing"] for point in selected],
                            replay_before=replay_before, replay_after=replay_after, validation=[])
            mined.append((course, evidence))
            if len(mined) >= max_courses:
                break
        if len(mined) >= max_courses:
            break
    if not mined:
        raise ValueError("No replayable successful trajectory contains safe points for requested extra buoys")
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(train=[evidence for _, evidence in mined], val=[], metadata=dict(
        family="learner-path-buoy-collision-mining-v1", source_run=str(previous_run),
        intended_use="training-only hard negatives; never held-out validation",
        old_validation_reused_as_training=True, requires_fresh_validation_layouts=True,
        added_count=add_count, min_start_distance_m=12., min_existing_buoy_distance_m=7.,
        min_waypoint_disc_extra_m=3., buoy_diameter_m=.6))
    for course, evidence in mined:
        (output / evidence["file"]).write_text(json.dumps(to_dict(course), indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous_run", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--add-count", type=int, default=1)
    parser.add_argument("--max-courses", type=int, default=4)
    args = parser.parse_args()
    manifest = mine_buoy_courses(args.previous_run, args.out, add_count=args.add_count,
                                 max_courses=args.max_courses)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
