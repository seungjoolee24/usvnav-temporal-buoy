"""Short physically continuous sensor clips; privileged labels stay offline.

These safe observer probes are not expert demonstrations or scored missions.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from usvnav.collide import CourseIndex
from usvnav.coursefile import load, to_dict
from usvnav.look import world_coords
from usvnav.plant import DT, Vessel
from usvnav.render import TRACK1, class_map, class_names, render, scene_from_course
from usvnav.world import MOORED
from training.collect_perception import ROOT, course_identity, read_split_plan
from training.perception_schema import DETAIL_CLASSES, labels_from_raster
from training.tracking import world_to_pixel


def ship_states(course, t):
    """Stable simulator IDs even when open-lane vessels enter or leave the scene."""
    states = {}
    for i, body in enumerate(course.bodies):
        if body.cls == MOORED:
            states[f"static:{i}"] = (0, body.shape, np.zeros(2))
    for i, route in enumerate(course.traffic):
        states[f"traffic:{i}"] = (1, route.rect_at(t), route.velocity_at(t))
    for i, lane in enumerate(course.lanes):
        for (k, _), (rect, velocity) in zip(lane.alive(t), lane.moving_at(t), strict=True):
            states[f"lane:{i}:{k}"] = (1, rect, np.asarray(velocity))
    return states


def fully_visible(rect, pose):
    corners = world_to_pixel(rect.corners(), pose)
    return bool(np.all((corners >= 0.0) & (corners <= 200.0)))


def instance_map(states, id_lookup, pose, semantic):
    """Pixel-centre rectangles, with official class-map painter precedence."""
    x, y = world_coords(TRACK1, pose, 200, 200)
    instances = np.full((200, 200), -1, dtype=np.int16)
    for identity, (_, rect, _) in states.items():
        u, v = rect.axes()
        dx, dy = x - rect.x, y - rect.y
        mask = ((np.abs(dx * u[0] + dy * u[1]) <= rect.length / 2)
                & (np.abs(dx * v[0] + dy * v[1]) <= rect.width / 2))
        instances[mask] = id_lookup[identity]
    vessel = semantic == DETAIL_CLASSES.index("vessel")
    instances[~vessel] = -1
    if np.any(vessel & (instances < 0)):
        raise RuntimeError("Simulator vessel pixels missing from instance truth")
    return instances


def continuous_states(course, initial, start_tick, frames, action, min_clearance=.5):
    """Use the official dynamics and full-hull safety at every tick, no teleporting."""
    vessel, index = Vessel(*initial), CourseIndex(course)
    poses, velocities, clearances = [], [], []
    for f in range(frames):
        clearance, hit, inside = index.tick(vessel.hull(), (start_tick + f) * DT, cap=10.0)
        if hit is not None or not inside or clearance < min_clearance:
            return None
        poses.append([vessel.x, vessel.y, vessel.psi])
        velocities.append([vessel.u, vessel.v, vessel.r])
        clearances.append(clearance)
        if f + 1 < frames:
            vessel.step(action)
    return np.asarray(poses), np.asarray(velocities), np.asarray(clearances)


def sample_clip(course, rng, *, frames, profile, min_clearance, max_tick):
    action = np.array([0.0, 0.0]) if profile == "stationary" else np.array([.6, rng.choice([-.12, .12])])
    low, high = np.min(course.boundary, axis=0), np.max(course.boundary, axis=0)
    for attempt in range(1, 10001):
        tick = int(rng.integers(0, max_tick - frames + 2))
        ships = ship_states(course, tick * DT)
        # Anchor each profile near its intended motion type when available.
        preferred = [r for kind, r, _ in ships.values() if kind == int(profile != "stationary")]
        anchors = preferred or [r for _, r, _ in ships.values()]
        if anchors:
            anchor = anchors[int(rng.integers(len(anchors)))]
            angle, radius = rng.uniform(-math.pi, math.pi), rng.uniform(10.0, 25.0)
            x, y = anchor.x + radius * math.cos(angle), anchor.y + radius * math.sin(angle)
        else:
            x, y = rng.uniform(low, high)
        pose = (x, y, rng.uniform(-math.pi, math.pi))
        states = continuous_states(course, pose, tick, frames, action, min_clearance)
        if states is None:
            continue
        if not any(fully_visible(r, pose) for _, r, _ in ships.values()):
            continue
        return tick, action, states, attempt
    raise RuntimeError("Could not sample a safe continuous clip")


def build_clip(course, tick, action, states):
    poses, velocities, clearances = states
    ticks = np.arange(tick, tick + len(poses), dtype=np.int32)
    truths = [ship_states(course, int(k) * DT) for k in ticks]
    identities = sorted({identity for state in truths for identity in state})
    if len(identities) >= 32767:
        raise ValueError("Too many instances for int16 label storage")
    lookup = {identity: i for i, identity in enumerate(identities)}
    f, n = len(poses), len(identities)
    gt_kind, gt_state = np.zeros(n, np.int8), np.zeros((f, n, 7), np.float64)
    gt_alive, gt_visible, gt_full = (np.zeros((f, n), bool) for _ in range(3))
    rgbs, semantics, instances = [], [], []
    for j, (pose, k, truth) in enumerate(zip(poses, ticks, truths)):
        scene = scene_from_course(course, Vessel(*pose), int(k) * DT)
        semantic = labels_from_raster(class_map(scene, TRACK1), class_names(TRACK1), scheme="detail")
        instance = instance_map(truth, lookup, pose, semantic)
        for identity, (kind, rect, velocity) in truth.items():
            i = lookup[identity]
            gt_kind[i], gt_alive[j, i] = kind, True
            gt_state[j, i] = [rect.x, rect.y, rect.heading, rect.length, rect.width, *velocity]
            gt_visible[j, i] = np.count_nonzero(instance == i) >= 6
            gt_full[j, i] = gt_visible[j, i] and fully_visible(rect, pose)
        rgbs.append(render(scene, TRACK1))
        semantics.append(semantic)
        instances.append(instance)
    return dict(rgb=np.stack(rgbs), semantic=np.stack(semantics), gt_instance=np.stack(instances),
                pose=poses.astype(np.float32), pose_exact=poses, vel=velocities.astype(np.float32), tick=ticks,
                actions=np.tile(action.astype(np.float32), (f - 1, 1)), clearance_m=clearances.astype(np.float32),
                gt_ids=np.array(identities, dtype="U64"), gt_kind=gt_kind, gt_state=gt_state,
                gt_alive=gt_alive, gt_visible=gt_visible, gt_fully_visible=gt_full)


def collect(plan, output, *, frames=41, clips_per_course=2, seed=43, min_clearance=.5, max_tick=6000):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if frames < 21 or clips_per_course < 2 or max_tick < frames - 1 or seed < 0 or not 0 <= min_clearance < 10:
        raise ValueError("Need >=21 frames, >=2 clips, valid seed/clearance/time horizon")
    assignments = read_split_plan(plan)
    courses = [(source, split, load(source)) for source, split in assignments.items()]
    hashes = [course_identity(course) for _, _, course in courses]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate course content in split plan")
    output.mkdir(parents=True)
    (output / "courses").mkdir()
    manifest = dict(format="usvnav-tracking/1", condition="1-2", seed=seed, dt_s=DT,
                    sample_kind="continuous_safe_observer_probes", frames_per_clip=frames,
                    clips_per_course=clips_per_course, classes=list(DETAIL_CLASSES), min_clearance_m=min_clearance,
                    max_tick=max_tick, input_fields=["rgb", "pose", "vel", "tick"],
                    privileged_fields=["semantic", "gt_instance", "gt_ids", "gt_kind", "gt_state",
                                       "gt_alive", "gt_visible", "gt_fully_visible"],
                    split_policy="Same explicit whole-course split as perception; duplicate contents rejected",
                    limitations="Short probes, no optimal driving actions, no mission score, no 1-4 disturbance",
                    courses=[], clips=[])
    started = time.perf_counter()
    for (source, split, course), identity in zip(courses, hashes):
        slug = f"{source.stem}-{identity[:8]}"
        snapshot = f"courses/{slug}.json"
        (output / snapshot).write_text(json.dumps(to_dict(course), indent=2), encoding="utf-8")
        manifest["courses"].append(dict(course_id=slug, split=split, sha256=identity, source=str(source), snapshot=snapshot))
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(identity[:16], 16)]))
        for j in range(clips_per_course):
            profile = "stationary" if j % 2 == 0 else "gentle_turn"
            tick, action, states, attempts = sample_clip(course, rng, frames=frames, profile=profile,
                                                        min_clearance=min_clearance, max_tick=max_tick)
            clip = build_clip(course, tick, action, states)
            name = f"{slug}-{j:02d}-{profile}"
            filename = name + ".npz"
            np.savez_compressed(output / filename, **clip)
            manifest["clips"].append(dict(clip_id=name, course_id=slug, split=split, path=filename,
                                          profile=profile, frames=frames, start_tick=tick, action=action.tolist(),
                                          attempts=attempts, min_clearance_m=float(clip["clearance_m"].min()),
                                          visible_object_frames=int(clip["gt_visible"].sum()),
                                          visible_moving_frames=int(clip["gt_visible"][:, clip["gt_kind"] == 1].sum())))
            print(f"{split} {name}: {frames} frames, visible={clip['gt_visible'].sum()}, "
                  f"moving={manifest['clips'][-1]['visible_moving_frames']}", flush=True)
    manifest["collection_seconds"] = time.perf_counter() - started
    manifest["total_frames"] = sum(c["frames"] for c in manifest["clips"])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "training/courses/perception-splits.json")
    parser.add_argument("--out", type=Path, default=ROOT / "training/data/tracking-pilot")
    parser.add_argument("--frames", type=int, default=41)
    parser.add_argument("--clips-per-course", type=int, default=2)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    collect(args.plan, args.out, frames=args.frames, clips_per_course=args.clips_per_course, seed=args.seed)


if __name__ == "__main__":
    main()
