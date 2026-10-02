"""Official RGB and rendered-buoy localization targets for training only.

These are independent, safe vessel views, not expert actions or RL transitions.
The network receives only RGB. Hidden geometry is used to sample views and to
create auxiliary targets; neither masks nor diagnostic geometry enter an actor.

The target includes the official two-pixel display marker. It localizes the
rendered buoy and must not be treated as the physical collision footprint.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from usvnav.collide import CourseIndex
from usvnav.coursefile import load
from usvnav.plant import DT, Vessel
from usvnav.render import TRACK1, class_map, class_names, render, scene_from_course
from usvnav.world import BUOY, EGO, UNOBSERVED
from training.collect_perception import course_identity


FORMAT = "usvnav-buoy-supervision/1"


def _targets_from_scene(scene):
    native = class_map(scene, TRACK1)
    names = class_names(TRACK1)
    if UNOBSERVED in names and np.any(native == names.index(UNOBSERVED)):
        raise ValueError("Track1-2 target unexpectedly contains unobserved pixels")
    valid200 = native != names.index(EGO)
    buoy200 = (native == names.index(BUOY)) & valid200
    valid = valid200.reshape(100, 2, 100, 2).all(axis=(1, 3))
    buoy = buoy200.reshape(100, 2, 100, 2).any(axis=(1, 3)) & valid
    return dict(buoy=buoy.astype(np.uint8), valid=valid.astype(np.uint8),
                buoy_pixels_200=int(buoy200.sum()))


def buoy_target_masks(course, vessel, time_s=0.):
    """Training-only100px masks for an already-rendered on-policy RGB frame.

    This avoids rendering a second RGB frame when adding an auxiliary loss to
    PPO. Course/geometry/masks remain outside actor observations.
    """
    return _targets_from_scene(scene_from_course(course, vessel, float(time_s)))


def render_buoy_targets(course, vessel, time_s=0.):
    """Return native CHW RGB, binary100x100 buoy mask, and valid100x100.

    RGB is precisely the official renderer output. The native200x200 class map
    supplies targets, with2x2 max pooling so small markers cannot disappear.
    Any2x2 block containing an ego pixel is excluded from loss.
    """
    scene = scene_from_course(course, vessel, float(time_s))
    return dict(rgb=render(scene, TRACK1).transpose(2, 0, 1).copy(), **_targets_from_scene(scene))


def _candidate_pose(course, rng, *, positive, mode, buoys):
    heading = float(rng.uniform(-math.pi, math.pi))
    focus = np.zeros(2, np.float32)
    if positive:
        body = buoys[int(rng.integers(len(buoys)))].shape
        if mode == 1:  # danger-relevant, bow-up forward views
            forward = float(rng.uniform(5., 30.))
            lateral = float(rng.uniform(-3., 3.) if rng.random() < .65 else rng.uniform(-15., 15.))
        elif mode == 2:  # some behind/side views prevent a front-only detector
            angle, distance = rng.uniform(-math.pi, math.pi), rng.uniform(3., 45.)
            forward, lateral = float(distance * math.cos(angle)), float(distance * math.sin(angle))
        else:  # marker clipping near the square sensor boundary
            forward, lateral = float(rng.uniform(48.5, 50.4)), float(rng.uniform(-30., 30.))
        c, s = math.cos(heading), math.sin(heading)
        x = float(body.x - (c * forward - s * lateral))
        y = float(body.y - (s * forward + c * lateral))
        focus[:] = [forward, lateral]
    else:
        low, high = np.min(course.boundary, axis=0), np.max(course.boundary, axis=0)
        x, y = map(float, rng.uniform(low, high))
    return Vessel(x, y, heading), focus


def generate_buoy_dataset(courses, *, count=768, seed=82, negative_fraction=.2,
                          front_fraction=.85, min_clearance_m=.15, max_tick=6000,
                          progress=False):
    """Sample safe official views and return NumPy arrays plus metadata.

    Main API: rgbuint8[N,3,200,200], buoyuint8[N,100,100],
    validuint8[N,100,100]. Normalize RGB by255 once when invoking a neural loss.
    For a12-channel temporal model, its auxiliary head can use this current
    frame directly; storing repeated histories wastes memory.

    Caller provides separate train/validation course lists. This function never
    mixes those splits. Negative frames must have a truly empty buoy target;
    positives must retain at least one valid output cell.
    """
    if (count < 1 or seed < 0 or not 0 <= negative_fraction <= 1
            or not 0 <= front_fraction <= 1 or not 0 <= min_clearance_m < 10 or max_tick < 0):
        raise ValueError("Need positive count, nonnegative seed/time and valid sampling fractions/clearance")
    courses = [load(path) if isinstance(path, (str, Path)) else path for path in courses]
    if not courses:
        raise ValueError("Provide at least one training course")
    buoy_bodies = [[body for body in course.bodies if body.cls == BUOY] for course in courses]
    positive_courses = [i for i, bodies in enumerate(buoy_bodies) if bodies]
    negative_count = int(round(count * negative_fraction))
    if negative_count < count and not positive_courses:
        raise ValueError("Positive scenes require courses containing actual buoys")
    clean_courses = [i for i, bodies in enumerate(buoy_bodies) if not bodies]
    indices = [CourseIndex(course) for course in courses]
    rng = np.random.default_rng(seed)
    positive_plan = np.ones(count, bool)
    positive_plan[:negative_count] = False
    rng.shuffle(positive_plan)
    data = dict(rgb=np.empty((count, 3, 200, 200), np.uint8),
                buoy=np.empty((count, 100, 100), np.uint8),
                valid=np.empty((count, 100, 100), np.uint8),
                pose=np.empty((count, 3), np.float64), course_index=np.empty(count, np.int32),
                tick=np.empty(count, np.int32), clearance_m=np.empty(count, np.float32),
                positive=positive_plan.copy(), scene_mode=np.empty(count, np.uint8),
                focus_body_m=np.empty((count, 2), np.float32),
                buoy_pixels_200=np.empty(count, np.int32))
    attempts, started = 0, time.perf_counter()
    for sample_index, positive in enumerate(positive_plan):
        if positive:
            mode_draw = rng.random()
            mode = 1 if mode_draw < front_fraction else (2 if mode_draw < front_fraction + .1 else 3)
        else:
            mode = 0
        for _ in range(1000):
            attempts += 1
            if positive:
                course_index = int(rng.choice(positive_courses))
            elif clean_courses and rng.random() < .7:
                course_index = int(rng.choice(clean_courses))
            else:
                course_index = int(rng.integers(len(courses)))
            course = courses[course_index]
            vessel, focus = _candidate_pose(course, rng, positive=positive, mode=mode, buoys=buoy_bodies[course_index])
            tick = int(rng.integers(max_tick + 1))
            clearance, hit, inside = indices[course_index].tick(vessel.hull(), tick * DT, cap=10.)
            if hit is not None or not inside or clearance < min_clearance_m:
                continue
            sample = render_buoy_targets(course, vessel, tick * DT)
            if bool(sample["buoy"].any()) != bool(positive):
                continue
            for key in ("rgb", "buoy", "valid", "buoy_pixels_200"):
                data[key][sample_index] = sample[key]
            data["pose"][sample_index] = [vessel.x, vessel.y, vessel.psi]
            data["course_index"][sample_index] = course_index
            data["tick"][sample_index] = tick
            data["clearance_m"][sample_index] = clearance
            data["scene_mode"][sample_index] = mode
            data["focus_body_m"][sample_index] = focus
            break
        else:
            raise RuntimeError(f"Could not sample safe {'positive' if positive else 'negative'} view{sample_index}; change course/sampling constraints")
        if progress and ((sample_index + 1) % 64 == 0 or sample_index + 1 == count):
            print(f"buoy data: {sample_index + 1}/{count}, attempts={attempts}, wall={time.perf_counter() - started:.1f}s", flush=True)
    data["metadata"] = dict(format=FORMAT, condition="1-2", seed=seed, count=count,
                            negatives=negative_count, positives=count - negative_count,
                            negative_fraction=negative_fraction, front_fraction=front_fraction,
                            min_clearance_m=min_clearance_m, max_tick=max_tick, sampling_attempts=attempts,
                            rgb_shape=[3, 200, 200], buoy_target_shape=[100, 100],
                            class_definition="Binary rendered localization target for detail class2 buoy; other valid pixels0",
                            marker_px=TRACK1.min_marker_px, m_per_px=TRACK1.m_per_px,
                            target_definition="Official200px class_map with display-marker inflation,2x2maxpool to100px; anyego-containingblockignored",
                            policy_input="Only rawRGB; masks, sourcecourse, hiddengeometry and diagnostics are trainingtargets/diagnostics, neveractorobservations",
                            sample_kind="Independent safe vessel poses; noexpert actions orRLtransitions/history",
                            course_sha256=[course_identity(course) for course in courses],
                            raw_array_bytes=sum(array.nbytes for array in data.values()),
                            elapsed_s=time.perf_counter() - started)
    return data


def save_buoy_dataset(data, path):
    """Write an allow_pickle=False compatible compressed cache without overwrite."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Dataset path must end in.npz")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in data.items() if key != "metadata"}
    np.savez_compressed(path, **arrays, metadata_json=np.array(json.dumps(data["metadata"], allow_nan=False)))
    return path


def load_buoy_dataset(path):
    with np.load(path, allow_pickle=False) as shard:
        data = {key: shard[key].copy() for key in shard.files if key != "metadata_json"}
        data["metadata"] = json.loads(str(shard["metadata_json"].item()))
    count = data["metadata"]["count"]
    if data["metadata"].get("format") != FORMAT:
        raise ValueError("Unknown buoy-supervision dataset format")
    for key, shape in (("rgb", (count, 3, 200, 200)), ("buoy", (count, 100, 100)), ("valid", (count, 100, 100))):
        if data[key].shape != shape or data[key].dtype != np.uint8:
            raise ValueError(f"Invalid{key} array")
    if not np.isin(data["buoy"], [0, 1]).all() or not np.isin(data["valid"], [0, 1]).all():
        raise ValueError("Buoy targets/validmask must be binary")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--courses", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=768)
    parser.add_argument("--seed", type=int, default=82)
    parser.add_argument("--negative-fraction", type=float, default=.2)
    args = parser.parse_args()
    data = generate_buoy_dataset(args.courses, count=args.count, seed=args.seed,
                                negative_fraction=args.negative_fraction, progress=True)
    path = save_buoy_dataset(data, args.out)
    print(json.dumps(dict(path=str(path.resolve()), **data["metadata"]), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
