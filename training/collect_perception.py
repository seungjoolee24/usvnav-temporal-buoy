"""Generate independent top-view/segmentation pairs without a driving agent.

Run from the repository root: python -m training.collect_perception --help
These are single-frame perception samples, NOT trajectories or RL experiences.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from usvnav.collide import CourseIndex
from usvnav.coursefile import load, to_dict
from usvnav.plant import DT, Vessel
from usvnav.png import write_png
from usvnav.render import TRACK1, class_map, class_names, scene_from_course, render
from training.perception_schema import COARSE_COLOURS, COARSE_TARGET_IDS, IGNORE, colour_labels, labels_from_raster

ROOT = Path(__file__).resolve().parents[1]
TARGET_IDS = COARSE_TARGET_IDS
LABEL_COLOURS = COARSE_COLOURS


def semantic_target(ids: np.ndarray, names: list[str]) -> np.ndarray:
    """Merge moored/moving ships into the observable vessel class; ignore ego."""
    return labels_from_raster(ids, names)


def course_identity(course) -> str:
    canonical = json.dumps(to_dict(course), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assign_splits(identities: list[str]) -> list[str]:
    """Hold out whole courses; detect identical course copies before writing data."""
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate course contents; copies must not cross data splits.")
    n = len(identities)
    if n < 3:
        raise ValueError("Provide at least three distinct courses for train/val/test.")
    held_out = max(1, n // 10)
    return ["train"] * (n - 2 * held_out) + ["val"] * held_out + ["test"] * held_out


def read_split_plan(path):
    """Explicit course membership, with paths relative to the split-plan file."""
    path = Path(path).resolve()
    plan = json.loads(path.read_text(encoding="utf-8"))
    if set(plan) != {"train", "val", "test"}:
        raise ValueError("Split plan must define exactly train, val and test")
    assignments = {}
    for split, entries in plan.items():
        if not isinstance(entries, list) or not entries or any(not isinstance(p, str) for p in entries):
            raise ValueError("Every split needs a nonempty list of course paths")
        for entry in entries:
            source = (path.parent / entry).resolve()
            if source in assignments:
                raise ValueError(f"Course listed twice in split plan: {source}")
            assignments[source] = split
    return assignments


def make_sample(course, vessel: Vessel, t: float, clearance: float) -> dict:
    scene = scene_from_course(course, vessel, t)
    native = class_map(scene, TRACK1)
    # Remove ONLY display marker inflation. Keep pixel coordinates and ego masking.
    # This is a pixel-centre geometry target, not exact continuous collision geometry.
    physical = class_map(scene, replace(TRACK1, min_marker_px=0.0))
    names = class_names(TRACK1)
    return {
        "rgb": render(scene, TRACK1),
        "semantic": semantic_target(native, names),
        "physical_semantic": semantic_target(physical, names),
        "raster_class": native.astype(np.uint8),
        "pose": np.array([vessel.x, vessel.y, vessel.psi], dtype=np.float64),
        "t": np.array(t, dtype=np.float64),
        "clearance_m": np.array(clearance, dtype=np.float32),
    }


def sample_course(course, rng, count: int, min_clearance: float, max_tick: int):
    """Mix uniform water coverage and neighbourhoods of obstacles, rejecting unsafe hulls."""
    index = CourseIndex(course)
    low, high = np.min(course.boundary, axis=0), np.max(course.boundary, axis=0)
    samples = []
    attempts = 0
    while len(samples) < count and attempts < max(10000, count * 1000):
        attempts += 1
        t = int(rng.integers(0, max_tick + 1)) * DT
        shapes = course.static_shapes() + course.traffic_rects(t)
        if shapes and rng.random() < 0.5:
            shape = shapes[int(rng.integers(len(shapes)))]
            angle, radius = rng.uniform(-math.pi, math.pi), rng.uniform(3.0, 32.0)
            x, y = shape.x + radius * math.cos(angle), shape.y + radius * math.sin(angle)
        else:
            x, y = rng.uniform(low, high)
        vessel = Vessel(x, y, rng.uniform(-math.pi, math.pi))
        clearance, hit, inside = index.tick(vessel.hull(), t, cap=10.0)
        if hit is not None or not inside or clearance < min_clearance:
            continue
        samples.append(make_sample(course, vessel, t, clearance))
    if len(samples) != count:
        raise RuntimeError(f"Only {len(samples)}/{count} safe poses after {attempts} attempts")
    return samples, attempts


def colour_target(target):
    return colour_labels(target)


def collect(paths, output: Path, *, samples_per_course=16, seed=42,
            min_clearance=0.5, max_tick=6000, split_by_path=None) -> dict:
    output = Path(output)
    if samples_per_course < 1 or max_tick < 0 or not 0.0 <= min_clearance < 10.0:
        raise ValueError("Positive sample count, nonnegative max tick, and clearance in [0,10) required")
    if seed < 0:
        raise ValueError("Seed must be nonnegative")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a fresh directory: {output}")
    sources = sorted((Path(p).resolve() for p in paths), key=lambda p: str(p))
    courses = [load(p) for p in sources]
    identities = [course_identity(c) for c in courses]
    splits = assign_splits(identities)
    if split_by_path is not None:
        assignments = {Path(p).resolve(): split for p, split in split_by_path.items()}
        if set(assignments) != set(sources) or set(assignments.values()) != {"train", "val", "test"}:
            raise ValueError("Explicit splits must cover all sources and all three splits")
        splits = [assignments[source] for source in sources]
    output.mkdir(parents=True)
    (output / "courses").mkdir()
    (output / "previews").mkdir()
    started = time.perf_counter()
    manifest = {
        "format": "usvnav-perception/1", "condition": "1-2", "seed": seed,
        "sample_kind": "independent_static_ego_frames", "has_actions": False,
        "labels": {"water": 0, "fixed_obstacle_including_bank": 1, "vessel": 2, "ignore_ego": IGNORE},
        "raster_classes": class_names(TRACK1), "shape_rgb": [200, 200, 3],
        "m_per_px": TRACK1.m_per_px, "frame": "bow_up",
        "semantic_definition": "Official class-map pixel centres, including minimum display markers; no shadows/decks as separate labels",
        "physical_semantic_definition": "Same pixel centres with minimum marker disabled; exact collision geometry stays in course snapshots",
        "min_clearance_m": min_clearance, "max_tick": max_tick,
        "sampling": "50% uniform bounding box, 50% obstacle neighbourhood; full-hull safety rejection",
        "split_policy": ("Explicit whole-course assignments; duplicate course contents rejected" if split_by_path is not None
                         else "Whole courses; sorted input order; duplicate canonical course contents rejected"),
        "courses": [], "total_samples": 0,
    }
    for source, course, identity, split in zip(sources, courses, identities, splits):
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(identity[:16], 16)]))
        samples, attempts = sample_course(course, rng, samples_per_course, min_clearance, max_tick)
        arrays = {key: np.stack([sample[key] for sample in samples]) for key in samples[0]}
        folder = output / split
        folder.mkdir(exist_ok=True)
        basename = identity[:16]
        shard = folder / f"{basename}.npz"
        np.savez_compressed(shard, **arrays)
        snapshot = output / "courses" / f"{basename}.json"
        # Preserve custom source precision; to_dict intentionally rounds coordinates.
        snapshot.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        preview_index = int(np.argmax(np.sum(arrays["semantic"] == 2, axis=(1, 2))))
        preview = output / "previews" / f"{basename}.png"
        write_png(preview, np.concatenate([
            arrays["rgb"][preview_index], colour_target(arrays["semantic"][preview_index]),
            colour_target(arrays["physical_semantic"][preview_index])], axis=1), scale=2)
        counts = np.bincount(arrays["semantic"].ravel(), minlength=256)
        manifest["courses"].append({
            "source": str(source), "sha256": identity, "split": split,
            "samples": samples_per_course, "sampling_attempts": attempts,
            "shard": shard.relative_to(output).as_posix(),
            "snapshot": snapshot.relative_to(output).as_posix(),
            "preview": preview.relative_to(output).as_posix(), "preview_sample_index": preview_index,
            "pixel_counts": {str(i): int(counts[i]) for i in (0, 1, 2, IGNORE)},
        })
        manifest["total_samples"] += samples_per_course
        print(f"{split}: {source.name}: {samples_per_course} samples ({attempts} attempts)", flush=True)
    manifest["elapsed_s"] = round(time.perf_counter() - started, 3)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--courses", nargs="+", type=Path, help="Distinct course JSON files; sorted before splitting")
    sources.add_argument("--split-file", type=Path, help="JSON {train:[paths],val:[paths],test:[paths]}; relative to JSON file")
    parser.add_argument("--out", type=Path, default=ROOT / "training/data/perception-pilot")
    parser.add_argument("--samples-per-course", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42, help="Pose/time sampling seed, NOT a simulator diversity seed")
    parser.add_argument("--min-clearance", type=float, default=0.5)
    parser.add_argument("--max-tick", type=int, default=6000)
    args = parser.parse_args()
    assignments = read_split_plan(args.split_file) if args.split_file else None
    paths = list(assignments) if assignments is not None else args.courses or sorted((ROOT / "sets/practice/courses").glob("*.json"))
    manifest = collect(paths, args.out, samples_per_course=args.samples_per_course,
                       seed=args.seed, min_clearance=args.min_clearance, max_tick=args.max_tick,
                       split_by_path=assignments)
    elapsed = manifest["elapsed_s"]
    print(f"Saved {manifest['total_samples']} samples in {elapsed:.3f}s to {args.out.resolve()}")


if __name__ == "__main__":
    main()
