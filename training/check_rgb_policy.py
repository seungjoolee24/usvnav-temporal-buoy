"""Compare one course with reference, normal, masked-RGB or hidden-buoy vision.

The reference uses the public-coordinate controller with zero learned residual.
Masked RGB replaces only the actor's RGB observation, preserving public state,
waypoints, goal, reference actions and history. All-zero images are outside the
training distribution: an action or outcome change shows image dependence,
not correct perception or successful obstacle reasoning. No privileged actor
features are added. Each phase gets separate, non-overwriting output files.
Hidden-buoy rendering removes buoy bodies from a temporary rendering copy only;
the true collision geometry, rewards and public-coordinate computation remain.
This intervention diagnoses visual dependence, not generalized avoidance. Once
actions differ, trajectories and later public observations may also differ.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

import numpy as np


class MaskedRgbPolicy:
    """Pass through prediction after zeroing a copy of only the RGB input."""

    def __init__(self, model):
        self.model = model

    def predict(self, observation, *args, **kwargs):
        masked = dict(observation)
        masked["rgb"] = np.zeros_like(observation["rgb"])
        return self.model.predict(masked, *args, **kwargs)


def course_without_buoys(course):
    """Make a rendering-only copy, retaining the true course and other bodies."""
    from usvnav.world import BUOY

    return replace(course, bodies=[body for body in course.bodies if body.cls != BUOY])


def hidden_buoy_renderer(original_renderer):
    """Remove buoys consistently whenever the scoped renderer builds an image."""
    def render(condition, vessel, course, t, raster=None):
        return original_renderer(condition, vessel, course_without_buoys(course), t, raster)

    return render


def phase_paths(output: Path, phase: str):
    if re.fullmatch(r"[A-Za-z0-9_-]+", phase) is None:
        raise ValueError("phase must contain only letters, digits, underscores or hyphens")
    paths = {key: output / f"{phase}-{suffix}.json" for key, suffix in (
        ("evaluation", "evaluation"), ("trace", "trace"), ("metadata", "diagnostic"))}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite this phase: " + ", ".join(existing))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reference", "normal", "masked-rgb", "hidden-buoy"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--course", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Existing run directory")
    parser.add_argument("--phase", required=True, help="New output phase, e.g. buoy-masked")
    parser.add_argument("--cutoff", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    if not 0 < args.cutoff <= 6000 or args.cutoff % 5:
        parser.error("cutoff must be 1..6000 ticks and divisible by 5")
    if args.mode != "reference" and args.checkpoint is None:
        parser.error("normal, masked-rgb and hidden-buoy modes require --checkpoint")
    if args.mode == "reference" and args.checkpoint is not None:
        parser.error("reference mode uses no checkpoint; omit --checkpoint")
    output, course = args.out.resolve(), args.course.resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    if not output.is_dir():
        parser.error("--out must name an existing run directory")
    if not course.is_file():
        parser.error("--course must name an existing course file")
    if checkpoint is not None and not checkpoint.is_file():
        parser.error("--checkpoint must name an existing policy ZIP file")
    try:
        paths = phase_paths(output, args.phase)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))

    metadata = dict(
        mode=args.mode, phase=args.phase, seed=args.seed, cutoff_ticks=args.cutoff,
        course=str(course), course_sha256=hashlib.sha256(course.read_bytes()).hexdigest(),
        checkpoint=str(checkpoint) if checkpoint else None,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None,
        device="cpu", torch_threads=2, status="started",
        masked_observation_keys=["rgb"] if args.mode == "masked-rgb" else [],
        render_only_hidden_class="buoy" if args.mode == "hidden-buoy" else None,
        physics_course_unchanged=True,
        reference_behavior="public-coordinate controller with zero learned residual",
        diagnostic_limit=(
            "Zero RGB is an out-of-distribution intervention. Differences show image dependence, "
            "not proof of correct perception or obstacle reasoning." if args.mode == "masked-rgb" else
            "Only buoy rendering is removed. Differences diagnose visual dependence, not proof "
            "of generalized obstacle avoidance." if args.mode == "hidden-buoy" else
            "A single-course evaluation is not a measure of generalization."),
        action_feedback_limit=("Once policy actions differ, the trajectory and subsequent public "
                               "coordinate/history observations may differ too."),
        actor_input_scope="Existing public RGB and coordinate observations only; no privileged features added.",
        outputs={key: path.name for key, path in paths.items()},
    )
    # Exclusive metadata creation reserves the phase before evaluation starts.
    with paths["metadata"].open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        # Lazy imports keep --help and the proxy test independent of Torch/SB3.
        import torch
        from stable_baselines3 import PPO
        from training.train_rgb_navigation import evaluate

        torch.set_num_threads(2)
        model = PPO.load(checkpoint, device="cpu") if checkpoint else None
        if args.mode == "masked-rgb":
            model = MaskedRgbPolicy(model)
        if args.mode == "hidden-buoy":
            from unittest.mock import patch
            from usvnav import sim

            original_renderer = sim._perception
            with patch.object(sim, "_perception", new=hidden_buoy_renderer(original_renderer)):
                evaluate(model, course, args.cutoff, output, args.phase, seed=args.seed)
        else:
            evaluate(model, course, args.cutoff, output, args.phase, seed=args.seed)
        metadata["status"] = "completed"
    except BaseException as error:
        metadata.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        paths["metadata"].write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
