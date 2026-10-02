"""Controlled auxiliary-only A/B on cloned saved policies and cached RGB.

Both arms use identical training minibatches and fresh Adam optimizers. Only
shared RGB convolutions and the buoy head are optimized. Heldout frames are
used for scoring only. This diagnoses localization, not navigation success;
changing shared visual features can also alter a saved actor's behavior.
The source checkpoint and active policy/trainer files are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO

from usvnav.coursefile import load
from usvnav.plant import DT, Vessel
from usvnav.render import TRACK1, class_map, class_names, scene_from_course
from training.buoy_loss_candidate import hard_negative_buoy_loss
from training.buoy_policy import global_negative_buoy_loss
from training.buoy_supervision import load_buoy_dataset
from training.collect_perception import course_identity
from training.review_buoy_perception import components, dilate_one


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validation_geometry(data, courses):
    bank_id = class_names(TRACK1).index("bank")
    targets, banks, component_pixels, expanded_component_pixels = [], [], [], []
    for index in range(len(data["rgb"])):
        target = data["buoy"][index].astype(bool) & data["valid"][index].astype(bool)
        native = class_map(scene_from_course(courses[int(data["course_index"][index])],
                                            Vessel(*data["pose"][index]), int(data["tick"][index]) * DT), TRACK1)
        bank = (native == bank_id).reshape(100, 2, 100, 2).any(axis=(1, 3))
        exact, expanded = [], []
        for points in components(target):
            mask = np.zeros_like(target)
            mask[points[:, 0], points[:, 1]] = True
            exact.append(np.flatnonzero(mask))
            expanded.append(np.flatnonzero(dilate_one(mask)))
        targets.append(target)
        banks.append(bank)
        component_pixels.append(exact)
        expanded_component_pixels.append(expanded)
    return dict(targets=np.asarray(targets), banks=np.asarray(banks),
                component_pixels=component_pixels, expanded_component_pixels=expanded_component_pixels)


def score(extractor, data, geometry):
    extractor.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(data["rgb"]), 8):
            rgb = torch.from_numpy(data["rgb"][start:start + 8]).float() / 255.
            probabilities.append(extractor.buoy_logits(rgb).sigmoid()[:, 0].numpy())
    predicted = (np.concatenate(probabilities) >= .5) & data["valid"].astype(bool)
    target = geometry["targets"]
    tp, fp, fn = int((predicted & target).sum()), int((predicted & ~target).sum()), int((~predicted & target).sum())
    hits = tolerant_hits = object_count = bank_fp = near_fp = 0
    for index, mask in enumerate(predicted):
        flat = mask.ravel()
        exact = geometry["component_pixels"][index]
        expanded = geometry["expanded_component_pixels"][index]
        object_count += len(exact)
        hits += sum(int(flat[pixels].any()) for pixels in exact)
        tolerant_hits += sum(int(flat[pixels].any()) for pixels in expanded)
        near = dilate_one(target[index])
        false = mask & ~target[index]
        near_fp += int((false & near).sum())
        bank_fp += int((false & ~near & geometry["banks"][index]).sum())
    pooled_pred = predicted.reshape(-1, 20, 5, 20, 5).any(axis=(2, 4))
    pooled_target = target.reshape(-1, 20, 5, 20, 5).any(axis=(2, 4))
    grid_tp, grid_fp, grid_fn = int((pooled_pred & pooled_target).sum()), int((pooled_pred & ~pooled_target).sum()), int((~pooled_pred & pooled_target).sum())
    empty = ~target.any(axis=(1, 2))
    return dict(pixel_precision=tp / max(tp + fp, 1), pixel_recall=tp / max(tp + fn, 1),
                true_positive_pixels=tp, false_positive_pixels=fp, missed_pixels=fn,
                gt_components=object_count, exact_component_hits=hits,
                exact_component_recall=hits / max(object_count, 1),
                one_cell_tolerance_recall=tolerant_hits / max(object_count, 1),
                false_positive_on_bank=bank_fp, false_positive_near_target=near_fp,
                negative_images=int(empty.sum()), negative_images_with_false_positive=int(predicted[empty].any(axis=(1, 2)).sum()),
                actor_20grid_precision=grid_tp / max(grid_tp + grid_fp, 1),
                actor_20grid_recall=grid_tp / max(grid_tp + grid_fn, 1),
                actor_20grid_false_positive_cells=grid_fp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=903)
    args = parser.parse_args()
    if not 0 < args.steps <= args.max_steps <= 128 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("Need positive steps<=max_steps<=128, batch size and learning rate")
    run = args.run.resolve()
    output = (args.out or run / "hard-negative-ab-01").resolve()
    checkpoint = run / "final-policy.zip"
    if not checkpoint.is_file():
        parser.error("Run must contain a completed final-policy.zip")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    train = load_buoy_dataset(run / "localization-train.npz")
    heldout = load_buoy_dataset(run / "localization-validation.npz")
    courses = [load(run / "courses" / row["file"]) for row in config["curriculum"]["val"]]
    if [course_identity(course) for course in courses] != heldout["metadata"]["course_sha256"]:
        raise RuntimeError("Heldout course hashes differ from cached views")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    geometry = validation_geometry(heldout, courses)
    models = {arm: PPO.load(checkpoint, device="cpu") for arm in ("global", "hard_negative")}
    optimizers = {}
    parameters = {}
    for arm, model in models.items():
        extractor = model.policy.features_extractor
        parameters[arm] = list(extractor.rgb_encoder.parameters()) + list(extractor.buoy_head.parameters())
        optimizers[arm] = torch.optim.Adam(parameters[arm], lr=args.learning_rate)
    before = score(models["global"].policy.features_extractor, heldout, geometry)
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(len(train["rgb"]), size=(args.max_steps, args.batch_size))
    np.save(output / "shared-minibatch-indices.npy", indices)
    losses, assessments = {arm: [] for arm in models}, []
    record = dict(source_checkpoint=str(checkpoint), source_checkpoint_sha256=source_hash,
                  train_images=len(train["rgb"]), validation_images=len(heldout["rgb"]),
                  train_only=True, steps=args.steps, max_steps=args.max_steps, seed=args.seed,
                  learning_rate=args.learning_rate, batch_size=args.batch_size, torch_threads=1,
                  hard_fraction=.02, hard_mix=.5, dice_weight=.1,
                  optimizers="Fresh Adam for each arm; identical minibatches; only RGB encoder and buoy head",
                  expansion_rule="After initial32 steps, extend to128 unless bank FP halves vs global arm, pixel precision improves, and exact component recall loses no more than2 percentage points vs global arm.",
                  before=before, assessments=assessments, losses=losses,
                  limitation="Auxiliary-only cloned-model localization test, not evidence of navigation improvement. Shared RGB changes may change actor behavior.")
    for step in range(args.max_steps):
        rgb = torch.from_numpy(train["rgb"][indices[step]]).float() / 255.
        labels = torch.from_numpy(train["buoy"][indices[step]]).float()
        valid = torch.from_numpy(train["valid"][indices[step]]).float()
        for arm, model in models.items():
            extractor = model.policy.features_extractor
            extractor.train()
            optimizer = optimizers[arm]
            optimizer.zero_grad()
            logits = extractor.buoy_logits(rgb)
            loss = (global_negative_buoy_loss(logits, labels, valid) if arm == "global" else
                    hard_negative_buoy_loss(logits, labels, valid))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite A/B loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters[arm], .5)
            optimizer.step()
            losses[arm].append(float(loss.detach()))
        if (step + 1) % 8 == 0:
            print(f"loss A/B: step={step + 1}, global={losses['global'][-1]:.5f}, hard={losses['hard_negative'][-1]:.5f}", flush=True)
        if step + 1 in (args.steps, args.max_steps):
            scores = {arm: score(model.policy.features_extractor, heldout, geometry) for arm, model in models.items()}
            candidate, baseline = scores["hard_negative"], scores["global"]
            useful = (candidate["false_positive_on_bank"] <= baseline["false_positive_on_bank"] * .5
                      and candidate["pixel_precision"] > baseline["pixel_precision"]
                      and candidate["exact_component_recall"] >= baseline["exact_component_recall"] - .02)
            assessments.append(dict(step=step + 1, meaningful_improvement_gate=useful, **scores))
            write_json(output / "metrics.json", record)
            print(json.dumps(assessments[-1], indent=2), flush=True)
            if useful or step + 1 == args.max_steps:
                break
    for arm, model in models.items():
        model.save(output / ("clone-global-policy" if arm == "global" else "clone-hard-negative-policy"))
    record.update(actual_steps=len(losses["global"]), wall_s=time.perf_counter() - started,
                  source_checkpoint_unchanged=hashlib.sha256(checkpoint.read_bytes()).hexdigest() == source_hash)
    write_json(output / "metrics.json", record)
    print(f"A/B diagnostics saved: {output}", flush=True)


if __name__ == "__main__":
    main()
