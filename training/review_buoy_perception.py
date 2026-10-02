"""Read-only pixel/component review of a saved buoy head on heldout RGB.

Object recall here means each rendered target's connected component has at
least one >=0.5 prediction overlapping it. The separate one-cell tolerance is
diagnostic, not a substitute for exact localization. Components are rendered
markers, not physical object IDs or collision footprints. Ground-truth masks,
pose and classes are used only for these offline diagnostics, never inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

from usvnav.coursefile import load
from usvnav.plant import DT, Vessel
from usvnav.render import TRACK1, class_map, class_names, scene_from_course
from training.buoy_supervision import load_buoy_dataset
from training.collect_perception import course_identity


def components(mask):
    """Eight-connected components as row/column coordinate arrays."""
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    height, width = mask.shape
    for row, col in zip(*np.nonzero(mask)):
        if seen[row, col]:
            continue
        seen[row, col] = True
        pending, points = [(int(row), int(col))], []
        while pending:
            r, c = pending.pop()
            points.append((r, c))
            for nr in range(max(0, r - 1), min(height, r + 2)):
                for nc in range(max(0, c - 1), min(width, c + 2)):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        pending.append((nr, nc))
        result.append(np.asarray(points, dtype=int))
    return result


def dilate_one(mask):
    padded = np.pad(mask, 1)
    return np.logical_or.reduce([padded[r:r + mask.shape[0], c:c + mask.shape[1]]
                                 for r in range(3) for c in range(3)])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prefix", default="perception-review")
    args = parser.parse_args()
    output = args.run.resolve()
    checkpoint = args.checkpoint or output / "initial-policy.zip"
    figure_path, metrics_path = output / f"{args.prefix}.png", output / f"{args.prefix}.json"
    if not args.prefix or Path(args.prefix).name != args.prefix:
        parser.error("prefix must be a plain filename stem")
    if figure_path.exists() or metrics_path.exists():
        parser.error("Refusing to overwrite existing perception review")
    data = load_buoy_dataset(output / "localization-validation.npz")
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    courses = [load(output / "courses" / row["file"]) for row in config["curriculum"]["val"]]
    if [course_identity(course) for course in courses] != data["metadata"]["course_sha256"]:
        raise RuntimeError("Cached validation course hashes do not match this run")
    torch.set_num_threads(1)
    model = PPO.load(checkpoint, device="cpu")
    extractor = model.policy.features_extractor
    extractor.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(data["rgb"]), 8):
            rgb = torch.from_numpy(data["rgb"][start:start + 8]).float() / 255.
            predictions.append(extractor.buoy_logits(rgb).sigmoid()[:, 0].cpu().numpy())
    probability = np.concatenate(predictions)
    names = class_names(TRACK1)
    bank_id = names.index("bank")
    rows = []
    for index, (prob, target, valid) in enumerate(zip(probability, data["buoy"], data["valid"])):
        target, valid = target.astype(bool), valid.astype(bool)
        predicted = (prob >= .5) & valid
        target &= valid
        gt_components = components(target)
        exact_hits = tolerant_hits = 0
        for points in gt_components:
            gt = np.zeros_like(target)
            gt[points[:, 0], points[:, 1]] = True
            exact_hits += int(predicted[gt].any())
            tolerant_hits += int(predicted[dilate_one(gt)].any())
        true_positive = int((predicted & target).sum())
        false_positive = predicted & ~target
        native = class_map(scene_from_course(courses[int(data["course_index"][index])],
                                            Vessel(*data["pose"][index]), int(data["tick"][index]) * DT), TRACK1)
        bank = (native == bank_id).reshape(100, 2, 100, 2).any(axis=(1, 3))
        near_target = false_positive & dilate_one(target)
        bank_false = false_positive & ~near_target & bank
        other_false = false_positive & ~near_target & ~bank
        predicted_components = components(predicted)
        target_points = np.argwhere(target)
        unmatched_distances = []
        for points in predicted_components:
            if not target[points[:, 0], points[:, 1]].any() and len(target_points):
                distances = np.sqrt(((points[:, None, :] - target_points[None, :, :]) ** 2).sum(axis=2))
                unmatched_distances.append(float(distances.min()))
        actor_target = target.reshape(20, 5, 20, 5).any(axis=(1, 3))
        actor_predicted = predicted.reshape(20, 5, 20, 5).any(axis=(1, 3))
        rows.append(dict(
            index=index, gt_components=len(gt_components), exact_hits=exact_hits,
            one_cell_tolerance_hits=tolerant_hits, target_pixels=int(target.sum()),
            predicted_pixels=int(predicted.sum()), true_positive_pixels=true_positive,
            false_positive_pixels=int(false_positive.sum()),
            missed_pixels=int((target & ~predicted).sum()),
            false_positive_near_target=int(near_target.sum()),
            false_positive_on_bank=int(bank_false.sum()),
            false_positive_elsewhere=int(other_false.sum()),
            predicted_components=len(predicted_components),
            predicted_components_with_exact_target=sum(int(target[points[:, 0], points[:, 1]].any())
                                                       for points in predicted_components),
            unmatched_prediction_min_target_distances_cells=unmatched_distances,
            actor_grid_true_positive_cells=int((actor_target & actor_predicted).sum()),
            actor_grid_false_positive_cells=int((~actor_target & actor_predicted).sum()),
            actor_grid_missed_cells=int((actor_target & ~actor_predicted).sum()),
            max_probability_on_target=float(prob[target].max()) if target.any() else None,
            mean_probability_on_target=float(prob[target].mean()) if target.any() else None,
            max_probability_elsewhere=float(prob[valid & ~target].max()),
        ))
    total = lambda key: sum(row[key] for row in rows)
    tp, fp, fn = (total(key) for key in ("true_positive_pixels", "false_positive_pixels", "missed_pixels"))
    gt_count = total("gt_components")
    positives = [row for row in rows if row["gt_components"]]
    negatives = [row for row in rows if not row["gt_components"]]
    actor_tp, actor_fp, actor_fn = (total(key) for key in (
        "actor_grid_true_positive_cells", "actor_grid_false_positive_cells", "actor_grid_missed_cells"))
    unmatched_distances = [distance for row in rows for distance in row["unmatched_prediction_min_target_distances_cells"]]
    summary = dict(
        checkpoint=str(checkpoint.resolve()), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        dataset=str((output / "localization-validation.npz").resolve()), images=len(rows),
        positive_images=len(positives), negative_images=len(negatives), threshold=.5,
        pixel_precision=tp / max(tp + fp, 1), pixel_recall=tp / max(tp + fn, 1),
        gt_components=gt_count, exact_component_hits=total("exact_hits"),
        exact_component_recall=total("exact_hits") / max(gt_count, 1),
        one_cell_tolerance_hits=total("one_cell_tolerance_hits"),
        one_cell_tolerance_recall=total("one_cell_tolerance_hits") / max(gt_count, 1),
        negative_images_with_false_positive=sum(row["predicted_pixels"] > 0 for row in negatives),
        true_positive_pixels=tp, false_positive_pixels=fp, missed_pixels=fn,
        false_positive_near_target=total("false_positive_near_target"),
        false_positive_on_bank=total("false_positive_on_bank"),
        false_positive_elsewhere=total("false_positive_elsewhere"),
        actor_20grid_precision=actor_tp / max(actor_tp + actor_fp, 1),
        actor_20grid_recall=actor_tp / max(actor_tp + actor_fn, 1),
        actor_20grid_true_positive_cells=actor_tp,
        actor_20grid_false_positive_cells=actor_fp,
        actor_20grid_missed_cells=actor_fn,
        unmatched_prediction_components_with_target_in_image=len(unmatched_distances),
        unmatched_prediction_components_within_1_5_cells=sum(distance <= 1.5 for distance in unmatched_distances),
        unmatched_prediction_components_within_3_cells=sum(distance <= 3 for distance in unmatched_distances),
        unmatched_prediction_components_beyond_3_cells=sum(distance > 3 for distance in unmatched_distances),
        diagnostic_limit="Rendered 8-connected marker coverage is not physical object-ID matching or avoidance. One-cell tolerance is a separate diagnostic. Independent validation poses are not on-policy rollouts.",
        per_image=rows,
    )
    selections = []
    criteria = [
        ("best target coverage", lambda row: (-row["exact_hits"], -row["true_positive_pixels"], row["index"])),
        ("lowest target confidence", lambda row: (row["max_probability_on_target"], row["index"])),
        ("most off-target pixels", lambda row: (-row["false_positive_pixels"], row["index"])),
    ]
    for label, key in criteria:
        candidates = sorted((row for row in positives if row["index"] not in [x[1] for x in selections]), key=key)
        if candidates:
            selections.append((label, candidates[0]["index"]))
    summary["example_selection"] = [dict(reason=reason, index=index) for reason, index in selections]
    fig, axes = plt.subplots(len(selections), 4, figsize=(14, 3.7 * len(selections)), squeeze=False)
    for row_number, (reason, index) in enumerate(selections):
        target, prob = data["buoy"][index], probability[index]
        rgb = data["rgb"][index].transpose(1, 2, 0)
        entry = rows[index]
        axes[row_number, 0].imshow(rgb)
        axes[row_number, 0].set_title(f"RGB #{index}: {reason}")
        axes[row_number, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[row_number, 1].set_title(f"GT marker: {entry['target_pixels']} cells")
        axes[row_number, 2].imshow(prob, cmap="magma", vmin=0, vmax=1)
        axes[row_number, 2].contour(target, levels=[.5], colors="cyan", linewidths=.5)
        axes[row_number, 2].set_title(f"Probability 0..1; GT max={entry['max_probability_on_target']:.3f}")
        r, c = np.argwhere(target).mean(axis=0) * 2
        r, c = int(r), int(c)
        y0, y1, x0, x1 = max(0, r - 14), min(200, r + 15), max(0, c - 14), min(200, c + 15)
        axes[row_number, 3].imshow(rgb[y0:y1, x0:x1], interpolation="nearest")
        axes[row_number, 3].set_title(f"RGB marker zoom; exact hits {entry['exact_hits']}/{entry['gt_components']}")
        for axis in axes[row_number]:
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(f"Saved buoy head ({checkpoint.name}): heldout views, threshold 0.5; examples include misses", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(figure_path, dpi=140)
    plt.close(fig)
    write_json(metrics_path, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "per_image"}, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
