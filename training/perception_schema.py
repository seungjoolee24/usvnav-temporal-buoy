"""Observable RGB classes; usable without PyTorch in a submission.

Motion is a separate temporal attribute. Rocks share the official `pier` class.
Shadows, shallow-water colours and decks do not change the geometry labels.
"""
from __future__ import annotations

import numpy as np

IGNORE = 255
COARSE_CLASSES = ("water", "fixed_obstacle", "vessel")
DETAIL_CLASSES = ("water", "bank", "buoy", "pier", "dock", "vessel")
CLASS_SCHEMES = {"coarse": COARSE_CLASSES, "detail": DETAIL_CLASSES}
COARSE_LABELS = {"water": 0, "fixed_obstacle_including_bank": 1, "vessel": 2, "ignore_ego": IGNORE}
COARSE_TARGET_IDS = {"water": 0, "bank": 1, "buoy": 1, "pier": 1,
                     "dock": 1, "vessel": 2, "ego": IGNORE, "unobserved": IGNORE}
COARSE_COLOURS = np.array([[35, 78, 108], [162, 109, 65], [38, 193, 168]], dtype=np.uint8)
DETAIL_COLOURS = np.array([[35, 78, 108], [176, 153, 107], [240, 200, 32],
                          [155, 101, 199], [184, 101, 53], [38, 193, 168]], dtype=np.uint8)


def scheme_for(classes):
    classes = tuple(classes)
    for name, known in CLASS_SCHEMES.items():
        if classes == known:
            return name
    raise ValueError(f"Unsupported perception classes: {classes}")


def labels_from_raster(ids, names, *, scheme="coarse"):
    """Map official pixel-centre ground truth to observable classes, never RGB input."""
    classes = CLASS_SCHEMES[scheme]
    ids = np.asarray(ids)
    if not np.issubdtype(ids.dtype, np.integer) or ids.size == 0 or ids.min() < 0 or ids.max() >= len(names):
        raise ValueError("Invalid raster class IDs")
    if len(set(names)) != len(names) or set(names) - COARSE_TARGET_IDS.keys():
        raise ValueError("Unknown or repeated official raster classes")
    if "unobserved" in names and np.any(ids == names.index("unobserved")):
        raise ValueError("Condition 1-2 must not contain unobserved pixels")
    mapping = COARSE_TARGET_IDS if scheme == "coarse" else {
        **{name: i for i, name in enumerate(classes)}, "ego": IGNORE, "unobserved": IGNORE,
    }
    return np.asarray([mapping[name] for name in names], dtype=np.uint8)[ids]


def colour_labels(target, classes=COARSE_CLASSES):
    scheme = scheme_for(classes)
    target = np.asarray(target)
    valid = target != IGNORE
    if np.any((target[valid] < 0) | (target[valid] >= len(classes))):
        raise ValueError("Invalid semantic class IDs")
    rgb = np.full((*target.shape, 3), 210, dtype=np.uint8)
    palette = COARSE_COLOURS if scheme == "coarse" else DETAIL_COLOURS
    rgb[valid] = palette[target[valid]]
    return rgb
