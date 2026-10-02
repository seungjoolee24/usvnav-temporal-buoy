"""Read generated NPZ perception shards without exposing world state to the model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from training.perception_schema import CLASS_SCHEMES, COARSE_LABELS, labels_from_raster


def read_manifest(root: Path):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format"] != "usvnav-perception/1" or manifest["condition"] != "1-2":
        raise ValueError("Expected a generated Track 1-2 perception dataset")
    identities = [c["sha256"] for c in manifest["courses"]]
    if len(set(identities)) != len(identities):
        raise ValueError("Repeated course identities; cannot guarantee disjoint splits")
    if set(c["split"] for c in manifest["courses"]) != {"train", "val", "test"}:
        raise ValueError("Dataset must contain train, val and test course splits")
    if manifest["labels"] != COARSE_LABELS:
        raise ValueError("Dataset label definitions differ from the model's classes")
    return manifest


class PerceptionDataset(Dataset):
    def __init__(self, root: Path, split: str, manifest=None, *, label_scheme="coarse"):
        root = Path(root)
        manifest = manifest or read_manifest(root)
        self.label_scheme = label_scheme
        self.classes = CLASS_SCHEMES[label_scheme]
        rgbs, labels, buoys = [], [], []
        self.courses = [c for c in manifest["courses"] if c["split"] == split]
        buoy_id = manifest["raster_classes"].index("buoy")
        for course in self.courses:
            with np.load(root / course["shard"], allow_pickle=False) as data:
                rgb, label, raster = data["rgb"], data["semantic"], data["raster_class"]
                n = course["samples"]
                if rgb.shape != (n, 200, 200, 3) or label.shape != (n, 200, 200) or raster.shape != label.shape:
                    raise ValueError(f"Bad shapes in {course['shard']}")
                if rgb.dtype != np.uint8 or label.dtype != np.uint8:
                    raise ValueError("RGB and semantic labels must be uint8")
                if not set(np.unique(label)) <= {0, 1, 2, 255} or np.any(np.all(label == 255, axis=(1, 2))):
                    raise ValueError("Unknown labels or an entirely ignored sample")
                if not np.array_equal(label, labels_from_raster(raster, manifest["raster_classes"])):
                    raise ValueError("Stored coarse labels disagree with official raster labels")
                label = labels_from_raster(raster, manifest["raster_classes"], scheme=label_scheme)
                rgbs.append(rgb.copy())
                labels.append(label.copy())
                buoys.append((raster == buoy_id) & (label != 255))
        if not rgbs:
            raise ValueError(f"Empty split: {split}")
        self.rgb = np.concatenate(rgbs)
        self.labels = np.concatenate(labels)
        self.buoys = np.concatenate(buoys)

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, i):
        x = torch.from_numpy(self.rgb[i]).permute(2, 0, 1).to(torch.float32) / 255.0
        y = torch.from_numpy(self.labels[i]).to(torch.long)
        return x, y, torch.from_numpy(self.buoys[i])

    def class_weights(self):
        valid = self.labels[self.labels != 255]
        counts = np.bincount(valid, minlength=len(self.classes))
        if np.any(counts == 0):
            missing = [name for i, name in enumerate(self.classes) if counts[i] == 0]
            raise ValueError(f"Training split is missing classes: {missing}")
        weights = 1.0 / np.sqrt(counts / counts.sum())
        return (weights / weights.mean()).astype(np.float32), counts.tolist()
