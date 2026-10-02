"""Single-frame perception using only NumPy and ONNX Runtime at inference time.

This module can later be copied into a submission; it imports no training framework.
It produces perception maps, not vessel control commands or object velocities.
"""
from __future__ import annotations

import json

import numpy as np
import onnxruntime as ort
from training.perception_schema import scheme_for


def prepare_rgb(rgb):
    rgb = np.asarray(rgb)
    if rgb.shape != (200, 200, 3) or rgb.dtype != np.uint8:
        raise ValueError("Expected official 200x200x3 uint8 top-view RGB")
    x = rgb.astype(np.float32) / np.float32(255.0)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


class OnnxPerception:
    def __init__(self, model_path, *, providers=None, threads=2):
        if threads < 1:
            raise ValueError("threads must be positive")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=options,
                                           providers=providers or ["CPUExecutionProvider"])
        metadata = self.session.get_modelmeta().custom_metadata_map
        classes = json.loads(metadata.get("classes", "null"))
        if not isinstance(classes, list):
            raise ValueError("Model is missing its class definitions")
        self.classes = tuple(classes)
        self.label_scheme = scheme_for(self.classes)
        self.vessel_class = self.classes.index("vessel")
        self.static_classes = [i for i, name in enumerate(self.classes) if name not in ("water", "vessel")]
        # Public ego geometry in bow-up pixels: 3x2 m hull, 0.5 m/px, centre at (100,100).
        # No simulator obstacle ground truth is used by this inference module.
        rows, cols = np.meshgrid(np.arange(200) + 0.5, np.arange(200) + 0.5, indexing="ij")
        forward, port = (100.0 - rows) * 0.5, (100.0 - cols) * 0.5
        self.valid = ~((np.abs(forward) <= 1.5) & (np.abs(port) <= 1.0))

    def predict(self, rgb):
        logits = self.session.run(["logits"], {"rgb": prepare_rgb(rgb)})[0][0]
        if logits.shape != (len(self.classes), 200, 200) or not np.isfinite(logits).all():
            raise RuntimeError("Unexpected or non-finite model output")
        scores = np.exp(logits - logits.max(axis=0, keepdims=True))
        probabilities = scores / scores.sum(axis=0, keepdims=True)
        labels = logits.argmax(axis=0).astype(np.uint8)
        labels[~self.valid] = 255
        static = probabilities[self.static_classes].sum(axis=0) * self.valid
        vessel = probabilities[self.vessel_class] * self.valid
        occupancy = static + vessel
        return {
            "labels": labels,
            "class_probabilities": probabilities.transpose(1, 2, 0).copy(),
            "occupancy_probability": occupancy,
            "static_occupancy_probability": static,
            "vessel_probability": vessel,
            "valid": self.valid.copy(),
        }
