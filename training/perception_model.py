"""Small spatial encoder/decoder and losses for Track 1-2 perception pretraining."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from training.perception_schema import COARSE_CLASSES, IGNORE, scheme_for

CLASSES = COARSE_CLASSES  # Preserve the original three-class checkpoint interface.


class ConvBlock(nn.Sequential):
    def __init__(self, inputs: int, outputs: int):
        super().__init__(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(4, outputs), nn.ReLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(4, outputs), nn.ReLU(),
        )


class TinyUNet(nn.Module):
    """Two downsampling stages, full-resolution skips, configurable logits/pixel.

    Input: N x 3 x H x W float RGB in [0,1]. No simulator ground truth as input.
    `encode` exposes spatial features for a later temporal perception module.
    """
    def __init__(self, base_channels: int = 8, classes=CLASSES):
        super().__init__()
        if base_channels < 4 or base_channels % 4:
            raise ValueError("base_channels must be a positive multiple of four, at least four")
        self.base_channels = base_channels
        self.classes = tuple(classes)
        self.label_scheme = scheme_for(self.classes)
        b = base_channels
        self.enc1 = ConvBlock(3, b)
        self.enc2 = ConvBlock(b, 2 * b)
        self.bottleneck = ConvBlock(2 * b, 4 * b)
        self.dec2 = ConvBlock(6 * b, 2 * b)
        self.dec1 = ConvBlock(3 * b, b)
        self.head = nn.Conv2d(b, len(self.classes), 1)

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        z = self.bottleneck(F.max_pool2d(e2, 2))
        return e1, e2, z

    def forward(self, x):
        e1, e2, z = self.encode(x)
        d2 = self.dec2(torch.cat([F.interpolate(z, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1))
        return self.head(d1)


class SegmentationLoss(nn.Module):
    """Weighted CE plus foreground Dice; all terms exclude the ego hull.

    Buoy pixel weights use simulator labels ONLY to supervise training, never as input.
    Class weights must be computed from the training split alone.
    """
    def __init__(self, class_weights, dice_weight=0.5, buoy_weight=5.0):
        super().__init__()
        self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        self.dice_weight, self.buoy_weight = float(dice_weight), float(buoy_weight)

    def forward(self, logits, target, buoy_pixels):
        valid = target != IGNORE
        safe = target.masked_fill(~valid, 0)
        weights = self.class_weights[safe] * torch.where(buoy_pixels, self.buoy_weight, 1.0) * valid
        ce = F.cross_entropy(logits, target, ignore_index=IGNORE, reduction="none")
        ce = (ce * weights).sum() / weights.sum().clamp_min(1e-8)
        probs = logits.softmax(dim=1) * valid[:, None]
        one_hot = F.one_hot(safe, len(self.class_weights)).permute(0, 3, 1, 2).to(logits.dtype) * valid[:, None]
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)[1:]
        target_area = one_hot.sum(dims)[1:]
        denominator = probs.sum(dims)[1:] + target_area
        dice = 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        present = target_area > 0
        dice = (dice * present).sum() / present.sum().clamp_min(1)
        return ce + self.dice_weight * dice


class SegmentationMetrics:
    """Aggregate a confusion matrix over all valid pixels, rather than image averages."""
    def __init__(self, classes=CLASSES):
        self.classes = tuple(classes)
        scheme_for(self.classes)
        self.n_classes = len(self.classes)
        self.buoy_class = self.classes.index("buoy" if "buoy" in self.classes else "fixed_obstacle")
        self.confusion = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
        self.buoy_correct = self.buoy_total = 0

    def update(self, prediction, target, buoy_pixels):
        pred, truth = np.asarray(prediction), np.asarray(target)
        valid = truth != IGNORE
        if np.any((truth[valid] < 0) | (truth[valid] >= self.n_classes)) or np.any((pred[valid] < 0) | (pred[valid] >= self.n_classes)):
            raise ValueError("Metric inputs contain unknown classes")
        self.confusion += np.bincount(self.n_classes * truth[valid].astype(np.int64) + pred[valid],
                                    minlength=self.n_classes ** 2).reshape(self.n_classes, self.n_classes)
        buoys = np.asarray(buoy_pixels, dtype=bool) & valid
        self.buoy_total += int(buoys.sum())
        self.buoy_correct += int((pred[buoys] == self.buoy_class).sum())

    def report(self):
        cm = self.confusion
        true_count, predicted_count = cm.sum(axis=1), cm.sum(axis=0)
        intersection = np.diag(cm)
        union = true_count + predicted_count - intersection
        iou = {name: float(intersection[i] / union[i]) if union[i] else None for i, name in enumerate(self.classes)}
        available = [v for v in iou.values() if v is not None]
        return {
            "miou": float(np.mean(available)) if available else None,
            "class_iou": iou,
            "class_recall": {name: float(intersection[i] / true_count[i]) if true_count[i] else None for i, name in enumerate(self.classes)},
            "class_true_pixels": {name: int(true_count[i]) for i, name in enumerate(self.classes)},
            "pixel_accuracy": float(intersection.sum() / cm.sum()) if cm.sum() else None,
            "buoy_pixel_recall": self.buoy_correct / self.buoy_total if self.buoy_total else None,
            "buoy_pixels": self.buoy_total, "valid_pixels": int(cm.sum()),
            "confusion_matrix_rows_true_columns_predicted": cm.tolist(),
        }
