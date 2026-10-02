"""Trainable RGB buoy localization and public-coordinate residual control.

The observation contract matches RgbNavigationEnv. For this static-buoy stage,
the image encoder uses the newest RGB frame at its original 200x200 resolution;
the public motion/history branch still receives the full supplied history.
A stride-one stem and local max pooling retain small image signals before a
learned 100x100 heatmap is pooled into a spatial 20x20 policy input. There is no
fixed color detector, and no object label or true geometry enters the actor.

SB3 normalize_images=True converts actor RGB to float/255 once. Standalone
supervised warmup must likewise pass already-normalized floating RGB. Teacher
labels are used only by segmentation_loss, never by forward or prediction.
This architecture is incompatible with old RgbNavigationFeatures checkpoints.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from training.buoy_loss_candidate import hard_negative_buoy_loss


IMAGE_SIZE = 200
HEATMAP_SIZE = 100
SPATIAL_GRID = 20


def global_negative_buoy_loss(logits, labels, valid=None, *, dice_weight=.1):
    """Original class-balanced loss, retained for controlled comparisons.

    Labels and optional valid masks have shape [B,H,W] or [B,1,H,W]. Positive
    and negative pixels are averaged separately per image so a one-pixel buoy
    does not disappear in a background-dominated mean. An all-negative image
    trains its negative pixels normally. Invalid/ego pixels are ignored, and
    an all-invalid batch gives differentiable zero loss. Dice is applied only
    to images containing a valid positive target.
    """
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("Expected buoy logits [B,1,H,W]")
    target = torch.as_tensor(labels, dtype=logits.dtype, device=logits.device)
    if target.ndim == 3:
        target = target.unsqueeze(1)
    mask = torch.ones_like(target) if valid is None else torch.as_tensor(
        valid, dtype=logits.dtype, device=logits.device)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if target.shape != logits.shape or mask.shape != logits.shape:
        raise ValueError("Labels and valid masks must match the buoy logit grid")
    if not torch.isfinite(target).all() or not torch.isfinite(mask).all():
        raise ValueError("Labels and valid masks must be finite")
    if ((target < 0) | (target > 1)).any() or ((mask < 0) | (mask > 1)).any():
        raise ValueError("Labels and valid masks must be in [0,1]")
    if dice_weight < 0:
        raise ValueError("dice_weight must be nonnegative")

    axes = (1, 2, 3)
    positive, negative = target * mask, (1. - target) * mask
    positive_count, negative_count = positive.sum(axes), negative.sum(axes)
    positive_loss = (F.softplus(-logits) * positive).sum(axes) / positive_count.clamp_min(1.)
    negative_loss = (F.softplus(logits) * negative).sum(axes) / negative_count.clamp_min(1.)
    has_positive, has_negative = positive_count > 0, negative_count > 0
    both = has_positive & has_negative
    balanced_bce = torch.where(both, .5 * (positive_loss + negative_loss),
                               torch.where(has_positive, positive_loss, negative_loss))
    probability = logits.sigmoid() * mask
    dice = 1. - (2. * (probability * target).sum(axes) + 1.) / (
        probability.sum(axes) + positive_count + 1.)
    per_image = balanced_bce + dice_weight * torch.where(has_positive, dice, 0.)
    has_valid = has_positive | has_negative
    return (per_image * has_valid.to(logits.dtype)).sum() / has_valid.sum().clamp_min(1)


def buoy_segmentation_loss(logits, labels, valid=None, *, dice_weight=.1,
                           hard_fraction=.02, hard_mix=.5):
    """Default binary auxiliary loss after a controlled bank-FP A/B check.

    The negative term mixes its global mean with its hardest valid 2% equally;
    positives and ignored ego cells do not enter the mined negative term.
    This changes supervision only: actor inputs and architecture are unchanged.
    """
    return hard_negative_buoy_loss(logits, labels, valid, dice_weight=dice_weight,
                                  hard_fraction=hard_fraction, hard_mix=hard_mix)


class BuoyNavigationFeatures(BaseFeaturesExtractor):
    """Learned high-resolution heatmap, scene features and direct public state."""

    def __init__(self, observation_space):
        rgb_shape = observation_space["rgb"].shape
        if (len(rgb_shape) != 3 or rgb_shape[0] < 3 or rgb_shape[0] % 3
                or rgb_shape[1:] != (IMAGE_SIZE, IMAGE_SIZE)):
            raise ValueError("Expected CHW RGB history with native 200x200 frames")
        waypoint_shape = observation_space["waypoints"].shape
        history_size = math.prod(observation_space["history"].shape)
        direct_size = (math.prod(observation_space["state"].shape) + waypoint_shape[-1]
                       + math.prod(observation_space["goal"].shape)
                       + math.prod(observation_space["reference"].shape) + history_size)
        super().__init__(observation_space, features_dim=64 + SPATIAL_GRID**2 + 32 + direct_size)
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 8, 3, stride=1, padding=1), nn.ReLU(),
        )
        self.buoy_head = nn.Conv2d(8, 1, 1)
        self.heatmap_pool = nn.AdaptiveMaxPool2d((SPATIAL_GRID, SPATIAL_GRID))
        self.scene_encoder = nn.Sequential(
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5)), nn.Flatten(), nn.Linear(400, 64), nn.ReLU(),
        )
        self.route_encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(math.prod(waypoint_shape), 32), nn.ReLU())
        self.latest_buoy_logits = None

    @staticmethod
    def _latest_rgb(rgb):
        if (rgb.ndim != 4 or rgb.shape[1] < 3 or rgb.shape[1] % 3
                or rgb.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE)):
            raise ValueError("Expected RGB [B,3*K,200,200]")
        if not rgb.is_floating_point():
            raise TypeError("Pass normalized floating RGB; divide raw uint8 by 255 exactly once")
        return rgb[:, -3:]

    def buoy_logits(self, rgb):
        """Predict [B,1,100,100] from normalized RGB or RGB history only."""
        return self.buoy_head(self.rgb_encoder(self._latest_rgb(rgb)))

    def segmentation_loss(self, labels, valid=None, *, rgb=None, dice_weight=.1):
        """Use explicit normalized RGB, or logits from the latest forward call.

        Supervised warmup should supply rgb so it does not depend on a cached
        inference graph. During a joint forward/auxiliary update, omitting rgb
        uses the current forward graph. Call before that graph is backpropagated.
        """
        logits = self.buoy_logits(rgb) if rgb is not None else self.latest_buoy_logits
        if logits is None:
            raise RuntimeError("Supply normalized rgb or call forward before segmentation_loss")
        return buoy_segmentation_loss(logits, labels, valid, dice_weight=dice_weight)

    def forward(self, obs):
        image = self.rgb_encoder(self._latest_rgb(obs["rgb"]))
        logits = self.buoy_head(image)
        self.latest_buoy_logits = logits
        heat_features = self.heatmap_pool(logits.sigmoid()).flatten(1)
        route = torch.where(obs["waypoints"][..., :1] > .5, obs["waypoints"], 0.)
        return torch.cat([
            self.scene_encoder(image), heat_features, self.route_encoder(route),
            obs["state"], route[:, 0], obs["goal"], obs["reference"], obs["history"].flatten(1),
        ], dim=1)


def buoy_policy_kwargs(*, log_std_init=-2.3):
    """SB3 policy settings; caller may zero the final residual mean for a prior."""
    return dict(features_extractor_class=BuoyNavigationFeatures, normalize_images=True,
                share_features_extractor=True, net_arch=dict(pi=[64, 64], vf=[64, 64]),
                log_std_init=log_std_init, activation_fn=nn.Tanh)
