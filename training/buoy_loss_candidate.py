"""Hard-negative auxiliary loss for rendered buoy localization masks.

Used by buoy_policy after a controlled cloned-model A/B comparison. It adds focused
supervision for false-positive shoreline pixels by mixing the mean negative
loss with the hardest valid negative pixels. Positive class weighting and Dice
are retained; teacher targets remain training-only. It changes no actor input
or architecture and does not imply that localization improves navigation.
"""
import math

import torch
from torch.nn import functional as F


def hard_negative_buoy_loss(logits, labels, valid=None, *, dice_weight=.1,
                            hard_fraction=.02, hard_mix=.5):
    """Balanced binary BCE/Dice with top-2%-negative mining per image.

    Input contracts match the existing loss for binary labels/valid masks.
    hard_mix=0 recovers its global negative mean; hard_mix=.5 gives half global
    and half hardest negatives. Mining excludes positive and invalid/ego cells.
    Negative-only images receive the same focused negative supervision. Empty
    valid masks return differentiable zero. No actor architecture is changed.
    """
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("Expected buoy logits [B,1,H,W]")
    if not 0 < hard_fraction <= 1 or not 0 <= hard_mix <= 1 or dice_weight < 0:
        raise ValueError("Need fraction in (0,1], mix in [0,1], and nonnegative Dice weight")
    target = torch.as_tensor(labels, dtype=logits.dtype, device=logits.device)
    if target.ndim == 3:
        target = target.unsqueeze(1)
    mask = torch.ones_like(target) if valid is None else torch.as_tensor(
        valid, dtype=logits.dtype, device=logits.device)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if target.shape != logits.shape or mask.shape != logits.shape:
        raise ValueError("Labels and valid masks must match the buoy logit grid")
    if not ((target == 0) | (target == 1)).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("Hard-negative candidate requires binary labels and valid masks")

    axes = (1, 2, 3)
    positive, negative = target * mask, (1. - target) * mask
    positive_count, negative_count = positive.sum(axes), negative.sum(axes)
    positive_loss = (F.softplus(-logits) * positive).sum(axes) / positive_count.clamp_min(1.)
    negative_losses = F.softplus(logits)
    global_negative_loss = (negative_losses * negative).sum(axes) / negative_count.clamp_min(1.)
    hard_losses = []
    for image_logits, image_negative, image_losses in zip(logits, negative, negative_losses):
        eligible = image_losses[image_negative.bool()]
        if eligible.numel():
            count = max(1, math.ceil(eligible.numel() * hard_fraction))
            hard_losses.append(torch.topk(eligible, count, sorted=False).values.mean())
        else:
            hard_losses.append(image_logits.sum() * 0.)
    hard_negative_loss = torch.stack(hard_losses)
    negative_loss = (1. - hard_mix) * global_negative_loss + hard_mix * hard_negative_loss
    has_positive, has_negative = positive_count > 0, negative_count > 0
    balanced_bce = torch.where(has_positive & has_negative, .5 * (positive_loss + negative_loss),
                               torch.where(has_positive, positive_loss, negative_loss))
    probability = logits.sigmoid() * mask
    dice = 1. - (2. * (probability * target).sum(axes) + 1.) / (
        probability.sum(axes) + positive_count + 1.)
    per_image = balanced_bce + dice_weight * torch.where(has_positive, dice, 0.)
    has_valid = has_positive | has_negative
    return (per_image * has_valid.to(logits.dtype)).sum() / has_valid.sum().clamp_min(1)
