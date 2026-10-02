"""Colour science for the palette constraint (4-V9) and the checks 4-V13 requires.

4-V13 says the general-position check "ships as a test in the renderer package, so
editing a colour later cannot silently break it". This module is the check; the
assertions on its output are in `tests/test_palette.py`.

**This checks the blends the renderer can actually emit, not an idealised segment.**
`render.render` averages the palette's sRGB values over `subsamples**2` subsample points
and rounds, so with the Track 1 default of 8 the reachable colour at a boundary pixel is
exactly `round(sum(k_i * c_i) / 64)` with the `k_i` summing to 64. That set is finite and
is enumerated exhaustively rather than sampled.

Two things follow from measuring the real thing:

*The margin is measured in CIELAB but the mixing is linear in sRGB*, so the reachable set
between two classes is a curve in CIELAB, not the straight segment `palette_check.py`
used. The curve can only bow away from or towards a third class, so the idealised number
is an estimate in an unknown direction and is reported separately.

*4-V9 is stated over segments, i.e. over two-class blends.* A pixel where three classes
meet carries a three-way blend, which lies inside a CIELAB triangle rather than on any of
its edges, so the segment check does not cover it. `palette_margins` measures mixes of up
to `max_mix` other classes for that reason.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

_XYZ_FROM_LINEAR = np.array([[0.4124, 0.3576, 0.1805],
                             [0.2126, 0.7152, 0.0722],
                             [0.0193, 0.1192, 0.9505]])
_D65 = np.array([0.95047, 1.0, 1.08883])


def srgb_to_linear(rgb) -> np.ndarray:
    c = np.asarray(rgb, dtype=float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin) -> np.ndarray:
    c = np.clip(np.asarray(lin, dtype=float), 0.0, 1.0)
    return 255.0 * np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def srgb_to_lab(rgb) -> np.ndarray:
    """sRGB in 0..255 -> CIELAB (D65). Works on any (..., 3) array."""
    xyz = srgb_to_linear(rgb) @ _XYZ_FROM_LINEAR.T / _D65
    eps = 216.0 / 24389.0
    f = np.where(xyz > eps, np.cbrt(np.maximum(xyz, 0.0)), (841.0 / 108.0) * xyz + 4.0 / 29.0)
    return np.stack([116.0 * f[..., 1] - 16.0,
                     500.0 * (f[..., 0] - f[..., 1]),
                     200.0 * (f[..., 1] - f[..., 2])], axis=-1)


def deuteranopia(rgb) -> np.ndarray:
    """Vienot (1999) deuteranope simulation, applied in linear RGB."""
    r, g, b = np.moveaxis(srgb_to_linear(rgb), -1, 0)
    L = 17.8824 * r + 43.5161 * g + 4.11935 * b
    M = 3.45565 * r + 27.1554 * g + 3.86714 * b
    S = 0.0299566 * r + 0.184309 * g + 1.46709 * b
    M2 = 0.494207 * L + 1.24827 * S
    out = np.stack([0.080944 * L - 0.130504 * M2 + 0.116721 * S,
                    -0.0102485 * L + 0.0540194 * M2 - 0.113615 * S,
                    -0.000365294 * L - 0.00412163 * M2 + 0.693513 * S], axis=-1)
    return linear_to_srgb(out)


def _compositions(total: int, parts: int) -> np.ndarray:
    """Every way to write `total` as `parts` strictly positive integers, as (N, parts)."""
    rows = [c for c in itertools.combinations(range(1, total), parts - 1)]
    if parts == 1:
        return np.array([[total]])
    cuts = np.array(rows, dtype=np.int64)
    bounds = np.concatenate([np.zeros((len(cuts), 1), np.int64), cuts,
                             np.full((len(cuts), 1), total, np.int64)], axis=1)
    return np.diff(bounds, axis=1)


def reachable_blends(colours: np.ndarray, subsamples: int) -> np.ndarray:
    """Every colour `render` can emit from a pixel covered by exactly these classes.

    `colours` is (k, 3). Returns (N, 3) float, before rounding -- the caller rounds, so
    the arithmetic matches `render.render` exactly.
    """
    k = len(colours)
    n = subsamples * subsamples
    if k == 1:
        return np.asarray(colours, dtype=float)
    w = _compositions(n, k) / n
    return w @ np.asarray(colours, dtype=float)


def palette_margins(palette, *, subsamples: int = 8, max_mix: int = 3) -> dict:
    """Measure every margin 4-V13 reports, plus the mixes 4-V9 does not cover.

    Returns metres-free numbers, all in CIELAB units:

    `general_position` -- the smallest distance from a class colour to any two-class
    blend of two *other* classes that the renderer can emit. This is 4-V9's margin,
    measured on the real blend space.

    `lab_segment` -- the same quantity computed on the idealised straight CIELAB
    segment, which is what `palette_check.py` reported. Kept so the recorded 19.0 stays
    reproducible and so the size of the idealisation is visible.

    `mix_<n>` -- the same, for blends of `n` other classes (n = 3 is the pixel where
    three regions meet).

    `lstar_gap` -- smallest gap in `L*`, i.e. survival of a grayscale conversion.

    `deuteranopia` -- smallest pairwise CIELAB distance after simulating deuteranopia.
    """
    names = list(palette.keys())
    rgb = np.array([palette[n] for n in names], dtype=float)
    lab = srgb_to_lab(rgb)
    k = len(names)

    out: dict[str, float] = {}
    worst: dict[str, tuple] = {}

    for m in range(2, max_mix + 1):
        best, where = math.inf, None
        for target in range(k):
            others = [i for i in range(k) if i != target]
            for combo in itertools.combinations(others, m):
                blends = np.rint(reachable_blends(rgb[list(combo)], subsamples))
                d = np.linalg.norm(srgb_to_lab(blends) - lab[target], axis=-1).min()
                if d < best:
                    best, where = float(d), (names[target], tuple(names[i] for i in combo))
        key = "general_position" if m == 2 else f"mix_{m}"
        out[key], worst[key] = best, where

    # The idealised straight-segment version, for comparison with the recorded figure.
    best, where = math.inf, None
    for target in range(k):
        for i, j in itertools.combinations([x for x in range(k) if x != target], 2):
            d = _point_segment(lab[target], lab[i], lab[j])
            if d < best:
                best, where = d, (names[target], (names[i], names[j]))
    out["lab_segment"], worst["lab_segment"] = best, where

    ls = np.sort(lab[:, 0])
    out["lstar_gap"] = float(np.min(np.diff(ls)))

    dl = srgb_to_lab(deuteranopia(rgb))
    best, where = math.inf, None
    for i, j in itertools.combinations(range(k), 2):
        d = float(np.linalg.norm(dl[i] - dl[j]))
        if d < best:
            best, where = d, (names[i], names[j])
    out["deuteranopia"], worst["deuteranopia"] = best, where

    out["_worst"] = worst
    return out


def _point_segment(p, a, b) -> float:
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom == 0.0 else float((p - a) @ ab) / denom
    t = min(1.0, max(0.0, t))
    return float(np.linalg.norm(p - (a + t * ab)))


def nearest_class(rgb, palette) -> np.ndarray:
    """Nearest palette entry in CIELAB, as an index into `list(palette)`.

    A plain colour-lookup classifier. Since 2026-09-16 the 1-2 picture is the studio's
    (`usvnav.look`): flat base colours inside a class, with shoreline, shadow, outline and
    anti-aliased edge tones around them and no noise unless the participant adds some --
    so nearest colour to the base palette is a fair first classifier, not an exact one.
    """
    names = list(palette.keys())
    ref = srgb_to_lab(np.array([palette[n] for n in names], dtype=float))
    lab = srgb_to_lab(np.asarray(rgb, dtype=float))
    d = np.linalg.norm(lab[..., None, :] - ref, axis=-1)
    return np.argmin(d, axis=-1)
