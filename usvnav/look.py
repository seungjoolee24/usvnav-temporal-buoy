"""The look of a river: how a class map becomes the picture the studio shows and -- since
2026-09-16, by the owner's decision -- the picture a Track 1 agent receives in 1-2 and 1-4.

One code path for both. `figure.map_view` paints the whole course with it for the eye;
`render.render` paints the agent's 100 m window with it when `RasterConfig.look` is set,
which it is for Track 1. What a team sees in the studio is therefore what its agent gets,
not a flat-coloured stand-in for it.

Every treatment is stated in **metres**, not pixels, so the map at 0.2 m/px and the raster's
fine pass at 0.25 m/px draw the same river:

1. **Noise, only if asked for.** The scorer renders with none (4-V14, as revised
   2026-09-16): the 1-2 / 1-4 picture a submission is scored on is flat inside every class.
   Noise is the participant's to add -- Track 2's renders carry it; here it is an exercise
   -- through `RasterConfig.noise`, a function `(class_name, x_world, y_world) -> offsets`
   applied before the treatments below. The kit ships one example, `grain`: value noise
   per class in blocks of a fixed size in metres, anchored to **world** coordinates by a
   hash, so it moves with the scene and is a pure function of position.
2. **A shoreline.** Wet ground within `WATERLINE_M` of the water, shallower murkier water
   within `SHALLOW_M` of the bank.
3. **Shadows, decks, outlines.** Bodies drop a soft shadow down-right (`SHADOW_M`), piers,
   vessels and docks carry a lighter inset deck, and every body is outlined.

The renderer applies this at `subsamples` times the raster's resolution and box-downsamples,
which anti-aliases every edge (4-V8, as revised 2026-09-16).
"""

from __future__ import annotations

import math
import zlib

import numpy as np

from .world import BANK, BUOY, DOCK, EGO, OUTSIDE_WORLD, PIER, UNOBSERVED, WATER

#: `render.VESSEL`; spelled here because this module must not import the renderer.
VESSEL = "vessel"

#: The base colours: Track 1's palette (`render.PALETTE_TRACK1` is this dict).
BASE = {
    UNOBSERVED: (0x00, 0x00, 0x00),
    WATER:      (0x16, 0x30, 0x3B),
    DOCK:       (0x6B, 0x4F, 0x35),
    PIER:       (0x7E, 0x7B, 0x73),
    BANK:       (0x77, 0x72, 0x5C),
    VESSEL:     (0x53, 0x60, 0x6B),
    EGO:        (0xF2, 0x6E, 0x7A),
    BUOY:       (0xF0, 0xC8, 0x20),
}

#: Per class: base colour and the outline / waterline colours. `outside_world` is a
#: map-only class (4-V12).
MAP_LOOK = {
    WATER:  dict(rgb=BASE[WATER]),
    BANK:   dict(rgb=BASE[BANK], waterline=(0x3E, 0x3B, 0x30)),
    PIER:   dict(rgb=BASE[PIER], edge=(0x2E, 0x2C, 0x28)),
    VESSEL: dict(rgb=BASE[VESSEL], edge=(0x1E, 0x25, 0x2B)),
    DOCK:   dict(rgb=BASE[DOCK], edge=(0x2A, 0x1F, 0x14)),
    BUOY:   dict(rgb=BASE[BUOY], edge=(0x3A, 0x30, 0x08)),
    EGO:    dict(rgb=BASE[EGO], edge=(0x40, 0x18, 0x1C)),
    OUTSIDE_WORLD: dict(rgb=(0x10, 0x12, 0x10)),
    UNOBSERVED:    dict(rgb=BASE[UNOBSERVED]),
}

#: The kit's one example of renderer noise: per class, the block size in metres and the
#: amplitude in uint8 levels of `grain`. Nothing applies it unless asked (see `NOISES`).
GRAIN_EXAMPLE = {
    WATER: (0.6, 4), BANK: (1.0, 11), PIER: (0.4, 6), VESSEL: (0.4, 5), DOCK: (0.4, 9),
    OUTSIDE_WORLD: (1.2, 5),
}

#: Classes that are objects on or in the water: they get a shadow and an outline. The bank
#: is the ground and gets neither.
BODIES = (PIER, VESSEL, DOCK, BUOY, EGO)

#: Small bodies wear their outline *outside* the footprint, as a ring on the water: a 1.0 m
#: buoy marker with a 0.2 m outline inside it would be three-quarters outline and lose its
#: colour after downsampling. The ring adds 0.2 m to what the raster over-states already
#: (the minimum marker), in the same conservative direction; collision keeps the true size.
RINGED = (BUOY, EGO)

WATERLINE_M = 2.5          # wet ground on the bank side of the shoreline
SHALLOW_M = 3.5            # murkier water on the water side
SHALLOW_RGB = (0x24, 0x44, 0x48)
SHADOW_M = 0.6             # drop-shadow offset, down-right, one light direction
DECK_INSET_M = 1.2         # the lighter deck panel inside piers, vessels and docks
OUTLINE_M = 0.2            # outline thickness


def world_coords(cfg, observer, h: int, w: int):
    """World `(x, y)` of every pixel centre of an `h x w` image drawn with `cfg` from
    `observer = (x, y, psi)`. Inverts 1-W3 for the `bow_up` frame and the map frame."""
    ox, oy, psi = (float(v) for v in observer)
    o, s = cfg.origin_px, cfg.m_per_px
    rows = np.arange(h, dtype=float) + 0.5
    cols = np.arange(w, dtype=float) + 0.5
    if cfg.frame == "map":
        py = (o - rows)[:, None] * s
        px = (cols - o)[None, :] * s
        return ox + px + 0.0 * py, oy + py + 0.0 * px
    xb = (o - rows)[:, None] * s            # along the bow
    yb = (o - cols)[None, :] * s            # to port
    c, sn = math.cos(psi), math.sin(psi)
    return ox + xb * c - yb * sn, oy + xb * sn + yb * c


def _hash(ix: np.ndarray, iy: np.ndarray, salt: int) -> np.ndarray:
    """A 64-bit mix of two block indices and a per-class salt. Pure; wraps on overflow."""
    with np.errstate(over="ignore"):
        h = (ix.astype(np.int64).astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
             ^ iy.astype(np.int64).astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
             ^ np.uint64((salt * 0x165667B19E3779F9) & 0xFFFFFFFFFFFFFFFF))
        h ^= h >> np.uint64(31); h *= np.uint64(0x7FB5D329728EA185)
        h ^= h >> np.uint64(27); h *= np.uint64(0x81DADEF4BC2DD44D)
        h ^= h >> np.uint64(33)
    return h


def grain_field(xw: np.ndarray, yw: np.ndarray, block_m: float, amplitude: int, salt: int) -> np.ndarray:
    """Blocky value noise in `[-amplitude, amplitude]`, a pure function of world position."""
    if amplitude <= 0:
        return np.zeros(xw.shape, dtype=np.int16)
    ix = np.floor(xw / block_m); iy = np.floor(yw / block_m)
    h = _hash(ix, iy, salt)
    return (h % np.uint64(2 * amplitude + 1)).astype(np.int16) - np.int16(amplitude)


def grain(class_name: str, xw: np.ndarray, yw: np.ndarray) -> np.ndarray:
    """**The example noise function** -- the shape every noise function has.

    Takes the class name of the pixels being painted and the world `x`, `y` of every pixel
    centre (metres, arrays of the picture's shape); returns integer offsets of that shape,
    added to the class's base colour before the shoreline, shadows and outlines are drawn.
    This one is value noise in world-anchored blocks (`GRAIN_EXAMPLE`), so it travels with
    the scene and repeats exactly for the same place. Write your own with this signature
    and hand it to `RasterConfig.noise`, `--noise module:function`, or the studio.
    """
    spec = GRAIN_EXAMPLE.get(class_name)
    if spec is None:
        return np.zeros(xw.shape, dtype=np.int16)
    block_m, amplitude = spec
    return grain_field(xw, yw, block_m, amplitude, zlib.crc32(class_name.encode()) & 0xFFFF)


#: What `--noise` and the studio's selector accept by name. `"none"` is the scorer's setting.
NOISES = {"none": None, "grain": grain}


def resolve_noise(spec):
    """`None` / `"none"` -> no noise; `"grain"` -> the example; a callable -> itself;
    `"module:function"` -> that function, imported (the caller puts its folder on `sys.path`)."""
    if spec is None or callable(spec):
        return spec
    spec = str(spec).strip()
    if spec in NOISES:
        return NOISES[spec]
    if ":" in spec:
        import importlib
        mod, _, fn = spec.partition(":")
        target = getattr(importlib.import_module(mod), fn, None)
        if not callable(target):
            raise ValueError(f"{spec!r}: {mod} has no function {fn}")
        return target
    raise ValueError(f"unknown noise {spec!r}: one of {', '.join(NOISES)} or module:function")


def paint(ids: np.ndarray, names, cfg, observer, noise=None) -> np.ndarray:
    """A class-id map -> a picture of a river. `names[i]` is the class of id `i`; `noise`,
    if given, is a function `(class_name, x_world, y_world) -> offsets` (see `grain`)."""
    h, w = ids.shape
    names = list(names)
    xw, yw = world_coords(cfg, observer, h, w)
    out = np.zeros((h, w, 3), dtype=np.int16)

    # 1 -- base colour, plus the caller's noise if any
    for i, name in enumerate(names):
        look = MAP_LOOK.get(name)
        if look is None:
            continue
        m = ids == i
        if not m.any():
            continue
        out[m] = look["rgb"]
        if noise is not None:
            out[m] += np.asarray(noise(name, xw, yw), dtype=np.int16)[m][:, None]

    px = lambda metres: max(1, int(round(metres / cfg.m_per_px)))

    # 2 -- the shoreline, both sides of it
    if WATER in names and BANK in names:
        water = ids == names.index(WATER)
        bank = ids == names.index(BANK)
        if bank.any() and water.any():
            wet = _dilate(water, px(WATERLINE_M)) & bank
            out[wet] = out[wet] * 0.30 + np.array(MAP_LOOK[BANK]["waterline"], dtype=np.int16) * 0.70
            shallow = _dilate(bank, px(SHALLOW_M)) & water
            out[shallow] = out[shallow] * 0.45 + np.array(SHALLOW_RGB, dtype=np.int16) * 0.55

    body = np.zeros((h, w), dtype=bool)
    for name in BODIES:
        if name in names:
            body |= ids == names.index(name)

    # 3 -- one light direction, so the shadow offset is the same for every body
    if body.any():
        off = px(SHADOW_M)
        shade = np.zeros_like(body)
        shade[off:, off:] = body[:-off, :-off]
        shade &= ~body
        out[shade] = (out[shade] * 0.62).astype(np.int16)

    # 3b -- a deck: an inset panel, lightened, so a hull reads as a hull
    for name in BODIES:
        if name not in names or name in (BUOY, EGO):
            continue
        m = ids == names.index(name)
        if m.any():
            inner = _erode(m, px(DECK_INSET_M))
            if inner.any():
                out[inner] = np.minimum(out[inner] + 24, 255)

    # 4 -- outlines, `OUTLINE_M` thick, in the class's edge colour: inside the footprint
    #      for piers, vessels and docks; a ring outside it for the small bodies (`RINGED`)
    if body.any():
        diff = np.zeros_like(body)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            diff |= np.roll(ids, (dr, dc), (0, 1)) != ids
        rim = _dilate(diff & body, px(OUTLINE_M) - 1) & body
        for name in BODIES:
            if name not in names:
                continue
            edge = MAP_LOOK[name].get("edge")
            if edge is None:
                continue
            m = ids == names.index(name)
            if name in RINGED:
                m = _dilate(m, px(OUTLINE_M)) & ~body
            else:
                m &= rim
            if m.any():
                out[m] = edge

    return np.clip(out, 0, 255).astype(np.uint8)


def _window_any(mask: np.ndarray, r: int, axis: int) -> np.ndarray:
    """`mask` OR-ed over a window of half-width `r` along `axis` (edges padded False)."""
    pad = [(0, 0), (0, 0)]; pad[axis] = (r, r)
    padded = np.pad(mask, pad)
    win = np.lib.stride_tricks.sliding_window_view(padded, 2 * r + 1, axis=axis)
    return win.any(axis=-1)


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Square dilation by `r` pixels, separable: two sliding-window ORs instead of `r`
    rounds of four rolls -- the same cost at r = 1 and ten times less at r = 10."""
    if r <= 0:
        return mask.copy()
    return _window_any(_window_any(mask, r, 0), r, 1)


def _erode(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask.copy()
    return ~_dilate(~mask, r)


def downsample(img: np.ndarray, ss: int) -> np.ndarray:
    """Box mean over `ss x ss` blocks: the anti-aliasing of the painted picture."""
    if ss == 1:
        return img
    h, w = img.shape[0] // ss * ss, img.shape[1] // ss * ss
    a = img[:h, :w].reshape(h // ss, ss, w // ss, ss, 3).astype(np.uint16)
    return a.mean(axis=(1, 3)).round().astype(np.uint8)
