"""Drawing a course as a picture -- the `usvnav view` command (7-A2).

**Not an observation.** Everything here is an annotation on a debug image: the trajectory,
the waypoint rings, the start marker, the lanes (arrowed polylines). None of it shares code
with the rasterizer, whose edge behaviour is a decision (4-V8), and the map view is allowed
colours the Track 1 palette does not have because it frames terrain 4-V12 never promised
(`outside_world`).

This ships. It used to live in `tools/figures.py`, which also builds the decision log's
figures by driving the reference agent -- so `usvnav view`, a command the README tells
participants to run, imported an internal module through a lazy import that the bundle
would have had to keep. Splitting the drawing from the figure *set* is what makes the
command shippable; `tools/figures.py` now imports these.
"""

from __future__ import annotations

import math

import numpy as np

from . import render
from .plant import Vessel
from .png import write_png
from .render import VESSEL, RasterConfig
from .world import (BANK, BUOY, DOCK, EGO, OUTSIDE_WORLD, PIER, UNOBSERVED, WATER)

# The map view is not an observation, so it may have colours Track 1's palette does not:
# `outside_world` exists here because a 340 m square around a 300 x 150 m reach shows
# terrain the generator did not make -- 4-V12 guarantees it only out to the raster's own
# reach, which is a far smaller box than this view.
MAP_PALETTE = dict(render.PALETTE_TRACK1, outside_world=(0x1A, 0x1A, 0x1A))
TRACK_RGB = (0xFF, 0xE0, 0x40)          # the trajectory: not a class, an annotation
WAYPOINT_RGB = (0xFF, 0xFF, 0xFF)
START_RGB = (0x60, 0xFF, 0x60)
LANE_RGB = (0x5C, 0x9E, 0x94)           # a lane: an annotation, like a loop's line

# --------------------------------------------------------------------------- the map look
#
# The look lives in `usvnav.look` and is shared with the renderer: since 2026-09-16 (owner)
# the agent's raster is painted by the same code, so the map here and the observation the
# agent receives are one picture. These names stay importable from here.
from .look import (BODIES as _BODIES, MAP_LOOK, SHALLOW_M, SHALLOW_RGB,  # noqa: E402,F401
                   WATERLINE_M, downsample as _downsample, paint as _paint)


def paint(ids, cfg, observer=(0.0, 0.0, 0.0), noise=None):
    """`usvnav.look.paint` with the renderer's class-name order; kept for callers."""
    return _paint(ids, render.class_names(cfg), cfg, observer, noise)


# --------------------------------------------------------------------------- overlays

def _to_px(pts, cfg: RasterConfig, centre, offset=(0, 0)) -> np.ndarray:
    """World metres -> continuous pixel coordinates, in the config's frame.

    `offset` is the crop the map view applies, in `(row, col)`.
    """
    p = np.asarray(pts, dtype=float).reshape(-1, 2) - np.asarray(centre, dtype=float)
    o, s = cfg.origin_px, cfg.m_per_px
    if cfg.frame == "map":
        rc = np.column_stack([o - p[:, 1] / s, o + p[:, 0] / s])
    else:
        rc = np.column_stack([o - p[:, 0] / s, o - p[:, 1] / s])
    return rc - np.asarray(offset, dtype=float)


def _plot(img, rc, rgb, radius=0.0):
    """Stamp a disc of `radius` px at continuous pixel coordinates `rc`."""
    h, w = img.shape[:2]
    k = int(math.ceil(radius))
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            if math.hypot(dr, dc) > radius + 1e-9:
                continue
            r = np.rint(rc[:, 0] + dr).astype(int)
            c = np.rint(rc[:, 1] + dc).astype(int)
            ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
            img[r[ok], c[ok]] = rgb


def _ring(img, centre_rc, radius_px, rgb, width=1.0):
    """An outline circle. Waypoints are drawn as rings rather than filled discs so the
    arrival radius is legible and does not hide what is under it."""
    n = max(24, int(6.3 * radius_px))
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    pts = np.column_stack([centre_rc[0] + radius_px * np.sin(t),
                           centre_rc[1] + radius_px * np.cos(t)])
    _plot(img, pts, rgb, width)


def _polyline(img, rc, rgb, width=0.6):
    """Draw a polyline by dense sampling. Deliberately crude: it is an annotation on a
    debug picture, not part of any observation, so it must not share code with the
    rasterizer whose edge behaviour is a decision (4-V8)."""
    for a, b in zip(rc[:-1], rc[1:]):
        n = max(2, int(math.hypot(*(b - a)) * 2) + 2)
        t = np.linspace(0.0, 1.0, n)[:, None]
        _plot(img, a[None, :] * (1 - t) + b[None, :] * t, rgb, width)


# --------------------------------------------------------------------------- figures

def map_view(course, m_per_px=0.4, bank_m=13.0, ss=2, noise=None):
    """The whole course as a map: `+x` right, `+y` up. **Not an observation.**

    Cropped to the water plus a strip of bank, so the river runs off the left and right
    edges of the picture with a shore above and below it. Framing the whole of 1-W2's
    box instead put the reach in the middle of a large beige rectangle, which is a
    picture of a pond in a field.

    Uses the renderer's view-only `map` frame so this and `tools/editor.html` agree about
    which way the reach runs; a bow-up raster puts `+x` at the top, which is right for a
    sensor and confusing for an author comparing the two.

    Painted from the renderer's **class map** rather than from its colours, so the look
    lives here and the observation palette is untouched. Rendered at `ss` times the final
    scale and box-downsampled, which anti-aliases the shoreline for free and lets the
    texture be finer than a final pixel.
    """
    b = course.boundary
    x0, x1 = float(b[:, 0].min()), float(b[:, 0].max())
    y0, y1 = float(b[:, 1].min()) - bank_m, float(b[:, 1].max()) + bank_m
    centre = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    # The renderer is square, so render a square that covers the window and slice.
    span = max(x1 - x0, y1 - y0)
    size = int(round(span / m_per_px))
    fine = RasterConfig(size_px=size * ss, m_per_px=m_per_px / ss, range_m=span,
                        frame="map", circular_mask=False, draw_ego=False,
                        occlusion=False, palette=MAP_PALETTE)
    scene = render.scene_from_course(course, Vessel(centre[0], centre[1], 0.0), 0.0,
                                     include_ego=False)
    img = _downsample(paint(render.class_map(scene, fine), fine, (centre[0], centre[1], 0.0), noise), ss)

    # The config the overlays are placed with is the *final* one, not the fine one.
    cfg = RasterConfig(size_px=size, m_per_px=m_per_px, range_m=span,
                       frame="map", circular_mask=False, draw_ego=False,
                       occlusion=False, palette=MAP_PALETTE)
    w = int(round((x1 - x0) / m_per_px))
    h = int(round((y1 - y0) / m_per_px))
    r0, c0 = (size - h) // 2, (size - w) // 2
    return img[r0:r0 + h, c0:c0 + w], cfg, centre, (r0, c0)


def draw_lanes(img, course, cfg, centre, offset=(0, 0)):
    """Every lane as an arrowed polyline in `LANE_RGB`, onto a `map_view` picture. Shared with
    `tools/figures.py`, which composes its own panels from `map_view` and the annotation helpers."""
    for lane in course.lanes:
        pts = np.vstack([lane.points, lane.points[:1]]) if lane.closed else lane.points
        _polyline(img, _to_px(pts, cfg, centre, offset), LANE_RGB, 0.6)
        # three arrowheads along the lane, pointing the way the vessels go
        for frac in (0.25, 0.5, 0.75):
            x, y, h = lane.pose_at_arc(frac * lane.length)
            tip = np.array([x, y])
            for side in (+1.0, -1.0):
                back = tip - 4.0 * np.array([math.cos(h + side * 0.5), math.sin(h + side * 0.5)])
                _polyline(img, _to_px(np.vstack([tip, back]), cfg, centre, offset), LANE_RGB, 0.9)


def course_image(course, result=None, noise=None):
    """The map picture as an RGB array: shoreline, bodies, traffic loops and lanes, waypoint
    chain, and -- if a run is supplied -- the trajectory it took. `noise` as in
    `usvnav.look.paint`. `figure_course` writes this to a file."""
    img, cfg, centre, off = map_view(course, noise=noise)
    for route in course.traffic:
        loop = np.vstack([route.points, route.points[:1]])
        _polyline(img, _to_px(loop, cfg, centre, off), (0x40, 0x70, 0x70), 0.5)
    draw_lanes(img, course, cfg, centre, off)
    if result is not None and result.trace:
        tr = np.array([[x, y] for _, x, y, _, _, _ in result.trace])
        _polyline(img, _to_px(tr, cfg, centre, off), TRACK_RGB, 1.2)
    # A 0.6 m buoy is 1.5 px at map scale, so the class map physically cannot show it --
    # the same problem the minimum-marker rule solves for the observation (follow-up item
    # 5), except that on a debug map the honest fix is to draw it as an annotation and
    # say so rather than to inflate the body.
    from .world import BUOY as _BUOY
    marks = np.array([[b.shape.x, b.shape.y] for b in course.bodies if b.cls == _BUOY])
    if len(marks):
        rc = _to_px(marks, cfg, centre, off)
        _plot(img, rc, MAP_LOOK[_BUOY]["edge"], 2.6)
        _plot(img, rc, MAP_LOOK[_BUOY]["rgb"], 1.4)
    for wp, rad in zip(course.waypoints, course.arrival_radii):
        rc = _to_px([wp], cfg, centre, off)[0]
        _ring(img, rc, float(rad) / cfg.m_per_px, WAYPOINT_RGB, 0.8)
        _plot(img, rc[None, :], WAYPOINT_RGB, 1.2)
    _plot(img, _to_px([course.start[:2]], cfg, centre, off), START_RGB, 3.5)
    return img


def figure_course(course, path, result=None, span_m=340.0, noise=None):
    """Write `course_image` to `path` as a PNG; returns the path."""
    return write_png(path, course_image(course, result, noise))
