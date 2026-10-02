"""Top-view raster renderer -- the observation for conditions 1-2 and 1-4 (4-V6), and
the one piece of logic Track 1 shares with Track 2-2 (4-V3).

**Shared, because it is a property of the logic and not of any number** (follow-up
items 3, 13, 14): the rendering procedure, the world-state input schema (`Scene`), the
public entry point (`render`), the world -> raster index mapping of 1-W3, the ten-class
vocabulary, and the minimum-marker rule (item 5).

**Per-track configuration** (`RasterConfig`): raster size, metres per pixel, sensing
range, per-class colours, minimum marker size, frame convention, and whether bodies
occlude. Track 1's configuration is `TRACK1`.

Four properties of this implementation are decisions rather than coding choices:

*Native frame transform* (item 4). Geometry is transformed into the observer's frame and
then rasterized. A world-aligned raster is never rendered and rotated: resampling blurs
every edge.

*The raster is the studio's picture* (4-V8 and 4-V13 revised 2026-09-16, owner). With
`RasterConfig.look` set -- Track 1's default -- the class map is drawn at `subsamples`
times the resolution by `usvnav.look.paint`, the same code that draws the studio map
(base colours, a shoreline, shadows, decks, outlines; noise only if `RasterConfig.noise`
asks for it, which scoring never does), and box-downsampled, which anti-aliases every edge. What a team sees in the studio is what its
agent receives. The owner's question was why the agent's picture should be a flat
stand-in for the author's; the measured answer was that nothing the flat picture
protected -- an exact-colour class map, a phantom-free blend -- ever stopped a
participant, so the trade goes to realism. Without `look`, the renderer is the plain
class-colour raster: interiors exact, edge pixels blended over `subsamples**2` samples
when `subsamples` > 1 (Track 2 renders that way with `PALETTE_GENERAL_POSITION`, whose
4-V9 general-position margins `tests/test_palette.py` keeps).

*Square extent, and nothing occludes* (4-V6 revised, 4-V17, both signed off 2026-09-08). The raster is the
whole 100 m x 100 m square with no circular mask, and a body does not hide what is behind it -- occlusion
exists in Track 1 only in 1-3's ray cast, where it is physics. Two consequences the code depends on: the
square reaches **70.71 m** into its corners against 50 m along its axes, which is what `reach_m` is for and
what 4-V12's terrain margin now has to cover; and `unobserved` **cannot appear in a Track 1 raster** any more,
since it had no other source. The `occlusion` flag is kept, defaulting off: it is what makes the 4-V17 decision
auditable -- `tools/figures.py` can still regenerate the comparison the decision was made on -- and Track 2
composes arbitrary scenes from the same vocabulary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Sequence

import numpy as np

from . import look as _look
from .geometry import Circle, Rect, wrap
from .world import (BANK, BUOY, DOCK, EGO, MOORED, MOVING, OUTSIDE_WORLD, PIER,
                    UNOBSERVED, WATER)

# The raster colour class shared by moored and moving vessels (4-V13). It is a *colour*
# class, not an object class: `world.MOORED` and `world.MOVING` stay distinct everywhere
# else, and 1-1 reports them apart.
VESSEL = "vessel"

#: The blend-safe palette (4-V9, 4-V13). Eight entries, not nine; values computed, not
#: chosen -- see `../hackaton/track_1/palette_check.py` for the search and
#: `tests/test_palette.py` for the check -- so that no two- or three-class blend of an
#: anti-aliased edge lands on a third class. This is what a render with `subsamples` > 1
#: needs; Track 2 renders with it. Track 1 stopped rendering with it on 2026-09-16.
PALETTE_GENERAL_POSITION: Mapping[str, tuple[int, int, int]] = {
    UNOBSERVED: (0x00, 0x00, 0x00),
    WATER:      (0x12, 0x26, 0x3E),
    DOCK:       (0x80, 0x00, 0x30),
    PIER:       (0xBE, 0x00, 0x91),
    BANK:       (0x8C, 0x78, 0x54),
    VESSEL:     (0x22, 0xB5, 0xA5),
    EGO:        (0xF8, 0xB2, 0xBA),
    BUOY:       (0xC2, 0xFB, 0x10),
}

#: Track 1's palette (4-V13 revised 2026-09-16, owner): the studio map's base colours,
#: owned by `usvnav.look` and drawn there with the map's textures, so the raster the agent
#: receives is the picture its author sees and the two cannot drift apart.
PALETTE_TRACK1: Mapping[str, tuple[int, int, int]] = dict(_look.BASE)
assert _look.VESSEL == VESSEL

#: Object class -> raster colour class. Absent means "same name".
COLOUR_OF = {MOORED: VESSEL, MOVING: VESSEL}

#: Painting order for bodies; later wins where footprints overlap. Buoys are painted
#: last because they are the class the minimum-marker rule exists for -- a 0.6 m disc
#: must not be swallowed by a rectangle it happens to touch.
BODY_PAINT_ORDER = (DOCK, PIER, MOORED, MOVING, BUOY)


class PaletteError(KeyError):
    """A class had to be painted and the configured palette has no colour for it.

    Raised rather than substituted, because the substitution would be silent. Track 1's
    palette has no `outside_world` entry: 4-V12 guarantees the class never appears, and
    this is what turns that guarantee into a check that fires.
    """


@dataclass(frozen=True)
class RasterConfig:
    """Per-track raster configuration (4-V3). Defaults are Track 1's (4-V6)."""

    size_px: int = 200
    m_per_px: float = 0.5
    range_m: float = 50.0                 # 4-V5; the raster's half-width, not a radius
    frame: str = "bow_up"                 # 4-V2; `world_aligned` is for map views
    circular_mask: bool = False           # 4-V6 revised: the extent is the square
    draw_ego: bool = True                 # 4-V6
    occlusion: bool = False               # 4-V17: no Track 1 condition occludes here
    min_marker_px: float = 2.0            # follow-up item 5; see module note below
    subsamples: int = 8                   # plain raster: 8x8 coverage levels on edge pixels;
                                          # with `look`: the picture is painted at this
                                          # multiple of the resolution and box-downsampled
    look: bool = False                    # paint the studio's picture (`usvnav.look`) instead
                                          # of flat class colours -- Track 1's setting since
                                          # 2026-09-16 (4-V8 / 4-V13 revised)
    noise: Callable | None = field(default=None, compare=False)
                                          # renderer noise for `look`: a function
                                          # `(class_name, x_world, y_world) -> offsets`; None
                                          # for scoring, always (4-V14 revised). The kit's
                                          # example is `usvnav.look.grain`.
    shadow_bins: int = 2880               # 0.125 deg; 0.22 px of arc at 50 m
    palette: Mapping[str, tuple[int, int, int]] = field(
        default_factory=lambda: dict(PALETTE_TRACK1))

    @property
    def origin_px(self) -> float:
        """`r0` = `c0` of 1-W3: the observer sits on the shared corner of the four
        central pixels, which for an even `size_px` is exactly half of it."""
        return self.size_px / 2.0

    @property
    def mask_radius_px(self) -> float:
        return self.range_m / self.m_per_px

    @property
    def reach_m(self) -> float:
        """Farthest a painted pixel can be from the observer.

        Under 4-V6's square extent that is the raster's **half-diagonal**, not its
        half-width: a 200 x 200 raster at 0.5 m/px reaches 70.71 m into its corners
        against 50.0 m along its axes. This is the number 4-V12's terrain margin has to
        cover, and the number the body cull uses.
        """
        half = 0.5 * self.size_px * self.m_per_px
        return self.range_m if self.circular_mask else half * math.sqrt(2.0)


#: Track 1's condition 1-2 / 1-4 raster: the studio's picture, painted at 0.25 m/px and
#: box-downsampled 2x to 0.5 m/px (anti-aliased).
TRACK1 = RasterConfig(look=True, subsamples=2)


@dataclass
class Scene:
    """The world-state input schema -- the shared half of 4-V3.

    Deliberately not a `Course`: the entry point must tolerate world states outside
    anything the Track 1 generator emits (follow-up item 3), which is what lets Track 2
    compose arbitrary scenes from the same vocabulary.

    `observer` is `(x, y, psi)` in world metres/radians. `bodies` is a sequence of
    `(class, Circle | Rect)`. `boundary` is the navigable polygon -- the shoreline
    (2-E6); water inside, bank outside. `terrain_extent` is `(xmin, ymin, xmax, ymax)`
    of generated terrain; beyond it is `outside_world` (4-V12). `None` for either means
    "unbounded": no water, or terrain everywhere.
    """

    observer: tuple[float, float, float]
    bodies: Sequence[tuple[str, object]] = ()
    boundary: np.ndarray | None = None
    terrain_extent: tuple[float, float, float, float] | None = None
    ego_hull: Rect | None = None


# --------------------------------------------------------------------------- public

#: Organiser-side acceleration seam (`usvnav.accel`, held back from the bundle). When set, `render`
#: hands the whole frame to it and `_in_polygon` its point test; both are native kernels verified
#: byte-identical to the Python below, so nothing a participant sees or a scorer stores changes.
#: `None` -- the shipped state -- means the Python here runs. Nothing in the bundle sets them.
_native_render = None
_native_in_polygon = None


def render(scene: Scene, cfg: RasterConfig = TRACK1) -> np.ndarray:
    """The public, documented, stable entry point (follow-up item 3).

    Returns an `(size_px, size_px, 3)` `uint8` raster. Row 0 is the top of the image;
    under `bow_up` the bow points at it (1-W3).
    """
    if _native_render is not None:
        return _native_render(scene, cfg)
    return _render_python(scene, cfg)


def _render_python(scene: Scene, cfg: RasterConfig) -> np.ndarray:
    """`render`, in the Python of the kit: the reference every native kernel is measured against."""
    if cfg.look:
        return _render_look(scene, cfg)
    fr = _Frame(scene, cfg)
    n = cfg.size_px

    rows, cols = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5, indexing="ij")
    ids = fr.classify(*fr.pixel_to_frame(rows.ravel(), cols.ravel())).reshape(n, n)

    rgb = fr.lut[ids]                          # interiors get the exact class colour
    edge = _edge_pixels(ids)
    ei, ej = np.nonzero(edge)
    if len(ei):
        s = cfg.subsamples
        off = (np.arange(s) + 0.5) / s
        dr, dc = np.meshgrid(off, off, indexing="ij")
        sr = (ei[:, None] + dr.ravel()[None, :]).ravel()
        sc = (ej[:, None] + dc.ravel()[None, :]).ravel()
        sub = fr.classify(*fr.pixel_to_frame(sr, sc)).reshape(len(ei), s * s)
        rgb[ei, ej] = np.rint(fr.lut[sub].mean(axis=1)).astype(np.uint8)
    return rgb


def _render_look(scene: Scene, cfg: RasterConfig) -> np.ndarray:
    """The studio's picture in the agent's window: paint the class map at `subsamples` times
    the resolution with `usvnav.look.paint`, then box-downsample (4-V8 revised)."""
    ss = max(1, int(cfg.subsamples))
    fine = replace(cfg, size_px=cfg.size_px * ss, m_per_px=cfg.m_per_px / ss,
                   min_marker_px=cfg.min_marker_px * ss, subsamples=1, look=False)
    ids = class_map(scene, fine)
    return _look.downsample(_look.paint(ids, class_names(fine), fine, scene.observer, cfg.noise), ss)


def class_map(scene: Scene, cfg: RasterConfig = TRACK1) -> np.ndarray:
    """The class id at each pixel centre, no anti-aliasing. Ground truth for tests and
    for the metric extractor (`T1-SCO-07`), which must never read agent observations."""
    fr = _Frame(scene, cfg)
    n = cfg.size_px
    rows, cols = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5, indexing="ij")
    return fr.classify(*fr.pixel_to_frame(rows.ravel(), cols.ravel())).reshape(n, n)


def class_names(cfg: RasterConfig = TRACK1) -> list[str]:
    """Index -> class name, matching the ids in `class_map`."""
    return list(cfg.palette.keys())


def scene_from_course(course, vessel, t: float, *, include_ego: bool = True) -> Scene:
    """Adapter: Track 1's `Course` plus a vessel state at time `t` -> `Scene`.

    The adapter is where Track 1's world model meets the shared schema; keeping it out
    of `render` is what makes the entry point usable by Track 2.
    """
    bodies = [(b.cls, b.shape) for b in course.bodies]
    bodies += [(MOVING, r) for r in course.traffic_rects(t)]
    return Scene(
        observer=(vessel.x, vessel.y, vessel.psi),
        bodies=bodies,
        boundary=course.boundary,
        terrain_extent=getattr(course, "terrain_extent", None),
        ego_hull=vessel.hull() if include_ego else None,
    )


def top_view(course, vessel, t: float, cfg: RasterConfig = TRACK1) -> np.ndarray:
    """Condition 1-2 / 1-4's observation for one tick."""
    return render(scene_from_course(course, vessel, t), cfg)


# --------------------------------------------------------------------------- internals

def _edge_pixels(ids: np.ndarray) -> np.ndarray:
    """Pixels adjacent to a class boundary, 8-connected.

    Diagonals are included as well as edges: a feature narrow enough to touch only
    diagonal neighbours would otherwise render with a hard edge while its neighbours
    are anti-aliased, and the minimum-marker rule does not bound orientation.
    """
    e = np.zeros(ids.shape, dtype=bool)
    for a, b in ((np.s_[:-1, :], np.s_[1:, :]),          # vertical
                 (np.s_[:, :-1], np.s_[:, 1:]),          # horizontal
                 (np.s_[:-1, :-1], np.s_[1:, 1:]),       # diagonal
                 (np.s_[:-1, 1:], np.s_[1:, :-1])):      # anti-diagonal
        d = ids[a] != ids[b]
        e[a] |= d
        e[b] |= d
    e[0, :] = e[-1, :] = e[:, 0] = e[:, -1] = True     # the mask meets the border
    return e


class _Frame:
    """Geometry transformed into the observer's frame once per rendered frame.

    This class is follow-up item 4: the transform happens here, on the geometry, and
    nothing downstream ever rotates a finished raster.
    """

    def __init__(self, scene: Scene, cfg: RasterConfig):
        self.cfg = cfg
        self.scene = scene

        names = list(cfg.palette.keys())
        self.ids = {name: i for i, name in enumerate(names)}
        self.lut = np.array([cfg.palette[n] for n in names], dtype=np.uint8)

        ox, oy, psi = scene.observer
        if cfg.frame == "bow_up":
            rot = float(psi)
        elif cfg.frame in ("world_aligned", "map"):
            rot = 0.0
        else:
            raise ValueError(
                f"frame must be 'bow_up', 'world_aligned' or 'map', got {cfg.frame!r}")
        c, s = math.cos(rot), math.sin(rot)
        self._R = np.array([[c, s], [-s, c]])       # world -> frame
        self._o = np.array([float(ox), float(oy)])

        self.reach = cfg.reach_m

        # --- geometry, transformed and marker-dilated, in frame metres ---
        #
        # Bodies outside `reach` are dropped here, once, by a scalar test. A body whose
        # bounding circle misses the painted disc can affect no pixel and cast no shadow
        # into it, because a shadow starts *past* its own far surface. Without this every
        # body was bounding-boxed against all 40,000 pixel centres and again against
        # every subsample of every edge pixel, and a 300 m reach carries ~26 bodies of
        # which ~6 are ever in range.
        self._bodies: dict[str, list] = {}
        for cls, shape in scene.bodies:
            local = _min_marker(self._to_frame_shape(shape), cfg)
            if math.hypot(local.x, local.y) - _bounding_radius(local) > self.reach:
                continue
            self._bodies.setdefault(cls, []).append(local)
        self._ego = (self._to_frame_shape(scene.ego_hull)
                     if (cfg.draw_ego and scene.ego_hull is not None) else None)
        self._poly = (self.to_frame_points(scene.boundary)
                      if scene.boundary is not None else None)
        self._extent = (self._extent_quad(scene.terrain_extent)
                        if scene.terrain_extent is not None else None)
        if self._extent is not None and self._quad_contains_reach(self._extent):
            self._extent = None      # every painted pixel is inside it; 4-V12's usual case

        # Only bodies cast shadows -- see `RasterConfig.occlusion`.
        self._shadow = None
        if cfg.occlusion:
            shapes = [sh for lst in self._bodies.values() for sh in lst]
            if shapes:
                self._shadow = _ShadowProfile(shapes, cfg.shadow_bins)

    # --- transforms ---------------------------------------------------------
    def to_frame_points(self, pts) -> np.ndarray:
        return (np.asarray(pts, dtype=float) - self._o) @ self._R.T

    def _to_frame_shape(self, shape):
        if isinstance(shape, Circle):
            p = self.to_frame_points([[shape.x, shape.y]])[0]
            return Circle(float(p[0]), float(p[1]), shape.r)
        if isinstance(shape, Rect):
            p = self.to_frame_points([[shape.x, shape.y]])[0]
            rot = math.atan2(self._R[0, 1], self._R[0, 0])
            return Rect(float(p[0]), float(p[1]), shape.length, shape.width,
                        float(wrap(shape.heading - rot)))
        raise TypeError(f"not a collision primitive (2-E3): {type(shape)}")

    def _quad_contains_reach(self, quad: np.ndarray) -> bool:
        """Is every point within `reach` of the origin inside this quadrilateral?

        True exactly when the origin is at least `reach` inside each of the four edges.
        When it holds, the terrain test is a constant and is dropped -- which is the
        normal case under 4-V12, whose whole point is that `outside_world` is
        unreachable. Dropping it is therefore free in the common case and the check
        still fires in the uncommon one.
        """
        for i in range(4):
            a, b = quad[i], quad[(i + 1) % 4]
            e = b - a
            n = float(np.hypot(e[0], e[1]))
            if n == 0.0:
                return False
            # signed distance from the origin to the edge line, positive inside for
            # counter-clockwise winding (same convention as `_in_quad`)
            if (e[0] * (0.0 - a[1]) - e[1] * (0.0 - a[0])) / n < self.reach:
                return False
        return True

    def _extent_quad(self, extent) -> np.ndarray:
        x0, y0, x1, y1 = (float(v) for v in extent)
        return self.to_frame_points([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

    def pixel_to_frame(self, row, col):
        """1-W3, inverted: `row = r0 - x/s`, `col = c0 - y/s`.

        The two minus signs are the thing most likely to be got wrong, which is why
        1-W3 states the formula and why `tests/test_render.py` asserts the sign of a
        port-bow object.

        `frame="map"` is the exception and is **not an observation mode**: it puts `+x`
        to the right and `+y` up, the orientation a person expects of a map and the one
        the course editor draws in. It exists so a debug picture and the editor cannot
        disagree about which way the reach runs -- 1-W3's mapping points `+x` at the top
        of the image, which is right for a bow-up sensor and disorienting for an
        authoring tool. No condition uses it, and `RasterConfig.frame` is per-track
        configuration under 4-V3, so a view-only third mode costs the contract nothing.
        """
        s, o = self.cfg.m_per_px, self.cfg.origin_px
        row = np.asarray(row, dtype=float)
        col = np.asarray(col, dtype=float)
        if self.cfg.frame == "map":
            return (col - o) * s, (o - row) * s
        return (o - row) * s, (o - col) * s

    # --- classification -----------------------------------------------------
    def classify(self, xb, yb) -> np.ndarray:
        """Class id for arbitrary points in frame metres.

        One code path serves both the pixel-centre pass and the edge-refinement pass, so
        the anti-aliased edge can never disagree with the interior it borders --
        4-V16's structural-equivalence requirement, enforced by construction.
        """
        xb = np.asarray(xb, dtype=float)
        yb = np.asarray(yb, dtype=float)
        out = np.empty(xb.shape, dtype=np.int16)

        # The mask is computed first and applied last. Computing it first matters
        # because a 200 x 200 square reaches 50*sqrt(2) = 70.7 m into its corners while
        # 4-V12 guarantees terrain to only 50 m: those corner pixels are outside the
        # generated world and are also masked, so `outside_world` must not be demanded
        # for them. It is the circular mask that makes one sensing range exactly enough.
        masked = ((xb * xb + yb * yb > self.cfg.range_m ** 2) if self.cfg.circular_mask
                  else np.zeros(xb.shape, dtype=bool))

        # --- regions: outside_world / bank / water ---
        inside_terrain = (_in_quad(xb, yb, self._extent) if self._extent is not None
                          else np.ones(xb.shape, dtype=bool))
        out[:] = self._id(BANK)
        beyond = ~inside_terrain & ~masked
        if beyond.any():
            out[beyond] = self._id(OUTSIDE_WORLD)
        if self._poly is not None:
            water = _in_polygon(xb, yb, self._poly) & inside_terrain
            out[water] = self._id(WATER)

        # --- bodies, in painting order ---
        for cls in BODY_PAINT_ORDER:
            shapes = self._bodies.get(cls)
            if not shapes:
                continue
            cid = self._id(COLOUR_OF.get(cls, cls))
            for shape in shapes:
                sel = _inside(xb, yb, shape)
                if sel is not None:
                    out[sel] = cid
        for cls, shapes in self._bodies.items():           # classes not in the order
            if cls in BODY_PAINT_ORDER:
                continue
            cid = self._id(COLOUR_OF.get(cls, cls))
            for shape in shapes:
                sel = _inside(xb, yb, shape)
                if sel is not None:
                    out[sel] = cid

        # --- occlusion: everything beyond the nearest body along its ray ---
        if self._shadow is not None:
            out[self._shadow.shadowed(xb, yb)] = self._id(UNOBSERVED)

        # --- the ego is drawn last: it is never occluded and never occludes (4-V6) ---
        if self._ego is not None:
            sel = _inside(xb, yb, self._ego)
            if sel is not None:
                out[sel] = self._id(EGO)

        # --- the circular mask, outermost (4-V6) ---
        if self.cfg.circular_mask:
            out[masked] = self._id(UNOBSERVED)
        return out

    def _id(self, cls: str) -> int:
        try:
            return self.ids[cls]
        except KeyError:
            raise PaletteError(
                f"class {cls!r} must be painted but the configured palette has no "
                f"colour for it; palette has {sorted(self.ids)}") from None


def _bounding_radius(shape) -> float:
    if isinstance(shape, Circle):
        return shape.r
    return 0.5 * math.hypot(shape.length, shape.width)


def _min_marker(shape, cfg: RasterConfig):
    """Follow-up item 5: a class whose true extent is below the minimum is drawn at it.

    Two independent floors, and `min_marker_px = 2.0` clears both. (a) A feature must
    register in the pixel-centre pass at all: pixel centres are 1 px apart, so a disc
    needs a radius above sqrt(2)/2 = 0.71 px to be certain of covering one, i.e. a
    diameter above 1.41 px. (b) Under an anti-aliased render (Track 2) a boundary pixel's
    blend must still classify as the class: a 0.6 m buoy is 1.2 px across, so at worst --
    centred on a pixel corner -- it puts 28% coverage into each of four pixels, and a 28%
    blend towards water is nearer water than buoy in CIELAB, so a colour-lookup
    segmentation would read the buoy as water. At a 2 px diameter the worst case is 79%
    in each of four pixels, which reads as buoy. Without anti-aliasing (Track 1 since
    2026-09-16) only (a) binds, and 2 px still clears it.

    Rendering only. Collision and the 1-1 object list keep the true size, so the raster
    over-states a buoy's radius by 0.2 m. That is deliberate and one-directional: a
    conservative agent gives it 0.2 m more room than it needs.
    """
    m = cfg.min_marker_px * cfg.m_per_px
    if isinstance(shape, Circle):
        return Circle(shape.x, shape.y, max(shape.r, m / 2.0))
    if isinstance(shape, Rect):
        return Rect(shape.x, shape.y, max(shape.length, m), max(shape.width, m),
                    shape.heading)
    raise TypeError(type(shape))


# --- vectorised point tests ------------------------------------------------

def _inside(xb, yb, shape):
    """Boolean mask, or `None` when the shape cannot touch these points at all.

    The bounding-box reject is what keeps one code path affordable: the refinement pass
    evaluates 64 subsamples per edge pixel, and without the reject every body would be
    tested against every one of them.
    """
    if isinstance(shape, Circle):
        x0, x1 = shape.x - shape.r, shape.x + shape.r
        y0, y1 = shape.y - shape.r, shape.y + shape.r
    else:
        h = 0.5 * math.hypot(shape.length, shape.width)
        x0, x1, y0, y1 = shape.x - h, shape.x + h, shape.y - h, shape.y + h
    near = (xb >= x0) & (xb <= x1) & (yb >= y0) & (yb <= y1)
    if not near.any():
        return None
    if isinstance(shape, Circle):
        dx, dy = xb - shape.x, yb - shape.y
        return near & (dx * dx + dy * dy <= shape.r * shape.r)
    c, s = math.cos(shape.heading), math.sin(shape.heading)
    dx, dy = xb - shape.x, yb - shape.y
    a = dx * c + dy * s
    b = -dx * s + dy * c
    return near & (np.abs(a) <= shape.length / 2.0) & (np.abs(b) <= shape.width / 2.0)


def _in_polygon(xb, yb, poly: np.ndarray) -> np.ndarray:
    """Crossing-number test, vectorised. Same convention as `geometry.point_in_polygon`
    so the renderer and the collision test agree on the shoreline."""
    if _native_in_polygon is not None:
        return _native_in_polygon(xb, yb, poly)
    return _in_polygon_python(xb, yb, poly)


def _in_polygon_python(xb, yb, poly: np.ndarray) -> np.ndarray:
    inside = np.zeros(xb.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        crosses = (y1 > yb) != (y2 > yb)
        if not crosses.any():
            continue
        dy = y2 - y1
        xint = x1 + (yb - y1) * (x2 - x1) / (dy if dy != 0.0 else 1.0)
        inside ^= crosses & (xb < xint)
    return inside


def _in_quad(xb, yb, quad: np.ndarray) -> np.ndarray:
    """Inside a convex quadrilateral given counter-clockwise."""
    inside = np.ones(xb.shape, dtype=bool)
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        cross = (b[0] - a[0]) * (yb - a[1]) - (b[1] - a[1]) * (xb - a[0])
        inside &= cross >= 0.0
    return inside


class _ShadowProfile:
    """Where the view ends along each bearing.

    For every bearing bin this holds the **exit** range of the nearest body along it, so
    the nearest body is drawn as a whole footprint and everything behind it is
    `unobserved`. That is the bird's-eye convention: a sensor sees the near face, a
    tracker fills in the footprint, and what is behind is unknown.

    The shoreline is not in the profile, deliberately. Land behind land is land, so the
    only thing shadowing the bank would change is hiding *bodies* behind a headland --
    and the shoreline polygon is disclosed exactly at reset (4-V1), so an agent can
    compute that region itself. Marking it unobserved would remove information the agent
    already has, which is the mechanism 4-V4 rejects for 1-1.
    """

    def __init__(self, shapes, nbins: int):
        self.n = int(nbins)
        mid = -math.pi + 2.0 * math.pi * (np.arange(self.n) + 0.5) / self.n
        d = np.stack([np.cos(mid), np.sin(mid)], axis=1)

        entry = np.full(self.n, np.inf)
        self.exit = np.full(self.n, np.inf)
        for shape in shapes:
            t0, t1 = _ray_shape(d, shape)
            take = t0 < entry
            entry = np.where(take, t0, entry)
            self.exit = np.where(take, t1, self.exit)

    def shadowed(self, xb, yb) -> np.ndarray:
        rho = np.hypot(xb, yb)
        th = np.arctan2(yb, xb)
        k = np.minimum(((th + math.pi) / (2.0 * math.pi) * self.n).astype(np.int64),
                       self.n - 1)
        return rho > self.exit[k]


def _ray_shape(d: np.ndarray, shape):
    """Entry and exit range of `shape` along unit directions `d` from the origin.

    `inf` where the shape is not hit. Entry is clamped at 0, so an observer inside a
    body shadows everything beyond that body's far surface rather than nothing.
    """
    if isinstance(shape, Circle):
        f = np.array([-shape.x, -shape.y])
        b = d @ f
        cc = float(f @ f) - shape.r ** 2
        disc = b * b - cc
        ok = disc >= 0.0
        root = np.sqrt(np.where(ok, disc, 0.0))
        t0, t1 = -b - root, -b + root
    elif isinstance(shape, Rect):
        c, s = math.cos(shape.heading), math.sin(shape.heading)
        R = np.array([[c, s], [-s, c]])
        o = R @ np.array([-shape.x, -shape.y])
        dl = d @ R.T
        he = np.array([shape.length / 2.0, shape.width / 2.0])
        safe = np.where(np.abs(dl) < 1e-15, 1e-15, dl)
        ta = (-he[None, :] - o[None, :]) / safe
        tb = (he[None, :] - o[None, :]) / safe
        t0 = np.minimum(ta, tb).max(axis=1)
        t1 = np.maximum(ta, tb).min(axis=1)
        ok = t1 >= np.maximum(t0, 0.0)
    else:
        raise TypeError(type(shape))
    ok = ok & (t1 > 0.0)
    return (np.where(ok, np.maximum(t0, 0.0), np.inf),
            np.where(ok, t1, np.inf))
