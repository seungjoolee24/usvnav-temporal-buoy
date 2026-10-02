"""Broad phase for the per-tick geometry, and the one place a tick's geometry is done.

`DECISIONS.md` §9 lists performance among the things the build would contradict, and it
did: the episode loop was recomputing the same exact surface distances twice per tick
(once for the collision test, once for the clearance statistic) and testing every body
and every shoreline edge every time. 30 teams fit the operations plan; 1000 did not.

**Exactness is unchanged and that is the point.** The index only decides which exact
test to *skip*, using a centre-distance bound that can never exceed the true
surface-to-surface distance, plus an early break once no remaining candidate can beat
the best distance found. The exact primitives of `geometry` are still what decides a
collision (2-E3, 1-W4): nothing is replaced by its circumscribed circle -- the
circumscribed radius appears only as a *lower bound* used to reject.
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import Circle, Rect, _seg_seg_dist, body_distance, point_in_polygon
from .plant import HULL_LENGTH, HULL_WIDTH
from .world import MOVING

HULL_RADIUS = 0.5 * math.hypot(HULL_LENGTH, HULL_WIDTH)     # 1.80 m (1-W4's arithmetic)


def bounding_radius(shape) -> float:
    if isinstance(shape, Circle):
        return shape.r
    if isinstance(shape, Rect):
        return 0.5 * math.hypot(shape.length, shape.width)
    raise TypeError(type(shape))


class CourseIndex:
    """Per-course, built once and reused for every tick of every episode on it."""

    def __init__(self, course):
        self.course = course
        self.shapes = [b.shape for b in course.bodies]
        self.classes = [b.cls for b in course.bodies]
        if self.shapes:
            self.centres = np.array([[s.x, s.y] for s in self.shapes], dtype=float)
            self.radii = np.array([bounding_radius(s) for s in self.shapes], dtype=float)
        else:
            self.centres = np.empty((0, 2))
            self.radii = np.empty(0)

        b = np.asarray(course.boundary, dtype=float)
        self.poly = b
        self.edge_a = b
        self.edge_b = np.roll(b, -1, axis=0)
        self.edge_v = self.edge_b - self.edge_a
        self.edge_len2 = np.einsum("ij,ij->i", self.edge_v, self.edge_v)

        # Lanes add and remove vessels over time, so the moving set is taken per tick
        # from `Course.traffic_rects`; the bounding radii come with the rectangles.
        self.has_traffic = bool(course.traffic) or bool(course.lanes)

    # --- the whole per-tick geometry, once -------------------------------------
    def tick(self, hull: Rect, t: float, cap: float):
        """Return `(clearance, hit_class_or_None, inside_bounds)` for one tick.

        `clearance` is the minimum surface-to-surface distance to any body *or* the
        shoreline, capped at `cap` (5-S13 caps per tick, before the episode statistic).
        `hit_class_or_None` is the class of a body in contact, which is what separates
        `static_collision` from `dynamic_collision` (5-S11). `inside_bounds` is 5-S11's
        out-of-bounds test: the hull leaves when *any part* of it does.
        """
        centre = np.array([hull.x, hull.y])
        best = cap
        hit = None

        best, hit = self._nearest(hull, centre, self.centres, self.radii,
                                  self.shapes, self.classes, best, hit)
        if hit is not None and best <= 0.0:
            return 0.0, hit, True

        if self.has_traffic:
            rects = self.course.traffic_rects(t)
            if rects:
                tc = np.array([[r.x, r.y] for r in rects], dtype=float)
                tr = np.array([0.5 * math.hypot(r.length, r.width) for r in rects], dtype=float)
                best, hit = self._nearest(hull, centre, tc, tr, rects,
                                          [MOVING] * len(rects), best, hit)
                if hit is not None and best <= 0.0:
                    return 0.0, hit, True

        d_poly = self._boundary_distance(hull, centre, best)
        inside = (d_poly > 0.0) and point_in_polygon(centre, self.poly)
        return min(best, d_poly, cap), None, inside

    def _nearest(self, hull, centre, centres, radii, shapes, classes, best, hit):
        """Exact nearest surface among `shapes`, skipping what cannot win."""
        if not len(shapes):
            return best, hit
        lower = np.linalg.norm(centres - centre, axis=1) - radii - HULL_RADIUS
        order = np.argsort(lower)
        for k in order:
            if lower[k] >= best:
                break                       # and neither can anything after it
            d = body_distance(hull, shapes[k])
            if d < best:
                best, hit = d, classes[k]
                if best <= 0.0:
                    return 0.0, classes[k]
        return best, hit

    def _boundary_distance(self, hull: Rect, centre, cap: float) -> float:
        """Hull-to-shoreline distance, capped, exact where it matters.

        Only edges whose distance from the hull *centre* could put a surface within
        `cap` are tested exactly, and `cap` is positive, so an uncapped result being
        returned as `cap` cannot turn a crossing into a clearance.
        """
        w = centre - self.edge_a
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.einsum("ij,ij->i", w, self.edge_v) / np.where(
                self.edge_len2 == 0.0, 1.0, self.edge_len2)
        s = np.clip(s, 0.0, 1.0)
        foot = self.edge_a + s[:, None] * self.edge_v
        d_centre = np.linalg.norm(centre - foot, axis=1)

        near = np.nonzero(d_centre - HULL_RADIUS < cap)[0]
        if not len(near):
            return cap
        corners = hull.corners()
        best = cap
        for i in near:
            a, b = self.edge_a[i], self.edge_b[i]
            for j in range(4):
                d = _seg_seg_dist(corners[j], corners[(j + 1) % 4], a, b)
                if d < best:
                    best = d
                    if best <= 0.0:
                        return 0.0
        return best
