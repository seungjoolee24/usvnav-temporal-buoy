"""Geometry primitives and exact distance/overlap tests.

Conventions (track_1/DECISIONS.md 1-W3):
  World frame  : right-handed 2D, navigable area inside x in [0,400], y in [0,150] m (the reach; rules.REACH_X).
  Body frame   : +x through the bow, +y to port. Positive yaw is counter-clockwise.
  Angles       : radians, wrapped to (-pi, pi].

Collision primitives (2-E3): circle, and oriented rectangle. Nothing is ever replaced
by its circumscribed circle -- including the ego hull (1-W4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * math.pi


def wrap(a):
    """Wrap angle(s) to (-pi, pi]."""
    a = np.asarray(a, dtype=float)
    out = -np.mod(-a + math.pi, TWO_PI) + math.pi
    return float(out) if out.ndim == 0 else out


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    r: float

    def corners(self):  # pragma: no cover - circles have none
        return np.empty((0, 2))


@dataclass(frozen=True)
class Rect:
    """Oriented rectangle. `length` runs along the heading, `width` across it."""

    x: float
    y: float
    length: float
    width: float
    heading: float

    def axes(self):
        c, s = math.cos(self.heading), math.sin(self.heading)
        return np.array([c, s]), np.array([-s, c])  # along-length, along-width

    def corners(self):
        u, v = self.axes()
        c = np.array([self.x, self.y])
        hl, hw = self.length / 2.0, self.width / 2.0
        return np.array([c + hl * u + hw * v, c + hl * u - hw * v,
                         c - hl * u - hw * v, c - hl * u + hw * v])


def _seg_point_dist(p, a, b):
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom == 0.0 else float((p - a) @ ab) / denom
    t = min(1.0, max(0.0, t))
    return float(np.linalg.norm(p - (a + t * ab)))


def _seg_seg_dist(p1, p2, q1, q2):
    if _segments_intersect(p1, p2, q1, q2):
        return 0.0
    return min(_seg_point_dist(p1, q1, q2), _seg_point_dist(p2, q1, q2),
               _seg_point_dist(q1, p1, p2), _seg_point_dist(q2, p1, p2))


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_intersect(p1, p2, q1, q2):
    d1, d2 = _cross(q1, q2, p1), _cross(q1, q2, p2)
    d3, d4 = _cross(p1, p2, q1), _cross(p1, p2, q2)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    for d, o, a, b, p in ((d1, q1, q2, None, p1), (d2, q1, q2, None, p2),
                          (d3, p1, p2, None, q1), (d4, p1, p2, None, q2)):
        if d == 0.0 and _on_segment(o, a, p):
            return True
    return False


def _on_segment(a, b, p):
    return (min(a[0], b[0]) - 1e-12 <= p[0] <= max(a[0], b[0]) + 1e-12 and
            min(a[1], b[1]) - 1e-12 <= p[1] <= max(a[1], b[1]) + 1e-12)


def point_in_rect(p, r: Rect):
    u, v = r.axes()
    d = np.asarray(p, float) - np.array([r.x, r.y])
    return abs(float(d @ u)) <= r.length / 2.0 and abs(float(d @ v)) <= r.width / 2.0


def rect_rect_distance(a: Rect, b: Rect) -> float:
    """0.0 if they overlap, else the surface-to-surface distance."""
    ca, cb = a.corners(), b.corners()
    if _sat_overlap(ca, cb):
        return 0.0
    best = math.inf
    for i in range(4):
        for j in range(4):
            best = min(best, _seg_seg_dist(ca[i], ca[(i + 1) % 4], cb[j], cb[(j + 1) % 4]))
    return best


def rect_circle_distance(r: Rect, c: Circle) -> float:
    """0.0 if they overlap, else the surface-to-surface distance."""
    p = np.array([c.x, c.y])
    if point_in_rect(p, r):
        return 0.0
    corners = r.corners()
    d = min(_seg_point_dist(p, corners[i], corners[(i + 1) % 4]) for i in range(4))
    return max(0.0, d - c.r)


def _sat_overlap(ca, cb) -> bool:
    for poly in (ca, cb):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            ax = np.array([-e[1], e[0]])
            n = float(np.linalg.norm(ax))
            if n == 0.0:
                continue
            ax = ax / n
            pa, pb = ca @ ax, cb @ ax
            if pa.max() < pb.min() - 1e-12 or pb.max() < pa.min() - 1e-12:
                return False
    return True


def body_distance(hull: Rect, body) -> float:
    if isinstance(body, Rect):
        return rect_rect_distance(hull, body)
    if isinstance(body, Circle):
        return rect_circle_distance(hull, body)
    raise TypeError(type(body))


# --- polygon (navigable boundary, 2-E6) ---

def point_in_polygon(p, poly: np.ndarray) -> bool:
    x, y = float(p[0]), float(p[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xint:
                inside = not inside
    return inside


def polygon_distance(hull: Rect, poly: np.ndarray) -> float:
    """Surface-to-surface distance from the hull to the polygon boundary.

    Returns 0.0 when the hull touches or crosses it.
    """
    corners = hull.corners()
    n = len(poly)
    best = math.inf
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        for j in range(4):
            best = min(best, _seg_seg_dist(corners[j], corners[(j + 1) % 4], a, b))
    return best


def hull_inside_polygon(hull: Rect, poly: np.ndarray) -> bool:
    """True when every corner is inside and no edge crosses the boundary.

    Out-of-bounds triggers when *any part* of the hull leaves (5-S11).
    """
    corners = hull.corners()
    if not all(point_in_polygon(c, poly) for c in corners):
        return False
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        for j in range(4):
            if _segments_intersect(corners[j], corners[(j + 1) % 4], a, b):
                return False
    return True
