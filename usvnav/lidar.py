"""Condition 1-3: 2D LiDAR (4-V7).

360 degrees, 180 beams at 2 degree spacing, index 0 at the bow increasing
counter-clockwise (1-W3), first return only with true occlusion, and **no-return
encoded as the maximum range** -- never 0, which would be indistinguishable from a
return at zero distance. Single planar cast, no mount height: the world has no vertical
dimension.
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import Circle, Rect

N_BEAMS = 180
MAX_RANGE = 50.0                       # 4-V5, shared with 1-1's list and 1-2/1-4's mask
_ANGLES = np.arange(N_BEAMS) * (2.0 * math.pi / N_BEAMS)


#: Organiser-side acceleration seam (`usvnav.accel`, held back from the bundle): a native cast verified
#: bit-identical to `_scan_python`. `None`, the shipped state, means the Python below runs.
_native_scan = None


def scan(vessel, course, t: float) -> np.ndarray:
    """All beams against all surfaces at once: (beams, segments) broadcasting."""
    if _native_scan is not None:
        return _native_scan(vessel, course, t)
    return _scan_python(vessel, course, t)


def _scan_python(vessel, course, t: float) -> np.ndarray:
    ox, oy = vessel.x, vessel.y
    ang = _ANGLES + vessel.psi
    dx, dy = np.cos(ang), np.sin(ang)

    p, q, circles = _surfaces(course, t)
    best = np.full(N_BEAMS, MAX_RANGE)

    if len(p):
        ax, ay = p[:, 0], p[:, 1]
        vx, vy = q[:, 0] - ax, q[:, 1] - ay
        den = dx[:, None] * vy[None, :] - dy[:, None] * vx[None, :]
        wx, wy = ax[None, :] - ox, ay[None, :] - oy
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (wx * vy[None, :] - wy * vx[None, :]) / den
            ss = (wx * dy[:, None] - wy * dx[:, None]) / den
        hit = (np.abs(den) > 1e-12) & (tt >= 0.0) & (tt <= MAX_RANGE) & (ss >= 0.0) & (ss <= 1.0)
        tt = np.where(hit, tt, np.inf)
        best = np.minimum(best, tt.min(axis=1))

    for c in circles:
        fx, fy = ox - c.x, oy - c.y
        b = fx * dx + fy * dy
        cc = fx * fx + fy * fy - c.r * c.r
        disc = b * b - cc
        ok = disc >= 0.0
        tt = np.full(N_BEAMS, np.inf)
        tt[ok] = -b[ok] - np.sqrt(disc[ok])
        tt = np.where((tt >= 0.0) & (tt <= MAX_RANGE), tt, np.inf)
        best = np.minimum(best, tt)

    return best.astype(np.float32)


def _surfaces(course, t: float):
    """Rectangle edges as (P, Q) arrays plus the circle list. Boundary included (2-E6)."""
    segs_p, segs_q, circles = [], [], []
    shapes = list(course.static_shapes()) + course.traffic_rects(t)
    for s in shapes:
        if isinstance(s, Rect):
            c = s.corners()
            for i in range(4):
                segs_p.append(c[i])
                segs_q.append(c[(i + 1) % 4])
        elif isinstance(s, Circle):
            circles.append(s)
    b = course.boundary
    for i in range(len(b)):
        segs_p.append(b[i])
        segs_q.append(b[(i + 1) % len(b)])
    P = np.array(segs_p, dtype=float) if segs_p else np.empty((0, 2))
    Q = np.array(segs_q, dtype=float) if segs_q else np.empty((0, 2))
    return P, Q, circles
