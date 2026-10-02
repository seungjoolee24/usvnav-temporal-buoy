"""Free space, clearance and the narrowest-passage measurement.

One construction serves three callers, which is why it lives here:

  * 3-K9's geometric validity check -- start -> waypoints -> goal connectivity.
  * 3-K4's narrowest-passage axis, measured on the finished candidate (3-K1).
  * The reference agent's distance-to-goal field (7-A5), which is a cost term.

Free space is the navigable polygon minus static obstacle footprints. Traffic is
excluded throughout: it passes, and 2-E7 keeps traffic routes out of gate apertures.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from .geometry import Circle, Rect
from .plant import HULL_WIDTH


class FreeSpace:
    """A grid of clearance-to-nearest-static-surface, in metres.

    `clearance[i, j]` is the distance from that cell to the closest obstacle surface or
    to the boundary, and is <= 0 outside the navigable polygon. A hull centred on a cell
    with clearance c fits at any heading when c >= the hull half-diagonal, and fits when
    aligned with a passage when c >= the hull half-width.
    """

    def __init__(self, course, cell: float = 0.5):
        self.cell = cell
        b = course.boundary
        self.x0 = float(b[:, 0].min()) - 2.0
        self.y0 = float(b[:, 1].min()) - 2.0
        self.nx = int((float(b[:, 0].max()) + 2.0 - self.x0) / cell) + 1
        self.ny = int((float(b[:, 1].max()) + 2.0 - self.y0) / cell) + 1
        self.gx, self.gy = np.meshgrid(
            self.x0 + cell * np.arange(self.nx),
            self.y0 + cell * np.arange(self.ny), indexing="ij")

        inside = _inside_polygon(self.gx, self.gy, b)
        d_bound = _distance_to_polygon(self.gx, self.gy, b)
        clearance = np.where(inside, d_bound, -d_bound)

        for body in course.bodies:
            clearance = np.minimum(clearance, _signed_distance(self.gx, self.gy, body.shape))
        self.clearance = clearance

    # --- indexing -----------------------------------------------------------
    def cell_of(self, x, y):
        return (int(round((x - self.x0) / self.cell)), int(round((y - self.y0) / self.cell)))

    def in_bounds(self, c):
        return 0 <= c[0] < self.nx and 0 <= c[1] < self.ny

    def passable(self, margin: float):
        """Cells where a hull centre can sit with at least `margin` of clearance."""
        return self.clearance >= margin

    # --- 3-K9 ---------------------------------------------------------------
    def connected(self, a, b, margin: float, snap_a: float = 0.0,
                  snap_b: float = 0.0) -> bool:
        """Is there a route from `a` to `b` through cells with `margin` of clearance?

        `snap_a` / `snap_b` are how far, **in metres**, each endpoint may be relocated to
        find such a cell, and they are not free parameters: for a waypoint it is the
        arrival radius, because 3-K11 only requires the hull centre to come within that
        radius, and for the start pose it is zero, because the vessel is placed exactly
        there. Snapping by any other amount is what let this gate certify a course whose
        start was buried inside a moored vessel -- the endpoint was quietly moved 6 m
        into open water and the route from *there* was fine.
        """
        free = self.passable(margin)
        ca, cb = self.cell_of(*a), self.cell_of(*b)
        if not (self.in_bounds(ca) and self.in_bounds(cb)):
            return False
        ca = self._nearest_free(ca, free, int(math.floor(snap_a / self.cell)))
        cb = self._nearest_free(cb, free, int(math.floor(snap_b / self.cell)))
        if ca is None or cb is None:
            return False
        seen = np.zeros_like(free)
        stack = [ca]
        seen[ca] = True
        while stack:
            i, j = stack.pop()
            if (i, j) == cb:
                return True
            for di, dj in _N8:
                ni, nj = i + di, j + dj
                if (0 <= ni < self.nx and 0 <= nj < self.ny
                        and free[ni, nj] and not seen[ni, nj]):
                    seen[ni, nj] = True
                    stack.append((ni, nj))
        return False

    def _nearest_free(self, c, free, radius: int = 0):
        """The nearest passable cell within `radius` *cells*, or `None`.

        `radius = 0` means "here or nowhere". The default used to be 12 cells (6 m),
        which is the bug documented in `connected`.
        """
        if free[c]:
            return c
        for r in range(1, radius + 1):
            best, bestd = None, math.inf
            for di in range(-r, r + 1):
                for dj in (-r, r):
                    for i, j in ((c[0] + di, c[1] + dj), (c[0] + dj, c[1] + di)):
                        if 0 <= i < self.nx and 0 <= j < self.ny and free[i, j]:
                            d = di * di + dj * dj
                            if d < bestd:
                                best, bestd = (i, j), d
            if best is not None:
                return best
        return None

    # --- 3-K4 ---------------------------------------------------------------
    def bottleneck(self, chain, snaps=None) -> float:
        """Narrowest passage the route must pass through, as a *width* in metres.

        For each consecutive pair in `chain`, find the path maximising its minimum
        clearance (a widest-path problem), then take the tightest such value over the
        whole route. Passage width is twice the clearance, since clearance is measured
        to the hull centre.

        `snaps` is the per-point tolerance in metres, as in `connected` -- 0 for the
        start pose and the arrival radius for a waypoint. Without it, a waypoint centre
        legitimately sitting within half a hull width of a wall would report the wall as
        the route's bottleneck.
        """
        snaps = [0.0] * len(chain) if snaps is None else list(snaps)
        worst = math.inf
        for k, (a, b) in enumerate(zip(chain, chain[1:])):
            worst = min(worst, self._widest_path_clearance(a, b, snaps[k], snaps[k + 1]))
        return 2.0 * worst if math.isfinite(worst) else math.inf

    def _widest_path_clearance(self, a, b, snap_a=0.0, snap_b=0.0) -> float:
        """Maximin clearance over all paths from a to b. Dijkstra on min-clearance."""
        free = self.passable(0.0)
        ca, cb = self.cell_of(*a), self.cell_of(*b)
        if not (self.in_bounds(ca) and self.in_bounds(cb)):
            return -math.inf
        ca = self._nearest_free(ca, free, int(math.floor(snap_a / self.cell)))
        cb = self._nearest_free(cb, free, int(math.floor(snap_b / self.cell)))
        if ca is None or cb is None:
            return -math.inf

        best = np.full((self.nx, self.ny), -math.inf)
        best[ca] = self.clearance[ca]
        # max-heap via negated key
        heap = [(-best[ca], ca)]
        while heap:
            negk, (i, j) = heapq.heappop(heap)
            k = -negk
            if k < best[i, j]:
                continue
            if (i, j) == cb:
                return float(k)
            for di, dj in _N8:
                ni, nj = i + di, j + dj
                if not (0 <= ni < self.nx and 0 <= nj < self.ny):
                    continue
                nk = min(k, self.clearance[ni, nj])
                if nk > best[ni, nj]:
                    best[ni, nj] = nk
                    heapq.heappush(heap, (-nk, (ni, nj)))
        return float(best[cb])

    # --- §9.3's diagnostic ---------------------------------------------------
    def shortest_path_length(self, a, b, margin: float, snap_a: float = 0.0,
                             snap_b: float = 0.0) -> float:
        """Length in metres of the shortest route from `a` to `b` through cells with at
        least `margin` of clearance, 8-connected with diagonal steps at their true cost;
        `inf` if there is none. Snap tolerances as in `connected`.

        Two calls at two margins measure whether a course poses a route *decision*: the
        shortest path a hull will just fit through against the shortest path that stays a
        cautious distance off everything. On a straight reach the two coincide; around a
        headland, through a twin gate or a reef they do not, and the ratio is what §9.3
        asked the generator to produce. Computed on a 1 m grid rather than the 0.5 m one:
        a length is far less sensitive to the cell than a clearance is, and Dijkstra in
        pure Python is four times cheaper for it.
        """
        step = 2 if self.cell < 0.75 else 1
        free = self.passable(margin)[::step, ::step]
        cell = self.cell * step
        nx, ny = free.shape
        ca, cb = self.cell_of(*a), self.cell_of(*b)
        ca, cb = (ca[0] // step, ca[1] // step), (cb[0] // step, cb[1] // step)
        if not (0 <= ca[0] < nx and 0 <= ca[1] < ny and 0 <= cb[0] < nx and 0 <= cb[1] < ny):
            return math.inf
        ca = _nearest_free_in(free, ca, int(math.floor(snap_a / cell)))
        cb = _nearest_free_in(free, cb, int(math.floor(snap_b / cell)))
        if ca is None or cb is None:
            return math.inf
        dist = np.full((nx, ny), np.inf)
        dist[ca] = 0.0
        heap = [(0.0, ca)]
        diag = math.sqrt(2.0)
        while heap:
            d, (i, j) = heapq.heappop(heap)
            if d > dist[i, j]:
                continue
            if (i, j) == cb:
                return float(d * cell)
            for di, dj in _N8:
                ni, nj = i + di, j + dj
                if 0 <= ni < nx and 0 <= nj < ny and free[ni, nj]:
                    nd = d + (diag if di and dj else 1.0)
                    if nd < dist[ni, nj]:
                        dist[ni, nj] = nd
                        heapq.heappush(heap, (nd, (ni, nj)))
        return math.inf

    # --- 7-A5 ---------------------------------------------------------------
    def distance_field(self, target, margin: float) -> np.ndarray:
        """Grid distance to `target` through cells with at least `margin` clearance."""
        from collections import deque
        free = self.passable(margin)
        dist = np.full((self.nx, self.ny), np.inf)
        seed = self._nearest_free(self.cell_of(*target), free)
        if seed is None:
            return dist
        dist[seed] = 0.0
        q = deque([seed])
        while q:
            i, j = q.popleft()
            for di, dj in _N4:
                ni, nj = i + di, j + dj
                if (0 <= ni < self.nx and 0 <= nj < self.ny and free[ni, nj]
                        and dist[i, j] + self.cell < dist[ni, nj]):
                    dist[ni, nj] = dist[i, j] + self.cell
                    q.append((ni, nj))
        return dist


def _nearest_free_in(free, c, radius: int):
    """`FreeSpace._nearest_free` on an arbitrary mask (the coarse grid above)."""
    nx, ny = free.shape
    if free[c]:
        return c
    for r in range(1, radius + 1):
        best, bestd = None, math.inf
        for di in range(-r, r + 1):
            for dj in (-r, r):
                for i, j in ((c[0] + di, c[1] + dj), (c[0] + dj, c[1] + di)):
                    if 0 <= i < nx and 0 <= j < ny and free[i, j]:
                        d = di * di + dj * dj
                        if d < bestd:
                            best, bestd = (i, j), d
        if best is not None:
            return best
    return None


_N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_N8 = _N4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _inside_polygon(gx, gy, poly):
    """Vectorised even-odd test."""
    inside = np.zeros(gx.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if y1 == y2:
            continue
        straddles = (y1 > gy) != (y2 > gy)
        xint = x1 + (gy - y1) * (x2 - x1) / (y2 - y1)
        inside ^= straddles & (gx < xint)
    return inside


def _distance_to_polygon(gx, gy, poly):
    n = len(poly)
    best = np.full(gx.shape, np.inf)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        best = np.minimum(best, _distance_to_segment(gx, gy, a, b))
    return best


def _distance_to_segment(gx, gy, a, b):
    ab = b - a
    L2 = float(ab @ ab)
    if L2 == 0.0:
        return np.hypot(gx - a[0], gy - a[1])
    t = np.clip(((gx - a[0]) * ab[0] + (gy - a[1]) * ab[1]) / L2, 0.0, 1.0)
    return np.hypot(gx - (a[0] + t * ab[0]), gy - (a[1] + t * ab[1]))


def _signed_distance(gx, gy, shape):
    """Distance from each cell to the shape surface; negative inside."""
    if isinstance(shape, Circle):
        return np.hypot(gx - shape.x, gy - shape.y) - shape.r
    c, s = math.cos(shape.heading), math.sin(shape.heading)
    dx, dy = gx - shape.x, gy - shape.y
    a = np.abs(dx * c + dy * s) - shape.length / 2.0
    b = np.abs(-dx * s + dy * c) - shape.width / 2.0
    outside = np.hypot(np.maximum(a, 0.0), np.maximum(b, 0.0))
    inside = np.minimum(np.maximum(a, b), 0.0)
    return outside + inside
