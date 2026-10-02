"""The lane design rules (spec 2026-09-21-traffic-lanes-design.md §6, L1-L10) and their geometry.

`coursefile.validate` calls `lane_problems` when a course has lanes. Everything here is read off
the same objects the simulator uses -- `Lane` for the corridor and the schedule, `FreeSpace` for
the free width across a lane -- so a rule cannot disagree with the run.

Two notions recur. The **corridor** of a lane is the band swept by its widest vessel plus
`LANE_CLEAR` on each side (`Lane.corridor_half_width`). A **passage** is a stretch of the lane
where the free width across it, bank to body or body to body along the lane's normal, is below
the lane's coexist width `W_CO` (`Lane.coexist_width`): there the ego cannot pass beside a
vessel and waits for the gap the spacing guarantee (spec §2, L3/L4) promises.

End caps: the reach is a closed polygon and the water is cut off at the world box's edge on
both ends (`channel.build`). A lane vessel spawns with its centre on the cap, half of it
outside, by design (spec §4), so an open lane's shoreline test runs against a polygon whose
cap vertices are pushed outward, and the free-width profile reads `nan` where the water it
would measure is really the world's edge rather than a wall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import _seg_point_dist, _seg_seg_dist, body_distance, hull_inside_polygon
from .world import (A_MAX, EGO_SLOT, LANE_CLEAR, MOVING, SIZE_RANGES, TRAFFIC_SPEED_MAX,
                    TRAFFIC_SPEED_MIN, V_WAIT, Lane)

_EPS = 1e-6

#: L11 (spec §1 decision 12): the largest turn a lane may make at a vertex -- 2-E7's fillet rule,
#: for lanes. A vessel's heading is the segment direction, so a sharper corner snaps it.
LANE_TURN_MAX_DEG = 60.0

#: A berth (spec §1 decision 13) is a rectangular slip cut into the shoreline: its back wall is
#: an edge this long whose two neighbours (the side walls, this long) meet it near-perpendicularly,
#: with a convex corner at each end of the back wall and a reflex corner at each mouth.
NOTCH_WALL_M = (2.0, 14.0)
NOTCH_SIDE_M = (1.0, 20.0)
_PERP_COS = 0.35          # |cos| below this counts as perpendicular (turns between 70° and 110°)


# --------------------------------------------------------------------------- end caps

def cap_edges(boundary) -> list:
    """The boundary edges lying on `x = min` or `x = max` -- the reach's two end caps. A shape
    without a vertical edge at either extreme has none.

    This is a heuristic, not a semantic check: any polygon with a vertical edge at its minimum
    x and at its maximum x is treated as a reach with end caps, whatever the shape actually is.
    It is exact for the reaches the generator builds, where exactly two vertices sit at each
    extreme."""
    b = np.asarray(boundary, dtype=float)
    xmin, xmax = float(b[:, 0].min()), float(b[:, 0].max())
    out = []
    for i in range(len(b)):
        a, c = b[i], b[(i + 1) % len(b)]
        for x in (xmin, xmax):
            if abs(a[0] - x) < _EPS and abs(c[0] - x) < _EPS:
                out.append((a, c))
    return out


def cap_padded(boundary, pad: float) -> np.ndarray:
    """The boundary with every cap vertex moved `pad` metres outward along x; unchanged when the
    shape has no caps."""
    b = np.asarray(boundary, dtype=float).copy()
    if not cap_edges(b):
        return b
    xmin, xmax = float(b[:, 0].min()), float(b[:, 0].max())
    b[np.abs(b[:, 0] - xmin) < _EPS, 0] -= pad
    b[np.abs(b[:, 0] - xmax) < _EPS, 0] += pad
    return b


def distance_to_caps(p, caps) -> float:
    p = np.asarray(p, dtype=float)
    return min((_seg_point_dist(p, a, c) for a, c in caps), default=math.inf)


@dataclass(frozen=True)
class SpawnWall:
    """Where a lane may start or end (spec §4 with decision 13): an end cap or a berth's back wall.
    `outward` is the unit normal pointing out of the water -- past the cap, or into the land
    behind the berth; `sides` are a berth's two side walls, `()` for a cap."""

    a: np.ndarray
    b: np.ndarray
    outward: np.ndarray
    kind: str
    sides: tuple = ()


def _orientation(b) -> float:
    """+1 for a counter-clockwise polygon, -1 for clockwise (the shoelace sign)."""
    x, y = b[:, 0], b[:, 1]
    area2 = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return 1.0 if area2 > 0 else -1.0


def notch_walls(boundary) -> list:
    """Every rectangular notch in the shoreline, as the `SpawnWall` of its back wall.

    Walking the polygon, a notch cut into the land reads: a reflex corner (the first mouth
    vertex), a convex corner, a convex corner (the back wall's ends), a reflex corner (the
    second mouth vertex), with the three walls near-perpendicular to their neighbours and within
    `NOTCH_WALL_M`/`NOTCH_SIDE_M`. A pier-shaped bump into the water has the opposite signature
    and a bank's gentle bend has small turns, so neither is a wall."""
    b = np.asarray(boundary, dtype=float)
    n = len(b)
    if n < 6:
        return []
    sgn = _orientation(b)
    e = np.roll(b, -1, axis=0) - b                        # e[k] = b[k+1] - b[k]
    L = np.linalg.norm(e, axis=1)

    def cross(u, v):
        return float(u[0] * v[1] - u[1] * v[0])

    def perp(u, v):
        return abs(float(np.dot(u, v))) < _PERP_COS * max(float(np.linalg.norm(u)) * float(np.linalg.norm(v)), 1e-12)

    out = []
    for i in range(n):
        ip, ipp, inx, inn = (i - 1) % n, (i - 2) % n, (i + 1) % n, (i + 2) % n
        if not NOTCH_WALL_M[0] <= L[i] <= NOTCH_WALL_M[1]:
            continue
        if not (NOTCH_SIDE_M[0] <= L[ip] <= NOTCH_SIDE_M[1] and NOTCH_SIDE_M[0] <= L[inx] <= NOTCH_SIDE_M[1]):
            continue
        if not (perp(e[ip], e[i]) and perp(e[i], e[inx])):
            continue
        convex_ends = sgn * cross(e[ip], e[i]) > 0 and sgn * cross(e[i], e[inx]) > 0
        reflex_mouths = sgn * cross(e[ipp], e[ip]) < 0 and sgn * cross(e[inx], e[inn]) < 0
        if not (convex_ends and reflex_mouths):
            continue
        left = np.array([-e[i][1], e[i][0]]) / max(L[i], 1e-12)      # the interior side of a CCW edge
        outward = -sgn * left
        out.append(SpawnWall(b[i].copy(), b[inx].copy(), outward, "berth",
                             ((b[ip].copy(), b[i].copy()), (b[inx].copy(), b[inn].copy()))))
    return out


def spawn_walls(boundary) -> list:
    """The caps (as `SpawnWall`s pointing along ±x) followed by every berth's back wall."""
    b = np.asarray(boundary, dtype=float)
    xmin, xmax = float(b[:, 0].min()), float(b[:, 0].max())
    out = []
    for a, c in cap_edges(b):
        outward = np.array([-1.0, 0.0]) if abs(float(a[0]) - xmin) < _EPS else np.array([1.0, 0.0])
        out.append(SpawnWall(np.array(a, dtype=float), np.array(c, dtype=float), outward, "cap"))
    return out + notch_walls(b)


def spawn_padded(boundary, pad: float) -> np.ndarray:
    """The boundary with every spawn wall's two vertices moved `pad` metres along its outward
    normal -- caps along ±x exactly as `cap_padded`, a berth's back wall deeper into the land --
    so a vessel spawning half behind the wall is inside the padded shore (L1)."""
    b = np.asarray(boundary, dtype=float).copy()
    for wall in spawn_walls(b):
        for v in (wall.a, wall.b):
            hit = np.all(np.abs(b - v) < _EPS, axis=1)
            b[hit] += pad * wall.outward
    return b


def distance_to_walls(p, walls) -> float:
    p = np.asarray(p, dtype=float)
    return min((_seg_point_dist(p, w.a, w.b) for w in walls), default=math.inf)


def wall_edges(walls) -> list:
    """Every wall edge -- a berth's sides included -- as `(a, b)` pairs, for the free-width profile,
    which reads `nan` where the march is stopped by one of them (a spawn slip is not a passage)."""
    out = []
    for w in walls:
        out.append((w.a, w.b))
        out.extend(w.sides)
    return out


# --------------------------------------------------------------------------- polylines

def lane_chain(lane: Lane) -> np.ndarray:
    """The lane's vertices in order, the first repeated at the end when the lane is closed."""
    return np.vstack([lane.points, lane.points[:1]]) if lane.closed else np.asarray(lane.points, dtype=float)


def loop_chain(route) -> np.ndarray:
    return np.vstack([route.points, route.points[:1]])


def polyline_distance(chain_a, chain_b, *, stop_below: float | None = None) -> float:
    """Smallest distance between any segment of one polyline and any of the other.

    `_seg_seg_dist` is exact but there are `Na * Nb` pairs of segments and a brute-force
    double loop over it is too slow for long chains (0.65 s of `validate`'s 1.0 s on two
    200-vertex lanes was this, before). Instead, a vectorised lower bound -- the gap between
    each pair's axis-aligned bounding boxes, which is <= the true segment-segment distance --
    ranks the pairs, and `_seg_seg_dist` is evaluated exactly in that order, smallest lower
    bound first, stopping once the next lower bound is >= the best exact distance found so
    far: no pair that could beat the best is skipped, so the result equals the brute-force
    minimum to the last bit.

    `stop_below`, when given, returns as soon as the best exact distance found is below it --
    for a caller (L5) that only needs to know the true minimum is under some threshold, not
    its exact value; the distance returned is then still exact, just not necessarily the
    chains' true minimum."""
    a0, a1 = np.asarray(chain_a[:-1], dtype=float), np.asarray(chain_a[1:], dtype=float)
    b0, b1 = np.asarray(chain_b[:-1], dtype=float), np.asarray(chain_b[1:], dtype=float)
    amin, amax = np.minimum(a0, a1), np.maximum(a0, a1)
    bmin, bmax = np.minimum(b0, b1), np.maximum(b0, b1)
    gx = np.maximum(0.0, np.maximum(amin[:, None, 0], bmin[None, :, 0])
                          - np.minimum(amax[:, None, 0], bmax[None, :, 0]))
    gy = np.maximum(0.0, np.maximum(amin[:, None, 1], bmin[None, :, 1])
                          - np.minimum(amax[:, None, 1], bmax[None, :, 1]))
    lb = np.hypot(gx, gy)
    n_b = len(b0)
    order = np.argsort(lb, axis=None)
    flat_lb = lb.ravel()
    best = math.inf
    for idx in order:
        if flat_lb[idx] >= best:
            break
        i, j = divmod(int(idx), n_b)
        d = _seg_seg_dist(a0[i], a1[i], b0[j], b1[j])
        if d < best:
            best = d
            if best == 0.0 or (stop_below is not None and best < stop_below):
                return best
    return best


def point_to_polyline(p, chain) -> float:
    p = np.asarray(p, dtype=float)
    return min(_seg_point_dist(p, a, b) for a, b in zip(chain, chain[1:]))


def max_turn_deg(lane: Lane) -> float:
    """The largest change of direction between consecutive segments of the lane's chain, in
    degrees -- at interior vertices for an open lane, at every vertex (the closing corner too)
    for a closed one. `0.0` for a single segment."""
    segs = lane.segments()
    dirs = []
    for a, b, _, _ in segs:
        v = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
        dirs.append(v / max(float(np.linalg.norm(v)), 1e-12))
    pairs = list(zip(dirs, dirs[1:]))
    if lane.closed and len(dirs) > 1:
        pairs.append((dirs[-1], dirs[0]))
    worst = 0.0
    for u, w in pairs:
        cosang = float(np.clip(np.dot(u, w), -1.0, 1.0))
        worst = max(worst, math.degrees(math.acos(cosang)))
    return worst


# --------------------------------------------------------------------------- free width

def free_width_sides(lane: Lane, fs, caps=(), *, step: float = 1.0, cap_m: float = 130.0):
    """`(s, left, right)`: the free distance from the lane point along +normal (`left`) and −normal
    (`right`) at every `step` metres, `nan` on a side whose march is unbounded or stopped by a wall
    edge in `caps` (see `free_width_profile`, which is their sum).

    From the lane point, march along the normal on both sides in steps of one grid cell until a
    cell's clearance is <= 0 (a body's interior or its surface, or outside the water); the
    distance to that surface is the last free cell's own clearance added to its offset, which
    is exact when the surface is square to the normal and a lower bound otherwise. `cap_m`
    bounds the march (the widest channel is 120 m).

    A side is `nan` where no width exists to report: its march runs the full `cap_m` without
    finding a wall (the width there is unbounded, as far as the march can see, not `cap_m`), or
    its blocking point -- the lane point itself when the very first step is already blocked, else
    the blocked sample itself -- lies within one grid cell of a wall edge in `caps` -- the caps,
    and a berth's back and side walls (`wall_edges(spawn_walls(boundary))`). That is the wall
    reading as the world's edge or as the lane's own spawn slip: a lane's endpoints sit on spawn
    walls by design (spec §4, decision 13) and read `nan` this way, while a real passage a few
    metres away is still measured."""
    cell = fs.cell
    J = int(math.ceil(cap_m / cell))
    offsets = np.arange(1, J + 1) * cell
    n_s = int(math.floor(lane.length / step + 1e-9)) + 1
    s = np.minimum(np.arange(n_s) * step, lane.length)
    if s[-1] < lane.length - 1e-9:
        s = np.append(s, lane.length)
    left = np.full(len(s), np.nan)
    right = np.full(len(s), np.nan)
    for k, sk in enumerate(s):
        x, y, h = lane.pose_at_arc(float(sk))
        n = np.array([-math.sin(h), math.cos(h)])
        for side, out in ((1.0, left), (-1.0, right)):
            pts = np.array([x, y])[None, :] + offsets[:, None] * (side * n)[None, :]
            i = np.round((pts[:, 0] - fs.x0) / cell).astype(int)
            j = np.round((pts[:, 1] - fs.y0) / cell).astype(int)
            ok = (i >= 0) & (i < fs.nx) & (j >= 0) & (j < fs.ny)
            cl = np.full(J, -1.0)
            cl[ok] = fs.clearance[i[ok], j[ok]]
            blocked = np.nonzero(cl <= 0.0)[0]
            if len(blocked) == 0:
                continue                                  # unbounded, not a fabricated cap_m: nan
            jb = int(blocked[0])
            point = (x, y) if jb == 0 else pts[jb]
            if caps and distance_to_caps(point, caps) <= cell:
                continue                                   # the wall here is the world's edge: nan
            if jb == 0:
                i0, j0 = fs.cell_of(x, y)
                here = fs.clearance[i0, j0] if 0 <= i0 < fs.nx and 0 <= j0 < fs.ny else 0.0
                out[k] = max(float(here), 0.0)
            else:
                out[k] = float(offsets[jb - 1]) + max(float(cl[jb - 1]), 0.0)
    return s, left, right


def free_width_profile(lane: Lane, fs, caps=(), *, step: float = 1.0, cap_m: float = 130.0):
    """`(s, w)`: the free width across the lane at arc `s`, every `step` metres -- `left + right`
    from `free_width_sides`, `nan` if either side is (see there for the march and the nan rule)."""
    s, left, right = free_width_sides(lane, fs, caps, step=step, cap_m=cap_m)
    return s, left + right


@dataclass
class Stretch:
    """A passage on a lane: arc `s0` to `s1`, its narrowest free width and the lane's slowest
    speed inside it."""

    s0: float
    s1: float
    w_min: float
    v_min: float

    @property
    def depth(self) -> float:
        return self.s1 - self.s0


def narrow_stretches(lane: Lane, fs, threshold: float, caps=(), *, step: float = 1.0,
                      profile=None) -> list:
    """Maximal runs of profile samples with free width below `threshold`, each as a `Stretch`
    whose `s1` is the last sample plus `step` (so `depth` over-estimates by at most one step,
    which errs on the strict side of L4). `profile` is a precomputed `(s, w)` pair from
    `free_width_profile`, reused as-is when given -- and then `step` is read off it (`s[1] -
    s[0]`) rather than trusted from the argument, since a precomputed profile may have been
    sampled at a different step than the caller's default. `profile=None` (the default)
    computes the profile here with `caps` and `step`."""
    s, w = profile if profile is not None else free_width_profile(lane, fs, caps, step=step)
    if profile is not None and len(s) >= 2:
        step = float(s[1] - s[0])
    narrow = np.nan_to_num(w, nan=np.inf) < threshold
    out = []
    k = 0
    while k < len(s):
        if not narrow[k]:
            k += 1
            continue
        k0 = k
        while k + 1 < len(s) and narrow[k + 1]:
            k += 1
        inside = s[k0:k + 1]
        out.append(Stretch(float(s[k0]), float(s[k]) + step, float(np.min(w[k0:k + 1])),
                           min(lane.speed_at_arc(float(v)) for v in inside)))
        k += 1
    return out


# --------------------------------------------------------------------------- the rules

def corridor_conflicts(lane: Lane, course) -> list:
    """Every place the lane's corridor (the widest vessel plus LANE_CLEAR all round) touches a static
    body or leaves the water, as `(arc, what)` -- `what` a body index or `"shore"` -- swept by arc
    length with overlapping rectangles as L1 does; the shoreline test runs on the spawn-padded polygon
    for an open lane. `lane_problems` reports the first of each kind; the generator repairs them all."""
    L_max, W_max = lane.max_size()
    size = (L_max + 2.0 * LANE_CLEAR, W_max + 2.0 * LANE_CLEAR)
    shore = (spawn_padded(course.boundary, L_max / 2.0 + 2.0 * LANE_CLEAR) if not lane.closed
             else course.boundary)
    step = max(0.25, 0.5 * L_max)
    out = []
    for k in range(int(math.ceil(lane.length / step)) + 1):
        arc = min(k * step, lane.length)
        rect = lane.rect_at_arc(arc, size)
        for i, body in enumerate(course.bodies):
            if body_distance(rect, body.shape) <= 0.0:
                out.append((arc, i))
                break
        if not hull_inside_polygon(rect, shore):
            out.append((arc, "shore"))
    return out


INTERACT_BAND_M = 12.0


def _chain_distance(p, chain) -> float:
    a = chain[:-1]; b = chain[1:]; ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0.0, 1.0)
    return float(np.linalg.norm(a + ab * t[:, None] - p, axis=1).min())


def interaction_speed(lane, chain, band: float = INTERACT_BAND_M) -> float:
    """The slowest lane speed within `band` metres of the ego's waypoint chain -- the speed L3's gap
    is judged at. Where the chain never comes near the lane, the lane's median speed."""
    chain = np.asarray(chain, dtype=float).reshape(-1, 2)
    near = []
    for s in np.arange(0.0, float(lane.length), 1.0):
        x, y, _ = lane.pose_at_arc(float(s))
        if _chain_distance(np.array([x, y]), chain) <= band:
            near.append(lane.speed_at_arc(float(s)))
    return float(min(near)) if near else float(np.median(lane.speeds))


def lane_problems(course, *, fs=None) -> list:
    """Every lane rule of spec §6, as `(rule, message, severity)`; empty for a course without
    lanes. `fs` is a `FreeSpace` of the course, built here when not supplied."""
    if not course.lanes:
        return []
    from .freespace import FreeSpace
    from .sim import SENSING_RANGE
    fs = fs if fs is not None else FreeSpace(course)
    walls = spawn_walls(course.boundary)
    caps = wall_edges(walls)          # what the profile blanks on: caps and berth walls
    start_xy = tuple(float(v) for v in course.start[:2])
    out: list = []

    def bad(rule, message, severity="error"):
        out.append((rule, message, severity))

    for j, lane in enumerate(course.lanes):
        L_max, W_max = lane.max_size()

        # L1: the corridor -- the widest vessel plus LANE_CLEAR all round -- never touches a
        # static body or leaves the water (factored into `corridor_conflicts` so the generator's
        # repair loop can act on every conflict, not just the first of each kind). The shoreline
        # test runs on the spawn-padded polygon for an open lane, whose first and last vessel
        # spawn half behind a cap or a berth's back wall by design (decision 13); a closed lane
        # never spawns at a wall, so its shore is unpadded.
        hits = corridor_conflicts(lane, course)
        body_hit = next(((arc, w) for arc, w in hits if isinstance(w, int)), None)
        if body_hit is not None:
            arc, i = body_hit
            bad("L1", f"lanes[{j}]'s corridor ({W_max + 2.0 * LANE_CLEAR:.1f} m wide: the {W_max:.1f} m "
                      f"vessel plus LANE_CLEAR = {LANE_CLEAR} m each side) passes through "
                      f"bodies[{i}] ({course.bodies[i].cls}) near arc {arc:.0f} m")
        shore_hit = next(((arc, w) for arc, w in hits if w == "shore"), None)
        if shore_hit is not None:
            arc, _ = shore_hit
            bad("L1", f"lanes[{j}]'s corridor leaves the water (crosses the shoreline) near arc "
                      f"{arc:.0f} m")

        # L11: corners are filleted so heading and velocity are continuous (decision 12).
        turn = max_turn_deg(lane)
        if turn > LANE_TURN_MAX_DEG + 1e-9:
            bad("L11", f"lanes[{j}] turns {turn:.1f}° at one vertex; L11 allows {LANE_TURN_MAX_DEG:.0f}° -- "
                       f"fillet the corner at a radius of at least twice the longest vessel "
                       f"({2.0 * L_max:.0f} m) so heading and velocity stay continuous")

        # L2: speeds within 2-E2, acceleration within A_MAX.
        v_lo, v_hi = float(np.min(lane.speeds)), float(np.max(lane.speeds))
        if v_lo < TRAFFIC_SPEED_MIN - 1e-9 or v_hi > TRAFFIC_SPEED_MAX + 1e-9:
            bad("L2", f"lanes[{j}]'s speeds run {v_lo:.2f}-{v_hi:.2f} m/s; 2-E2 allows "
                      f"{TRAFFIC_SPEED_MIN}-{TRAFFIC_SPEED_MAX}")
        accel = lane.max_accel()
        if accel > A_MAX + 1e-9:
            bad("L2", f"lanes[{j}] accelerates at {accel:.2f} m/s² on one segment (|v·dv/ds|); "
                      f"A_MAX is {A_MAX} -- lengthen the segment or flatten the speed step")

        # L3: the space inequality -- an ego slot between consecutive vessels where the ego meets
        # the lane (PoC 2026-09-22: was the lane's global minimum speed, which a berth's 0.5 m/s
        # pin always set and which lifted every headway to ~50 s).
        v_ref = interaction_speed(lane, [course.start[:2], *course.waypoints])
        gap = lane.headway() * v_ref - L_max
        if gap < EGO_SLOT - 1e-9:
            bad("L3", f"lanes[{j}]: at {v_ref:.2f} m/s (where the route meets the lane) a headway of "
                      f"{lane.headway():.0f} s leaves {gap:.1f} m between a {L_max:.0f} m vessel and the "
                      f"next; the ego needs EGO_SLOT = {EGO_SLOT:.0f} m (spec §2, headway·v − L ≥ EGO_SLOT)")

        # L4: the time inequality at every passage narrower than W_CO -- the vessel's occupancy
        # of the passage plus the ego's transit at V_WAIT must fit inside one headway (spec §2).
        # L9 (sizes): every vessel inside 2-E5's moving-vessel range, immediately followed by
        # L9 (width): the widest vessel leaves a metre of the lane's narrowest passage -- both
        # of L9's clauses sit together, in the block that computes the profile L4 also needs.
        w_co = lane.coexist_width()
        s_prof, widths = free_width_profile(lane, fs, caps)
        stretches = narrow_stretches(lane, fs, w_co, caps, profile=(s_prof, widths))
        for st in stretches:
            occupied = (L_max + st.depth) / st.v_min
            needed = (st.depth + EGO_SLOT) / V_WAIT
            if lane.headway() - occupied < needed - 1e-9:
                bad("L4", f"lanes[{j}]: a {L_max:.0f} m vessel at {st.v_min:.2f} m/s holds the "
                          f"{st.w_min:.1f} m passage at arc {st.s0:.0f}-{st.s1:.0f} m for {occupied:.0f} s "
                          f"of every {lane.headway():.0f} s, leaving {lane.headway() - occupied:.0f} s; "
                          f"the ego needs {needed:.0f} s to get through at V_WAIT = {V_WAIT} m/s "
                          f"(spec §2, headway − (L + D)/v ≥ (D + EGO_SLOT)/V_WAIT)")
        (l_lo, l_hi), (w_lo, w_hi) = SIZE_RANGES[MOVING]
        for k, (L, W) in enumerate(lane.vessels):
            if not l_lo - 1e-6 <= L <= l_hi + 1e-6 or not w_lo - 1e-6 <= W <= w_hi + 1e-6:
                bad("L9", f"lanes[{j}].vessels[{k}] is {L:.1f} × {W:.1f} m; 2-E5 allows "
                          f"{l_lo:.0f}-{l_hi:.0f} m by {w_lo:.1f}-{w_hi:.0f} m for a moving vessel")
        if np.any(~np.isnan(widths)):
            w_min = float(np.nanmin(widths))
            if W_max > w_min - 1.0 + 1e-9:
                bad("L9", f"lanes[{j}]'s widest vessel is {W_max:.1f} m; the lane's narrowest passage "
                          f"is {w_min:.1f} m and L9 leaves a metre of it: {w_min - 1.0:.1f} m at most "
                          f"(spec §5.3)")

        # L8: nothing within one sensing range of the start pose at t = 0 (2-E7's rule, for lanes).
        for r in lane.rects_at(0.0):
            d0 = math.dist((r.x, r.y), start_xy)
            if d0 < SENSING_RANGE:
                bad("L8", f"lanes[{j}] has a vessel {d0:.1f} m from the ego start pose at t = 0; "
                          f"2-E7 requires at least one sensing range ({SENSING_RANGE:.0f} m) -- "
                          f"shift offset_s")
                break

        # L10 (warning): an open lane's vessels enter and leave at a spawn wall -- an end cap or a
        # berth's back wall (spec §4, decision 13) -- so none appears out of nowhere inside an
        # observation. A course without any wall is not judged.
        if not lane.closed and walls:
            for end, name in ((lane.points[0], "starts"), (lane.points[-1], "ends")):
                d = distance_to_walls(end, walls)
                if d > 1.0:
                    bad("L10", f"lanes[{j}] {name} {d:.0f} m from the nearest end cap or berth wall; vessels "
                               f"would appear or vanish in open water (spec §4 spawns them on the walls)",
                        "warning")

    # L5: no two corridors share any water -- lanes with lanes, and lanes with the legacy loops.
    # Two loops without a lane are 2-E7's concern (phase-separated where they cross), not L5's.
    from .collide import HULL_RADIUS
    corridors = [(f"lanes[{j}]", lane_chain(lane), lane.corridor_half_width())
                 for j, lane in enumerate(course.lanes)]
    corridors += [(f"traffic[{j}]", loop_chain(route), route.width / 2.0 + LANE_CLEAR)
                  for j, route in enumerate(course.traffic)]
    n_lanes = len(course.lanes)
    for a in range(n_lanes):
        for b in range(a + 1, len(corridors)):
            name_a, chain_a, half_a = corridors[a]
            name_b, chain_b, half_b = corridors[b]
            d = polyline_distance(chain_a, chain_b, stop_below=half_a + half_b)
            if d < half_a + half_b - 1e-9:
                bad("L5", f"{name_a} and {name_b} run {d:.1f} m apart; their corridors "
                          f"({2 * half_a:.1f} m and {2 * half_b:.1f} m wide) overlap -- lanes never "
                          f"share water (spec §1 decision 7)")

    # L7: corridors stay out of the start pose's neighbourhood and the goal's arrival disc (2-E11's
    # discs, each grown by the corridor's half-width). Intermediate waypoints are deliberately
    # dangerous places (owner, 2026-09-22): a lane may pass right beside one.
    discs = [(start_xy, HULL_RADIUS, "the start pose")]
    if len(course.waypoints):
        w, r = course.waypoints[-1], course.arrival_radii[-1]
        discs.append((tuple(float(v) for v in w), float(r) + HULL_RADIUS, "the goal's arrival disc"))
    for j, lane in enumerate(course.lanes):
        chain, half = lane_chain(lane), lane.corridor_half_width()
        for centre, rad, what in discs:
            d = point_to_polyline(centre, chain)
            if d < rad + half - 1e-9:
                bad("L7", f"lanes[{j}]'s corridor passes {max(d - half, 0.0):.1f} m from {what}; "
                          f"2-E11 keeps {rad:.1f} m clear of it, so the lane's centreline must stay "
                          f"{rad + half:.1f} m off")

    return out
