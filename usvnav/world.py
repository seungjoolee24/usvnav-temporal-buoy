"""Course description -- the static world plus deterministic traffic.

The course file is the complete static world and carries no seed, no difficulty label
and no generator parameters (3-K6, follow-up item 6). Each traffic route's phase is fixed
in the course file; only the 1-4 disturbance is drawn from the episode seed.

Classes (2-E4). Nine appear in a Track 1 raster; `outside_world` is in the shared
vocabulary but never appears here (4-V12). Moored and moving vessels are distinct
*object* classes that share one raster colour (4-V13).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import Circle, Rect, wrap
from .plant import HULL_LENGTH, HULL_WIDTH, V_CRUISE

WATER = "water"
BANK = "bank"
BUOY = "buoy"
PIER = "pier"
MOORED = "moored_vessel"
MOVING = "moving_vessel"
DOCK = "dock"
UNOBSERVED = "unobserved"
EGO = "ego"
OUTSIDE_WORLD = "outside_world"

STATIC_CLASSES = (BUOY, PIER, MOORED, DOCK)

# 2-E5 size ranges, metres.
# 2-E5 size ranges, metres, as `(length_range, width_range)` for every rectangle -- the
# same order `geometry.Rect` takes them in, where `length` runs along the heading.
#
# The bridge-pier row used to be written the other way round, `(across, along)`, with a
# comment saying so. Nothing read the table until the course-file validator did, and it
# then reported every pier the generator had just drawn as out of range: the generator
# passes 8-20 m as the length directly and never consulted the table, so the two never
# had to agree. A lookup table whose rows are ordered differently from each other is a
# trap regardless of the comment, so the row was flipped rather than the reader.
SIZE_RANGES = {
    BUOY: (0.6, 0.6),                      # 2-E4 fixes the diameter; 4-V7 derives it
    PIER: ((8.0, 20.0), (4.0, 10.0)),      # 8-20 m along the flow, 4-10 m across
    MOORED: ((10.0, 40.0), (3.0, 10.0)),     # 2026-09-25: small moored craft too (the gap filler places 10-25 x 3-6 m)
    DOCK: ((10.0, 30.0), (5.0, 15.0)),
    MOVING: ((3.0, 20.0), (1.2, 6.0)),     # PoC 2026-09-22: small craft added (2-E5 said 6-20 x 2.5-6)
}

# 2-E2
TRAFFIC_SPEED_MIN, TRAFFIC_SPEED_MAX = 0.5, 2.5

#: Lanes (2-E14). `A_MAX` bounds a lane vessel's acceleration along its profile, as
#: `|v · dv/ds|` per segment; `LANE_CLEAR` is the margin kept beside a vessel's hull in
#: the corridor the validator and the generator sweep. Both are design defaults.
A_MAX = 0.2
LANE_CLEAR = 0.5

#: The ego's slot in a gap between two lane vessels: its length and a hull width fore and aft
#: (spec §2). `V_WAIT` is the speed the spacing guarantee assumes for the ego through a passage
#: it had to wait for -- half of cruise, since it starts from rest beside the corridor.
EGO_SLOT = HULL_LENGTH + 2.0 * HULL_WIDTH      # 7 m
V_WAIT = V_CRUISE / 2.0                        # 0.75 m/s

#: `Lane.moving_at` remembers this many distinct times. A look-ahead planner asks for 14
#: look-ahead instants per tick and the same instant again on the next three ticks; the
#: simulator, lidar and raster ask for the current one. Two 40-vertex lanes cost ~120 µs per
#: uncached call (measured 2026-09-21, phase 2a) -- more than the collision check itself.
MEMO_MAX = 256


@dataclass(frozen=True)
class StaticBody:
    cls: str
    shape: Circle | Rect


@dataclass
class TrafficRoute:
    """A closed-loop polyline travelled at constant speed with a phase offset (2-E7).

    Corners are filleted at generation time so heading and velocity are continuous;
    the vessel's heading is the path tangent.
    """

    cls: str
    length: float
    width: float
    points: np.ndarray          # (N,2), closed loop, already filleted
    speed: float
    phase: float = 0.0          # arc-length offset at t=0

    _cum: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        p = np.asarray(self.points, dtype=float)
        # Drop consecutive duplicate vertices. A zero-length segment makes `pose_at`
        # compute the heading as `atan2(0, 0) = 0` -- so at that one arc position the
        # vessel points along +x_world regardless of where its path is going. The
        # generator produced them: filleting a short side consumes it from both ends, so
        # two corners' fillets met at exactly the same point. Deduping here rather than
        # in the fillet covers hand-authored routes too, which the course file accepts.
        keep = np.ones(len(p), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-9
        if len(p) > 1 and np.linalg.norm(p[-1] - p[0]) <= 1e-9:
            keep[-1] = False
        p = p[keep]
        seg = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
        self.points = p
        self._cum = np.concatenate([[0.0], np.cumsum(seg)])

    @property
    def perimeter(self) -> float:
        return float(self._cum[-1])

    def pose_at_arc(self, s: float):
        """Exact pose at arc-length `s` along the loop, phase aside. Over one perimeter this
        visits every position the vessel ever occupies, for any phase -- which is what a
        static-collision check has to cover (the phase only chooses where t=0 lands)."""
        s = s % self.perimeter
        i = int(np.searchsorted(self._cum, s, side="right") - 1)
        i = min(max(i, 0), len(self.points) - 1)
        a = self.points[i]
        b = self.points[(i + 1) % len(self.points)]
        seg = float(self._cum[i + 1] - self._cum[i])
        frac = 0.0 if seg == 0.0 else (s - self._cum[i]) / seg
        p = a + (b - a) * frac
        head = math.atan2(b[1] - a[1], b[0] - a[0])
        return float(p[0]), float(p[1]), wrap(head)

    def pose_at(self, t: float):
        """Exact pose at time t. Deterministic: a pure function of t (2-E7)."""
        return self.pose_at_arc(self.phase + self.speed * t)

    def rect_at(self, t: float) -> Rect:
        x, y, h = self.pose_at(t)
        return Rect(x, y, self.length, self.width, h)

    def rect_at_arc(self, s: float) -> Rect:
        x, y, h = self.pose_at_arc(s)
        return Rect(x, y, self.length, self.width, h)

    def velocity_at(self, t: float):
        _, _, h = self.pose_at(t)
        return np.array([self.speed * math.cos(h), self.speed * math.sin(h)])


def _segment_time(d: float, a: float, b: float) -> float:
    """Seconds to cover `d` metres with the speed linear in arc length from `a` to `b`.
    `∫₀ᵈ ds / (a + (b − a)·s/d)` = `d·ln(b/a)/(b − a)`, or `d/a` when the speed is constant."""
    if abs(b - a) < 1e-12:
        return d / a
    return d * math.log(b / a) / (b - a)


def _segment_arc(tau: float, d: float, a: float, b: float) -> float:
    """Metres covered after `tau` seconds on the same segment -- the inverse of
    `_segment_time`: `a·d·(exp((b − a)·tau/d) − 1)/(b − a)`, or `a·tau` when constant."""
    if abs(b - a) < 1e-12:
        return a * tau
    return a * d * (math.exp((b - a) * tau / d) - 1.0) / (b - a)


@dataclass
class Lane:
    """A traffic lane (2-E14): a polyline with a speed profile and a vessel schedule.

    *Open*: vessels enter at `points[0]` every `headway_s` seconds starting at `offset_s`,
    ride the profile to the last point and leave; the `k`-th entry has the size
    `vessels[k mod len(vessels)]`. *Closed*: the polyline is a loop and exactly
    `len(vessels)` vessels circulate, equally spaced in time (`traversal_s / len(vessels)`
    apart), vessel `k+1` leading vessel `k`. Every vessel rides the same profile, so the
    time gap between consecutive vessels is constant and they never meet.

    The speed is piecewise-linear in arc length between the vertices, so the time along
    a segment and its inverse are closed-form (`_segment_time`, `_segment_arc`) and a
    pose is a pure function of time -- nothing depends on the ego or the episode seed
    (3-K6). Corners are filleted by whoever builds the polyline, as for `TrafficRoute`.
    """

    closed: bool
    points: np.ndarray                          # (N,2); closed: the last segment returns to points[0]
    speeds: np.ndarray                          # (N,) m/s at the vertices
    vessels: list                               # [(length_m, width_m), ...]
    headway_s: float | None = None              # open lanes only
    offset_s: float = 0.0
    cls: str = MOVING

    # Derived caches, not part of a `Lane`'s value: two lanes with the same points/speeds/
    # vessels/headway/offset are equal and interchangeable regardless of what `__post_init__`
    # happened to compute, so these are excluded from `__eq__` (and `__init__`'s signature,
    # and `repr`) rather than sitting there as ordinary fields a caller could pass in.
    _pseg: np.ndarray = field(default=None, init=False, compare=False, repr=False)   # vertices of the segment chain (N or N+1)
    _vseg: np.ndarray = field(default=None, init=False, compare=False, repr=False)
    _cum: np.ndarray = field(default=None, init=False, compare=False, repr=False)    # arc length at each chain vertex
    _T: np.ndarray = field(default=None, init=False, compare=False, repr=False)      # time at each chain vertex
    _memo: dict = field(default_factory=dict, init=False, compare=False, repr=False)  # round(t, 6) -> moving_at(t)

    def __post_init__(self):
        p = np.asarray(self.points, dtype=float)
        v = np.asarray(self.speeds, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2:
            raise ValueError("a lane needs at least two [x, y] points")
        if v.shape != (len(p),):
            raise ValueError("a lane's speeds must have one entry per point")
        if np.any(v <= 0.0):
            raise ValueError("a lane's speeds must be positive")
        self.vessels = [(float(L), float(W)) for L, W in self.vessels]
        if not self.vessels:
            raise ValueError("a lane needs at least one vessel size")
        if self.closed:
            if self.headway_s is not None:
                raise ValueError("a closed lane has no headway_s: its headway is traversal_s / len(vessels)")
            pseg, vseg = np.vstack([p, p[:1]]), np.concatenate([v, v[:1]])
        else:
            if self.headway_s is None or float(self.headway_s) <= 0.0:
                raise ValueError("an open lane needs headway_s > 0")
            self.headway_s = float(self.headway_s)
            pseg, vseg = p, v
        seg = np.linalg.norm(np.diff(pseg, axis=0), axis=1)
        if np.any(seg <= 1e-9):
            raise ValueError("a lane has a zero-length segment")
        self.points, self.speeds = p, v
        self.offset_s = float(self.offset_s)
        self._pseg, self._vseg = pseg, vseg
        self._cum = np.concatenate([[0.0], np.cumsum(seg)])
        times = [_segment_time(float(seg[i]), float(vseg[i]), float(vseg[i + 1]))
                 for i in range(len(seg))]
        self._T = np.concatenate([[0.0], np.cumsum(times)])

    # --- the profile -----------------------------------------------------------
    @property
    def length(self) -> float:
        return float(self._cum[-1])

    @property
    def traversal_s(self) -> float:
        return float(self._T[-1])

    def headway(self) -> float:
        """Seconds between consecutive vessels."""
        return self.traversal_s / len(self.vessels) if self.closed else float(self.headway_s)

    def _segment_of_arc(self, s: float) -> int:
        i = int(np.searchsorted(self._cum, s, side="right") - 1)
        return min(max(i, 0), len(self._cum) - 2)

    def speed_at_arc(self, s: float) -> float:
        i = self._segment_of_arc(s)
        d = float(self._cum[i + 1] - self._cum[i])
        frac = (s - float(self._cum[i])) / d
        return float(self._vseg[i] + (self._vseg[i + 1] - self._vseg[i]) * frac)

    def time_at_arc(self, s: float) -> float:
        i = self._segment_of_arc(s)
        d = float(self._cum[i + 1] - self._cum[i])
        a, b = float(self._vseg[i]), float(self._vseg[i + 1])
        part = s - float(self._cum[i])
        # speed at `part` along the segment, then the segment's own time integral up to it
        b_part = a + (b - a) * part / d
        return float(self._T[i]) + (_segment_time(part, a, b_part) if part > 0.0 else 0.0)

    def arc_at_time(self, tau: float) -> float:
        """Arc length reached `tau` seconds after passing `points[0]` (0 ≤ tau ≤ traversal_s)."""
        tau = min(max(tau, 0.0), self.traversal_s)
        i = int(np.searchsorted(self._T, tau, side="right") - 1)
        i = min(max(i, 0), len(self._T) - 2)
        d = float(self._cum[i + 1] - self._cum[i])
        s = float(self._cum[i]) + _segment_arc(tau - float(self._T[i]), d,
                                                float(self._vseg[i]), float(self._vseg[i + 1]))
        return min(max(s, 0.0), self.length)

    def pose_at_arc(self, s: float):
        i = self._segment_of_arc(s)
        a, b = self._pseg[i], self._pseg[i + 1]
        d = float(self._cum[i + 1] - self._cum[i])
        frac = (s - float(self._cum[i])) / d
        p = a + (b - a) * frac
        return float(p[0]), float(p[1]), wrap(math.atan2(b[1] - a[1], b[0] - a[0]))

    # --- the schedule ---------------------------------------------------------------
    def vessel(self, k: int):
        return self.vessels[k % len(self.vessels)]

    def alive(self, t: float) -> list:
        """`(k, s)` for every vessel on the lane at time `t`."""
        out = []
        if self.closed:
            n = len(self.vessels)
            for k in range(n):
                tau = (t - self.offset_s + k * self.traversal_s / n) % self.traversal_s
                out.append((k, self.arc_at_time(tau)))
            return out
        h = self.headway_s
        k_hi = math.floor((t - self.offset_s) / h + 1e-9)
        k_lo = max(0, math.ceil((t - self.offset_s - self.traversal_s) / h - 1e-9))
        for k in range(k_hi, k_lo - 1, -1):
            if k < 0:
                break
            tau = t - (self.offset_s + k * h)
            if 0.0 <= tau <= self.traversal_s:
                out.append((k, self.arc_at_time(tau)))
        return out

    def rects_at(self, t: float) -> list:
        return [r for r, _ in self.moving_at(t)]

    def moving_at(self, t: float) -> list:
        """`(rect, (vx, vy))` per alive vessel, the velocity over ground in world axes. A pure
        function of `t`, so it is computed once per distinct time (to a microsecond) and kept
        in a bounded memo; the list returned is the caller's to extend."""
        key = round(float(t), 6)
        got = self._memo.get(key)
        if got is None:
            got = []
            for k, s in self.alive(t):
                x, y, h = self.pose_at_arc(s)
                L, W = self.vessel(k)
                v = self.speed_at_arc(s)
                got.append((Rect(x, y, L, W, h), (v * math.cos(h), v * math.sin(h))))
            if len(self._memo) >= MEMO_MAX:
                del self._memo[next(iter(self._memo))]          # the oldest entry
            self._memo[key] = got
        return list(got)

    def max_accel(self) -> float:
        """The largest `|v · dv/ds|` on the lane -- `dv/dt` along a segment where `v` is
        linear in `s` is largest at the faster end."""
        worst = 0.0
        for i in range(len(self._cum) - 1):
            d = float(self._cum[i + 1] - self._cum[i])
            a, b = float(self._vseg[i]), float(self._vseg[i + 1])
            worst = max(worst, abs(max(a, b) * (b - a) / d))
        return worst

    # --- geometry the rules and the generator read (spec §2, §6) --------------------
    def max_size(self):
        """`(length, width)` of the largest vessel on the lane, each taken over all of them."""
        return (max(L for L, _ in self.vessels), max(W for _, W in self.vessels))

    def corridor_half_width(self) -> float:
        """Half the width the lane's corridor is swept with: the widest hull's half-width plus
        `LANE_CLEAR` (spec §6 L1)."""
        return self.max_size()[1] / 2.0 + LANE_CLEAR

    def coexist_width(self) -> float:
        """`W_CO` (spec §2): a passage narrower than this cannot hold the widest vessel and the
        ego side by side, so the ego waits for the gap there."""
        return self.max_size()[1] + HULL_WIDTH + 2.0 * LANE_CLEAR

    def rect_at_arc(self, s: float, size=None) -> Rect:
        """The rectangle a vessel of `size` (default: the largest) occupies at arc `s`."""
        L, W = self.max_size() if size is None else size
        x, y, h = self.pose_at_arc(s)
        return Rect(x, y, float(L), float(W), h)

    def segments(self) -> list:
        """`(a, b, s0, s1)` for every segment of the chain -- the return segment included when the
        lane is closed -- with `s0`, `s1` the arc length at its ends."""
        return [(self._pseg[i], self._pseg[i + 1], float(self._cum[i]), float(self._cum[i + 1]))
                for i in range(len(self._cum) - 1)]


@dataclass
class Course:
    """Everything static, plus the traffic: closed loops (`TrafficRoute`) and lanes (`Lane`).
    The episode seed drives only the 1-4 disturbance; loops' phases and lanes' schedules
    are fixed in the course file, not re-drawn per seed."""

    boundary: np.ndarray                 # navigable polygon == the shoreline (2-E6)
    bodies: list[StaticBody]
    waypoints: np.ndarray                # (K,2) ordered chain, 2-4 of them (3-K5)
    arrival_radii: np.ndarray            # (K,) -- 5 m intermediate, 3 m goal (3-K11)
    start: tuple[float, float, float]    # x, y, psi
    traffic: list[TrafficRoute] = field(default_factory=list)
    lanes: list[Lane] = field(default_factory=list)      # 2-E14; see `Lane`

    # 4-V12: generated terrain reaches at least one sensing range beyond the navigable
    # extent, so `outside_world` never enters a Track 1 raster. The renderer has no
    # colour for that class (4-V13 has eight entries), so it raises if it ever does --
    # which is what makes this field a checked guarantee rather than a claim.
    terrain_extent: tuple[float, float, float, float] | None = None

    # 3-K2's situation types present in this course, in order along the reach. Diagnostic
    # and generator-side only: follow-up item 6 forbids a course file from carrying a
    # difficulty label, seed or generator parameters, because participants can read any
    # course file they hold, and a situation list is a difficulty label by another name.
    # It must be stripped when a course is written out for a team.
    situations: tuple[str, ...] = ()

    @property
    def n_waypoints(self) -> int:
        return len(self.waypoints)

    def static_shapes(self):
        return [b.shape for b in self.bodies]

    def traffic_rects(self, t: float):
        """Every moving vessel's rectangle at time `t`: loop vessels, then lane vessels.
        The one path the collision check, the lidar, the raster and any planner
        take to the traffic, so a lane is seen wherever a loop is."""
        out = [r.rect_at(t) for r in self.traffic]
        for lane in self.lanes:
            out.extend(lane.rects_at(t))
        return out

    def moving(self, t: float):
        """`(rect, (vx, vy))` for every moving vessel at `t`, velocity over ground in
        world axes, in `traffic_rects`' order. The 1-1 object list reads this."""
        out = [(r.rect_at(t), tuple(float(v) for v in r.velocity_at(t))) for r in self.traffic]
        for lane in self.lanes:
            out.extend(lane.moving_at(t))
        return out
