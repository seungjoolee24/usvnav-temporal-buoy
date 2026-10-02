"""The course file format, and the checks that say whether a course is usable.

Follow-up item 6 makes this format the contract between four things: the generator emits
it, the editor authors it, the scorer consumes it, and a participant reads it. So it has
two hard properties.

**No hidden fields.** No difficulty label, no seed, no generator parameters, no situation
list. A participant can read any course file they hold, so anything in here is disclosed
whether or not it was meant to be. `Course.situations` exists in memory for the
generator's own bookkeeping and is deliberately **not** written.

**Everything needed to run an episode, and nothing derived.** The terrain extent is
optional, because 4-V12 derives it from the navigable extent; supplying it is for the
case where an author wants terrain somewhere else. Nothing else is optional.

**A scored course is the file, not the object the generator returned.** Coordinates are
written to four decimal places, which is 0.1 mm, so a round trip is not bit-exact.
Measured: every outcome and tick count is identical in all four conditions, the
clearance statistic moves by up to 6e-5 m, and distance by 3e-10 m without the 1-4
disturbance and 2e-5 m with it -- the drift amplifies a rounding difference about a
hundredfold. Full float precision in a file a person hand-edits is the wrong trade;
loading the file everywhere is the right one, and it makes the difference not exist
rather than merely be small. The scorer and every practice run should take the same path.

Units and conventions are 1-W3's throughout: metres, radians, angles wrapped to
(-pi, pi], `+x` along the reach, `+y` to port of it. The format uses radians because
1-W3 says every machine interface does; the editor converts for humans, which is the
right place for that -- two representations in the file would eventually disagree.

`validate` is the other half. A hand-authored course can be broken in ways the generator
cannot produce, so every rule the generator honours by construction is checked here
explicitly, each one naming the decision it comes from. The build has already shown that
this is not theoretical: 3-K9's gate, 2-E11's placement rule, 3-K5's leg lengths and
2-E7's traffic separation were each violated by generated courses at some point.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from .geometry import Circle, Rect, wrap
from .world import (BUOY, DOCK, MOORED, MOVING, PIER, SIZE_RANGES, Course, Lane, StaticBody,
                    TrafficRoute)

FORMAT = "usvnav-course/2"
#: Tags a course file may carry when read. `/1` is every file written before lanes existed
#: (2026-09-21) and has no `lanes` section; the third tag is what every course was written
#: with before the package was renamed on 2026-09-15 -- the frozen hidden sets still carry
#: it, and frozen means not rewritten. Nothing writes either any more.
READ_FORMATS = (FORMAT, "usvnav-course/1", "mallard-course/1")

#: Which primitive each class carries (2-E4). A file disagreeing with this is rejected
#: rather than coerced: 2-E3's whole point is that every body keeps its real primitive.
PRIMITIVE_OF = {BUOY: "circle", PIER: "rect", MOORED: "rect", MOVING: "rect", DOCK: "rect"}

BODY_CLASSES = tuple(PRIMITIVE_OF)
STATIC_BODY_CLASSES = (BUOY, PIER, MOORED, DOCK)


class CourseFileError(ValueError):
    """The file is not a course: wrong format, missing field, impossible primitive."""


# --------------------------------------------------------------------------- write

def to_dict(course: Course) -> dict:
    """The course as plain JSON-able data. Emits no hidden field (follow-up item 6)."""
    out = {
        "format": FORMAT,
        "world": {"boundary": [[round(float(x), 4), round(float(y), 4)]
                               for x, y in course.boundary]},
        "start": {"x": round(float(course.start[0]), 4),
                  "y": round(float(course.start[1]), 4),
                  "heading_rad": round(float(wrap(course.start[2])), 6)},
        "waypoints": [{"x": round(float(w[0]), 4), "y": round(float(w[1]), 4),
                       "arrival_radius_m": round(float(r), 4)}
                      for w, r in zip(course.waypoints, course.arrival_radii)],
        "bodies": [_body_to_dict(b) for b in course.bodies],
        "traffic": [_route_to_dict(r) for r in course.traffic],
        "lanes": [_lane_to_dict(l) for l in course.lanes],
    }
    if course.terrain_extent is not None:
        out["world"]["terrain_extent"] = [round(float(v), 4) for v in course.terrain_extent]
    return out


def _body_to_dict(body: StaticBody) -> dict:
    s = body.shape
    if isinstance(s, Circle):
        return {"class": body.cls, "primitive": "circle",
                "x": round(s.x, 4), "y": round(s.y, 4),
                "diameter_m": round(2.0 * s.r, 4)}
    return {"class": body.cls, "primitive": "rect",
            "x": round(s.x, 4), "y": round(s.y, 4),
            "length_m": round(s.length, 4), "width_m": round(s.width, 4),
            "heading_rad": round(float(wrap(s.heading)), 6)}


def _route_to_dict(route: TrafficRoute) -> dict:
    return {"class": route.cls, "length_m": round(route.length, 4),
            "width_m": round(route.width, 4), "speed_mps": round(route.speed, 4),
            "phase_m": round(float(route.phase), 4),
            "route": [[round(float(x), 4), round(float(y), 4)] for x, y in route.points]}


def _lane_to_dict(lane: Lane) -> dict:
    out = {"class": lane.cls, "closed": bool(lane.closed),
           "route": [[round(float(x), 4), round(float(y), 4)] for x, y in lane.points],
           "speeds_mps": [round(float(v), 4) for v in lane.speeds],
           "offset_s": round(float(lane.offset_s), 4),
           "vessels": [{"length_m": round(L, 4), "width_m": round(W, 4)} for L, W in lane.vessels]}
    if not lane.closed:
        out["headway_s"] = round(float(lane.headway_s), 4)
    return out


def save(course: Course, path) -> str:
    path = pathlib.Path(path)
    path.write_text(json.dumps(to_dict(course), indent=1) + "\n")
    return str(path)


# --------------------------------------------------------------------------- read

def from_dict(data: dict) -> Course:
    """Build a `Course`. Raises `CourseFileError` on anything structurally wrong.

    Structural errors are raised; *design* violations are reported by `validate`. The
    split matters for the editor: a half-finished course should still load and draw so
    the author can see what is wrong with it, and only a file that cannot become a
    course at all should fail to open.
    """
    if not isinstance(data, dict):
        raise CourseFileError(f"expected an object, got {type(data).__name__}")
    fmt = data.get("format")
    if fmt not in READ_FORMATS:
        raise CourseFileError(f"format is {fmt!r}, expected {FORMAT!r}")

    world = _need(data, "world", dict)
    boundary = _xy_points(_need(world, "boundary", list), "world.boundary", 3)

    st = _need(data, "start", dict)
    start = (float(_need(st, "x", (int, float))), float(_need(st, "y", (int, float))),
             float(wrap(float(_need(st, "heading_rad", (int, float))))))

    wps = _need(data, "waypoints", list)
    if not wps:
        raise CourseFileError("waypoints is empty; an episode needs a goal")
    waypoints = np.array([[float(_need(w, "x", (int, float))),
                           float(_need(w, "y", (int, float)))] for w in wps], dtype=float)
    radii = np.array([float(_need(w, "arrival_radius_m", (int, float))) for w in wps],
                     dtype=float)

    bodies = [_body_from_dict(b, i) for i, b in enumerate(data.get("bodies", []))]
    traffic = [_route_from_dict(r, i) for i, r in enumerate(data.get("traffic", []))]
    lanes = [_lane_from_dict(l, i) for i, l in enumerate(data.get("lanes", []))]

    extent = world.get("terrain_extent")
    if extent is None:
        extent = default_terrain_extent(boundary)
    else:
        if len(extent) != 4:
            raise CourseFileError("world.terrain_extent must be [xmin, ymin, xmax, ymax]")
        extent = tuple(float(v) for v in extent)

    return Course(boundary, bodies, waypoints, radii, start, traffic, lanes,
                  terrain_extent=extent)


def _body_from_dict(b: dict, i: int) -> StaticBody:
    cls = _need(b, "class", str, f"bodies[{i}]")
    if cls not in STATIC_BODY_CLASSES:
        raise CourseFileError(
            f"bodies[{i}].class is {cls!r}; a static body is one of "
            f"{list(STATIC_BODY_CLASSES)} (2-E4). A moving vessel goes in `traffic`.")
    want = PRIMITIVE_OF[cls]
    got = b.get("primitive", want)
    if got != want:
        raise CourseFileError(
            f"bodies[{i}] is a {cls} so its primitive is {want!r}, not {got!r} (2-E3/2-E4)")
    x = float(_need(b, "x", (int, float), f"bodies[{i}]"))
    y = float(_need(b, "y", (int, float), f"bodies[{i}]"))
    if want == "circle":
        d = float(b.get("diameter_m", SIZE_RANGES[BUOY][0]))
        return StaticBody(cls, Circle(x, y, d / 2.0))
    return StaticBody(cls, Rect(
        x, y,
        float(_need(b, "length_m", (int, float), f"bodies[{i}]")),
        float(_need(b, "width_m", (int, float), f"bodies[{i}]")),
        float(wrap(float(b.get("heading_rad", 0.0))))))


def _route_from_dict(r: dict, i: int) -> TrafficRoute:
    cls = r.get("class", MOVING)
    if cls != MOVING:
        raise CourseFileError(f"traffic[{i}].class is {cls!r}; only {MOVING!r} moves (2-E4)")
    pts = _xy_points(_need(r, "route", list, f"traffic[{i}]"), f"traffic[{i}].route (a closed loop, 2-E7)", 3)
    route = TrafficRoute(MOVING,
                         float(_need(r, "length_m", (int, float), f"traffic[{i}]")),
                         float(_need(r, "width_m", (int, float), f"traffic[{i}]")),
                         pts,
                         float(_need(r, "speed_mps", (int, float), f"traffic[{i}]")))
    route.phase = float(r.get("phase_m", 0.0))
    return route


def _lane_from_dict(d: dict, i: int) -> Lane:
    where = f"lanes[{i}]"
    cls = d.get("class", MOVING)
    if cls != MOVING:
        raise CourseFileError(f"{where}.class is {cls!r}; only {MOVING!r} moves (2-E4)")
    closed = d.get("closed", False)
    if not isinstance(closed, bool):
        raise CourseFileError(f"{where}.closed must be true or false")
    pts = _xy_points(_need(d, "route", list, where), f"{where}.route", 2)
    speeds = _need(d, "speeds_mps", list, where)
    if len(speeds) != len(pts):
        raise CourseFileError(f"{where}.speeds_mps must have one entry per route point "
                              f"({len(speeds)} for {len(pts)} points)")
    vessels_d = _need(d, "vessels", list, where)
    if not vessels_d:
        raise CourseFileError(f"{where}.vessels is empty; a lane needs at least one vessel size")
    vessels = [(float(_need(v, "length_m", (int, float), f"{where}.vessels[{j}]")),
                float(_need(v, "width_m", (int, float), f"{where}.vessels[{j}]")))
               for j, v in enumerate(vessels_d)]
    headway = d.get("headway_s")
    if closed and headway is not None:
        raise CourseFileError(f"{where} is closed and carries headway_s; a closed lane's headway is "
                              f"its lap time over its vessel count")
    if not closed and headway is None:
        raise CourseFileError(f"{where} is open and has no headway_s")
    try:
        return Lane(closed=closed, points=pts, speeds=np.asarray(speeds, dtype=float),
                    vessels=vessels, headway_s=None if closed else float(headway),
                    offset_s=float(d.get("offset_s", 0.0)))
    except ValueError as e:
        raise CourseFileError(f"{where}: {e}") from None


def load(path) -> Course:
    return from_dict(json.loads(pathlib.Path(path).read_text()))


def _need(d: dict, key: str, types, where: str = ""):
    if key not in d:
        raise CourseFileError(f"{where or 'course'} is missing {key!r}")
    v = d[key]
    if not isinstance(v, types):
        raise CourseFileError(f"{where or 'course'}.{key} should be "
                              f"{getattr(types, '__name__', types)}, got {type(v).__name__}")
    return v


def _xy_points(value, where: str, at_least: int) -> np.ndarray:
    """`value` as an `(N, 2)` float array with `N >= at_least`, or a `CourseFileError` naming
    `where` -- a ragged list, a string, a huge integer (`OverflowError`, from a JSON literal
    too big for a float) or a NaN inside is a file error, not numpy's or json's. Shape, count
    and finiteness are three separate conditions, each with its own message."""
    try:
        pts = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise CourseFileError(f"{where} must be a list of [x, y] number pairs") from None
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise CourseFileError(f"{where} must be a list of [x, y] number pairs")
    if len(pts) < at_least:
        raise CourseFileError(f"{where} must have at least {at_least} [x, y] pairs ({len(pts)} given)")
    if not np.all(np.isfinite(pts)):
        raise CourseFileError(f"{where} contains a non-finite coordinate (NaN or infinity)")
    return pts


def default_terrain_extent(boundary) -> tuple[float, float, float, float]:
    """4-V12's margin around a navigable polygon, when a file does not supply one."""
    from .rules import TERRAIN_MARGIN
    b = np.asarray(boundary, dtype=float)
    return (float(b[:, 0].min()) - TERRAIN_MARGIN, float(b[:, 1].min()) - TERRAIN_MARGIN,
            float(b[:, 0].max()) + TERRAIN_MARGIN, float(b[:, 1].max()) + TERRAIN_MARGIN)


# --------------------------------------------------------------------------- validate

def validate(course: Course, *, strict_sizes: bool = True) -> list[dict]:
    """Every design rule a hand-authored course can break, each naming its decision.

    Returns a list of `{"severity", "rule", "message"}`. `error` means the course cannot
    be run or scored as it stands; `warning` means it will run but is outside what the
    generator would emit, which an author may be doing on purpose.

    The generator honours all of these by construction, so this function is for the
    editor and for anyone hand-authoring. It is also the organizer's repair tool: item 6
    wants a generated course opened in the editor and *fixed* rather than discarded.
    """
    from .collide import CourseIndex, HULL_RADIUS
    from .freespace import FreeSpace
    from .rules import LEG_MAX, LEG_MIN, REACH_X, REACH_Y, chain_conforms
    from .geometry import _segments_intersect, body_distance
    from .plant import HULL_LENGTH, HULL_WIDTH, W_MAX
    from .sim import SENSING_RANGE
    from .world import TRAFFIC_SPEED_MAX, TRAFFIC_SPEED_MIN

    out: list[dict] = []

    def bad(rule, message, severity="error"):
        out.append({"severity": severity, "rule": rule, "message": message})

    b = np.asarray(course.boundary, dtype=float)

    # --- the navigable polygon itself (2-E6, 1-W2) ---
    if _polygon_self_intersects(b, _segments_intersect):
        bad("2-E6", "the navigable boundary crosses itself, so inside/outside is not "
                    "defined; the shoreline must be a simple polygon")
    if b[:, 0].min() < 0.0 or b[:, 1].min() < 0.0 or b[:, 0].max() > REACH_X or b[:, 1].max() > REACH_Y:
        bad("1-W2", f"the navigable area reaches "
                    f"x [{b[:, 0].min():.1f}, {b[:, 0].max():.1f}], "
                    f"y [{b[:, 1].min():.1f}, {b[:, 1].max():.1f}]; "
                    f"1-W2 places it inside x [0, {REACH_X:.0f}], y [0, {REACH_Y:.0f}]",
            "warning")

    # --- 4-V12: terrain must cover everything a raster can reach ---
    from .render import TRACK1
    if course.terrain_extent is not None:
        x0, y0, x1, y1 = course.terrain_extent
        need = TRACK1.reach_m
        short = min(b[:, 0].min() - x0, b[:, 1].min() - y0, x1 - b[:, 0].max(),
                    y1 - b[:, 1].max())
        # A centimetre of slack: the file stores four decimal places, so a margin that
        # is exactly the reach round-trips a hair short and this fired on every course
        # the generator emitted.
        if short < need - 0.01:
            bad("4-V12", f"generated terrain reaches only {short:.1f} m beyond the "
                         f"navigable extent; a {TRACK1.size_px}-pixel raster reaches "
                         f"{need:.2f} m into its corners, so `outside_world` would appear "
                         f"and Track 1 has no colour for it")

    # --- 3-K14: the start pose, tested exactly ---
    hull = Rect(course.start[0], course.start[1], HULL_LENGTH, HULL_WIDTH, course.start[2])
    index = CourseIndex(course)
    _, hit, inside = index.tick(hull, 0.0, 10.0)
    if hit is not None:
        bad("3-K14", f"the start pose is in contact with a {hit}; the vessel is placed "
                     f"exactly there, so it must be clear at t = 0")
    if not inside:
        bad("3-K14", "the start pose is not inside the navigable boundary")

    # --- 3-K5: the waypoint chain ---
    ok, why = chain_conforms(course)
    if not ok:
        bad("3-K5", why)
    for i, r in enumerate(course.arrival_radii):
        last = i == len(course.arrival_radii) - 1
        want = 3.0 if last else 5.0
        if abs(float(r) - want) > 1e-6:
            bad("3-K11", f"waypoint {i} has an arrival radius of {float(r):.1f} m; "
                         f"3-K11 uses {want:.0f} m for "
                         f"{'the goal' if last else 'an intermediate waypoint'}",
                "warning")

    # --- 2-E11: nothing inside the route's fixed points ---
    discs = [(np.asarray(course.start[:2], dtype=float), HULL_RADIUS)]
    discs += [(np.asarray(w, dtype=float), float(r) + HULL_RADIUS)
              for w, r in zip(course.waypoints, course.arrival_radii)]
    for j, body in enumerate(course.bodies):
        for k, (p, rad) in enumerate(discs):
            probe = Circle(float(p[0]), float(p[1]), rad)
            if body_distance(Rect(probe.x, probe.y, 1e-9, 1e-9, 0.0), body.shape) < rad:
                what = "the start pose" if k == 0 else f"waypoint {k - 1}'s arrival disc"
                bad("2-E11", f"bodies[{j}] ({body.cls}) enters {what}; inside that disc "
                             f"the hull cannot meet the arrival condition at any heading")
                break

    # --- 2-E5: size ranges, per class ---
    if strict_sizes:
        for j, body in enumerate(course.bodies):
            for msg in _size_problems(body, j):
                bad("2-E5", msg, "warning")

    # --- 3-K9: ordered connectivity through eroded free space ---
    fs = FreeSpace(course)
    chain = [course.start[:2]] + [tuple(w) for w in course.waypoints]
    snaps = [0.0] + [float(r) for r in course.arrival_radii]
    for i, (p, q) in enumerate(zip(chain, chain[1:])):
        if not fs.connected(p, q, HULL_WIDTH / 2.0, snaps[i], snaps[i + 1]):
            bad("3-K9", f"no route through static geometry from leg {i}'s start to its "
                        f"end with the hull half-width of clearance")

    # --- 2-E7: traffic ---
    start_xy = tuple(float(v) for v in course.start[:2])
    for j, route in enumerate(course.traffic):
        if not TRAFFIC_SPEED_MIN - 1e-9 <= route.speed <= TRAFFIC_SPEED_MAX + 1e-9:
            bad("2-E2", f"traffic[{j}] moves at {route.speed:.2f} m/s; 2-E2 allows "
                        f"{TRAFFIC_SPEED_MIN}-{TRAFFIC_SPEED_MAX}")
        x, y, _ = route.pose_at(0.0)
        d0 = math.dist((x, y), start_xy)
        if d0 < SENSING_RANGE:
            bad("2-E7", f"traffic[{j}] starts {d0:.1f} m from the ego start pose; 2-E7 "
                        f"requires at least one sensing range ({SENSING_RANGE:.0f} m), so "
                        f"no episode opens with a vessel already in view")
        for msg in _route_problems(route, course, j, index):
            bad("2-E7", msg)

    # --- lanes (spec 2026-09-21 §6, rules L1-L10; decision ids 2-E14/2-E15 proposed) ---
    if course.lanes:
        from .lanerules import lane_problems
        for rule, message, severity in lane_problems(course, fs=fs):
            bad(rule, message, severity)

    return out


def _size_problems(body: StaticBody, j: int):
    rng = SIZE_RANGES.get(body.cls)
    if rng is None:
        return
    s = body.shape
    if isinstance(s, Circle):
        lo, hi = rng
        d = 2.0 * s.r
        if not lo - 1e-6 <= d <= hi + 1e-6:
            yield (f"bodies[{j}] ({body.cls}) is {d:.2f} m across; 2-E5 fixes it at "
                   f"{lo:.1f} m")
        return
    (l_lo, l_hi), (w_lo, w_hi) = rng
    if not l_lo - 1e-6 <= s.length <= l_hi + 1e-6:
        yield (f"bodies[{j}] ({body.cls}) is {s.length:.1f} m long; 2-E5 allows "
               f"{l_lo:.0f}-{l_hi:.0f} m")
    if not w_lo - 1e-6 <= s.width <= w_hi + 1e-6:
        yield (f"bodies[{j}] ({body.cls}) is {s.width:.1f} m wide; 2-E5 allows "
               f"{w_lo:.0f}-{w_hi:.0f} m")


def _route_problems(route: TrafficRoute, course: Course, j: int, index):
    """2-E7's geometric requirements on a traffic loop."""
    from .geometry import body_distance

    # Never crosses a static body. Sample by arc length, not by vertex: a step shorter than the
    # vessel is long makes consecutive rectangles overlap, so the samples sweep a continuous band
    # with no gap a body could hide in between (a vertex stride missed mid-segment collisions).
    step = max(0.25, 0.5 * route.length)
    for k in range(max(1, math.ceil(route.perimeter / step))):
        rect = route.rect_at_arc(k * step)
        for body in course.bodies:
            if body_distance(rect, body.shape) <= 0.0:
                yield (f"traffic[{j}]'s route passes through bodies "
                       f"[{course.bodies.index(body)}] ({body.cls}); 2-E7 keeps routes "
                       f"off static bodies")
                return
    # Corners filleted to at least twice the vessel's length, so heading is continuous.
    p = np.asarray(route.points, dtype=float)
    v = np.roll(p, -1, axis=0) - p
    n = np.linalg.norm(v, axis=1, keepdims=True)
    u = v / np.maximum(n, 1e-9)
    cosang = np.clip((u * np.roll(u, -1, axis=0)).sum(axis=1), -1.0, 1.0)
    turn = np.degrees(np.arccos(cosang))
    if turn.max() > 60.0:
        yield (f"traffic[{j}]'s route turns {turn.max():.0f} deg at one vertex; 2-E7 "
               f"fillets corners at a radius of at least twice the vessel's length so "
               f"heading and velocity are continuous")


def _polygon_self_intersects(poly, segments_intersect) -> bool:
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        for k in range(i + 2, n):
            if i == 0 and k == n - 1:
                continue
            c, d = poly[k], poly[(k + 1) % n]
            if segments_intersect(a, b, c, d):
                return True
    return False


def _polygon_area(poly) -> float:
    """Signed area; positive counter-clockwise. Kept for the editor, which uses it to
    label a polygon's winding. Not validated: `point_in_polygon` is even-odd and so is
    SVG's fill rule, and no decision fixes a winding, so warning about it would be
    inventing a convention.
    """
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
