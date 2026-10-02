"""Condition 1-1 -- the structured object list (4-V4), and follow-up item 16.

1-1 is the **perception-free upper bound**: what an agent could do if seeing were free.
Everything here follows from that one role, and 4-V4 spells out the four consequences.

*Shape parameters, not a size scalar.* 2-E3 keeps every body's real primitive, so a row
carries the primitive and its dimensions. `extent` is `(length, width)` for a rectangle
and `(diameter, diameter)` for a circle, so nothing has to branch on the primitive to
know how big a body is -- and `primitive` is still there, because a 0.6 m disc and a
0.6 m square differ at the corners.

*Over-ground velocity in the vessel's axes.* Not relative velocity: the statement's
invariant "a static obstacle has velocity zero" only holds over ground, and a relative
figure would make a moored vessel indistinguishable from a moving one. The agent's own
velocity is in the common observation (4-V15), so relative is one subtraction away.

*Occluded objects are listed.* There is no visibility test at all. Anything that would
silently remove information defeats the condition's purpose.

*Fixed length with a validity mask*, capacity derived with headroom and asserted at
generation time. Nearest-first ordering, so the truncation that must never fire would
drop the farthest body rather than an arbitrary one.

**The region is the square, not the 50 m circle** -- follow-up item 16, settled here
because building 1-1 is what the item said would settle it. 4-V6 made 1-2's extent the
whole 100 x 100 m square, which reaches 70.71 m into its corners; leaving 1-1 radial
would put bodies in the raster that the object list does not carry, over 21% of the
frame, and a "perception-free upper bound" that sees *less* than the image condition is
not an upper bound. 1-3 stays radial because a ray cast has no other shape. A body is in
the list when its **footprint intersects the square**, which is the same test the
renderer applies, so 1-1 contains exactly what 1-2 draws plus velocity, identity and the
moored/moving distinction 4-V13 merges.
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import Circle, Rect, wrap
from .render import TRACK1
from .world import BUOY, DOCK, MOORED, MOVING, PIER

#: The object classes a row can carry, in a published order. The bank is absent (2-E6:
#: 4-V1 already discloses the boundary polygon exactly, so a row for it would carry zero
#: information), and so is the ego, which is the observer. Moored and moving vessels are
#: **separate** here even though 4-V13 gives them one raster colour -- that distinction is
#: part of what 1-1 has and 1-2 does not.
CLASSES = (BUOY, PIER, MOORED, MOVING, DOCK)
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}

CIRCLE, RECT = 0, 1

#: 4-V4's capacity. Derived rather than chosen: `tools/x2.py` and
#: `tests/test_objects.py` measure the densest course the generator can emit and assert
#: the headroom. The generator asserts it too, so a course that would overflow is a
#: generation-time failure rather than a silent truncation during scoring.
CAPACITY = 96


class CapacityExceeded(Exception):
    """4-V4's truncation fallback, which must never fire. Raised instead of truncating.

    Truncation is documented as a fallback, so the code has to have one -- but a silent
    one would remove information from the condition whose whole role is to have all of
    it. Raising makes the generator's assertion the thing that catches it, at generation
    time, where a course can still be redrawn.
    """


def _extent(shape):
    if isinstance(shape, Circle):
        d = 2.0 * shape.r
        return CIRCLE, d, d, 0.0
    return RECT, shape.length, shape.width, shape.heading


def _intersects_square(xb: float, yb: float, half_l: float, half_w: float,
                       heading_b: float, half_span: float, primitive: int) -> bool:
    """Does a body's footprint reach into the observation square?

    Separating-axis on the two frames' axes for a rectangle; for a circle the exact test
    is the distance to the square. Conservative in the rectangle case only in the corner
    of the corner, which costs at most one body that the raster also does not draw.
    """
    if primitive == CIRCLE:
        dx = max(abs(xb) - half_span, 0.0)
        dy = max(abs(yb) - half_span, 0.0)
        return math.hypot(dx, dy) <= half_l
    c, s = abs(math.cos(heading_b)), abs(math.sin(heading_b))
    reach_x = half_l * c + half_w * s
    reach_y = half_l * s + half_w * c
    return abs(xb) <= half_span + reach_x and abs(yb) <= half_span + reach_y


def object_list(course, vessel, t: float, *, capacity: int = CAPACITY,
                half_span: float | None = None):
    """4-V4's list for one tick. A dict of named arrays, as 4-V15 has the rest of it."""
    if half_span is None:
        half_span = 0.5 * TRACK1.size_px * TRACK1.m_per_px      # 50 m, the square's half

    cpsi, spsi = math.cos(vessel.psi), math.sin(vessel.psi)

    rows = []
    for shape, cls, vel in _bodies(course, t):
        primitive, length, width, heading_w = _extent(shape)
        dx, dy = shape.x - vessel.x, shape.y - vessel.y
        # World -> body axes: +x through the bow, +y to port (1-W3).
        xb = dx * cpsi + dy * spsi
        yb = -dx * spsi + dy * cpsi
        heading_b = wrap(heading_w - vessel.psi)
        if not _intersects_square(xb, yb, length / 2.0, width / 2.0, heading_b,
                                  half_span, primitive):
            continue
        vxb = vel[0] * cpsi + vel[1] * spsi
        vyb = -vel[0] * spsi + vel[1] * cpsi
        rows.append((xb * xb + yb * yb, CLASS_ID[cls], primitive, xb, yb, heading_b,
                     vxb, vyb, length, width))

    if len(rows) > capacity:
        raise CapacityExceeded(
            f"{len(rows)} bodies inside the observation square, capacity {capacity} "
            f"(4-V4). The generator asserts this bound, so reaching it here means a "
            f"course was built by something that does not.")

    rows.sort(key=lambda r: r[0])                # nearest first (4-V4)
    out = {
        "valid": np.zeros(capacity, dtype=bool),
        "cls": np.zeros(capacity, dtype=np.int8),
        "primitive": np.zeros(capacity, dtype=np.int8),
        "pos": np.zeros((capacity, 2), dtype=np.float32),
        "heading": np.zeros(capacity, dtype=np.float32),
        "vel": np.zeros((capacity, 2), dtype=np.float32),
        "extent": np.zeros((capacity, 2), dtype=np.float32),
    }
    for i, (_, cls, prim, xb, yb, hb, vxb, vyb, L, W) in enumerate(rows):
        out["valid"][i] = True
        out["cls"][i] = cls
        out["primitive"][i] = prim
        out["pos"][i] = (xb, yb)
        out["heading"][i] = hb
        out["vel"][i] = (vxb, vyb)
        out["extent"][i] = (L, W)
    return out


def _bodies(course, t: float):
    """Every body, static and moving, with its over-ground world velocity. Moving vessels
    -- loop and lane -- come through `Course.moving`, so a lane vessel is listed exactly
    while it exists."""
    for body in course.bodies:
        yield body.shape, body.cls, (0.0, 0.0)
    for rect, vel in course.moving(t):
        yield rect, MOVING, tuple(vel)


def peak_occupancy(course, samples: int = 240) -> int:
    """The most bodies any observation of this course can contain (4-V4's bound).

    Swept over the whole navigable area rather than over a driven path, because the
    capacity has to hold wherever the vessel goes -- including somewhere no agent in the
    calibration set happened to visit. A loop-only course is sampled at four phases of
    its slowest lap, as before lanes existed; once a lane is present, the resolution
    follows the ratio of the longest cycle to the shortest instead (see below), since a
    course's densest moment can be a phase rather than a place.
    """
    from .plant import Vessel

    b = np.asarray(course.boundary, dtype=float)
    xs = np.linspace(b[:, 0].min(), b[:, 0].max(), int(math.sqrt(samples)) + 1)
    ys = np.linspace(b[:, 1].min(), b[:, 1].max(), int(math.sqrt(samples)) + 1)
    periods = [r.perimeter / max(r.speed, 1e-6) for r in course.traffic]
    lane_periods = [lane.traversal_s + lane.headway() for lane in course.lanes]
    period = max(periods + lane_periods, default=1.0)
    if lane_periods:
        # Two schedules cannot alias to the same few instants: the resolution follows the
        # ratio of the longest cycle to the shortest, capped so the generator's per-candidate
        # assertion stays affordable.
        shortest = min(periods + lane_periods)
        steps = min(32, max(8, int(math.ceil(8 * period / shortest))))
    else:
        steps = 4          # the pre-lane sweep: loops only, four phases of the slowest lap
    worst = 0
    for x in xs:
        for y in ys:
            for t in np.linspace(0.0, period, steps, endpoint=False):
                got = object_list(course, Vessel(float(x), float(y), 0.0), float(t),
                                  capacity=10 ** 6)
                worst = max(worst, int(got["valid"].sum()))
    return worst
