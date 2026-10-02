"""The course rules a participant's tooling needs, without the generator (7-A8).

`coursefile.validate` and the editor check hand-authored courses against rules the
generator honours by construction -- the reach's extent (1-W2, 1-W8), the waypoint chain's
shape (3-K5), the terrain margin (4-V12). Those rules used to live in `generate.py` and
`channel.py` and were imported from there, which made the validator depend on the
generator. Since 2026-09-10 the generator does not ship (7-A8), so the rules the shipped
side needs live here and the generator imports them from here: one definition, on the
side of the line that has to have it.
"""

from __future__ import annotations

import numpy as np

REACH_X, REACH_Y = 400.0, 150.0          # 1-W2, x widened -- see 1-W8
LEG_MIN, LEG_MAX = 40.0, 160.0           # 3-K5 (PoC 2026-09-22: was 120; a U-turn leg back downstream needs the room)
WP_MIN, WP_MAX = 2, 10                   # 3-K5 (PoC 2026-09-22: was 2-4; PoC r13 2026-09-25: 10 -- an out-and-back route has the return's waypoints too)
RADIUS_INTERMEDIATE, RADIUS_GOAL = 5.0, 3.0   # 3-K11


def _terrain_margin() -> float:
    """4-V12's margin, derived from the raster rather than restated as a constant.

    It was 50.0 -- "one sensing range" -- and that was right only while 4-V6 masked the
    raster to a circle of that radius. With the square extent the corners reach
    70.71 m, so a 50 m margin leaves `outside_world` reachable in them, and Track 1's
    palette has no colour for that class on purpose (it raises). Deriving the margin
    from `reach_m` means changing the raster cannot silently reintroduce the gap.
    """
    from .render import TRACK1
    return TRACK1.reach_m


TERRAIN_MARGIN = _terrain_margin()       # 4-V12: 70.71 m under 4-V6's square extent


def chain_conforms(course):
    """3-K5 conformance: 2-4 waypoints, legs of 40-120 m from the start pose onward.

    Checked rather than assumed. The waypoint count follows from the situation chain and
    the leg lengths from the channel's curvature and, for a docking zone, from a lateral
    offset onto the dock face -- so the spec can be violated by geometry that no single
    placement rule owns. Rejecting and redrawing is cheaper than threading the
    constraint back through every builder; the validator applies the same test to a
    hand-authored course.
    """
    k = course.n_waypoints
    if not WP_MIN <= k <= WP_MAX:
        return False, f"{k} waypoints, 3-K5 allows {WP_MIN}-{WP_MAX}"
    chain = [np.asarray(course.start[:2], dtype=float)] + [np.asarray(w, dtype=float)
                                                           for w in course.waypoints]
    for i, (a, b) in enumerate(zip(chain, chain[1:])):
        d = float(np.linalg.norm(b - a))
        if not LEG_MIN - 1e-6 <= d <= LEG_MAX + 1e-6:
            return False, f"leg {i} is {d:.0f} m, 3-K5 allows {LEG_MIN:.0f}-{LEG_MAX:.0f} m"
    return True, "ok"
