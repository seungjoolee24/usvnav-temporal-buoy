"""Collision-rich, public-route-paired courses for learning buoy avoidance.

Only obstacle layout depends on ``seed``. A route ID identifies public start,
waypoints and shoreline, so changing an obstacle seed cannot leak its position
through waypoint coordinates. All buoy sizes and saved geometry are official.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from usvnav.coursefile import default_terrain_extent, from_dict, to_dict, validate
from usvnav.geometry import Circle
from usvnav.world import BUOY, Course, StaticBody
from training.collect_perception import course_identity


def _route_geometry(route_id):
    if not isinstance(route_id, (int, np.integer)) or not 0 <= route_id < 256:
        raise ValueError("route_id must be an integer in [0, 256)")
    # Many starts and mild turns, all well within the same wide channel. Every
    # leg has 45 m, with slack above the 40 m serialized official minimum.
    x = 50. + 1.2 * (route_id % 16)
    y = 65. + 1.4 * (route_id // 16)
    angle = math.radians((route_id % 3 - 1) * 10.)
    start = (x, y, 0.)
    middle = np.array([x + 45., y])
    goal = middle + 45. * np.array([math.cos(angle), math.sin(angle)])
    return start, np.array([middle, goal])


def make_buoy_course(seed, count=1, *, route_id=0):
    """Make a validated official course, with 0--4 independently placed buoys.

    The first encounter is 12--25 m after the start. Two-buoy courses exercise
    both legs; three/four-buoy courses also require repeated first-leg dodges.
    Most offsets intersect the reference hull corridor, while some pass outside
    it, so seeing a buoy does not always require a turn. Side and distance are
    drawn independently of the public route. The channel leaves ample room on
    both sides and all waypoint arrival discs remain clear.
    """
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(count, (int, np.integer)) or not 0 <= count <= 4:
        raise ValueError("count must be an integer in [0, 4]")
    start, waypoints = _route_geometry(route_id)
    boundary = np.array([[0., 20.], [400., 20.], [400., 130.], [0., 130.]])
    rng = np.random.default_rng(seed)
    # With two obstacles, encounter one on each leg. With three/four, put a
    # second obstacle on each corresponding leg, at least 7 m further along.
    allocations = {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (2, 1), 4: (2, 2)}[int(count)]
    chain = [np.asarray(start[:2]), *waypoints]
    bodies = []
    for leg_index, number in enumerate(allocations):
        direction = (chain[leg_index + 1] - chain[leg_index]) / 45.
        normal = np.array([-direction[1], direction[0]])
        for index in range(number):
            if number == 1:
                along = rng.uniform(12., 25.)
            else:
                along = rng.uniform(12., 20.) if index == 0 else rng.uniform(27., 34.)
            lateral = rng.uniform(-.9, .9) if rng.random() < .8 else rng.choice([-1., 1.]) * rng.uniform(1.7, 2.8)
            center = chain[leg_index] + along * direction + lateral * normal
            bodies.append(StaticBody(BUOY, Circle(float(center[0]), float(center[1]), .3)))
    course = Course(boundary, bodies, waypoints, np.array([5., 3.]), start,
                    terrain_extent=default_terrain_extent(boundary))
    # All checks are applied after official JSON precision/size round-tripping.
    course = from_dict(to_dict(course))
    problems = validate(course)
    if problems:
        raise ValueError(f"Invalid buoy curriculum course: {problems}")
    return course


def write_buoy_curriculum(output, stage=1, *, seed=82, train_layouts=32,
                          val_layouts=8, clean_fraction=.2):
    """Write independent layout splits and an auditable hash manifest.

    Stage 1 has one buoy, stage 2 has two, stage 3 varies three/four. The default
    train pool has 32 buoy courses and 8 clean courses (80/20). Validation uses
    independent obstacle seeds on exactly the same public routes, avoiding a
    test that could be solved just by memorizing each route's obstacle position.
    Validation contains buoy courses; clean retention can be evaluated directly
    on a clean route separately. The ``train``/``val`` manifest rows have the
    same ``file``/``sha256`` convention as the existing RGB curriculum.
    """
    if stage not in (1, 2, 3):
        raise ValueError("stage must be 1, 2 or 3")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(train_layouts, int) or train_layouts < 4 or not isinstance(val_layouts, int) or val_layouts < 1:
        raise ValueError("Need at least four train layouts and one validation layout")
    if not 0. <= clean_fraction <= .5 or not math.isfinite(clean_fraction):
        raise ValueError("clean_fraction must be finite and in [0, .5]")
    clean_count = int(round(train_layouts * clean_fraction / (1. - clean_fraction)))
    route_count = max(1, clean_count) if clean_count else min(4, train_layouts // 2)
    if route_count > 256 or route_count > train_layouts // 2:
        raise ValueError("Need at least two buoy layouts per public route; lower clean_fraction or pool size")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(train=[], val=[], metadata=dict(
        family="randomized-buoy-avoidance-v1", stage=stage, seed=int(seed),
        train_buoy_layouts=train_layouts, train_clean_routes=clean_count,
        train_clean_fraction=clean_count / (train_layouts + clean_count),
        val_buoy_layouts=val_layouts, route_count=route_count,
        split_rule="independent obstacle seeds; paired public routes; disjoint serialized hashes",
        buoy_diameter_m=.6))
    hashes, seeds = set(), set()

    def emit(split, index, count, route_id, layout_seed):
        course = make_buoy_course(layout_seed or 0, count, route_id=route_id)
        identity = course_identity(course)
        if identity in hashes:
            raise ValueError("Duplicate serialized layout across curriculum splits")
        hashes.add(identity)
        name = f"buoy{count}-{split}-r{route_id:02d}-{index:03d}.json"
        (output / name).write_text(json.dumps(to_dict(course), indent=2) + "\n", encoding="utf-8")
        manifest[split].append(dict(file=name, stage=stage, kind="buoy" if count else "clean",
                                    count=count, route_id=route_id, layout_seed=layout_seed,
                                    sha256=identity, validation=[]))

    for split_id, (split, number) in enumerate((("train", train_layouts), ("val", val_layouts))):
        for index in range(number):
            layout_seed = int(np.random.SeedSequence([int(seed), stage, split_id, index]).generate_state(1)[0])
            if layout_seed in seeds:
                raise ValueError("Generated duplicate layout seed across curriculum splits")
            seeds.add(layout_seed)
            count = stage if stage < 3 else 3 + int(np.random.default_rng(layout_seed).integers(2))
            emit(split, index, count, index % route_count, layout_seed)
    for route_id in range(clean_count):
        emit("train", train_layouts + route_id, 0, route_id, None)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def reference_probe(course, *, tick_limit=1200):
    """Check genuine reference-controller physics without RGB or NN rendering."""
    from usvnav import sim
    from training.navigation_env import NavigationEpisode
    from training.rgb_navigation_env import base_command

    episode = NavigationEpisode(course, tick_limit=tick_limit, seed=0)
    while episode.outcome is None:
        public = sim.observation(episode.vessel, course, episode.tick,
                                 episode.wp_index, episode.prev_applied)
        episode.advance(base_command(public))
    return dict(outcome=episode.outcome, ticks=episode.tick, elapsed_s=episode.tick * .1,
                waypoints_reached=episode.wp_index,
                min_clearance_m=float(min(episode.per_tick_clearance)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--seed", type=int, default=82)
    parser.add_argument("--train-layouts", type=int, default=32)
    parser.add_argument("--val-layouts", type=int, default=8)
    args = parser.parse_args()
    manifest = write_buoy_curriculum(args.out, args.stage, seed=args.seed,
                                      train_layouts=args.train_layouts, val_layouts=args.val_layouts)
    print(json.dumps(manifest["metadata"], indent=2))


if __name__ == "__main__":
    main()
