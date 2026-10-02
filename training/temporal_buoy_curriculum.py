"""Reviewable static-buoy curriculum: paired public routes, 2/4/6 buoys.

All serialized courses use original geometry, sizes and official validation.
Writing a draft never starts RL. Public route IDs are independent of layout
seeds, and mirrored obstacle layouts retain identical public route information.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from usvnav.coursefile import default_terrain_extent, from_dict, to_dict, validate
from usvnav.geometry import Circle
from usvnav.world import BUOY, Course, StaticBody
from training.collect_perception import course_identity


LEVELS = {0: ('clean', 0, 40.), 1: ('two-buoys', 2, 40.),
          2: ('four-buoys', 4, 30.), 3: ('six-buoys', 6, 24.)}


def make_temporal_buoy_course(seed, level, *, route_id=0, mirror=False):
    if level not in LEVELS or not 0 <= route_id < 256 or seed < 0:
        raise ValueError('Need level0..3, route_id0..255 and nonnegative seed')
    _, count, width = LEVELS[level]
    rng = np.random.default_rng(seed)
    y = 75.
    x = 50. + .8 * (route_id % 8)
    turn = math.radians((route_id % 3 - 1) * 5.)
    waypoints = np.array([[x + 45., y], [x + 45. + 45. * math.cos(turn), y + 45. * math.sin(turn)]])
    start = (x, y, math.radians((route_id % 5 - 2) * 4.))
    boundary = np.array([[0., y - width / 2], [230., y - width / 2],
                         [230., y + width / 2], [0., y + width / 2]])
    direction = np.array([math.cos(turn), math.sin(turn)])
    normal = np.array([-direction[1], direction[0]])
    centers = []
    side = 1. if rng.integers(2) else -1.
    if level == 1:
        for leg in range(2):
            base = np.array(start[:2]) if leg == 0 else waypoints[0]
            along = rng.uniform(16., 26.)
            lateral = rng.uniform(-.65, .65)
            d, n = (np.array([1., 0.]), np.array([0., 1.])) if leg == 0 else (direction, normal)
            centers.append(base + along * d + lateral * n)
    elif level == 2:
        # One centerline blocker plus one on a possible bypass, repeated on leg2.
        for leg in range(2):
            base = np.array(start[:2]) if leg == 0 else waypoints[0]
            d, n = (np.array([1., 0.]), np.array([0., 1.])) if leg == 0 else (direction, normal)
            along, offset = rng.uniform(17., 25.), rng.uniform(-.25, .25)
            for lateral in (offset, offset + (side if leg == 0 else -side) * rng.uniform(3.2, 4.3)):
                centers.append(base + along * d + lateral * n)
    elif level == 3:
        # Two alternating clusters before WP1: a short zigzag, still leaving
        # a broad bypass around each cluster. Banks limit unlimited detours.
        for group, along in enumerate((rng.uniform(16., 18.), rng.uniform(30., 32.))):
            sign, offset = side * (1. if group == 0 else -1.), rng.uniform(-.2, .2)
            for lateral in (0., sign * 3.25, sign * 6.5):
                centers.append(np.array(start[:2]) + [along, lateral + offset])
    if mirror:
        # Reflect obstacle offsets about their public leg, not the route itself.
        reflected = []
        for index, center in enumerate(centers):
            leg = index if level == 1 else index // 2 if level == 2 else 0
            base = np.array(start[:2]) if leg == 0 else waypoints[0]
            d = np.array([1., 0.]) if leg == 0 else direction
            delta = center - base
            reflected.append(base + 2. * np.dot(delta, d) * d - delta)
        centers = reflected
    bodies = [StaticBody(BUOY, Circle(float(point[0]), float(point[1]), .3)) for point in centers]
    assert len(bodies) == count
    course = Course(boundary, bodies, waypoints, np.array([5., 3.]), start,
                    terrain_extent=default_terrain_extent(boundary))
    serialized = from_dict(to_dict(course))
    problems = validate(serialized)
    if problems:
        raise ValueError(f'Invalid temporal curriculum course: {problems}')
    return serialized


def write_temporal_buoy_draft(output, *, seed=20261002):
    """60train(12clean+18/18/12),30val,60test +6review examples, all disjoint."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(train=[], val=[], test=[], review=[], metadata=dict(
        family='temporal-static-buoys-v2', approval_status='pending_user_confirmation',
        training_started=False, seed=seed, condition='1-2', history_frames=4,
        image_size=[200, 200], sensor_extent_m=[100., 100.], decision_dt=.5,
        history_span_s=1.5, physics_dt=.1, cutoff_ticks=1800,
        buoy_diameter_m=.6, hull_size_m=[3., 2.],
        objects=['static buoy', 'water', 'bank', 'ego vessel'],
        route_legs_m=[45., 45.], arrival_radii_m=[5., 3.],
        levels={str(key): dict(name=name, buoys=count, channel_width_m=width)
                for key, (name, count, width) in LEVELS.items()},
        sampling_percent={
            '1': {'clean': 20, '1': 80},
            '2': {'clean': 20, '1': 25, '2': 55},
            '3': {'clean': 20, '1': 15, '2': 20, '3': 45}},
        proposed_promotion={'validation_layouts_per_level': 10, 'min_goal_fraction': .9,
                            'target_full_hull_clearance_m': .8,
                            'checks': ['no prior-level regression', 'changed-layout and hidden-buoy probes']},
        split_rule='Independent seeds/hashes; paired public routes; review examples excluded from training/test',
        supervision='Rendered buoy masks for shared-CNN auxiliary loss only; no GT actor input'))
    hashes = set()

    def emit(split, level, index, layout_seed, route_id, mirror=False):
        course = make_temporal_buoy_course(layout_seed, level, route_id=route_id, mirror=mirror)
        digest = course_identity(course)
        if digest in hashes:
            raise ValueError('Duplicate course across splits')
        hashes.add(digest)
        name = f'{split}-l{level}-{index:03d}.json'
        (output / name).write_text(json.dumps(to_dict(course), indent=2) + '\n', encoding='utf-8')
        manifest[split].append(dict(file=name, level=level, count=LEVELS[level][1],
                                    kind='buoy' if level else 'clean', route_id=route_id,
                                    layout_seed=layout_seed, mirror=mirror, sha256=digest, validation=[]))

    for split_id, split in enumerate(('train', 'val', 'test')):
        counts = {0: 12, 1: 18, 2: 18, 3: 12} if split == 'train' else {1: 10, 2: 10, 3: 10} if split == 'val' else {1: 20, 2: 20, 3: 20}
        index = 0
        for level, number in counts.items():
            for number_index in range(number):
                layout_seed = int(np.random.SeedSequence([seed, split_id, level, number_index]).generate_state(1)[0])
                # Clean route geometry must also be disjoint across splits.
                route_id = number_index % 8 if level else number_index
                emit(split, level, index, layout_seed, route_id, bool(number_index % 2))
                index += 1
    for level in (1, 2, 3):
        for mirrored in (False, True):
            emit('review', level, len(manifest['review']), seed + 500 + level, 1, mirrored)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest
