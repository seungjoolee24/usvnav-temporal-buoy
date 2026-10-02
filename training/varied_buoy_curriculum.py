"""Waypoint and obstacle variations for continuing the temporal buoy policy.

Public routes and private obstacle layouts use independent seeds. Mirrored
pairs retain every public coordinate and change only buoy locations. Courses
use the original simulator geometry, arrival discs and collision checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from training.collect_perception import course_identity
from usvnav.coursefile import default_terrain_extent, from_dict, load, to_dict, validate
from usvnav.geometry import Circle
from usvnav.world import BUOY, Course, StaticBody


WIDTHS = {0: 40., 1: 40., 2: 34.}


def make_varied_buoy_course(route_seed, layout_seed, level, *, mirror=False):
    if level not in WIDTHS or route_seed < 0 or layout_seed < 0:
        raise ValueError('Need level 0..2 and nonnegative independent seeds')
    route_rng = np.random.default_rng(route_seed)
    layout_rng = np.random.default_rng(layout_seed)
    width = WIDTHS[level]
    boundary = np.array([[0., 75.-width/2], [300., 75.-width/2],
                         [300., 75.+width/2], [0., 75.+width/2]])
    start_xy = np.array([route_rng.uniform(45., 65.), 75.+route_rng.uniform(-4., 4.)])
    number = 3 if route_seed % 3 == 0 else 2
    amplitude = min(10., width/2 - 8.5)
    waypoints, x = [], start_xy[0]
    for _ in range(number):
        x += route_rng.uniform(45., 57.)
        waypoints.append([x, 75.+route_rng.uniform(-amplitude, amplitude)])
    waypoints = np.asarray(waypoints)
    first = waypoints[0] - start_xy
    heading = math.atan2(first[1], first[0]) + math.radians(route_rng.uniform(-18., 18.))
    start = (*start_xy, heading)
    chain = np.vstack([start_xy, waypoints])
    centers = []
    for encounter, leg in enumerate((0, number-1)):
        if level == 0:
            break
        delta = chain[leg+1] - chain[leg]
        length = np.linalg.norm(delta)
        direction = delta / length
        normal = np.array([-direction[1], direction[0]])
        if level == 1:
            placements = [(layout_rng.uniform(13., min(31., length-12.)),
                           layout_rng.uniform(-2.4, 2.4))]
        else:
            along = layout_rng.uniform(15., 24.)
            offset = layout_rng.uniform(-.5, .5)
            side = (1. if layout_rng.integers(2) else -1.)
            placements = [(along, offset),
                          (along+layout_rng.uniform(4., 7.),
                           offset+side*layout_rng.uniform(2., 4.5))]
        for along, lateral in placements:
            if mirror:
                lateral = -lateral
            centers.append(chain[leg] + along*direction + lateral*normal)
    bodies = [StaticBody(BUOY, Circle(float(p[0]), float(p[1]), .3)) for p in centers]
    radii = np.full(number, 5.)
    radii[-1] = 3.
    course = from_dict(to_dict(Course(boundary, bodies, waypoints, radii, start,
                                    terrain_extent=default_terrain_extent(boundary))))
    if problems := validate(course):
        raise ValueError(f'Invalid varied course: {problems}')
    return course


def write_varied_buoy_curriculum(output, legacy_source, *, seed=20261003):
    """Write an immutable draft; caller records the user's explicit approval.

    64 train courses: 8 clean, 24 two-buoy, 32 four-buoy. Validation has
    six paired four-buoy courses, four paired two-buoy courses, two clean
    routes and four original heldout routes. The final 20 tests stay unused.
    """
    output, legacy_source = Path(output), Path(legacy_source)
    output.mkdir(parents=True, exist_ok=False)
    legacy = json.loads((legacy_source/'manifest.json').read_text(encoding='utf-8'))
    plan = dict(train=[], val=[], test=[], review=[], metadata=dict(
        family='temporal-varied-waypoint-buoys-v1', seed=seed,
        approval_status='pending_user_confirmation', training_started=False,
        condition='1-2', history_frames=4, history_span_s=1.5,
        physics_dt=.1, decision_dt=.5, cutoff_ticks=2400,
        image_size=[200, 200], sensor_extent_m=[100., 100.],
        buoy_diameter_m=.6, hull_size_m=[3., 2.],
        waypoint_count=[2, 3], route_leg_dx_m=[45., 57.],
        waypoint_y_variation_m=[-10., 10.], initial_heading_perturb_deg=[-18., 18.],
        channel_width_m={str(k): v for k,v in WIDTHS.items()},
        sampling_percent={'1': {'clean': 20, '1': 80},
                          '2': {'clean': 20, '1': 25, '2': 55}},
        split_rule='Independent route/layout seeds; disjoint hashes; exact same-route mirrored pairs',
        legacy_source_manifest_sha256=hashlib.sha256((legacy_source/'manifest.json').read_bytes()).hexdigest(),
        promotion_checks=['separate clean/legacy/varied validation', 'paired layouts',
                          'minimum full-hull clearance 0.8m', 'image/history intervention']))
    hashes = set()

    def emit(split, course, *, level, kind, **details):
        digest = course_identity(course)
        if digest in hashes:
            raise ValueError('Duplicate serialized course in train/validation/test/review')
        hashes.add(digest)
        name = f'{split}-varied-{len(plan[split]):03d}.json'
        (output/name).write_text(json.dumps(to_dict(course), indent=2)+'\n', encoding='utf-8')
        row = dict(file=name, sha256=digest, level=level, count=len(course.bodies),
                   kind=kind, validation=[], **details)
        if split == 'val':
            row['evaluation_stages'] = [2]
        plan[split].append(row)

    # Earlier training courses retain familiarity; original validation remains
    # diagnostic only and never enters the training or auxiliary-label pools.
    for level, number in ((0, 4), (1, 8)):
        for row in [r for r in legacy['train'] if r['level'] == level][:number]:
            course = load(legacy_source/row['file'])
            if course_identity(course) != row['sha256']:
                raise ValueError('Changed legacy training course')
            emit('train', course, level=level, kind='clean' if level == 0 else 'buoy',
                 group='legacy_train', legacy_file=row['file'])
    for row in [r for r in legacy['val'] if r['level'] == 1][:4]:
        course = load(legacy_source/row['file'])
        if course_identity(course) != row['sha256']:
            raise ValueError('Changed legacy validation course')
        emit('val', course, level=1, kind='buoy', group='legacy_two', legacy_file=row['file'])

    counts = {'train': {0:4, 1:16, 2:32}, 'val': {0:2, 1:4, 2:6},
              'test': {0:2, 1:6, 2:12}, 'review': {2:4}}
    for split_id, (split, levels) in enumerate(counts.items()):
        for level, count in levels.items():
            for index in range(count):
                pair = index if level == 0 else index//2
                route_seed, layout_seed = map(int, np.random.SeedSequence(
                    [seed, split_id, level, pair]).generate_state(2))
                mirror = level != 0 and bool(index % 2)
                course = make_varied_buoy_course(route_seed, layout_seed, level, mirror=mirror)
                emit(split, course, level=level, kind='clean' if level == 0 else 'buoy',
                     group=f'varied_{"clean" if level == 0 else "two" if level == 1 else "four"}',
                     route_seed=route_seed, layout_seed=layout_seed, mirror=mirror,
                     pair_id=f'{split}-l{level}-p{pair:02d}')
    (output/'manifest.json').write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')
    return plan


def preview_curriculum(directory):
    """Render exact course maps and unmodified official top-view examples."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle as PlotCircle
    from usvnav.plant import Vessel
    from usvnav.render import top_view
    directory = Path(directory)
    plan = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    examples = [r for r in plan['review']][:4]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for col, row in enumerate(examples):
        course = load(directory/row['file'])
        ax = axes[0,col]
        b = course.boundary
        ax.fill(b[:,0], b[:,1], color='#dff1f6')
        route = np.vstack([course.start[:2], course.waypoints])
        ax.plot(route[:,0], route[:,1], '--', color='#2862ab')
        ax.scatter(*course.start[:2], marker='>', color='#dc646e', s=45)
        for index, (p,r) in enumerate(zip(course.waypoints, course.arrival_radii)):
            ax.add_patch(PlotCircle(p,r,fill=False,color='#208366'))
            ax.text(*p,str(index+1),ha='center',va='center',fontsize=8)
        for body in course.bodies:
            ax.add_patch(PlotCircle((body.shape.x,body.shape.y),body.shape.r,color='#a48300'))
        ax.set(xlim=(route[:,0].min()-5,route[:,0].max()+6),
               ylim=(b[:,1].min()-2,b[:,1].max()+2),aspect='equal',xlabel='world x (m)')
        ax.set_title(f'Pair {col//2+1} / {"mirrored" if row["mirror"] else "original"}\n{course.n_waypoints} targets, 4 buoys')
        axes[1,col].imshow(top_view(course,Vessel(*course.start),0.),interpolation='nearest')
        axes[1,col].set_title('Actual 200 x 200 RGB input')
        axes[1,col].axis('off')
    fig.suptitle('Varied targets / independent buoy layouts / identical-route mirrored pairs')
    fig.tight_layout()
    path = directory/'varied-course-examples.png'
    fig.savefig(path,dpi=130)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--legacy-source', type=Path, required=True)
    args = parser.parse_args()
    plan = write_varied_buoy_curriculum(args.out,args.legacy_source)
    preview_curriculum(args.out)
    print(json.dumps({key:len(plan[key]) for key in ('train','val','test','review')}))


if __name__ == '__main__':
    main()
