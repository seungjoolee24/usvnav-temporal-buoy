"""Official-format simple-to-complex navigation course families."""
import json
import math
from pathlib import Path

import numpy as np
from usvnav.coursefile import default_terrain_extent, from_dict, to_dict, validate
from usvnav.geometry import Circle
from usvnav.world import BUOY, Course, StaticBody
from training.collect_perception import course_identity


def make_rgb_course(stage, variant):
    if stage not in (0, 1, 2):
        raise ValueError("Implemented stages: 0 straight, 1 turns, 2 single buoy")
    boundary = np.array([[0., 20.], [400., 20.], [400., 130.], [0., 130.]])
    x = 45 + variant * 3
    y = 69 + (variant % 3) * 4
    heading = (variant % 5 - 2) * .025
    angle = 0. if stage == 0 else (1 if variant % 2 else -1) * math.radians(12 + variant % 3 * 3)
    # Leave margin above the official40m minimum for serialized angled legs.
    # Rounding a nominal40m diagonal can otherwise make it microscopically short.
    leg = 40. if stage == 0 else 41.
    points = np.array([[x + 40, y], [x + 40 + leg * math.cos(angle), y + leg * math.sin(angle)]])
    obstacle_position = points[0] + (.45 + .05 * (variant % 3)) * (points[1] - points[0])
    bodies = [] if stage < 2 else [StaticBody(BUOY, Circle(*map(float, obstacle_position), .3))]
    course = Course(boundary, bodies, points, np.array([5., 3.]), (x, y, heading),
                    terrain_extent=default_terrain_extent(boundary))
    course = from_dict(to_dict(course))
    problems = validate(course)
    if problems:
        raise ValueError(problems)
    return course


def write_rgb_curriculum(output, stage):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan, seen = {}, set()
    for split, variants in (("train", range(4)), ("val", range(10, 11))):
        plan[split] = []
        tasks = [(stage, variant) for variant in variants]
        if split == "train" and stage > 0:
            tasks.append((stage - 1, 20))
        if split == "train" and stage > 1:
            tasks.append((0, 21))
        for task_stage, variant in tasks:
            course = make_rgb_course(task_stage, variant)
            identity = course_identity(course)
            if identity in seen:
                raise ValueError("Duplicate train/validation course")
            seen.add(identity)
            name = f"stage{task_stage}-{split}-{variant:02d}.json"
            (output / name).write_text(json.dumps(to_dict(course), indent=2) + "\n", encoding="utf-8")
            plan[split].append(dict(file=name, stage=task_stage, sha256=identity, validation=[]))
    (output / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan
