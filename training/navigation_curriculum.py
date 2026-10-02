"""Small, separate easy train/validation courses; no moving traffic at stage 0."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from usvnav.coursefile import default_terrain_extent, from_dict, to_dict, validate
from usvnav.geometry import Circle, Rect
from usvnav.world import BUOY, DOCK, MOORED, PIER, Course, StaticBody
from training.collect_perception import course_identity


def make_course(x, y, bend, heading, layout):
    boundary = np.array([[0., 35.], [400., 35.], [400., 115.], [0., 115.]])
    # Objects provide visual context outside the nominal corridor. Avoidance is
    # deliberately a later curriculum stage, not a conclusion of this pilot.
    bodies = [StaticBody(MOORED, Rect(x + 37 + layout, 97, 12, 4, 0)),
              StaticBody(BUOY, Circle(x + 62, 48 + layout, .3)),
              StaticBody(PIER, Rect(x + 108, 100, 12, 4, .2)),
              StaticBody(DOCK, Rect(x + 76 - layout, 104, 15, 6, 0))]
    course = Course(boundary, bodies, np.array([[x + 40, y + bend], [x + 80, y + 2 * bend]]),
                    np.array([5., 3.]), (x, y, heading), terrain_extent=default_terrain_extent(boundary))
    # Train on precisely the serialized official course, including rounding.
    course = from_dict(to_dict(course))
    violations = validate(course)
    if violations:
        raise ValueError(violations)
    return course


def write_curriculum(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    specs = {"train": [(45, 70, 0, 0, 0), (52, 77, 0, .025, 2),
                       (48, 67, 2, .025, 4), (55, 81, -2, -.025, 6)],
             "val": [(63, 73, 0, -.015, 3), (58, 79, -3, -.04, 5)]}
    manifest = {"stage": "easy-0", "condition": "1-2", "traffic": "none", "splits": {}}
    identities = set()
    for split, rows in specs.items():
        manifest["splits"][split] = []
        for i, row in enumerate(rows):
            course = make_course(*row)
            identity = course_identity(course)
            if identity in identities:
                raise ValueError("Duplicate course across curriculum splits")
            identities.add(identity)
            path = directory / f"{split}-{i + 1:02d}.json"
            path.write_text(json.dumps(to_dict(course), indent=2) + "\n", encoding="utf-8")
            manifest["splits"][split].append(dict(file=path.name, sha256=identity, validation=validate(course)))
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
