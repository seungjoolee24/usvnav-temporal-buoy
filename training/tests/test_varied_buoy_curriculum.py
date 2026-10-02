"""Route diversity, paired image challenges and heldout retention checks."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.colab_runtime import approve_curriculum_copy
from training.temporal_buoy_curriculum import write_temporal_buoy_draft
from training.train_buoy_navigation import load_reviewed_curriculum
from training.varied_buoy_curriculum import make_varied_buoy_course, write_varied_buoy_curriculum
from usvnav.coursefile import load, validate


class VariedBuoyCurriculumTests(unittest.TestCase):
    def test_route_and_layout_seeds_are_independent_and_mirrors_keep_public_targets(self):
        a=make_varied_buoy_course(120,31,2)
        b=make_varied_buoy_course(120,31,2,mirror=True)
        c=make_varied_buoy_course(120,99,2)
        for other in (b,c):
            self.assertEqual(a.start,other.start)
            np.testing.assert_array_equal(a.waypoints,other.waypoints)
            np.testing.assert_array_equal(a.boundary,other.boundary)
        self.assertFalse(np.array_equal([[x.shape.x,x.shape.y] for x in a.bodies],
                                       [[x.shape.x,x.shape.y] for x in b.bodies]))
        self.assertEqual(a.n_waypoints,3)
        self.assertEqual(len(a.bodies),4)
        for course in (a,b,c):
            self.assertEqual(validate(course),[])
            self.assertTrue(all(body.shape.r==.3 for body in course.bodies))

    def test_multiple_waypoints_bends_lengths_and_start_heading_remain_official(self):
        courses=[make_varied_buoy_course(130+i,90+i,2) for i in range(12)]
        self.assertEqual({c.n_waypoints for c in courses},{2,3})
        self.assertGreater(np.ptp([c.waypoints[0,1] for c in courses]),8.)
        self.assertGreater(np.ptp([c.waypoints[0,0]-c.start[0] for c in courses]),5.)
        for c in courses:
            chain=np.vstack([c.start[:2],c.waypoints])
            self.assertTrue(np.all(np.linalg.norm(np.diff(chain,axis=0),axis=1)>=40.))
            self.assertEqual(validate(c),[])

    def test_stage_two_evaluates_clean_prior_and_varied_courses_without_training_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            write_temporal_buoy_draft(root/'legacy')
            draft=write_varied_buoy_curriculum(root/'draft',root/'legacy')
            self.assertEqual([len(draft[k]) for k in ('train','val','test','review')],[64,16,20,4])
            hashes=[r['sha256'] for k in ('train','val','test','review') for r in draft[k]]
            self.assertEqual(len(hashes),len(set(hashes)))
            approved=approve_curriculum_copy(root/'draft',root/'approved',confirmed=True)
            plan=load_reviewed_curriculum(approved,root/'used',2,4)
            self.assertEqual(len(plan['val']),16)
            self.assertEqual(len(plan['train']),100)
            self.assertEqual(len({r['sha256'] for r in plan['train']}),64)
            self.assertEqual({r['group'] for r in plan['val']},
                             {'legacy_two','varied_two','varied_four','varied_clean'})
            self.assertTrue({r['sha256'] for r in plan['train']}.isdisjoint(
                            {r['sha256'] for r in plan['val']}))
            self.assertFalse((root/'used'/draft['test'][0]['file']).exists())
            for row in draft['review']:
                self.assertEqual(validate(load(root/'draft'/row['file'])),[])


if __name__=='__main__':
    unittest.main()
