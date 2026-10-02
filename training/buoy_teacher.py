"""Training-only geometric buoy teacher; never a deployed policy input.

This oracle can label genuine RGB/public-state observations for action warmup.
It uses private buoy geometry offline and supplies only a normalized residual
target. Its supported scope is static buoys in a wide rectangular channel.
It deliberately rejects traffic and other bodies rather than inventing labels
for cases its path planner does not model.
"""
from __future__ import annotations

import heapq
import math
from pathlib import Path

import numpy as np

from usvnav import sim
from usvnav.coursefile import load
from usvnav.plant import HULL_LENGTH, HULL_WIDTH
from usvnav.world import BUOY
from training.collect_perception import course_identity
from training.navigation_env import NavigationEpisode
from training.rgb_navigation_env import RgbNavigationEnv, base_command


class TrainingBuoyTeacher:
    """Private-geometry visibility graph and low-speed waypoint tracking.

    Graph edges avoid a circumscribed full hull plus 0.7 m static clearance.
    Only labels and training diagnostics can expose this class's path/geometry.
    Network forward/predict must continue receiving the existing public dict.
    """
    def __init__(self, course, *, speed=.85, clearance_margin=.7):
        if not 0. < speed <= 1.3 or not .3 <= clearance_margin <= 2.:
            raise ValueError("Need slow positive speed and a conservative clearance margin")
        boundary = np.asarray(course.boundary)
        low, high = boundary.min(0), boundary.max(0)
        corners = {(float(x), float(y)) for x, y in boundary}
        if (len(boundary) != 4 or corners != {(float(x), float(y)) for x in (low[0], high[0])
                                            for y in (low[1], high[1])}
                or course.traffic or course.lanes or any(body.cls != BUOY for body in course.bodies)):
            raise ValueError("Teacher supports only static buoys and a rectangular channel")
        self.course, self.speed = course, float(speed)
        self.hull_margin = math.hypot(HULL_LENGTH, HULL_WIDTH) / 2. + clearance_margin
        self.low, self.high = low + self.hull_margin, high - self.hull_margin
        self.all_centers = np.array([[body.shape.x, body.shape.y] for body in course.bodies], float).reshape(-1, 2)
        self.all_radii = np.array([body.shape.r + self.hull_margin for body in course.bodies], float)
        self.centers, self.radii = self.all_centers, self.all_radii
        self.path, self.wp_index, self.node_index, self.visible_signature = None, None, 1, None

    def _point_safe(self, point):
        return (np.all(point >= self.low) and np.all(point <= self.high)
                and np.all(np.linalg.norm(self.centers - point, axis=1) > self.radii))

    def _segment_safe(self, start, goal):
        delta = goal - start
        distance2 = float(np.dot(delta, delta))
        if distance2 == 0.:
            return self._point_safe(start)
        fraction = np.clip((self.centers - start) @ delta / distance2, 0., 1.)
        distance = np.linalg.norm(self.centers - (start + fraction[:, None] * delta), axis=1)
        return bool(np.all(distance >= self.radii))

    def plan(self, start, goal):
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        if not self._point_safe(start) or not self._point_safe(goal):
            raise ValueError("Teacher needs endpoints clear of its inflated full-hull safety margin")
        nodes = [start, goal]
        angles = np.arange(16) * (2. * math.pi / 16)
        for center, radius in zip(self.centers, self.radii):
            for angle in angles:
                point = center + (radius + 1.) * np.array([math.cos(angle), math.sin(angle)])
                if self._point_safe(point):
                    nodes.append(point)
        edges = [[] for _ in nodes]
        for index, point in enumerate(nodes):
            for other in range(index):
                if self._segment_safe(point, nodes[other]):
                    length = float(np.linalg.norm(point - nodes[other]))
                    edges[index].append((other, length))
                    edges[other].append((index, length))
        queue, distances, parents = [(0., 0)], [math.inf] * len(nodes), {}
        distances[0] = 0.
        while queue:
            distance, index = heapq.heappop(queue)
            if distance > distances[index]:
                continue
            if index == 1:
                path = [nodes[index]]
                while index:
                    index = parents[index]
                    path.append(nodes[index])
                return np.array(path[::-1])
            for other, length in edges[index]:
                candidate = distance + length
                if candidate < distances[other]:
                    distances[other], parents[other] = candidate, index
                    heapq.heappush(queue, (candidate, other))
        raise ValueError("No teacher route exists with the requested full-hull margin")

    def residual(self, public):
        pose = np.asarray(public["pose"], float)
        wp = int(public["wp_index"])
        # Geometry beyond the current 100x100m bow-up RGB square must not
        # determine an action target that an image-based actor cannot infer.
        delta = self.all_centers - pose[:2]
        c, s = math.cos(pose[2]), math.sin(pose[2])
        body_xy = np.column_stack([delta[:, 0] * c + delta[:, 1] * s,
                                   -delta[:, 0] * s + delta[:, 1] * c])
        visible = tuple(np.flatnonzero((np.abs(body_xy) <= 50.).all(axis=1)).tolist())
        if self.path is None or self.wp_index != wp or self.visible_signature != visible:
            self.centers = self.all_centers[list(visible)]
            self.radii = self.all_radii[list(visible)]
            self.path = self.plan(pose[:2], self.course.waypoints[wp])
            self.wp_index, self.node_index = wp, 1
            self.visible_signature = visible
        while self.node_index < len(self.path) - 1 and np.linalg.norm(pose[:2] - self.path[self.node_index]) < .8:
            self.node_index += 1
        offset = self.path[self.node_index] - pose[:2]
        bearing = math.atan2(offset[1], offset[0]) - pose[2]
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        yaw = float(np.clip(1.4 * bearing - .3 * float(public["vel"][2]), -.6, .6))
        surge = self.speed * max(.15, math.cos(bearing))
        surge = min(surge, max(.25, .3 * float(np.linalg.norm(offset))))
        physical = np.array([surge, yaw])
        return np.clip((physical - base_command(public)) / [2., 1.2], -1., 1.).astype(np.float32)


def teacher_rollout(course, *, seed=101, cutoff=1200, capture_observations=False):
    """Run held-for-five-ticks residual labels over the real simulator.

    With capture_observations=True, each sample contains precisely the existing
    RGB/public-coordinate observation and a teacher residual action. No hidden
    geometry is included in that input dict. Otherwise, avoid rendering entirely
    for a quick dynamics/teacher-quality check.
    """
    if isinstance(course, (str, Path)):
        course = load(course)
    teacher = TrainingBuoyTeacher(course)
    observations, labels, trace = [], [], []
    if capture_observations:
        env = RgbNavigationEnv([course], training_tick_limit=cutoff)
        obs, _ = env.reset(seed=seed)
        episode = env.episode
    else:
        env, episode = None, NavigationEpisode(course, seed=seed, tick_limit=6000)
    while episode.outcome is None and episode.tick < cutoff:
        public = sim.observation(episode.vessel, course, episode.tick, episode.wp_index, episode.prev_applied)
        residual = teacher.residual(public)
        if env is not None:
            observations.append({key: value.copy() for key, value in obs.items()})
            labels.append(residual.copy())
            obs, _, terminated, truncated, _ = env.step(residual)
        else:
            for _ in range(5):
                public = sim.observation(episode.vessel, course, episode.tick, episode.wp_index, episode.prev_applied)
                episode.advance(base_command(public) + residual * [2., 1.2])
                if episode.outcome is not None or episode.tick >= cutoff:
                    break
        trace.append([episode.tick, episode.vessel.x, episode.vessel.y, episode.vessel.psi])
    result = dict(outcome=episode.outcome or "training_cutoff", ticks=episode.tick,
                  elapsed_s=episode.tick * .1, waypoints_reached=episode.wp_index,
                  min_clearance_m=float(min(episode.per_tick_clearance)),
                  course_sha256=course_identity(course), teacher="training-only geometric oracle",
                  inference_input="Original RGB/public dict; geometry only supplies action labels")
    if env is not None:
        env.close()
    return dict(result=result, observations=observations,
                actions=np.array(labels, np.float32).reshape(-1, 2), trace=np.array(trace))
