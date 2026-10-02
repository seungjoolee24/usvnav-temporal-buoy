"""Gymnasium bridge to the public simulator, with explicit offline reward terms.

The official tick's precedence is preserved: collision, bank exit, ONE waypoint,
goal, task timeout. Full-hull geometry, commands, dynamics, sensors and metrics
reuse usvnav. A shorter training rollout is a truncation, not an official timeout.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import traceback

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from usvnav import sim
from usvnav.collide import CourseIndex
from usvnav.coursefile import load
from usvnav.plant import (DT, HULL_LENGTH, HULL_WIDTH, InvalidAction, NoDisturbance,
                         OUDisturbance, V_MAX, V_MIN, W_MAX, Vessel, sanitize)
from usvnav.world import MOVING


@dataclass(frozen=True)
class RewardConfig:
    progress_per_m: float = 1.0
    waypoint: float = 5.0
    goal: float = 50.0
    failure: float = -100.0
    timeout: float = -10.0
    time_per_s: float = .1
    effort: float = .01
    danger_per_s: float = .5
    danger_distance_m: float = 2.0


class NavigationEpisode:
    """Stepwise version of sim.run_episode; no policy and no process time cap."""
    def __init__(self, course, *, condition="1-2", seed=0, tick_limit=sim.TICK_LIMIT,
                 d_cap=10.0, clearance_percentile=5.0):
        if condition not in ("1-2", "1-4") or tick_limit < 1 or d_cap <= 0:
            raise ValueError("Need RGB condition, positive task horizon and clearance cap")
        self.course, self.condition, self.seed = course, condition, int(seed)
        self.tick_limit, self.d_cap, self.clearance_percentile = int(tick_limit), d_cap, clearance_percentile
        self.vessel = Vessel(*course.start)
        self.drift = OUDisturbance(self.seed ^ 0x9E3779B9) if condition == "1-4" else NoDisturbance()
        self.index = CourseIndex(course)
        self.tick, self.wp_index, self.distance, self.fuel = 0, 0, 0.0, 0.0
        self.prev_applied = np.zeros(2, dtype=float)
        self.per_tick_clearance, self.trace = [], []
        self.outcome, self.detail, self.tier = None, "", "navigational"
        self.clearance, self.last_effort = 0.0, 0.0
        self._inspect()

    def meta(self):
        return dict(boundary=self.course.boundary.copy(), waypoints=self.course.waypoints.copy(),
                    arrival_radii=self.course.arrival_radii.copy(), hull=(HULL_LENGTH, HULL_WIDTH),
                    observation_mode=self.condition)

    def _inspect(self):
        self.clearance, hit, inside = self.index.tick(self.vessel.hull(), self.tick * DT, self.d_cap)
        self.per_tick_clearance.append(self.clearance)
        if hit is not None:
            self.outcome = "dynamic_collision" if hit == MOVING else "static_collision"
        elif not inside:
            self.outcome = "out_of_bounds"
        else:
            if self.wp_index < self.course.n_waypoints:
                target = self.course.waypoints[self.wp_index]
                if math.hypot(self.vessel.x - target[0], self.vessel.y - target[1]) <= self.course.arrival_radii[self.wp_index]:
                    self.wp_index += 1
            if self.wp_index >= self.course.n_waypoints:
                self.outcome = "goal"
            elif self.tick == self.tick_limit:
                self.outcome = "timeout"

    def observe(self):
        public = sim.observation(self.vessel, self.course, self.tick, self.wp_index, self.prev_applied)
        public["perception"] = sim._perception(self.condition, self.vessel, self.course, self.tick * DT)
        return public

    def remaining_route_m(self):
        if self.wp_index >= self.course.n_waypoints:
            return 0.0
        points = self.course.waypoints[self.wp_index:]
        return float(np.linalg.norm(points[0] - [self.vessel.x, self.vessel.y])
                     + np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

    def advance(self, action):
        if self.outcome is not None:
            raise RuntimeError("Episode has ended; create/reset an episode")
        try:
            applied = sanitize(action)
        except InvalidAction as exc:
            self.outcome, self.tier, self.detail = "invalid_action", "submission_fault", str(exc)
            self.last_effort = 0.0
            return
        except Exception:
            self.outcome, self.tier, self.detail = "crash", "submission_fault", traceback.format_exc()
            self.last_effort = 0.0
            return
        self.last_effort = float(((applied[0] - self.prev_applied[0]) / V_MAX) ** 2
                                 + ((applied[1] - self.prev_applied[1]) / W_MAX) ** 2)
        self.fuel += self.last_effort
        v = self.vessel
        self.trace.append((self.tick, v.x, v.y, v.psi, float(applied[0]), float(applied[1])))
        old = (v.x, v.y)
        v.step(applied, self.drift.step())
        self.distance += math.hypot(v.x - old[0], v.y - old[1])
        self.prev_applied = applied
        self.tick += 1
        self._inspect()

    def result(self):
        if self.outcome is None:
            return None
        return sim.Result(outcome=self.outcome, tier=self.tier, ticks=self.tick, distance=self.distance,
                          elapsed=self.tick * DT, clearance=sim.clearance_stat(self.per_tick_clearance, self.clearance_percentile),
                          waypoints_reached=self.wp_index, fuel=self.fuel, trace=list(self.trace), detail=self.detail,
                          pose=(self.vessel.x, self.vessel.y, self.vessel.psi), per_tick_clearance=list(self.per_tick_clearance))


class NavigationEnv(gym.Env):
    """Physical action (m/s, rad/s), official RGB/common observation, 0.1 s steps.

Collision/goal/6000-tick task timeout terminate. Optional shorter training_tick_limit
truncates an unfinished task, keeps its final observation and gives no failure bonus.
Reward uses privileged geometry offline; the observation contains public data only.
"""
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, courses, *, condition="1-2", tick_limit=sim.TICK_LIMIT,
                 training_tick_limit=None, reward_config=None, render_mode=None):
        super().__init__()
        self.courses = [load(p) if isinstance(p, (str, Path)) else p for p in courses]
        if not self.courses or condition not in ("1-2", "1-4") or tick_limit < 1:
            raise ValueError("Provide courses, an RGB condition and a positive task horizon")
        if training_tick_limit is not None and not 0 < training_tick_limit <= tick_limit:
            raise ValueError("Training cutoff must be within the task horizon")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Only rgb_array rendering is supported")
        self.condition, self.tick_limit = condition, int(tick_limit)
        self.training_tick_limit, self.render_mode = training_tick_limit, render_mode
        self.reward_config = reward_config or RewardConfig()
        if self.reward_config.danger_distance_m <= 0 or not all(math.isfinite(v) for v in asdict(self.reward_config).values()):
            raise ValueError("Invalid reward configuration")
        self.action_space = spaces.Box(np.array([V_MIN, -W_MAX], np.float32), np.array([V_MAX, W_MAX], np.float32))
        bound = np.finfo(np.float32).max
        self.observation_space = spaces.Dict({
            "t": spaces.Box(0, self.tick_limit, shape=(), dtype=np.int32),
            "pose": spaces.Box(-bound, bound, (3,), np.float32),
            "vel": spaces.Box(-bound, bound, (3,), np.float32),
            "wp_index": spaces.Box(0, max(c.n_waypoints for c in self.courses) - 1, shape=(), dtype=np.int32),
            "wp_polar": spaces.Box(np.array([0, -math.pi], np.float32), np.array([bound, math.pi], np.float32)),
            "prev_action": self.action_space,
            "perception": spaces.Box(0, 255, (200, 200, 3), np.uint8),
        })
        self.episode, self._observation, self._done = None, None, True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._done, self._observation = True, None
        options = options or {}
        course_index = int(options.get("course_index", self.np_random.integers(len(self.courses))))
        if not 0 <= course_index < len(self.courses):
            raise ValueError("Invalid course index")
        episode_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**32))
        self.episode = NavigationEpisode(self.courses[course_index], condition=self.condition,
                                        seed=episode_seed, tick_limit=self.tick_limit)
        if self.episode.outcome is not None:
            raise ValueError(f"Course starts in a terminal state: {self.episode.outcome}")
        self._done, self._course_index = False, course_index
        self._observation = self.episode.observe()
        info = self._info()
        info["meta"] = self.episode.meta()
        return self._copy_observation(), info

    def _copy_observation(self):
        return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in self._observation.items()}

    def _info(self, *, truncated=False, terms=None):
        ep = self.episode
        info = dict(tick=ep.tick, seed=ep.seed, course_index=self._course_index,
                    outcome="training_cutoff" if truncated else ep.outcome, official_outcome=ep.outcome,
                    waypoints_reached=ep.wp_index, waypoints_total=ep.course.n_waypoints,
                    clearance_m=ep.clearance, distance_m=ep.distance, fuel=ep.fuel,
                    elapsed_s=ep.tick * DT, applied_action=ep.prev_applied.copy(),
                    reward_terms=terms or {})
        result = ep.result()
        if result is not None:
            info["official_result"] = asdict(result)
        return info

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset before step or after termination/truncation")
        ep, config = self.episode, self.reward_config
        old_tick, old_wp, old_remaining = ep.tick, ep.wp_index, ep.remaining_route_m()
        ep.advance(action)
        seconds = (ep.tick - old_tick) * DT
        danger = max(0., 1. - ep.clearance / config.danger_distance_m) ** 2
        terms = dict(progress=config.progress_per_m * (old_remaining - ep.remaining_route_m()),
                     waypoint=config.waypoint * (ep.wp_index - old_wp),
                     goal=config.goal if ep.outcome == "goal" else 0.,
                     failure=config.failure if ep.outcome in ("static_collision", "dynamic_collision", "out_of_bounds", "invalid_action", "crash") else 0.,
                     timeout=config.timeout if ep.outcome == "timeout" else 0.,
                     time=-config.time_per_s * seconds, effort=-config.effort * ep.last_effort,
                     danger=-config.danger_per_s * seconds * danger)
        terminated = ep.outcome is not None
        truncated = not terminated and self.training_tick_limit is not None and ep.tick >= self.training_tick_limit
        self._done = bool(terminated or truncated)
        # Invalid commands end at the same tick; don't rebuild an identical sensor frame.
        if ep.tick != old_tick:
            self._observation = ep.observe()
        return self._copy_observation(), float(sum(terms.values())), bool(terminated), bool(truncated), self._info(truncated=truncated, terms=terms)

    def render(self):
        if self._observation is None:
            raise RuntimeError("Call reset before render")
        return self._observation["perception"].copy()

    def close(self):
        self._done = True
