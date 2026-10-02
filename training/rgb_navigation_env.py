"""Raw-RGB navigation with real 10 Hz physics and slower policy decisions.

Public-coordinate reference steering + held learned residual. No segmentation
model, tracker, ground-truth object features or image resizing enter the policy.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
import math
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from usvnav import sim
from usvnav.coursefile import load
from usvnav.plant import DT
from training.navigation_env import NavigationEpisode, RewardConfig
from training.policy_observation import STATE_FIELDS, WAYPOINT_FIELDS
from training.tracking import world_to_body


def base_command(public):
    distance, bearing = map(float, public["wp_polar"])
    rate = float(public["vel"][2])
    yaw = np.clip(1.2 * bearing - .2 * rate, -.6, .6)
    surge = 1.3 * max(.12, math.cos(bearing))
    surge = min(surge, max(.35, .3 * distance))
    return np.array([surge, yaw], dtype=float)


class RgbNavigationEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 2}

    def __init__(self, courses, *, condition="1-2", action_repeat=5, training_tick_limit=1200,
                 tick_limit=6000, history_frames=4, reward_config=None):
        super().__init__()
        self.courses = [load(p) if isinstance(p, (str, Path)) else p for p in courses]
        if (not self.courses or not 1 <= action_repeat <= 10 or history_frames < 1
                or training_tick_limit <= 0 or training_tick_limit > tick_limit
                or training_tick_limit % action_repeat):
            raise ValueError("Need courses, valid repeat/history, and cutoff divisible by repeat within task horizon")
        self.condition, self.action_repeat = condition, int(action_repeat)
        self.training_tick_limit, self.tick_limit = int(training_tick_limit), int(tick_limit)
        self.history_frames, self.reward_config = int(history_frames), reward_config or RewardConfig()
        self.action_space = spaces.Box(-1., 1., (2,), np.float32)
        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(0, 255, (3 * history_frames, 200, 200), np.uint8),
            "state": spaces.Box(-10., 10., (len(STATE_FIELDS),), np.float32),
            "waypoints": spaces.Box(-10., 10., (32, len(WAYPOINT_FIELDS)), np.float32),
            "goal": spaces.Box(-10., 10., (4,), np.float32),
            "reference": spaces.Box(-1., 1., (2,), np.float32),
            "history": spaces.Box(-10., 10., (history_frames, 8), np.float32),
        })
        self.frames = deque(maxlen=history_frames)
        self._done = True

    def _public(self):
        ep = self.episode
        return sim.observation(ep.vessel, ep.course, ep.tick, ep.wp_index, ep.prev_applied)

    def _observe(self, *, initial=False):
        public = self._public()
        rgb = sim._perception(self.condition, self.episode.vessel, self.episode.course, self.episode.tick * DT)
        snapshot = (rgb.transpose(2, 0, 1).copy(), public)
        if initial:
            self.frames.clear()
            self.frames.extend([snapshot] * self.history_frames)
        else:
            self.frames.append(snapshot)
        pose, vel, previous = (np.asarray(public[k], dtype=float) for k in ("pose", "vel", "prev_action"))
        wp = int(public["wp_index"])
        remaining = world_to_body(self.meta["waypoints"][wp:], pose)
        distance, bearing = public["wp_polar"]
        count = len(self.meta["waypoints"])
        state = np.array([pose[0] / 400, pose[1] / 150, math.sin(pose[2]), math.cos(pose[2]),
                          vel[0] / 2, vel[1] / 2, vel[2] / .6, previous[0] / 2, previous[1] / .6,
                          distance / 500, math.sin(bearing), math.cos(bearing),
                          remaining[0, 0] / 100, remaining[0, 1] / 100, wp / count, len(remaining) / 32,
                          max(0., 1 - self.episode.tick / self.tick_limit), .3, .2], np.float32)
        waypoints = np.zeros((32, 4), np.float32)
        waypoints[:len(remaining), 0] = 1
        waypoints[:len(remaining), 1:3] = remaining / 400
        waypoints[:len(remaining), 3] = self.meta["arrival_radii"][wp:] / 10
        goal = np.array([1., *remaining[-1] / 400, self.meta["arrival_radii"][-1] / 10], np.float32)
        history = []
        for _, frame_public in self.frames:
            x, y, heading = frame_public["pose"]
            u, v, r = frame_public["vel"]
            history.append([x / 400, y / 150, math.sin(heading), math.cos(heading), u / 2, v / 2, r / .6,
                            (self.episode.tick - int(frame_public["t"])) / 10])
        self._rgb, self._last_public = rgb, public
        return dict(rgb=np.concatenate([frame[0] for frame in self.frames], axis=0), state=state,
                    waypoints=waypoints, goal=goal, history=np.array(history, np.float32),
                    reference=(base_command(public) / [2., .6]).astype(np.float32))

    def _info(self, *, truncated=False, terms=None, actions=()):
        ep = self.episode
        info = dict(tick=ep.tick, course_index=self._course_index, seed=ep.seed,
                    outcome="training_cutoff" if truncated else ep.outcome, official_outcome=ep.outcome,
                    waypoints_reached=ep.wp_index, waypoints_total=ep.course.n_waypoints,
                    elapsed_s=ep.tick * DT, distance_m=ep.distance, fuel=ep.fuel, clearance_m=ep.clearance,
                    reward_terms=terms or {}, applied_actions=list(actions), physics_ticks_advanced=len(actions))
        if ep.outcome is not None:
            info["official_result"] = asdict(ep.result())
        return info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._done = True
        self._course_index = int((options or {}).get("course_index", self.np_random.integers(len(self.courses))))
        if not 0 <= self._course_index < len(self.courses):
            raise ValueError("Bad course index")
        episode_seed = int(seed) if seed is not None else int(self.np_random.integers(2**32))
        self.episode = NavigationEpisode(self.courses[self._course_index], condition=self.condition,
                                        seed=episode_seed, tick_limit=self.tick_limit)
        if self.episode.outcome:
            raise ValueError(f"Initial terminal state: {self.episode.outcome}")
        self.meta = self.episode.meta()
        if not 0 < len(self.meta["waypoints"]) <= 32:
            raise ValueError("Need 1..32 public waypoints")
        self._done = False
        return self._observe(initial=True), self._info()

    def step(self, action):
        if self._done:
            raise RuntimeError("Reset before stepping")
        residual = np.asarray(action, dtype=float).reshape(-1)
        valid = residual.shape == (2,) and np.isfinite(residual).all()
        if valid:
            residual = np.clip(residual, -1., 1.)
        terms, actions = {}, []
        ep, config = self.episode, self.reward_config
        for _ in range(self.action_repeat):
            old_tick, old_wp, old_remaining = ep.tick, ep.wp_index, ep.remaining_route_m()
            command = base_command(self._public()) + residual * [2., 1.2] if valid else residual
            ep.advance(command)
            seconds = (ep.tick - old_tick) * DT
            danger = max(0., 1 - ep.clearance / config.danger_distance_m) ** 2
            tick_terms = dict(progress=config.progress_per_m * (old_remaining - ep.remaining_route_m()),
                              waypoint=config.waypoint * (ep.wp_index - old_wp), goal=config.goal if ep.outcome == "goal" else 0.,
                              failure=config.failure if ep.outcome in ("static_collision", "dynamic_collision", "out_of_bounds", "invalid_action", "crash") else 0.,
                              timeout=config.timeout if ep.outcome == "timeout" else 0., time=-config.time_per_s * seconds,
                              effort=-config.effort * ep.last_effort, danger=-config.danger_per_s * seconds * danger)
            for key, value in tick_terms.items():
                terms[key] = terms.get(key, 0.) + value
            if ep.tick != old_tick:
                actions.append(ep.prev_applied.tolist())
            if ep.outcome is not None or ep.tick >= self.training_tick_limit:
                break
        terminated = ep.outcome is not None
        truncated = not terminated and ep.tick >= self.training_tick_limit
        self._done = bool(terminated or truncated)
        return self._observe(), float(sum(terms.values())), bool(terminated), bool(truncated), self._info(truncated=truncated, terms=terms, actions=actions)

    def render(self):
        return self._rgb.copy()

    def close(self):
        self._done = True
