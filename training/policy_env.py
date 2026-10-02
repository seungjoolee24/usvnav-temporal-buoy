"""Normalized-action Gym wrapper using public-only frozen perception features."""
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from training.policy_observation import PolicyEncoder, SHIP_FIELDS, STATE_FIELDS, WAYPOINT_FIELDS


def physical_action(action):
    action = np.asarray(action, dtype=float).reshape(-1)
    if action.size != 2 or not np.isfinite(action).all():
        return action  # Official sanitizer classifies the invalid command.
    normalized = np.clip(action, -1., 1.)
    return np.array([.75 + 1.25 * normalized[0], .6 * normalized[1]], dtype=float)


class PolicyEnv(gym.Wrapper):
    def __init__(self, env, perception, *, max_ships=32, max_waypoints=32):
        super().__init__(env)
        self.encoder = PolicyEncoder(perception, max_ships=max_ships, max_waypoints=max_waypoints, tick_limit=env.tick_limit)
        self.action_space = spaces.Box(-1., 1., (2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "map": spaces.Box(0., 1., (7, 50, 50), np.float32),
            "state": spaces.Box(-10., 10., (len(STATE_FIELDS),), np.float32),
            "ships": spaces.Box(-10., 10., (max_ships, len(SHIP_FIELDS)), np.float32),
            "waypoints": spaces.Box(-10., 10., (max_waypoints, len(WAYPOINT_FIELDS)), np.float32),
        })

    def reset(self, *, seed=None, options=None):
        raw, info = self.env.reset(seed=seed, options=options)
        self.encoder.reset(info["meta"])
        return self.encoder.encode(raw), info

    def step(self, action):
        raw, reward, terminated, truncated, info = self.env.step(physical_action(action))
        features = self.encoder.encode(raw)
        info["tracked_ships"] = len(self.encoder.last_tracks)
        info["dropped_tracks"] = self.encoder.dropped_tracks
        return features, reward, terminated, truncated, info
