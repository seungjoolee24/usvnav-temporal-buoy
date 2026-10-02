"""NumPy policy features built ONLY from public metadata and RGB inference.

No course or simulator imports. Waypoints and state bypass the perception CNN.
Max pooling of foreground probabilities preserves tiny predicted obstacles; the
pooled channels are features, not a mutually exclusive probability distribution.
"""
from __future__ import annotations

import math

import numpy as np

from training.perception_schema import DETAIL_CLASSES
from training.tracking import VesselTracker, world_to_body

STATE_FIELDS = ("x_over_400", "y_over_150", "sin_heading", "cos_heading", "u_over_2", "v_over_2", "yaw_rate_over_0.6",
                "previous_surge_over_2", "previous_yaw_over_0.6", "wp_distance_over_500", "sin_wp_bearing", "cos_wp_bearing",
                "next_wp_forward_over_100", "next_wp_port_over_100", "reached_wp_fraction", "remaining_wp_over_capacity",
                "remaining_task_time_fraction", "hull_length_over_10", "hull_width_over_10")
SHIP_FIELDS = ("present", "forward_over_50", "port_over_50", "ground_forward_speed_over_2.5", "ground_port_speed_over_2.5",
               "velocity_valid", "length_over_40", "width_over_10", "sin_double_axis_angle", "cos_double_axis_angle",
               "stationary", "moving", "unknown", "observed", "missing_time_over_0.5", "segmentation_confidence")
WAYPOINT_FIELDS = ("present", "forward_over_400", "port_over_400", "arrival_radius_over_10")


def pool_maps(maps):
    probabilities = np.asarray(maps["class_probabilities"], np.float32)
    valid = np.asarray(maps["valid"], bool)
    if probabilities.shape != (200, 200, 6) or valid.shape != (200, 200):
        raise ValueError("Expected 6-class official top-view maps")
    pixels = probabilities * valid[..., None]
    blocks = pixels.reshape(50, 4, 50, 4, 6)
    pooled = blocks.max(axis=(1, 3))
    pooled[..., 0] = blocks[..., 0].mean(axis=(1, 3))
    visible = valid.reshape(50, 4, 50, 4).mean(axis=(1, 3)).astype(np.float32)
    return np.concatenate([pooled.transpose(2, 0, 1), visible[None]], axis=0).astype(np.float32)


class PolicyEncoder:
    def __init__(self, perception, *, max_ships=32, max_waypoints=32, tick_limit=6000):
        if tuple(perception.classes) != DETAIL_CLASSES or max_ships < 1 or max_waypoints < 1 or tick_limit < 1:
            raise ValueError("Need 6-class perception and positive feature capacities/horizon")
        self.perception = perception
        self.max_ships, self.max_waypoints, self.tick_limit = max_ships, max_waypoints, tick_limit
        self.tracker = VesselTracker(perception.classes)
        self._ready, self._tick, self._cached = False, None, None

    def reset(self, meta):
        self.waypoints = np.asarray(meta["waypoints"], dtype=float).copy()
        self.arrival_radii = np.asarray(meta["arrival_radii"], dtype=float).copy()
        self.hull = np.asarray(meta["hull"], dtype=float).copy()
        n = len(self.waypoints)
        if (self.waypoints.shape != (n, 2) or not 0 < n <= self.max_waypoints
                or self.arrival_radii.shape != (n,) or self.hull.shape != (2,)
                or not all(np.isfinite(a).all() for a in (self.waypoints, self.arrival_radii, self.hull))):
            raise ValueError("Invalid public metadata or too many waypoints")
        self.tracker.reset()
        self._ready, self._tick, self._cached = True, None, None
        self.last_tracks, self.dropped_tracks = [], 0

    def encode(self, public):
        if not self._ready:
            raise RuntimeError("Reset encoder with public metadata first")
        tick = int(public["t"])
        if self._tick == tick:
            return {k: v.copy() for k, v in self._cached.items()}
        pose, vel, previous = (np.asarray(public[k], dtype=float) for k in ("pose", "vel", "prev_action"))
        if pose.shape != (3,) or vel.shape != (3,) or previous.shape != (2,):
            raise ValueError("Invalid public vessel state")
        wp = int(public["wp_index"])
        if not 0 <= wp < len(self.waypoints):
            raise ValueError("Invalid public waypoint index")
        distance, bearing = np.asarray(public["wp_polar"], dtype=float)
        remaining = world_to_body(self.waypoints[wp:], pose)
        state = np.array([pose[0] / 400., pose[1] / 150., math.sin(pose[2]), math.cos(pose[2]),
                          vel[0] / 2., vel[1] / 2., vel[2] / .6, previous[0] / 2., previous[1] / .6,
                          distance / 500., math.sin(bearing), math.cos(bearing), remaining[0, 0] / 100., remaining[0, 1] / 100.,
                          wp / len(self.waypoints), len(remaining) / self.max_waypoints,
                          max(0., 1. - tick / self.tick_limit), self.hull[0] / 10., self.hull[1] / 10.], dtype=np.float32)
        waypoints = np.zeros((self.max_waypoints, len(WAYPOINT_FIELDS)), np.float32)
        waypoints[:len(remaining), 0] = 1
        waypoints[:len(remaining), 1:3] = remaining / 400.
        waypoints[:len(remaining), 3] = self.arrival_radii[wp:] / 10.
        maps = self.perception.predict(public["perception"])
        tracks = self.tracker.update(maps["labels"], pose, tick, ego_velocity=vel, probabilities=maps["vessel_probability"])
        tracks.sort(key=lambda t: np.linalg.norm(t["position_body_m"]))
        self.last_tracks, self.dropped_tracks = tracks, max(0, len(tracks) - self.max_ships)
        ships = np.zeros((self.max_ships, len(SHIP_FIELDS)), np.float32)
        for i, track in enumerate(tracks[:self.max_ships]):
            axis = track["axis_heading_world_rad"] - pose[2]
            velocity = track["velocity_body_mps"] or [0., 0.]
            ships[i] = [1., track["position_body_m"][0] / 50., track["position_body_m"][1] / 50.,
                        velocity[0] / 2.5, velocity[1] / 2.5, float(track["velocity_valid"]),
                        track["length_m"] / 40., track["width_m"] / 10., math.sin(2 * axis), math.cos(2 * axis),
                        float(track["motion_state"] == "stationary"), float(track["motion_state"] == "moving"),
                        float(track["motion_state"] == "unknown"), float(track["observed"]),
                        min(1., track["missing_s"] / .5), track["segmentation_confidence"]]
        features = dict(map=pool_maps(maps), state=state, ships=ships, waypoints=waypoints)
        if not all(np.isfinite(a).all() for a in features.values()):
            raise RuntimeError("Non-finite policy features")
        self._tick, self._cached = tick, {k: v.copy() for k, v in features.items()}
        return features
