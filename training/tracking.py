"""RGB vessel instances and temporal tracking; NumPy only, no simulator state.

Inputs are segmentation, the public ego pose/velocity and control tick. Estimate
velocity over ground in world axes, then express it in the ego axes when needed.
Unobserved, clipped and immature tracks keep an unknown motion state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

TICK_SECONDS = 0.1
METRES_PER_PIXEL = 0.5


def body_to_world(points, pose):
    points, pose = np.asarray(points, dtype=float), np.asarray(pose, dtype=float)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return points @ np.array([[c, s], [-s, c]]) + pose[:2]


def world_to_body(points, pose):
    points, pose = np.asarray(points, dtype=float), np.asarray(pose, dtype=float)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return (points - pose[:2]) @ np.array([[c, -s], [s, c]])


def pixel_to_world(rows, cols, pose):
    body = np.column_stack([(100.0 - (np.asarray(rows) + .5)) * METRES_PER_PIXEL,
                            (100.0 - (np.asarray(cols) + .5)) * METRES_PER_PIXEL])
    return body_to_world(body, pose)


def world_to_pixel(points, pose):
    body = world_to_body(points, pose)
    return np.column_stack([100.0 - body[:, 0] / METRES_PER_PIXEL,
                           100.0 - body[:, 1] / METRES_PER_PIXEL])  # row, col


@dataclass
class Detection:
    rows: np.ndarray
    cols: np.ndarray
    position: np.ndarray
    length: float
    width: float
    axis_heading: float
    truncated: bool
    confidence: float
    track_id: int | None = None

    @property
    def area(self):
        return len(self.rows) * METRES_PER_PIXEL ** 2


def vessel_detections(labels, pose, vessel_class, *, probabilities=None, min_pixels=6):
    """8-connected components, full pixel-centre transform and a PCA extent.

    Touching vessels can merge; components are not guaranteed object identities.
    Clipped component centroids must never supervise a velocity estimate.
    """
    labels = np.asarray(labels)
    if labels.shape != (200, 200):
        raise ValueError("Expected a 200x200 segmentation")
    if probabilities is not None and np.shape(probabilities) != labels.shape:
        raise ValueError("Expected a 200x200 vessel probability map")
    mask = labels == vessel_class
    seen = np.zeros_like(mask)
    detections = []
    for start_row, start_col in zip(*np.nonzero(mask)):
        if seen[start_row, start_col]:
            continue
        stack, pixels = [(int(start_row), int(start_col))], []
        seen[start_row, start_col] = True
        while stack:
            r, c = stack.pop()
            pixels.append((r, c))
            for nr in range(max(0, r - 1), min(200, r + 2)):
                for nc in range(max(0, c - 1), min(200, c + 2)):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
        if len(pixels) < min_pixels:
            continue
        rows, cols = np.asarray(pixels).T
        points = pixel_to_world(rows, cols, pose)
        position = points.mean(axis=0)
        centred = points - position
        _, axes = np.linalg.eigh(centred.T @ centred / len(points))
        axis = axes[:, -1]
        normal = np.array([-axis[1], axis[0]])
        projected = centred @ np.column_stack([axis, normal])
        # Each square pixel has a projected width between 0.5 and sqrt(2)*0.5 m.
        body_axes = world_to_body(np.array([pose[:2] + axis, pose[:2] + normal]), pose)
        pixel_widths = np.abs(body_axes).sum(axis=1) * METRES_PER_PIXEL
        lengths = np.ptp(projected, axis=0) + pixel_widths
        confidence = float(np.mean(probabilities[rows, cols])) if probabilities is not None else 1.0
        detections.append(Detection(rows, cols, position, float(lengths[0]), float(lengths[1]),
                                    math.atan2(axis[1], axis[0]) % math.pi,
                                    bool(np.any((rows == 0) | (rows == 199) | (cols == 0) | (cols == 199))), confidence))
    return detections


def minimum_assignment(cost):
    """Hungarian assignment for finite N x M costs, N <= M; deterministic ties."""
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or cost.shape[0] > cost.shape[1] or not np.isfinite(cost).all():
        raise ValueError("Assignment needs finite costs with rows <= columns")
    n, m = cost.shape
    if n == 0:
        return []
    u, v, p, way = np.zeros(n + 1), np.zeros(m + 1), np.zeros(m + 1, dtype=int), np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        distances, used = np.full(m + 1, np.inf), np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = np.inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    value = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if value < distances[j]:
                        distances[j], way[j] = value, j0
                    if distances[j] < delta:
                        delta, j1 = distances[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    distances[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j]]


@dataclass
class _Track:
    identity: int
    detection: Detection
    first_time: float
    last_time: float
    samples: list = field(default_factory=list)
    hits: int = 0
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    velocity: np.ndarray | None = None
    velocity_sigma: float | None = None


class VesselTracker:
    def __init__(self, classes, *, history_s=2.0, max_missing_s=.5, position_noise_m=.15,
                 motion_threshold_mps=.25, min_motion_span_s=1.2, min_pixels=6):
        self.classes = tuple(classes)
        self.vessel_class = self.classes.index("vessel")
        if (not 0 < min_motion_span_s <= history_s or max_missing_s <= 0 or position_noise_m <= 0
                or min_pixels < 1 or motion_threshold_mps <= 0):
            raise ValueError("Invalid tracker configuration")
        self.history_s, self.max_missing_s, self.position_noise_m = history_s, max_missing_s, position_noise_m
        self.motion_threshold_mps, self.min_motion_span_s, self.min_pixels = motion_threshold_mps, min_motion_span_s, min_pixels
        self.reset()

    def reset(self):
        self._tracks = {}
        self._next_id, self._last_tick = 1, None
        self.last_detections = []

    def _estimate(self, track, now):
        track.samples = [(t, p) for t, p in track.samples if t >= now - self.history_s - 1e-9]
        track.velocity, track.velocity_sigma = None, None
        if len(track.samples) < 5:
            return
        times = np.array([t for t, _ in track.samples])
        if times[-1] - times[0] < self.min_motion_span_s - 1e-9:
            return
        points = np.array([p for _, p in track.samples])
        centred_times = times - times.mean()
        denominator = float(centred_times @ centred_times)
        velocity = (centred_times[:, None] * (points - points.mean(axis=0))).sum(axis=0) / denominator
        residual = points - (points.mean(axis=0) + centred_times[:, None] * velocity)
        residual_sigma = math.sqrt(float(np.max((residual * residual).sum(axis=0)) / max(1, len(times) - 2)))
        track.velocity = velocity
        track.velocity_sigma = max(self.position_noise_m, residual_sigma) / math.sqrt(denominator)
        # Position estimate at the latest frame; reduces centroid quantization noise.
        track.position = points.mean(axis=0) + (now - times.mean()) * velocity

    def update(self, labels, pose, tick, *, ego_velocity=None, probabilities=None):
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all() or int(tick) != tick or tick < 0:
            raise ValueError("Need a finite public pose and nonnegative integer tick")
        tick = int(tick)
        if self._last_tick is not None and tick <= self._last_tick:
            raise ValueError("Ticks must increase; call reset for a new episode")
        if ego_velocity is not None:
            ego_velocity = np.asarray(ego_velocity, dtype=float)
            if ego_velocity.shape != (3,) or not np.isfinite(ego_velocity).all():
                raise ValueError("Expected finite public body velocity (u, v, r)")
        self._last_tick = tick
        now = tick * TICK_SECONDS
        self._tracks = {i: t for i, t in self._tracks.items() if now - t.last_time <= self.max_missing_s + 1e-9}
        detections = vessel_detections(labels, pose, self.vessel_class, probabilities=probabilities, min_pixels=self.min_pixels)
        old = list(self._tracks.values())
        # A separate dummy column for each track permits globally optimal unmatched choices.
        costs = np.full((len(old), len(detections) + len(old)), 4.0)
        for i, track in enumerate(old):
            predicted = track.position + (track.velocity if track.velocity is not None else np.zeros(2)) * (now - track.last_time)
            for j, detection in enumerate(detections):
                distance = float(np.linalg.norm(predicted - detection.position))
                gate = 2.0 + 3.0 * (now - track.last_time)
                ratio = detection.area / max(track.detection.area, .25)
                if distance <= gate and .45 <= ratio <= 2.2:
                    costs[i, j] = distance + .5 * abs(math.log(ratio))
                else:
                    costs[i, j] = 1e6
        matched = set()
        for i, j in minimum_assignment(costs):
            if j >= len(detections) or costs[i, j] >= 4.0:
                continue
            track, detection = old[i], detections[j]
            matched.add(j)
            track.detection, track.last_time, track.hits = detection, now, track.hits + 1
            detection.track_id = track.identity
            if not detection.truncated:
                track.position = detection.position.copy()
                track.samples.append((now, detection.position.copy()))
                self._estimate(track, now)
            else:
                # Discard a clipped centroid's velocity; do not declare it stationary.
                track.samples.clear()
                track.position = detection.position.copy()
                track.velocity = track.velocity_sigma = None
        for j, detection in enumerate(detections):
            if j in matched:
                continue
            identity = self._next_id
            self._next_id += 1
            detection.track_id = identity
            samples = [] if detection.truncated else [(now, detection.position.copy())]
            self._tracks[identity] = _Track(identity, detection, now, now, samples, 1, detection.position.copy())
        self.last_detections = detections
        outputs = []
        for track in self._tracks.values():
            observed = abs(track.last_time - now) < 1e-9
            velocity_valid = observed and track.velocity is not None and not track.detection.truncated
            speed = float(np.linalg.norm(track.velocity)) if velocity_valid else None
            motion = "unknown"
            if velocity_valid:
                lower, upper = max(0.0, speed - 2 * track.velocity_sigma), speed + 2 * track.velocity_sigma
                if lower > self.motion_threshold_mps:
                    motion = "moving"
                elif upper < self.motion_threshold_mps:
                    motion = "stationary"
            position = track.position + (track.velocity if track.velocity is not None else np.zeros(2)) * (now - track.last_time)
            ground_body = world_to_body(np.array([pose[:2] + track.velocity]), pose)[0] if velocity_valid else None
            relative_body = ground_body - np.asarray(ego_velocity)[:2] if velocity_valid and ego_velocity is not None else None
            position_body = world_to_body(position[None], pose)[0]
            # A rotating body's coordinate derivative also includes -yaw_rate * J * position.
            position_rate_body = (relative_body + ego_velocity[2] * np.array([position_body[1], -position_body[0]])
                                  if relative_body is not None else None)
            outputs.append({
                "track_id": track.identity, "position_world_m": position.tolist(),
                "position_body_m": position_body.tolist(),
                "velocity_world_mps": track.velocity.tolist() if velocity_valid else None,
                "velocity_body_mps": ground_body.tolist() if ground_body is not None else None,
                "relative_velocity_body_mps": relative_body.tolist() if relative_body is not None else None,
                "position_rate_body_mps": position_rate_body.tolist() if position_rate_body is not None else None,
                "velocity_valid": velocity_valid, "velocity_sigma_mps": track.velocity_sigma if velocity_valid else None,
                "speed_mps": speed, "motion_state": motion, "observed": observed,
                "age_s": now - track.first_time, "missing_s": now - track.last_time, "hits": track.hits,
                "length_m": track.detection.length, "width_m": track.detection.width,
                "axis_heading_world_rad": track.detection.axis_heading,
                "truncated": track.detection.truncated, "segmentation_confidence": track.detection.confidence,
            })
        return outputs
