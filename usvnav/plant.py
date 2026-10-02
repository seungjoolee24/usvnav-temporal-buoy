"""Vessel dynamics (track_1/DECISIONS.md 1-W5, amended 2026-09-29 §9.33) and the action contract (1-W6).

The command is still a *velocity setpoint* `(v, w)`: surge speed and yaw rate. Since 2026-09-29 (owner) it is the
setpoint of an autopilot driving **Track 2's public nominal plant** -- the vessel Track 2 ships in its kit
(hackaton-sim `usvsim/plant.py` and `params.json`, T2-RES-07/08): a planar 3-DOF rigid body with linear damping on
each axis, a stern azimuth thruster and a bow tunnel thruster, forward Euler at 0.05 s. Both tracks sail the same
boat. **Only the public model**: Track 2's reference vessel (`usvsim/internal/vessel.py`) is the answer to its
subproblem 2-1, and this file ships in Track 1's kit. If Track 2 changes its public parameters, change them here.

What the autopilot makes of it: yaw comes from thrust, not from a commanded rate. It turns mainly with the stern
azimuth, as an outboard does, so a turn pushes the stern sideways and the hull slips (sway is nonzero without any
drift); the stern thruster's 400 N are shared between surge and turning, so a hard turn at full speed bleeds speed
and yaw authority falls with speed; the bow thruster takes half the turning moment below 0.5 m/s (a pivot) and
35 % above it, which keeps the slip of a hard turn near ten degrees (the public model has no Coriolis term and low
sway damping: a stern-only turn slips ~35 degrees); reverse thrust is capped at 40 %; a lower speed setpoint throttles
back rather than reversing, so the hull coasts. The allocation is tuned, not derived (DECISIONS §9.33).
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import Rect, wrap

# 1-W4 / 1-W5
HULL_LENGTH = 3.0
HULL_WIDTH = 2.0
V_MIN, V_MAX = -0.5, 2.0
V_CRUISE = 1.5
W_MAX = 0.6
SLEW_V = 0.5           # per tick
SLEW_W = 0.3           # per tick
DT = 0.1               # 10 Hz


class InvalidAction(ValueError):
    """Non-finite or structurally malformed agent output (1-W6 -> `invalid_action`)."""


def sanitize(action):
    """Clip a finite out-of-range command; reject NaN/inf and bad shapes (1-W6).

    Returns the command *actually applied*, which is what `prev_action` reports (4-V15).
    """
    a = np.asarray(action, dtype=float).reshape(-1)
    if a.size != 2 or not np.all(np.isfinite(a)):
        raise InvalidAction(f"expected 2 finite values, got {action!r}")
    return np.array([min(V_MAX, max(V_MIN, a[0])),
                     min(W_MAX, max(-W_MAX, a[1]))])


# Track 2's public nominal parameters (usvsim/params.json) and thruster geometry (usvsim/plant.py).
MASS, IZZ = 250.0, 270.8
X_U, Y_V, N_R = 200.0, 300.0, 900.0
ARM, K_STERN, K_BOW, DELTA_MAX = 1.2, 400.0, 150.0, math.pi / 2
SUB_DT = 0.05
SUBSTEPS = int(round(DT / SUB_DT))

# The autopilot: feed-forward on the damping plus a proportional term, per axis.
K_U = 1.2              # 1/s, surge-speed error gain
K_R = 4.0              # 1/s, yaw-rate error gain
BOW_BELOW = 0.5        # m/s, below this the bow thruster takes half the turning moment (a pivot)
BOW_SHARE = 0.35       # above it, the bow thruster's share: the net side force is (1 - 2 x share) of a stern-only turn
REVERSE_MAX = 0.4      # an outboard's reverse is weaker than its ahead thrust


def thruster_map(T_stern, delta, T_bow):
    fy_stern = K_STERN * T_stern * math.sin(delta)
    fy_bow = K_BOW * T_bow
    return K_STERN * T_stern * math.cos(delta), fy_stern + fy_bow, ARM * (fy_bow - fy_stern)


#: The plant's own damping time constants (mass / X_u, izz / N_r). Not the closed loop's response: the
#: autopilot's gains act on top of them. Kept for agents that want a first-order approximation.
TAU_SURGE = MASS / X_U     # 1.25 s
TAU_YAW = IZZ / N_R        # 0.30 s


class Vessel:
    """World pose, over-ground body velocity `(u, v, r)` (4-V15/1-W7), the slew-limited setpoints in force, and the
    plant's own through-water body velocities `u_tw`, `v_tw`."""

    def __init__(self, x: float, y: float, psi: float, u: float = 0.0, v: float = 0.0, r: float = 0.0):
        # `u, v, r` let a recorded state be rebuilt (studio.observe); with no drift they are also the through-water state.
        self.x, self.y, self.psi = float(x), float(y), float(psi)
        self.u, self.v, self.r = float(u), float(v), float(r)          # reported: over ground, body axes
        self.u_tw, self.v_tw = self.u, self.v                       # the plant's own (through-water) body velocities
        self.v_set = self.w_set = 0.0
        self.last_command = (0.0, 0.0, 0.0)     # (T_stern, delta, T_bow), for figures

    def hull(self) -> Rect:
        return Rect(self.x, self.y, HULL_LENGTH, HULL_WIDTH, self.psi)

    def autopilot(self):
        fx = X_U * self.v_set + MASS * K_U * (self.v_set - self.u_tw)
        if self.v_set >= 0.0 and fx < 0.0:           # slowing down is throttling back, not reversing; reverse needs v < 0
            fx = 0.0
        n = N_R * self.w_set + IZZ * K_R * (self.w_set - self.r)
        share = 0.5 if abs(self.u_tw) < BOW_BELOW else BOW_SHARE
        T_bow = float(np.clip(share * n / (ARM * K_BOW), -1.0, 1.0))
        s = -(n - ARM * K_BOW * T_bow) / ARM          # stern lateral force that gives the rest of the moment
        c = fx
        mag = math.hypot(c, s)
        if mag > K_STERN:                            # the stern thruster saturates: surge and turn share it
            c, s = c * K_STERN / mag, s * K_STERN / mag
            mag = K_STERN
        sgn = 1.0 if c >= 0.0 else -1.0
        if sgn < 0.0 and mag > REVERSE_MAX * K_STERN:  # reverse is capped lower
            c, s = c * REVERSE_MAX * K_STERN / mag, s * REVERSE_MAX * K_STERN / mag
            mag = REVERSE_MAX * K_STERN
        T_stern = sgn * mag / K_STERN
        delta = math.atan2(s * sgn, abs(c)) if mag > 0.0 else 0.0
        return T_stern, float(np.clip(delta, -DELTA_MAX, DELTA_MAX)), T_bow

    def step(self, applied, drift_world=(0.0, 0.0)):
        v_cmd, w_cmd = float(applied[0]), float(applied[1])
        self.v_set += max(-SLEW_V, min(SLEW_V, v_cmd - self.v_set))
        self.w_set += max(-SLEW_W, min(SLEW_W, w_cmd - self.w_set))
        for _ in range(SUBSTEPS):
            cmd = self.autopilot()
            self.last_command = cmd
            fx, fy, n = thruster_map(*cmd)
            c, s = math.cos(self.psi), math.sin(self.psi)
            self.x += (self.u_tw * c - self.v_tw * s + drift_world[0]) * SUB_DT
            self.y += (self.u_tw * s + self.v_tw * c + drift_world[1]) * SUB_DT
            self.psi = wrap(self.psi + self.r * SUB_DT)
            self.u_tw += (fx - X_U * self.u_tw) / MASS * SUB_DT
            self.v_tw += (fy - Y_V * self.v_tw) / MASS * SUB_DT
            self.r += (n - N_R * self.r) / IZZ * SUB_DT
        c, s = math.cos(self.psi), math.sin(self.psi)
        self.u = self.u_tw + drift_world[0] * c + drift_world[1] * s
        self.v = self.v_tw - drift_world[0] * s + drift_world[1] * c
        return self


class OUDisturbance:
    """Ornstein-Uhlenbeck drift in the world frame (4-V10): tau = 20 s, bounded so the
    induced steady lateral speed stays at or below `cap` m/s."""

    TAU = 20.0
    CAP = 0.35

    def __init__(self, seed: int, cap: float | None = None):
        self.rng = np.random.default_rng(seed)
        self.cap = self.CAP if cap is None else cap
        self.state = np.zeros(2)
        self.sigma = self.cap * math.sqrt(2.0 / self.TAU) * 0.5

    def step(self):
        a = math.exp(-DT / self.TAU)
        noise = self.rng.normal(0.0, self.sigma * math.sqrt(1.0 - a * a) * math.sqrt(self.TAU / 2.0), 2)
        self.state = a * self.state + noise
        n = float(np.linalg.norm(self.state))
        if n > self.cap:
            self.state *= self.cap / n
        return self.state.copy()


class NoDisturbance:
    def step(self):
        return np.zeros(2)
