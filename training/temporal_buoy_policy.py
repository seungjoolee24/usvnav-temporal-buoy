"""Sliding-window visual navigation with public ego motion and direct goals.

Four native bow-up RGB frames are read oldest to newest using one shared CNN.
Past buoy probabilities are warped into the newest vessel frame using only the
public pose history. A GRUCell combines aligned spatial and scene features.
Hidden state starts at zero for EACH observed window: standard PPO minibatches
need no cross-episode recurrent state. Only past/present observations are used.

The old single-frame policy remains loadable under its original class name.
Actor weights are not interchangeable; RGB stem/head can be transferred.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from training.buoy_policy import BuoyNavigationFeatures, SPATIAL_GRID


TEMPORAL_SIZE = 64
HEAT_TOKEN_SIZE = 64
# In the 100x100 one-metre buoy grid: 35m ahead to 10m behind,
# 20m to each side. The scene branch still reads the entire native view.
LOCAL_ROWS = slice(15, 60)
LOCAL_COLS = slice(30, 70)


def ego_alignment(history):
    """Return current-view -> past-view affine grids and public motion tokens.

    history=[world_x/400,world_y/150,sin(heading),cos(heading),
             surge/2,sway/2,yaw_rate/.6,age_seconds/10].
    Bow-up raster: row=origin-forward/mpp, col=origin-port/mpp.
    affine_grid coordinates are [column,row], so both axis signs matter.
    """
    if history.ndim != 3 or history.shape[-1] != 8:
        raise ValueError('Expected public history [B,T,8]')
    xy = history[..., :2] * history.new_tensor([400., 150.])
    sine, cosine = history[..., 2], history[..., 3]
    current_sine, current_cosine = sine[:, -1:], cosine[:, -1:]
    cosine_delta = current_cosine * cosine + current_sine * sine
    sine_delta = current_sine * cosine - current_cosine * sine
    delta = xy[:, -1:, :] - xy
    tx = cosine * delta[..., 0] + sine * delta[..., 1]
    ty = -sine * delta[..., 0] + cosine * delta[..., 1]
    theta = history.new_zeros((*history.shape[:2], 2, 3))
    theta[..., 0, 0], theta[..., 0, 1] = cosine_delta, sine_delta
    theta[..., 1, 0], theta[..., 1, 1] = -sine_delta, cosine_delta
    theta[..., 0, 2], theta[..., 1, 2] = -ty / 50., -tx / 50.
    # Older pose relative to current vessel, rather than absolute world position.
    back = -delta
    forward = current_cosine * back[..., 0] + current_sine * back[..., 1]
    port = -current_sine * back[..., 0] + current_cosine * back[..., 1]
    motion = torch.stack([forward / 20., port / 20., -sine_delta, cosine_delta,
                          history[..., 4], history[..., 5], history[..., 6],
                          history[..., 7] * 5.], dim=-1)
    return theta, motion


def align_buoy_history(probability, history):
    """Warp [B,T,1,100,100] with public motion, keeping gradients to all frames."""
    if probability.ndim != 5 or probability.shape[2:] != (1, 100, 100):
        raise ValueError('Expected buoy history [B,T,1,100,100]')
    if probability.shape[:2] != history.shape[:2]:
        raise ValueError('RGB and public history lengths must match')
    batch, frames = probability.shape[:2]
    theta, motion = ego_alignment(history)
    flat = probability.reshape(batch * frames, 1, 100, 100)
    grid = F.affine_grid(theta.reshape(-1, 2, 3), flat.shape, align_corners=False)
    aligned = F.grid_sample(flat, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return aligned.reshape(batch, frames, 1, 100, 100), motion


class TemporalBuoyNavigationFeatures(BuoyNavigationFeatures):
    """Shared frame CNN -> ego-aligned local heatmaps -> window GRU -> PPO."""

    def __init__(self, observation_space):
        super().__init__(observation_space)
        self.frames = observation_space['rgb'].shape[0] // 3
        if self.frames < 2 or observation_space['history'].shape != (self.frames, 8):
            raise ValueError('Need at least two RGB frames and one public history row per frame')
        self._features_dim += TEMPORAL_SIZE
        self.heat_token = nn.Sequential(nn.Linear(SPATIAL_GRID ** 2, HEAT_TOKEN_SIZE), nn.ReLU())
        self.temporal_cell = nn.GRUCell(64 + HEAT_TOKEN_SIZE + 8, TEMPORAL_SIZE)

    def forward(self, obs):
        rgb = obs['rgb']
        if rgb.ndim != 4 or rgb.shape[1:] != (self.frames * 3, 200, 200):
            raise ValueError('Need the complete chronological RGB window')
        if not rgb.is_floating_point():
            raise TypeError('SB3 must normalize RGB exactly once before the extractor')
        batch = rgb.shape[0]
        image = self.rgb_encoder(rgb.reshape(batch * self.frames, 3, 200, 200))
        logits = self.buoy_head(image).reshape(batch, self.frames, 1, 100, 100)
        self.latest_buoy_logits = logits[:, -1]
        aligned, motion = align_buoy_history(logits.sigmoid(), obs['history'])
        spatial = self.heatmap_pool(aligned.reshape(-1, 1, 100, 100)[..., LOCAL_ROWS, LOCAL_COLS])
        spatial = spatial.reshape(batch, self.frames, SPATIAL_GRID ** 2)
        scene = self.scene_encoder(image).reshape(batch, self.frames, 64)
        tokens = torch.cat([scene, self.heat_token(spatial), motion], dim=-1)
        # Reset repeats the first image. Ignore duplicate timestamps rather than
        # presenting those copies as four genuine observations of a trajectory.
        ages = obs['history'][..., 7]
        valid = torch.ones((batch, self.frames), dtype=torch.bool, device=rgb.device)
        valid[:, :-1] = (ages[:, :-1] - ages[:, 1:]).abs() > 1e-6
        memory = rgb.new_zeros((batch, TEMPORAL_SIZE))
        for index in range(self.frames):
            candidate = self.temporal_cell(tokens[:, index], memory)
            memory = torch.where(valid[:, index, None], candidate, memory)
        route = torch.where(obs['waypoints'][..., :1] > .5, obs['waypoints'], 0.)
        return torch.cat([scene[:, -1], spatial[:, -1], memory, self.route_encoder(route),
                          obs['state'], route[:, 0], obs['goal'], obs['reference'],
                          obs['history'].flatten(1)], dim=1)


def temporal_buoy_policy_kwargs(*, log_std_init=-3.):
    return dict(features_extractor_class=TemporalBuoyNavigationFeatures,
                normalize_images=True, share_features_extractor=True,
                net_arch=dict(pi=[64, 64], vf=[64, 64]),
                log_std_init=log_std_init, activation_fn=nn.Tanh)
