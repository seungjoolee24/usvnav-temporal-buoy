"""Trainable actor/critic features; the RGB perception ONNX stays frozen.

Only the four public PolicyEncoder tensors enter this module. Ship pooling is
order independent; waypoint order is retained. Raw state and next waypoint
coordinates bypass the learned perception/features before the policy heads.
"""
from __future__ import annotations

import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class NavigationFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        state_size = observation_space["state"].shape[0]
        ship_size = observation_space["ships"].shape[-1]
        route_shape = observation_space["waypoints"].shape
        super().__init__(observation_space, features_dim=64 + 64 + 32 + state_size + route_shape[-1])
        self.map_net = nn.Sequential(
            nn.Conv2d(7, 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(16 * 7 * 7, 64), nn.ReLU(),
        )
        self.ship_net = nn.Sequential(nn.Linear(ship_size - 1, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.route_net = nn.Sequential(nn.Flatten(), nn.Linear(route_shape[0] * route_shape[1], 32), nn.ReLU())

    def forward(self, observations):
        ships = observations["ships"]
        present = ships[..., :1] > .5
        # Mask before and after the MLP: padding values and layer biases must
        # contribute neither to the mean nor to the maximum.
        ship_features = self.ship_net(torch.where(present, ships[..., 1:], 0.))
        count = present.sum(dim=1)
        mean = (ship_features * present).sum(dim=1) / count.clamp(min=1)
        maximum = ship_features.masked_fill(~present, torch.finfo(ship_features.dtype).min).amax(dim=1)
        maximum = torch.where(count > 0, maximum, 0.)
        waypoints = observations["waypoints"]
        waypoints = torch.where(waypoints[..., :1] > .5, waypoints, 0.)
        return torch.cat((self.map_net(observations["map"]), mean, maximum,
                          self.route_net(waypoints), observations["state"], waypoints[:, 0]), dim=1)


def policy_kwargs():
    return dict(features_extractor_class=NavigationFeatures, normalize_images=False,
                share_features_extractor=True, activation_fn=nn.Tanh,
                net_arch=dict(pi=[64, 64], vf=[64, 64]), log_std_init=-1.6)
