"""Jointly learned image history representation and residual actor/critic."""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class RgbNavigationFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        history_size = int(torch.tensor(observation_space["history"].shape).prod())
        super().__init__(observation_space, features_dim=64 + 32 + 19 + 4 + 4 + 2 + history_size)
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(observation_space["rgb"].shape[0], 4, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(4, 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5)), nn.Flatten(), nn.Linear(400, 64), nn.ReLU(),
        )
        self.route_encoder = nn.Sequential(nn.Flatten(), nn.Linear(128, 32), nn.ReLU())

    def forward(self, obs):
        route = torch.where(obs["waypoints"][..., :1] > .5, obs["waypoints"], 0.)
        # SB3 has already divided the uint8 RGB by 255 exactly once.
        return torch.cat([self.rgb_encoder(obs["rgb"]), self.route_encoder(route), obs["state"],
                          route[:, 0], obs["goal"], obs["reference"], obs["history"].flatten(1)], dim=1)


def rgb_policy_kwargs():
    return dict(features_extractor_class=RgbNavigationFeatures, normalize_images=True,
                share_features_extractor=True, net_arch=dict(pi=[64, 64], vf=[64, 64]),
                log_std_init=-2.8, activation_fn=nn.Tanh)
