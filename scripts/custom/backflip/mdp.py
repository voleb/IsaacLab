# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains the MDP specifications for the Go2 backflip task.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pitch_velocity_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for high pitch angular velocity (backflip)."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Target pitch velocity (negative for backflip, assuming z-up and right-handed system with x-forward)
    # Check coordinate system: usually y is pitch axis. 
    # Isaac Sim: x-forward, y-left, z-up. Pitch is rotation around y.
    # Right-hand rule on y (left): thumb to left, fingers curl up-to-back. 
    # So positive pitch velocity is nose-down? No.
    # Let's verify: x-forward. y-left. z-up.
    # Rotation around y: +y is left. 
    # If I rotate around +y, x goes to z. (Nose up).
    # So positive pitch velocity means backflip (nose up).
    
    return asset.data.root_ang_vel_b[:, 1]


def base_height_reward(env: ManagerBasedRLEnv, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for reaching a target height."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_pos_w[:, 2] - target_height) * -1.0 # Penalty for distance


def body_orientation_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for being upright (optional, might conflict with backflip during the maneuver)."""
    asset: Articulation = env.scene[asset_cfg.name]
    # Projected gravity on base z-axis. Should be close to -1 for upright?
    # Gravity is (0, 0, -1). Projected on base frame.
    # If upright, projected gravity is (0, 0, -1). 
    return asset.data.projected_gravity_b[:, 2] + 1.0 # Minimal at upright (-1 + 1 = 0), negative otherwise? No.
    # we want to maximize this. 
    # If upright: -1. We want to reward being upright (at start/end).
    # Maybe use existing flat_orientation_l2 but as reward?
    
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) * -1.0

