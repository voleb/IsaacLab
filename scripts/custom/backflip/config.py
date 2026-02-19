# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab.utils import configclass

import os
import sys

# Add the scripts directory to sys.path
scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg, UnitreeGo2RoughEnvCfg_PLAY
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg

import scripts.custom.backflip.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp_std

@configclass
class UnitreeGo2BackflipEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # 1. Disable velocity commands (we want the robot to learn a specific motion)
        self.commands.base_velocity = None
        self.actions.joint_pos.scale = 0.5 # Increase control authority?
        
        # 2. Change Terrain to Flat (easier for learning acrobatics first)
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # 3. Custom Rewards
        # Remove tracking rewards
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.lin_vel_z_l2 = None
        self.rewards.ang_vel_xy_l2 = None # We want angular velocity!
        self.rewards.feet_air_time = None # Depends on base_velocity command

        # Remove velocity command observation
        self.observations.policy.velocity_commands = None
        
        # Add backflip rewards
        self.rewards.pitch_velocity = RewTerm(
            func=mdp.pitch_velocity_reward,
            weight=1.0,
        )
        # Encouraging jump height
        # self.rewards.jump_height = RewTerm(
        #     func=mdp.base_height_reward,
        #     weight=0.5,
        #     params={"target_height": 1.0}, # Target 1m height?
        # )

        # Add penalty for falling sideways (roll)
        # self.rewards.no_roll = RewTerm(
        #     func=mdp_std.ang_vel_xy_l2, # This penalizes pitch and roll. We only want to penalize roll.
        #     weight=0.0
        # )
        
        # Terminations
        # Make termination harder so it tries longer?
        # Or easier so it fails fast?
        self.terminations.base_contact = None # Allow body contact during roll? Maybe not.
        
        self.episode_length_s = 5.0 # Short episodes for backflip

@configclass
class UnitreeGo2BackflipPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_backflip"
        self.max_iterations = 2000
        # Adjust PPO params if needed?


@configclass
class UnitreeGo2BackflipEnvCfg_PLAY(UnitreeGo2BackflipEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
