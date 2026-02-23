
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

# Import custom terrain
try:
    from .terrains import HfRailwayOnHillTerrainCfg, railway_on_hill_terrain
except ImportError:
    from terrains import HfRailwayOnHillTerrainCfg, railway_on_hill_terrain

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

@configclass
class UnitreeGo2RailwayEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------
        # Scene settings
        # ------------------------------
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        
        # Configure Terrain
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=10, # Fewer columns for testing
            horizontal_scale=0.05, # Finer resolution for rails (5cm)
            vertical_scale=0.005,
            slope_threshold=0.75,
            difficulty_range=(0.0, 1.0), # Enable difficulty scaling
            use_cache=False,
            sub_terrains={
                "railway": HfRailwayOnHillTerrainCfg(
                    function=railway_on_hill_terrain,
                    proportion=1.0,
                    rail_height=0.174,
                    rail_width=0.145,
                    rail_head_width=0.065,
                    track_gauge=1.435,
                    hill_height=0.3, # Start with a smaller hill
                    hill_width=4.0,
                    slope_width=2.0,
                    noise_range=(0.0, 0.02),
                )
            },
        )

        # ------------------------------
        # Commands
        # ------------------------------
        # Restrict to lateral movement (side-stepping) towards the positive rail
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0) # No forward/backward
        self.commands.base_velocity.ranges.lin_vel_y = (0.3, 0.6) # Force side stepping towards rail
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0) # No turning
        self.commands.base_velocity.ranges.heading = (0.0, 0.0) # Face forward relative to rail
        
        # ------------------------------
        # Events
        # ------------------------------
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-2.0, 2.0), "y": (0.2, 0.4), "yaw": (-0.1, 0.1)}, # Start next to the rail
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # ------------------------------
        # Rewards
        # ------------------------------
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.5 # Encourage picking up feet high
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0, # Penalize chassis/thigh collision
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*thigh"), "threshold": 1.0},
        )
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=0.0, # Will set positive below
            params={"target_height": 0.40},
        )
        self.rewards.base_height_l2.weight = 2.0
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7
        
        # Add penalty for non-flat orientation (keep body level)
        self.rewards.flat_orientation_l2.weight = -0.5
        
        # ------------------------------
        # Terminations
        # ------------------------------
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class UnitreeGo2RailwayEnvCfg_PLAY(UnitreeGo2RailwayEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
