
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for Unitree Go2 walking task."""

import math

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
from isaaclab.terrains import (
    TerrainImporterCfg,
    TerrainGeneratorCfg,
    MeshPyramidStairsTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshRandomGridTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfInvertedPyramidSlopedTerrainCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class UnitreeGo2WalkingEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for Unitree Go2 forward walking task on rough terrain.

    The robot is trained to walk forward at a target velocity while
    maintaining stability on rough terrain.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Increase PhysX buffer sizes for multiple environments and complex terrains
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**16
        self.sim.physx.gpu_max_contact_pairs = 10 * 2**16

        # ------------------------------
        # Scene settings
        # ------------------------------
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # Expand terrain generator to include stairs, rough terrain, and slopes
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "pyramid_stairs": MeshPyramidStairsTerrainCfg(
                    proportion=0.2,
                    step_height_range=(0.05, 0.23),
                    step_width=0.3,
                    platform_width=1.0,
                    border_width=0.0,
                ),
                "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.2,
                    step_height_range=(0.05, 0.23),
                    step_width=0.3,
                    platform_width=1.0,
                    border_width=0.0,
                ),
                "boxes": MeshRandomGridTerrainCfg(
                    proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
                ),
                "random_rough": HfRandomUniformTerrainCfg(
                    proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
                ),
                "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
                    proportion=0.1, slope_range=(0.0, 0.4), border_width=0.25
                ),
                "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
                    proportion=0.1, slope_range=(0.0, 0.4), border_width=0.25
                ),
            },
        )

        # ------------------------------
        # Commands — forward walking
        # ------------------------------
        # Train to walk forward; allow small lateral and heading corrections
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.5)   # Forward velocity [m/s]
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)  # Small lateral adjustment
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)  # Yaw rate [rad/s]
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        # ------------------------------
        # Events
        # ------------------------------
        # Keep mild pushing to build robustness
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.5, 0.5),
                    "roll": (-0.5, 0.5),
                    "pitch": (-0.5, 0.5),
                    "yaw": (-0.5, 0.5),
                }
            },
        )
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        }

        # ------------------------------
        # Rewards
        # ------------------------------
        # Velocity tracking (primary objectives)
        self.rewards.track_lin_vel_xy_exp.weight = 2.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75

        # Locomotion style rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.3  # Mildly increased for better ground clearance

        # Penalty terms for smooth, efficient motion
        self.rewards.lin_vel_z_l2.weight = -0.5          # Relaxed to allow climbing
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -0.0001     # Reverted to allow more torque on rough terrain
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.flat_orientation_l2.weight = -0.5    # Relaxed to allow pitching on slopes

        # Base height — keep robot at natural standing height (~0.34 m for Go2)
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=-1.0,
            params={"target_height": 0.34},
        )

        # Penalize body/thigh contacts with the ground (not feet)
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*thigh"),
                "threshold": 1.0,
            },
        )

        # ------------------------------
        # Terminations
        # ------------------------------
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

        # ------------------------------
        # Curriculum
        # ------------------------------
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class UnitreeGo2WalkingEnvCfg_PLAY(UnitreeGo2WalkingEnvCfg):
    """Play (evaluation) configuration — fewer envs, no randomisation."""

    def __post_init__(self):
        super().__post_init__()

        # Smaller scene for visualisation
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Spawn randomly in the grid instead of by terrain difficulty
        self.scene.terrain.max_init_terrain_level = None

        # Reduce terrain size to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Disable sensor noise
        self.observations.policy.enable_corruption = False

        # Disable disturbance events
        self.events.base_external_force_torque = None
        self.events.push_robot = None
