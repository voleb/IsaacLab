# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to control the Unitree Go2 robot.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/custom/go2.py

"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to control the Unitree Go2 robot.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Robot
    origin = [0.0, 0.0, 0.0]
    # Create the robot articulation
    robot_cfg = UNITREE_GO2_CFG.replace(prim_path="/World/Robot")
    robot_cfg.init_state.pos = (0.0, 0.0, 0.6)  # Start a bit high to drop
    robot = Articulation(robot_cfg)

    # return the scene information
    scene_entities = {
        "go2": robot,
    }
    return scene_entities, origin


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origin: list[float]):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    robot = entities["go2"]

    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset robot
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += torch.tensor(origin, device=sim.device)
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            
            # reset joints
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            
            # reset the internal state
            robot.reset()
            print("[INFO]: Resetting robot state...")

        # apply default actions (standing)
        # generate random joint positions around default
        # joint_pos_target = robot.data.default_joint_pos + torch.randn_like(robot.data.joint_pos) * 0.1
        joint_pos_target = robot.data.default_joint_pos.clone()
        
        # apply action to the robot
        robot.set_joint_position_target(joint_pos_target)
        # write data to sim
        robot.write_data_to_sim()
        
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        robot.update(sim_dt)


def main():
    """Main function."""

    # Initialize the simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    # Set main camera
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_entities, origin = design_scene()
    
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, origin)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
