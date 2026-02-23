# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Keyboard teleop control for the Unitree Go2 robot in Isaac Lab.

Key bindings (ROS teleop-style):
    Arrow Up    / Numpad 8  : Move forward
    Arrow Down  / Numpad 2  : Move backward
    Arrow Left  / Numpad 4  : Move left (strafe)
    Arrow Right / Numpad 6  : Move right (strafe)
    Z           / Numpad 7  : Rotate left (yaw +)
    X           / Numpad 9  : Rotate right (yaw -)
    L                       : Stop / reset velocities

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/custom/go2_control/go2.py

"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleop control for the Unitree Go2 robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


# ── Velocity command limits ──────────────────────────────────────────────────
V_X_MIN,  V_X_MAX  = -1.0,  1.0   # m/s forward / backward
V_Y_MIN,  V_Y_MAX  = -0.5,  0.5   # m/s lateral
W_Z_MIN,  W_Z_MAX  = -1.0,  1.0   # rad/s yaw

# Gait parameters (simple trot – sinusoidal joint offsets driven by velocity)
GAIT_FREQ   = 2.0   # Hz
GAIT_SCALE  = 0.3   # joint-space amplitude (rad)


def design_scene() -> tuple[dict, list[float]]:
    """Build the scene: ground + light + Go2."""
    # Ground plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Light
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Go2 robot
    robot_cfg = UNITREE_GO2_CFG.replace(prim_path="/World/Robot")
    robot_cfg.init_state.pos = (0.0, 0.0, 0.5)
    robot = Articulation(robot_cfg)

    return {"go2": robot}, [0.0, 0.0, 0.0]


def compute_gait_joints(
    default_pos: torch.Tensor,
    vel_cmd: torch.Tensor,
    sim_time: float,
) -> torch.Tensor:
    """
    Generate a simple sinusoidal trot motion modulated by the command velocity.
    """
    v_x      = float(vel_cmd[0, 0])
    v_y      = float(vel_cmd[0, 1])
    omega_z  = float(vel_cmd[0, 2])

    speed = math.sqrt(v_x**2 + v_y**2 + (omega_z*0.3)**2)
    if speed < 0.01:
        return default_pos.clone()

    dir_x = 1.0 if v_x >= 0 else -1.0
    amp_thigh = min(speed * GAIT_SCALE, GAIT_SCALE)
    amp_calf  = amp_thigh * 1.5  # Lift leg higher during swing

    phase = 2.0 * math.pi * GAIT_FREQ * sim_time * dir_x
    
    # Pair A: FL (0,1,2), RR (9,10,11)
    phase_a = phase
    # Pair B: FR (3,4,5), RL (6,7,8)
    phase_b = phase + math.pi

    # Thigh: cos(phase). Moving - -> + is stance (pushing backward).
    thigh_a = -amp_thigh * math.cos(phase_a)
    thigh_b = -amp_thigh * math.cos(phase_b)

    # Calf: lift during swing phase (-sin > 0). Calf needs negative offset to lift foot.
    calf_a = -amp_calf * max(0.0, -math.sin(phase_a))
    calf_b = -amp_calf * max(0.0, -math.sin(phase_b))

    joint_offsets = default_pos.clone()

    # Apply offsets
    joint_offsets[0, 1]  += thigh_a
    joint_offsets[0, 10] += thigh_a
    joint_offsets[0, 4]  += thigh_b
    joint_offsets[0, 7]  += thigh_b

    joint_offsets[0, 2]  += calf_a
    joint_offsets[0, 11] += calf_a
    joint_offsets[0, 5]  += calf_b
    joint_offsets[0, 8]  += calf_b

    # Add very basic yaw/lateral via hips
    lat_yaw = v_y * 0.2 + omega_z * 0.1
    joint_offsets[0, 0] += lat_yaw
    joint_offsets[0, 9] += lat_yaw
    joint_offsets[0, 3] -= lat_yaw
    joint_offsets[0, 6] -= lat_yaw

    return joint_offsets


def run_simulator(
    sim: sim_utils.SimulationContext,
    entities: dict[str, Articulation],
    origin: list[float],
    keyboard: Se2Keyboard,
):
    """Main simulation loop."""
    sim_dt   = sim.get_physics_dt()
    sim_time = 0.0
    count    = 0
    robot    = entities["go2"]

    print("\n" + str(keyboard) + "\n")
    print("[INFO]: Simulation running. Use keys above to control the Go2.")
    print("[INFO]: Close the window or press Ctrl+C to quit.\n")

    while simulation_app.is_running():
        # ── Read keyboard ────────────────────────────────────────────────────
        raw_cmd = keyboard.advance()          # shape: (3,)  [vx, vy, wz]

        # clamp and reshape to (1, 3) for batch dimension
        vx   = float(raw_cmd[0].clamp(V_X_MIN, V_X_MAX))
        vy   = float(raw_cmd[1].clamp(V_Y_MIN, V_Y_MAX))
        wz   = float(raw_cmd[2].clamp(W_Z_MIN, W_Z_MAX))
        vel_cmd = torch.tensor([[vx, vy, wz]], device=sim.device)
        
        # Debug print so user knows keyboard is responding
        if count % 20 == 0 and (vx != 0 or vy != 0 or wz != 0):
            print(f"[INFO]: Capturing command -> vx: {vx:.2f}, vy: {vy:.2f}, wz: {wz:.2f}")

        # ── Reset every N steps ──────────────────────────────────────────────
        if count % 1000 == 0:
            keyboard.reset()
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += torch.tensor(origin, device=sim.device)
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos, joint_vel = (
                robot.data.default_joint_pos.clone(),
                robot.data.default_joint_vel.clone(),
            )
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            sim_time = 0.0
            count    = 0
            print("[INFO]: Resetting robot...")

        # ── Compute gait joint targets ────────────────────────────────────
        default_pos = robot.data.default_joint_pos.clone()
        joint_targets = compute_gait_joints(default_pos, vel_cmd, sim_time)

        # ── Apply actions ─────────────────────────────────────────────────
        robot.set_joint_position_target(joint_targets)
        robot.write_data_to_sim()

        # ── Step simulation ───────────────────────────────────────────────
        sim.step()
        sim_time += sim_dt
        count    += 1
        robot.update(sim_dt)


def main():
    """Entry point."""
    # Simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    sim.set_camera_view(eye=[3.0, 3.0, 2.5], target=[0.0, 0.0, 0.0])

    # Scene
    scene_entities, origin = design_scene()
    sim.reset()

    # Keyboard device (Se2Keyboard = ROS teleop-style SE(2) velocity commands)
    keyboard_cfg = Se2KeyboardCfg(
        v_x_sensitivity=0.8,
        v_y_sensitivity=0.4,
        omega_z_sensitivity=1.0,
        sim_device=sim.device,
    )
    keyboard = keyboard_cfg.class_type(keyboard_cfg)

    # Add 'r' key callback to reset manually
    keyboard.add_callback("R", lambda: print("[INFO]: Manual reset requested (will apply on next cycle)"))

    print("[INFO]: Setup complete. Starting simulation...")
    run_simulator(sim, scene_entities, origin, keyboard)


if __name__ == "__main__":
    main()
    simulation_app.close()
