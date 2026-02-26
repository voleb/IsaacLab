
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play (evaluate) a trained Go2 walking RL checkpoint."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# Add rsl_rl scripts to path
rsl_rl_scripts_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../reinforcement_learning/rsl_rl")
)
sys.path.append(rsl_rl_scripts_path)
import cli_args  # noqa: E402

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play a trained Go2 walking policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Rough-Go2-Walking-v0",
    help="Registered gym task name.",
)
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports (after sim launch)
# ---------------------------------------------------------------------------
import gymnasium as gym
import torch
import yaml
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    RslRlOnPolicyRunnerCfg,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.utils import configclass

# Local env config
try:
    from env_cfg import UnitreeGo2WalkingEnvCfg_PLAY
except ImportError:
    from scripts.custom.go2_walking.env_cfg import UnitreeGo2WalkingEnvCfg_PLAY

# ---------------------------------------------------------------------------
# Gym registration (play variant)
# ---------------------------------------------------------------------------
gym.register(
    id="Isaac-Velocity-Rough-Go2-Walking-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UnitreeGo2WalkingEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": None,
    },
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Run the trained Go2 walking policy."""
    env_cfg = UnitreeGo2WalkingEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # Resolve checkpoint path
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", "unitree_go2_walking"))
    print(f"[INFO] Looking for checkpoints in: {log_root_path}")

    if args_cli.checkpoint and os.path.isabs(args_cli.checkpoint):
        resume_path = args_cli.checkpoint
    else:
        resume_path = get_checkpoint_path(
            log_root_path,
            args_cli.load_run if args_cli.load_run else ".*",
            args_cli.checkpoint if args_cli.checkpoint else "model_.*\.pt",
        )
    log_dir = os.path.dirname(resume_path)

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Optional video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # RSL-RL wrapper
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    # Load agent config from saved yaml (falls back to defaults)
    agent_yaml_path = os.path.join(log_dir, "params", "agent.yaml")
    if os.path.exists(agent_yaml_path):
        with open(agent_yaml_path, "r") as f:
            agent_cfg_dict = yaml.safe_load(f)
        print(f"[INFO] Loaded agent config from {agent_yaml_path}")
    else:
        print(f"[WARNING] agent.yaml not found at {agent_yaml_path}. Using default config.")
        agent_cfg_dict = RslRlOnPolicyRunnerCfg(experiment_name="unitree_go2_walking").to_dict()

    # Runner
    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device=args_cli.device)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)

    # Inference loop
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs, _ = env.reset()

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
