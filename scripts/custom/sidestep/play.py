
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# local imports
# Add rsl_rl scripts to path to import cli_args
rsl_rl_scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../reinforcement_learning/rsl_rl"))
sys.path.append(rsl_rl_scripts_path)
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Go2-Railway-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import logging
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner, DistillationRunner

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.utils import configclass

# Local imports
try:
    from env_cfg import UnitreeGo2RailwayEnvCfg_PLAY
except ImportError:
    from scripts.custom.sidestep.env_cfg import UnitreeGo2RailwayEnvCfg_PLAY

# Register (if not already registered by train.py import, but we need it here likely)
gym.register(
    id="Isaac-Velocity-Rough-Go2-Railway-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UnitreeGo2RailwayEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": None, # Agent config not strictly needed for play if loading from checkpoint?
    }
)

def main():
    """Play with RSL-RL agent."""
    # Resolve the config
    env_cfg = UnitreeGo2RailwayEnvCfg_PLAY()
    
    # override configurations with CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    
    # load previously trained model
    agent_cfg = RslRlOnPolicyRunnerCfg(experiment_name="unitree_go2_railway") # Minimal config to deduce path
    # But usually we need to know the run name

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # find checkpoint
    if args_cli.checkpoint and os.path.isabs(args_cli.checkpoint):
        resume_path = args_cli.checkpoint
    else:
        resume_path = get_checkpoint_path(
            log_root_path,
            args_cli.load_run if args_cli.load_run else ".*",
            args_cli.checkpoint if args_cli.checkpoint else ".*"
        )
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0, # Record first episode
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=1.0) # Default clip actions? 
    # Agent config should probably be loaded from yaml to get clip_actions
    
    # create runner from rsl-rl
    # We need to construct a runner to load the policy.
    # OnPolicyRunner needs alg_cfg, obs_cfg etc.
    # Usually we load these from `agent.yaml` in the log dir.
    import yaml
    agent_yaml_path = os.path.join(log_dir, "params", "agent.yaml")
    if os.path.exists(agent_yaml_path):
        with open(agent_yaml_path, 'r') as f:
            agent_cfg_dict = yaml.safe_load(f)
        # We need to convert dict back to config object or just pass dict?
        # Runner takes dict.
    else:
        print(f"[WARNING] Could not find agent.yaml at {agent_yaml_path}. Using default config.")
        # fallback
        agent_cfg_dict = RslRlOnPolicyRunnerCfg(experiment_name="unitree_go2_railway").to_dict()

    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device=args_cli.device)
    
    # load the checkpoint
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)

    # run execution
    print("[INFO]: Starting execution...")
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs = env.get_observations()
    
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
        
        obs, _, _, _ = env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
