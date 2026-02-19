# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play with trained RL agent for Go2 backflip.
"""

import argparse
import sys
import os

# Add the scripts directory to sys.path
scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play with trained RL agent for Go2 backflip.")
parser.add_argument("--num_envs", type=int, default=50, help="Number of environments to simulate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg

# Register the environment
import scripts.custom.backflip.config as backflip_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunner

def main():
    """Play with RSL-RL agent."""
    # create env config
    env_cfg = backflip_cfg.UnitreeGo2BackflipEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs

    # create agent config
    agent_cfg = backflip_cfg.UnitreeGo2BackflipPPORunnerCfg()
    
    # create environment
    from isaaclab.envs import ManagerBasedRLEnv
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # wrap around environment for rsl-rl
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner
    log_dir = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    # Check if we have a checkpoint
    # Simple check for latest run
    # list directories in log_dir
    # ...
    # For now assume we load the default latest? RslRlOnPolicyRunner handles load?
    # We need to explicitly load.
    
    runner = RslRlOnPolicyRunner(env, agent_cfg, log_dir=log_dir, device=env.device)
    runner.load_or_resume_latest() # Attempt to load latest

    # reset environment
    obs, _ = env.reset()

    # simulate
    while simulation_app.is_running():
        # unsqueeze obs if needed? rsl_rl output is actions
        with torch.no_grad():
            actions = runner.get_inference_policy(obs)
        
        obs, _, _, _, _ = env.step(actions)

    # close environment
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
