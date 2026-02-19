# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent for Go2 backflip.
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
parser = argparse.ArgumentParser(description="Train RL agent for Go2 backflip.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
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
from rsl_rl.runners import OnPolicyRunner

def main():
    """Train with RSL-RL agent."""
    # create env config
    env_cfg = backflip_cfg.UnitreeGo2BackflipEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    # create agent config
    agent_cfg = backflip_cfg.UnitreeGo2BackflipPPORunnerCfg()
    agent_cfg.seed = args_cli.seed
    
    # create environment
    from isaaclab.envs import ManagerBasedRLEnv
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # wrap around environment for rsl-rl
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner
    log_dir = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close environment
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
