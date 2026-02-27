from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from go2.go2_env import Go2RSLEnvCfg
import go2.go2_ctrl as go2_ctrl
import env.sim_env as sim_env

go2_env_cfg = Go2RSLEnvCfg()
go2_env_cfg.scene.num_envs = 1
go2_ctrl.init_base_vel_cmd(1)
env, policy = go2_ctrl.get_rsl_flat_policy(go2_env_cfg)

# Print joint names
joint_names = env.unwrapped.scene["unitree_go2"].data.joint_names
print("JOINT NAMES:", joint_names)
default_pos = env.unwrapped.scene["unitree_go2"].data.default_joint_pos
print("DEFAULT POS:", default_pos)
import torch
print("ACTION CFG:", env.action_manager.action_term_dict['joint_pos'])
print("ACTION CFG SCALE:", env.action_manager.action_term_dict['joint_pos']._scale)

simulation_app.close()
