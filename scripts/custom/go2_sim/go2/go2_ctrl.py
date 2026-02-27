import os
import torch
import carb
import gymnasium as gym
from isaaclab.envs import ManagerBasedEnv
from go2.go2_ctrl_cfg import unitree_go2_flat_cfg, unitree_go2_rough_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner

base_vel_cmd_input = None
camera_follow_enabled = True
control_mode = "policy" # "policy" or "unitree"
lowcmd_q_input = None # Store target joint positions from unitree LowCmd
reset_requested = False # Flag to trigger simulation reset

# Initialize base_vel_cmd_input as a tensor when created
def init_base_vel_cmd(num_envs):
    global base_vel_cmd_input
    global lowcmd_q_input
    base_vel_cmd_input = torch.zeros((num_envs, 3), dtype=torch.float32)
    lowcmd_q_input = torch.zeros((num_envs, 12), dtype=torch.float32)

# Modify base_vel_cmd to use the tensor directly
def base_vel_cmd(env: ManagerBasedEnv) -> torch.Tensor:
    global base_vel_cmd_input
    return base_vel_cmd_input.clone().to(env.device)

# Update sub_keyboard_event to modify specific rows of the tensor based on key inputs
def sub_keyboard_event(event) -> bool:
    global base_vel_cmd_input
    global camera_follow_enabled
    global control_mode
    global reset_requested
    lin_vel = 1.5
    ang_vel = 1.5
    
    if base_vel_cmd_input is not None:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Toggle control mode (Policy vs Unitree)
            if event.input.name == 'M':
                control_mode = "unitree" if control_mode == "policy" else "policy"
                print(f"\n[INFO] Control Mode: {control_mode.upper()}")

            # Toggle camera follow
            if event.input.name == 'V':
                camera_follow_enabled = not camera_follow_enabled
                print(f"\n[INFO] Camera Follow: {'Enabled' if camera_follow_enabled else 'Disabled'}")
                
            # Request Environment Reset
            if event.input.name == 'R':
                reset_requested = True
                print("\n[INFO] Environment reset requested...")
                
            # Update tensor values for environment 0 individually
            if event.input.name == 'W':
                base_vel_cmd_input[0][0] = lin_vel
            elif event.input.name == 'S':
                base_vel_cmd_input[0][0] = -lin_vel
            elif event.input.name == 'A':
                base_vel_cmd_input[0][1] = lin_vel
            elif event.input.name == 'D':
                base_vel_cmd_input[0][1] = -lin_vel
            elif event.input.name == 'Z':
                base_vel_cmd_input[0][2] = ang_vel
            elif event.input.name == 'C':
                base_vel_cmd_input[0][2] = -ang_vel
            
            # If there are multiple environments, handle inputs for env 1
            if base_vel_cmd_input.shape[0] > 1:
                if event.input.name == 'I':
                    base_vel_cmd_input[1][0] = lin_vel
                elif event.input.name == 'K':
                    base_vel_cmd_input[1][0] = -lin_vel
                elif event.input.name == 'J':
                    base_vel_cmd_input[1][1] = lin_vel
                elif event.input.name == 'L':
                    base_vel_cmd_input[1][1] = -lin_vel
                elif event.input.name == 'M':
                    base_vel_cmd_input[1][2] = ang_vel
                elif event.input.name == '>':
                    base_vel_cmd_input[1][2] = -ang_vel
        
        # Reset specific axis commands on key release
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in ('W', 'S'):
                base_vel_cmd_input[0][0] = 0.0
            elif event.input.name in ('A', 'D'):
                base_vel_cmd_input[0][1] = 0.0
            elif event.input.name in ('Z', 'C'):
                base_vel_cmd_input[0][2] = 0.0

            if base_vel_cmd_input.shape[0] > 1:
                if event.input.name in ('I', 'K'):
                    base_vel_cmd_input[1][0] = 0.0
                elif event.input.name in ('J', 'L'):
                    base_vel_cmd_input[1][1] = 0.0
                elif event.input.name in ('M', '>'):
                    base_vel_cmd_input[1][2] = 0.0
                    
    return True

def get_rsl_flat_policy(env_cfg, sim_cfg):
    env_cfg.observations.policy.height_scan = None
    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # Low level control: rsl control policy
    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_flat_cfg
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "ckpts", sim_cfg.checkpoint)
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    ppo_runner.load(ckpt_path)
    policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])
    return env, policy

def get_rsl_rough_policy(cfg):
    env = gym.make("Isaac-Velocity-Rough-Unitree-Go2-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env)

    # Low level control: rsl control policy
    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_rough_cfg
    ckpt_path = get_checkpoint_path(log_path=os.path.join(os.path.dirname(__file__), "..", "ckpts"), 
                                    run_dir=agent_cfg["load_run"], 
                                    checkpoint=agent_cfg["load_checkpoint"])
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    ppo_runner.load(ckpt_path)
    policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])
    return env, policy