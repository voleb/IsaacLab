import os
import hydra
import torch
import time
import math
import argparse
import carb
import omni
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Standalone Keyboard Control for Go2")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from go2.go2_env import Go2RSLEnvCfg, camera_follow
import env.sim_env as sim_env
import go2.go2_ctrl as go2_ctrl

FILE_PATH = os.path.join(os.path.dirname(__file__), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def run_simulator(cfg):

    # Go2 Environment setup
    go2_env_cfg = Go2RSLEnvCfg()
    go2_env_cfg.scene.num_envs = cfg.num_envs
    go2_env_cfg.decimation = math.ceil(1./go2_env_cfg.sim.dt/cfg.freq)
    go2_env_cfg.sim.render_interval = go2_env_cfg.decimation
    go2_ctrl.init_base_vel_cmd(cfg.num_envs)
    
    # We load the flat policy model as requested (flat_model_6800.pt)
    env, policy = go2_ctrl.get_rsl_flat_policy(go2_env_cfg)

    # Simulation environment logic
    if (cfg.env_name == "obstacle-dense"):
        sim_env.create_obstacle_dense_env()
    elif (cfg.env_name == "obstacle-medium"):
        sim_env.create_obstacle_medium_env()
    elif (cfg.env_name == "obstacle-sparse"):
        sim_env.create_obstacle_sparse_env()
    elif (cfg.env_name == "warehouse"):
        sim_env.create_warehouse_env()
    elif (cfg.env_name == "warehouse-forklifts"):
        sim_env.create_warehouse_forklifts_env()
    elif (cfg.env_name == "warehouse-shelves"):
        sim_env.create_warehouse_shelves_env()
    elif (cfg.env_name == "full-warehouse"):
        sim_env.create_full_warehouse_env()

    # Keyboard control hook
    print("[INFO]: Setting up Keyboard Input")
    system_input = carb.input.acquire_input_interface()
    system_input.subscribe_to_keyboard_events(
        omni.appwindow.get_default_app_window().get_keyboard(), go2_ctrl.sub_keyboard_event)
    
    print("[INFO]: Keyboard Controls:")
    print("        W - Forward")
    print("        S - Backward")
    print("        A - Left")
    print("        D - Right")
    print("        Z - Rotate Left")
    print("        C - Rotate Right")
    print("        V - Toggle Camera Follow")

    # Run simulation loop
    sim_step_dt = float(go2_env_cfg.sim.dt * go2_env_cfg.decimation)
    obs, _ = env.reset()

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():            
            # get actions from policy
            actions = policy(obs)

            # step the environment
            obs, _, _, _ = env.step(actions)

            # Camera follow if enabled
            if go2_ctrl.camera_follow_enabled:
                camera_follow(env)

            # sleep to maintain loop rate
            elapsed_time = time.time() - start_time
            if elapsed_time < sim_step_dt:
                sleep_duration = sim_step_dt - elapsed_time
                time.sleep(sleep_duration)
                
        actual_loop_time = time.time() - start_time
        rtf = min(1.0, sim_step_dt/actual_loop_time)
        print(f"\rStep time: {actual_loop_time*1000:.2f}ms, Real Time Factor: {rtf:.2f}", end='', flush=True)

    simulation_app.close()

if __name__ == "__main__":
    run_simulator()
