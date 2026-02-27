from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaacsim.core.utils.extensions import enable_extension
enable_extension("omni.isaac.ros2_bridge")
import rclpy
try:
    from unitree_go.msg import LowCmd
    with open("import_result.txt", "w") as f:
        f.write("SUCCESS IMPORTING LOWCMD\n")
except Exception as e:
    with open("import_result.txt", "w") as f:
        f.write(f"ERROR: {e}\n")

simulation_app.close()
