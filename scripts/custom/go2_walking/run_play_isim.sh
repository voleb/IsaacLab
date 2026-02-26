#!/bin/bash
# Script to play (evaluate) a trained Go2 walking policy via Isaac Sim
source ~/.bashrc

# Isaac Sim environment variables
export ISAACSIM_PATH=$HOME/isaac-sim
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
export ROS_DOMAIN_ID=147
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ISAACSIM_PATH/exts/isaacsim.ros2.bridge/humble/lib
export ISAAC_ROS_WS="/home/$USER/ws_isaacsim/IsaacSim-ros_workspaces"

# Source ROS setups if available
if [ -f "$ISAAC_ROS_WS/build_ws/humble/humble_ws/install/local_setup.bash" ]; then
    source $ISAAC_ROS_WS/build_ws/humble/humble_ws/install/local_setup.bash
fi
if [ -f "$ISAAC_ROS_WS/build_ws/humble/isaac_sim_ros_ws/install/local_setup.bash" ]; then
    source $ISAAC_ROS_WS/build_ws/humble/isaac_sim_ros_ws/install/local_setup.bash
fi

echo "[Isaac-sim environment is activated]"
echo "[Starting Go2 walking play...]"

# Run play script
# Example: ./run_play_isim.sh --load_run 2024-01-01_00-00-00 --num_envs 50
./isaaclab.sh -p scripts/custom/go2_walking/play.py "$@"
