#!/bin/bash
# Source the user's bashrc to get the isim function
source ~/.bashrc

# Manually export the variables from isim function since we can't easily call the function from non-interactive shell in some cases
export ISAACSIM_PATH=$HOME/isaac-sim
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
export ROS_DOMAIN_ID=147
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ISAACSIM_PATH/exts/isaacsim.ros2.bridge/humble/lib
export ISAAC_ROS_WS="/home/$USER/ws_isaacsim/IsaacSim-ros_workspaces"

# Source ROS setups if they exist
if [ -f "$ISAAC_ROS_WS/build_ws/humble/humble_ws/install/local_setup.bash" ]; then
    source $ISAAC_ROS_WS/build_ws/humble/humble_ws/install/local_setup.bash
fi
if [ -f "$ISAAC_ROS_WS/build_ws/humble/isaac_sim_ros_ws/install/local_setup.bash" ]; then
    source $ISAAC_ROS_WS/build_ws/humble/isaac_sim_ros_ws/install/local_setup.bash
fi

echo "[Isaac-sim environment is activated]"

# Run the training script using the python executable defined by isim or fallback to isaaclab.sh
# The user said "use isim command". isim exports ISAACSIM_PYTHON_EXE.
# IsaacLab usually uses ./isaaclab.sh which wraps the python call.
# ./isaaclab.sh likely uses the python.sh from isaac sim as well.

# Let's try running via isaaclab.sh but with the environment variables set.
./isaaclab.sh -p scripts/custom/sidestep/train.py "$@"
