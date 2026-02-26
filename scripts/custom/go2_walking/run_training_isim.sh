#!/bin/bash
# Script to launch Go2 walking RL training via Isaac Sim
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
echo "[Starting Go2 walking training...]"

# Check if --distributed is passed
if [[ "$*" == *"--distributed"* ]]; then
    echo "[INFO] Running multi-GPU distributed training using torchrun..."
    ./isaaclab.sh -p -m torch.distributed.run --nnodes=1 --nproc_per_node=4 scripts/custom/go2_walking/train.py "$@" --headless
else
    # Run standard single-GPU training
    # Example: ./run_training_isim.sh --num_envs 4096 --headless
    ./isaaclab.sh -p scripts/custom/go2_walking/train.py "$@"
fi
