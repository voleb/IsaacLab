import sys
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from unitree_go.msg import LowCmd
import go2.go2_ctrl as go2_ctrl
import torch

class CmdVelSubscriber(Node):
    def __init__(self):
        super().__init__('go2_ros2_bindings')
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10)
            
        self.lowcmd_sub = self.create_subscription(
            LowCmd,
            '/lowcmd',
            self.lowcmd_callback,
            10)
            
        self.get_logger().info("Subscribed to /cmd_vel and /lowcmd for teleop control")

    def cmd_vel_callback(self, msg):
        # Update the velocity tensor for environment 0 (and others if multiple)
        if go2_ctrl.base_vel_cmd_input is not None:
            # We map linear.x to forward/backward (X)
            # We map linear.y to left/right (Y)
            # We map angular.z to rotation (Z)
            
            lin_x = msg.linear.x
            lin_y = msg.linear.y
            ang_z = msg.angular.z
            
            num_envs = go2_ctrl.base_vel_cmd_input.shape[0]
            for i in range(num_envs):
                go2_ctrl.base_vel_cmd_input[i][0] = lin_x
                go2_ctrl.base_vel_cmd_input[i][1] = lin_y
                go2_ctrl.base_vel_cmd_input[i][2] = ang_z

    def lowcmd_callback(self, msg):
        if go2_ctrl.control_mode == "unitree" and go2_ctrl.lowcmd_q_input is not None:
            # Safely extract 12 joint targets from Unitree LowCmd motor_cmd array
            num_motors_in_msg = len(msg.motor_cmd)
            u_q = [0.0] * 12
            for j in range(min(12, num_motors_in_msg)):
                u_q[j] = msg.motor_cmd[j].q
                
            num_envs = go2_ctrl.lowcmd_q_input.shape[0]
            for i in range(num_envs):
                # Map Unitree order to Isaac Sim order
                # Unitree: 0:FR_hip, 1:FR_thigh, 2:FR_calf, 3:FL_hip, 4:FL_thigh, 5:FL_calf...
                # Isaac Sim: 0:FL_hip, 1:FR_hip, 2:RL_hip, 3:RR_hip...
                
                # HIP
                go2_ctrl.lowcmd_q_input[i][0] = u_q[3] # FL_hip
                go2_ctrl.lowcmd_q_input[i][1] = u_q[0] # FR_hip
                go2_ctrl.lowcmd_q_input[i][2] = u_q[9] # RL_hip
                go2_ctrl.lowcmd_q_input[i][3] = u_q[6] # RR_hip
                
                # THIGH
                go2_ctrl.lowcmd_q_input[i][4] = u_q[4] # FL_thigh
                go2_ctrl.lowcmd_q_input[i][5] = u_q[1] # FR_thigh
                go2_ctrl.lowcmd_q_input[i][6] = u_q[10] # RL_thigh
                go2_ctrl.lowcmd_q_input[i][7] = u_q[7] # RR_thigh
                
                # CALF
                go2_ctrl.lowcmd_q_input[i][8] = u_q[5] # FL_calf
                go2_ctrl.lowcmd_q_input[i][9] = u_q[2] # FR_calf
                go2_ctrl.lowcmd_q_input[i][10] = u_q[11] # RL_calf
                go2_ctrl.lowcmd_q_input[i][11] = u_q[8] # RR_calf

