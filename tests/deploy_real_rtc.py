#!/usr/bin/env python3
"""Real-robot FastWAM checkpoint inference entrypoint.

This script is intentionally dry-run first: it can run from image files and a
state JSON file, save predicted actions, and exposes small provider/controller
classes that can be replaced by real robot IO.

Example command:
uv run python tests/deploy_real_ckpt_temporal_ensembling.py \
    --ckpt "your_ckpt_path" \
    --dataset-stats "your_<dataset_stats.json>_path" \
    --prompt "your_task_prompt" \
    --num-replans 5000 \
    --execute-steps 1 \
    --output-actions predicted_actions.npy \
    --output-actions-json predicted_actions.json \
    --context-cache-dir your_text_embedding_cache_dir \
    --no-dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from qi.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from qi.datasets.dataset_utils import CenterCrop, ResizeSmallestSideAspectPreserving
from qi.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from qi.utils.config_resolvers import register_default_resolvers
from qi.utils.logging_config import get_logger, setup_logging

import sys
import time
import signal
from collections import deque, defaultdict
import threading
from pyAgxArm import create_agx_arm_config, AgxArmFactory
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

register_default_resolvers()
logger = get_logger(__name__)

#==================== 异步推理全局变量 ==========================
_action_prod_thread = None
_action_stop_event = threading.Event()
_action_lock = threading.Lock()

_ACTION_CHUNK_SIZE_ORIGIN = 50  # 模型一次推理的步数
_ACTION_DIM = 14

_action_chunks = {}
_control_t = 0

# ========== 线程停止信号 (用于通知producer线程) ===============
stop_signal = threading.Event() 

#====================== 异步推理函数 ============================
def clear_action_buffer(): # 清空缓存
    global _control_t, _action_chunks
    with _action_lock:
        _control_t = 0
        _action_chunks = {} # {cursor: chunk} 

def temporal_ensembled_action(current_time):
    """取当前时刻所有已推断的动作softmax加权平均"""
    global _ACTION_CHUNK_SIZE_ORIGIN
    with _action_lock:
        relevant = {}
        to_delete = []
        before_keys = sorted(_action_chunks.keys())

        for cursor, chunk in _action_chunks.items():
            end = cursor + _ACTION_CHUNK_SIZE_ORIGIN # 模型一次推理的步数
            if cursor <= current_time < end:
                # 还覆盖当前时刻 → 参与加权
                relevant[cursor] = chunk[current_time - cursor]
            elif end <= current_time:
                # 整个chunk都过期了 → 标记删除
                to_delete.append(cursor)

        # 清理过期chunk，并打印关键状态用于排查是否正确删除
        print(
            f"[action_chunks] t={current_time} before={before_keys} "
            f"delete={sorted(to_delete)}"
        )
        for k in to_delete:
            del _action_chunks[k]
        print(f"[action_chunks] t={current_time} after={sorted(_action_chunks.keys())}")
        
        if not relevant: 
            return None

        # older -> lower weight, newer -> higher weight.
        sorted_items = sorted(relevant.items(), key=lambda x: x[0])
        cur_actions = np.asarray([action for _, action in sorted_items])

    exp_weights = np.exp(0.5 * np.arange(len(cur_actions)))
    exp_weights = (exp_weights / exp_weights.sum())[:, None]
    sub_action = (cur_actions * exp_weights).sum(axis=0)
    return sub_action

def _enqueue_chunk_to_expected_queues(chunk, cursor):
    """将一个 chunk 存入字典"""
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != _ACTION_DIM:
        raise ValueError(f"Expected action chunk shape [T, {_ACTION_DIM}], got {chunk.shape}")
    with _action_lock:
        _action_chunks[cursor] = chunk

class RosOperator:
    def __init__(self, args):
        self.img_front_deque = None
        self.img_front = None
        self.img_left = None
        self.img_right = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.args = args
        self.ctrl_state = False
        self.ctrl_state_lock = threading.Lock()
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
    
    def get_img(self):
        
        if len(self.img_front_deque) == 0 :
            return False
        frame_time = self.img_front_deque[-1].header.stamp.to_sec()

        if len(self.img_front_deque) == 0 < frame_time:
            return False
        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')
        return img_front
    def get_frame(self):
        # if len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or \
        #         (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
        #     return False
        if len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or len(self.img_left_deque) == 0 or \
            (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
            return False
        if self.args.use_depth_image:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec(),
                              self.img_left_depth_deque[-1].header.stamp.to_sec(), self.img_right_depth_deque[-1].header.stamp.to_sec(), self.img_front_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(),self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False


        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')


        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)
        self.img_left = msg

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)
        self.img_right = msg

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)
        self.img_front = msg

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def ctrl_callback(self, msg):
        self.ctrl_state_lock.acquire()
        self.ctrl_state = msg.data
        self.ctrl_state_lock.release()

    def get_ctrl_state(self):
        self.ctrl_state_lock.acquire()
        state = self.ctrl_state
        self.ctrl_state_lock.release()
        return state

    def init_ros(self):
        rospy.init_node('joint_state_publisher', anonymous=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)

class InferController:
    def __init__(self):
        self.client = None
        self.left_channel = "can_left"
        self.right_channel = "can_right"
        self.bitrate = 1000000
        
        # === 机械臂参数 ===
        self.speed_pct = 15           # 速度百分比
        self.max_linear_vel = 0.5     # m/s
        self.max_angular_vel = 0.1  # rad/s
        self.max_linear_acc = 0.1     # m/s2
        self.max_angular_acc = 0.05    # rad/s2

        # === 夹爪对象 ===
        self.left_gripper = None
        self.right_gripper = None
        
        # === 初始位置 ===
        self.LEFT_INIT_POSITION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05]
        self.RIGHT_INIT_POSITION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05]

        # === 安全阈值 ===
        self.ACTION_SAFETY_THRESHOLD = 1.5
        self.STATE_SAFETY_THRESHOLD = 0.3

    def connect_arms(self):
        """连接双臂并初始化夹爪"""
        print("连接双臂...")
        try:
            # --- 左臂 ---
            cfg_l = create_agx_arm_config(
                robot="piper", comm="can", channel=self.left_channel, bitrate=self.bitrate
            )
            self.left_arm = AgxArmFactory.create_arm(cfg_l)
            # 初始化夹爪
            self.left_gripper = self.left_arm.init_effector(
                self.left_arm.OPTIONS.EFFECTOR.AGX_GRIPPER
            )
            self.left_arm.connect()
            
            # --- 右臂 ---
            cfg_r = create_agx_arm_config(
                robot="piper", comm="can", channel=self.right_channel, bitrate=self.bitrate
            )
            self.right_arm = AgxArmFactory.create_arm(cfg_r)
            # 初始化夹爪
            self.right_gripper = self.right_arm.init_effector(
                self.right_arm.OPTIONS.EFFECTOR.AGX_GRIPPER
            )
            self.right_arm.connect()
            
            time.sleep(0.5)
            if not (self.left_arm.is_ok() and self.right_arm.is_ok()):
                raise Exception("机械臂连接状态检查失败")
            
            # 设置速度和使能
            for name, arm in [("左臂", self.left_arm), ("右臂", self.right_arm)]:
                arm.set_flange_vel_acc_limits(
                    max_linear_vel=self.max_linear_vel,
                    max_angular_vel=self.max_angular_vel,
                    max_linear_acc=self.max_linear_acc,
                    max_angular_acc=self.max_angular_acc,
                    timeout=1.0
                )
                arm.set_speed_percent(self.speed_pct)
                
                enabled = False
                for _ in range(5):
                    if arm.enable():
                        enabled = True
                        break
                    time.sleep(0.5)
                if not enabled:
                    raise Exception(f"{name} 使能超时")
                
            print("连接成功\n")
            return True
            
        except Exception as e:
            print(f"连接错误：{e}")
            return False
    
    def get_status_and_state(self):
        """获取当前状态：12 维关节 (6+6) + 2 维夹爪 (归一化) = 14 维"""
        state = np.zeros(14, dtype=np.float32)
        try:
            # 1. 左臂关节
            ja_l = self.left_arm.get_joint_angles()
            if ja_l is not None: 
                state[0:6] = ja_l.msg
            
            # 2. 右臂关节
            ja_r = self.right_arm.get_joint_angles()
            if ja_r is not None: 
                state[7:13] = ja_r.msg
            
            # 3. 左臂夹爪
            if self.left_gripper:
                gs = self.left_gripper.get_gripper_status()
                if gs is not None:
                    state[6] = gs.msg.value
            
            # 4. 右臂夹爪
            if self.right_gripper:
                gs = self.right_gripper.get_gripper_status()
                if gs is not None:
                    state[13] = gs.msg.value
                    
        except Exception as e:
            print(f"状态读取异常：{e}")
        return state
        
    def move(self, position_state):
        """移动到特定位置"""
        
        left_arm_position = position_state[0:6].tolist()
        right_arm_position = position_state[7:13].tolist()
        left_gripper_position = max(0.0, min(position_state[6], 0.1)) # width 0.0-0.1
        right_gripper_position = max(0.0, min(position_state[13], 0.1))
        try:
            # 左臂
            self.left_arm.move_js(left_arm_position)
            self.left_gripper.move_gripper(width=left_gripper_position, force=1.0)
            # 右臂
            self.right_arm.move_js(right_arm_position)
            self.right_gripper.move_gripper(width=right_gripper_position, force=1.0)
            
            time.sleep(0.02)
        except Exception as e:
            print(f"移动失败：{e}")
    
    def move_initial(self, position_state):
        """移动到初始位置"""
        n=50
        left_arm_position = position_state[0:6].tolist()
        right_arm_position = position_state[7:13].tolist()
        left_gripper_position = max(0.0, min(position_state[6], 0.1)) # width 0.0-0.1
        right_gripper_position = max(0.0, min(position_state[13], 0.1))
        left_traj = np.linspace(self.get_status_and_state()[0:6], left_arm_position, n)
        right_traj = np.linspace(self.get_status_and_state()[7:13], right_arm_position, n)
        try:
           for i in range(n):
                print(left_traj[i],right_traj[i])
                # 左臂
                self.left_arm.move_js(left_traj[i].tolist())
                self.left_gripper.move_gripper(width=left_gripper_position, force=1.0)
                # 右臂
                self.right_arm.move_js(right_traj[i].tolist())
                self.right_gripper.move_gripper(width=right_gripper_position, force=1.0)
                time.sleep(0.02)
        except Exception as e:
            print(f"移动失败：{e}")
 



@dataclass
class RealObservation:
    cam_high: np.ndarray
    cam_left_wrist: np.ndarray
    cam_right_wrist: np.ndarray
    state: np.ndarray


class ObservationProvider(Protocol):
    def get_observation(self,ros_operator,infercontroller) -> RealObservation:
        rate = rospy.Rate(50)
        print_flag_local = True
        while not rospy.is_shutdown():
            # 采帧（必要时等待）
            result = ros_operator.get_frame()
            if not result:
                if print_flag_local:
                    print("async syn fail")
                    print_flag_local = False
                rate.sleep()
                continue
            break
        print_flag_local = True
        (img_h,img_l, img_r, img_h_depth, img_l_depth, img_r_depth) = result
        
        # 构建obs
        return RealObservation(
            cam_high=img_h,
            cam_left_wrist=img_l,
            cam_right_wrist=img_r,
            state=infercontroller.get_status_and_state(),
        )


class RobotController(Protocol):
    def execute_action_chunk(self, actions: np.ndarray, execute_steps: int,infercontroller) -> None:
        raise NotImplementedError


class FileObservationProvider:
    def __init__(
        self,
        cam_high_path: str | Path,
        cam_left_wrist_path: str | Path,
        cam_right_wrist_path: str | Path,
        state_path: str | Path,
    ):
        self.cam_high_path = Path(cam_high_path)
        self.cam_left_wrist_path = Path(cam_left_wrist_path)
        self.cam_right_wrist_path = Path(cam_right_wrist_path)
        self.state_path = Path(state_path)

    def get_observation(self) -> RealObservation:
        return RealObservation(
            cam_high=load_rgb_image(self.cam_high_path),
            cam_left_wrist=load_rgb_image(self.cam_left_wrist_path),
            cam_right_wrist=load_rgb_image(self.cam_right_wrist_path),
            state=load_state_vector(self.state_path),
        )


class DryRunController:
    def execute_action_chunk(self, actions: np.ndarray, execute_steps: int, infercontroller=None) -> None:
        actions = np.asarray(actions)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        steps = min(int(execute_steps), int(actions.shape[0]))
        logger.info("Dry-run mode: not executing %d predicted actions.", steps)


class RealRobotObservationProvider:
    def get_observation(self,ros_operator,infercontroller) -> RealObservation:
        rate = rospy.Rate(50)
        print_flag_local = True
        while not rospy.is_shutdown():
            # 采帧（必要时等待）
            result = ros_operator.get_frame()
            if not result:
                if print_flag_local:
                    print("async syn fail")
                    print_flag_local = False
                rate.sleep()
                continue
            break
        print_flag_local = True
        (img_h,img_l, img_r, img_h_depth, img_l_depth, img_r_depth) = result
        
        # 构建obs
        return RealObservation(
            cam_high=img_h,
            cam_left_wrist=img_l,
            cam_right_wrist=img_r,
            state=infercontroller.get_status_and_state(),
        )

        # raise NotImplementedError(
        #     "Connect your camera/robot SDK here and return cam_high, "
        #     "cam_left_wrist, cam_right_wrist, and a 14-dim state vector."
        # )


class RealRobotController:
    def execute_action_chunk(self, actions: np.ndarray, execute_steps: int,infercontroller) -> None:
        rate=rospy.Rate(50)
        try:
            actions = np.asarray(actions, dtype=np.float32)
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)
            if actions.ndim != 2 or actions.shape[1] != _ACTION_DIM:
                raise ValueError(f"Expected actions shape [T, {_ACTION_DIM}], got {actions.shape}")
            steps = min(int(execute_steps), int(actions.shape[0]))
            for i in range(steps):
                infercontroller.move(actions[i])
                rate.sleep()
        except Exception as e:
            print(f"执行异常：{e}")
        # raise NotImplementedError(
        #     "Connect your robot controller here. Add action limits, workspace "
        #     "checks, speed limits, and emergency-stop handling before use."
        # )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FastWAM real-ckpt action inference.")
    parser.add_argument("--ckpt", required=True, help="Path to runs/.../checkpoints/weights/step_xxxxxx.pt.")
    parser.add_argument("--dataset-stats", required=True, help="Path to runs/.../dataset_stats.json.")
    parser.add_argument("--task", default="real_cleaning_uncond_3cam_384_1e-4")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--train-config", default=None, help="Optional saved runs/.../config.yaml.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--prompt", default="clean the table")
    parser.add_argument("--context-cache-dir", default=None)
    parser.add_argument("--use-text-encoder", action="store_true")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--cam-high", default=None)
    parser.add_argument("--cam-left-wrist", default=None)
    parser.add_argument("--cam-right-wrist", default=None)
    parser.add_argument("--state-json", default=None)
    parser.add_argument("--output-actions", default="predicted_actions.npy")
    parser.add_argument("--output-actions-json", default=None)
    parser.add_argument("--execute-steps", type=int, default=1)
    parser.add_argument("--num-replans", type=int, default=1)
    parser.add_argument("--replan-sleep", type=float, default=0.0)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    
    #ros
    parser.add_argument('--img_front_topic', action='store', type=str, default='/camera_h/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, default='/camera_r/color/image_raw', required=False)

    parser.add_argument('--img_front_depth_topic', action='store', type=str, default='/camera_h/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, default='/camera_r/depth/image_raw', required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, default=False, required=False)

    return parser.parse_args()


def dtype_from_mixed_precision(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "no":
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed precision: {mixed_precision}")


def load_config(args: argparse.Namespace) -> DictConfig:
    if args.train_config:
        cfg = OmegaConf.load(args.train_config)
        if not isinstance(cfg, DictConfig):
            raise ValueError(f"Expected DictConfig in {args.train_config}, got {type(cfg)}")
        if args.use_text_encoder:
            cfg.model.load_text_encoder = True
        return cfg

    config_dir = str(Path(args.config_dir).resolve())
    overrides = [f"task={args.task}", f"mixed_precision={args.mixed_precision}"]
    if args.use_text_encoder:
        overrides.append("model.load_text_encoder=true")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        return compose(config_name="train", overrides=overrides)


def load_rgb_image(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def load_state_vector(path: str | Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        if "state" not in payload:
            raise ValueError(f"{path} must contain a `state` key or be a raw list.")
        payload = payload["state"]
    state = np.asarray(payload, dtype=np.float32)
    if state.ndim != 1:
        raise ValueError(f"State must be a 1D vector, got shape {state.shape}")
    return state


def build_file_provider(args: argparse.Namespace) -> FileObservationProvider:
    missing = [
        name
        for name, value in {
            "--cam-high": args.cam_high,
            "--cam-left-wrist": args.cam_left_wrist,
            "--cam-right-wrist": args.cam_right_wrist,
            "--state-json": args.state_json,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(
            "File dry-run requires these arguments: "
            + ", ".join(missing)
            + ". For a real robot, replace RealRobotObservationProvider."
        )
    return FileObservationProvider(
        cam_high_path=args.cam_high,
        cam_left_wrist_path=args.cam_left_wrist,
        cam_right_wrist_path=args.cam_right_wrist,
        state_path=args.state_json,
    )


def rgb_to_float_chw(rgb_uint8: np.ndarray) -> torch.Tensor:
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"Expected RGB image as [H,W,3], got {rgb_uint8.shape}")
    rgb_uint8 = np.ascontiguousarray(rgb_uint8).copy()
    return torch.from_numpy(rgb_uint8).permute(2, 0, 1).to(torch.float32) / 255.0


def preprocess_real_images(observation: RealObservation, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    cam_high = transforms_F.resize(
        rgb_to_float_chw(observation.cam_high),
        [240, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_left = transforms_F.resize(
        rgb_to_float_chw(observation.cam_left_wrist),
        [240, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_right = transforms_F.resize(
        rgb_to_float_chw(observation.cam_right_wrist),
        [240, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_high = transforms_F.resize(
        cam_high,
        [256, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_left = transforms_F.resize(
        cam_left,
        [128, 160],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_right = transforms_F.resize(
        cam_right,
        [128, 160],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    bottom = torch.cat([cam_left, cam_right], dim=-1)
    image = torch.cat([cam_high, bottom], dim=-2)
    resize = ResizeSmallestSideAspectPreserving(args={"img_w": 320, "img_h": 384})
    crop = CenterCrop(args={"img_w": 320, "img_h": 384})
    image = crop(resize(image))
    image = image.mul(2.0).sub(1.0)
    return image.unsqueeze(0).to(device=device, dtype=dtype)


def normalize_proprio(processor: Any, state: np.ndarray) -> torch.Tensor:
    state_tensor = torch.as_tensor(state, dtype=torch.float32)
    expected_dim = int(processor.proprio_output_dim)
    if state_tensor.numel() != expected_dim:
        raise ValueError(f"Expected proprio/state dim {expected_dim}, got {state_tensor.numel()}")
    batch = {"state": {"default": state_tensor.view(1, -1)}}
    batch = processor.normalizer.forward(batch)
    return batch["state"]["default"].squeeze(0)


def denormalize_actions(processor: Any, action_norm: torch.Tensor) -> torch.Tensor:
    if action_norm.ndim == 3 and action_norm.shape[0] == 1:
        action_norm = action_norm.squeeze(0)
    if action_norm.ndim == 1:
        action_norm = action_norm.view(1, -1)
    if action_norm.ndim != 2:
        raise ValueError(f"Expected normalized action [T,D], got {tuple(action_norm.shape)}")
    dummy_state = torch.zeros(
        1,
        action_norm.shape[0],
        int(processor.proprio_output_dim),
        dtype=torch.float32,
        device=action_norm.device,
    )
    batch = {
        "action": {"default": action_norm.unsqueeze(0).to(torch.float32)},
        "state": {"default": dummy_state},
    }
    batch = processor.normalizer.backward(batch)
    return batch["action"]["default"].squeeze(0)


def format_prompt(task_prompt: str) -> str:
    return DEFAULT_PROMPT.format(task=task_prompt)


def load_cached_context(cache_dir: str | Path, formatted_prompt: str, context_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    cache_dir = Path(cache_dir)
    cache_hash = hashlib.sha256(formatted_prompt.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{cache_hash}.t5_len{context_len}.wan22ti2v5b.pt"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing text embedding cache: {cache_path}. "
            "Run scripts/precompute_text_embeds.py for this prompt, or pass --use-text-encoder."
        )
    payload = torch.load(cache_path, map_location="cpu")
    context = payload["context"]
    context_mask = payload["mask"].bool()
    if context.ndim != 2 or context_mask.ndim != 1:
        raise ValueError(f"Invalid cached context shapes in {cache_path}: {context.shape}, {context_mask.shape}")
    if context.shape[0] != context_len or context_mask.shape[0] != context_len:
        raise ValueError(
            f"Cached context length mismatch in {cache_path}: "
            f"context={context.shape[0]}, mask={context_mask.shape[0]}, expected={context_len}"
        )
    context = context.clone()
    context[~context_mask] = 0.0
    context_mask = torch.ones_like(context_mask, dtype=torch.bool)
    return context, context_mask


def resolve_text_condition(
    args: argparse.Namespace,
    cfg: DictConfig,
) -> tuple[str | None, torch.Tensor | None, torch.Tensor | None]:
    formatted_prompt = format_prompt(args.prompt)
    if args.use_text_encoder:
        return formatted_prompt, None, None

    cache_dir = args.context_cache_dir or cfg.data.train.get("text_embedding_cache_dir")
    if cache_dir is None:
        raise ValueError("No context cache dir found. Pass --context-cache-dir or --use-text-encoder.")
    context_len = int(cfg.data.train.get("context_len", 128))
    context, context_mask = load_cached_context(cache_dir, formatted_prompt, context_len)
    return None, context, context_mask


def load_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any, DictConfig]:
    cfg = load_config(args)
    model_dtype = dtype_from_mixed_precision(args.mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=args.device)
    model.load_checkpoint(args.ckpt)
    model.eval()

    processor = instantiate(cfg.data.train.processor)
    stats = load_dataset_stats_from_json(args.dataset_stats)
    processor.set_normalizer_from_stats(stats)
    processor.eval()
    return model, processor, cfg


def predict_action_chunk(
    model: Any,
    processor: Any,
    observation: RealObservation,
    prompt: str | None,
    context: torch.Tensor | None,
    context_mask: torch.Tensor | None,
    args: argparse.Namespace,
) -> torch.Tensor:
    image = preprocess_real_images(observation, model.device, model.torch_dtype)
    proprio = normalize_proprio(processor, observation.state)
    with torch.no_grad():
        output = model.infer_action(
            prompt=prompt,
            input_image=image,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            action_horizon=args.action_horizon,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            rand_device=args.rand_device,
        )
    return denormalize_actions(processor, output["action"])


def save_actions(actions: torch.Tensor | np.ndarray, output_npy: str | Path, output_json: str | Path | None) -> None:
    if isinstance(actions, torch.Tensor):
        actions_np = actions.detach().cpu().numpy().astype(np.float32)
    else:
        actions_np = np.asarray(actions, dtype=np.float32)
    np.save(output_npy, actions_np)
    logger.info("Saved predicted actions to %s with shape %s.", output_npy, actions_np.shape)
    if output_json:
        with open(output_json, "w", encoding="utf-8") as file:
            json.dump({"action": actions_np.tolist()}, file, ensure_ascii=False, indent=2)
        logger.info("Saved predicted actions JSON to %s.", output_json)


def build_runtime_io(args: argparse.Namespace) -> tuple[ObservationProvider, RobotController]:
    if args.dry_run:
        return build_file_provider(args), DryRunController()
    return RealRobotObservationProvider(), RealRobotController()


def run() -> None:

    args = parse_args()
    global _ACTION_CHUNK_SIZE_ORIGIN
    _ACTION_CHUNK_SIZE_ORIGIN = args.action_horizon
    setup_logging(log_level=logging.INFO)
    model, processor, cfg = load_model_and_processor(args)
    prompt, context, context_mask = resolve_text_condition(args, cfg)
    provider, controller = build_runtime_io(args)
    
    ros_operator = RosOperator(args)
    infer_controller = InferController()
    if not infer_controller.connect_arms():
        return False
    
    rate = rospy.Rate(50)
    initial_position = np.concatenate((infer_controller.LEFT_INIT_POSITION, infer_controller.RIGHT_INIT_POSITION)).astype(np.float32)
    infer_controller.move_initial(initial_position)

    clear_action_buffer()
    global _action_prod_thread, _control_t
    t = 0
    
    input("按回车键开始模型推理...")
    _action_prod_thread = threading.Thread(
        target=_action_producer_loop,
        args=(ros_operator, infer_controller, model, processor, prompt, provider, context, context_mask, args),
        daemon=True,
    )
    _action_prod_thread.start()
    last_action = initial_position
    try:
        while True:
            while not rospy.is_shutdown() and not stop_signal.is_set():
                with _action_lock:
                    _control_t = t
                # Temporal Ensembling提取当前t动作
                action = temporal_ensembled_action(t)
                if action is None or len(action) == 0:
                    print(f"[WAITING] Waiting for action...")
                    rate.sleep()
                    continue
                action = np.asarray(action, dtype=np.float32)
                if action.shape != (_ACTION_DIM,):
                    print(f"执行异常：Expected action shape [{_ACTION_DIM}], got {action.shape}")
                    break
                            
                # 安全检查
                prev_curr_l1 = np.mean(np.abs(action - last_action))
                if prev_curr_l1 > infer_controller.ACTION_SAFETY_THRESHOLD:
                    print(f"\033[31m 安全检查失败：动作变化过大 {prev_curr_l1:.3f}\033[0m")
                    break
                current_state =  infer_controller.get_status_and_state()
                # 检查动作与当前状态的差异 (防止跳变)
                prev_state_l1 = np.mean(np.abs(current_state - last_action))
                if prev_state_l1 >  infer_controller.STATE_SAFETY_THRESHOLD:
                    print(f"\033[31m 安全检查失败：状态差异过大 {prev_state_l1:.3f}\033[0m")
                    break
                last_action = action
                print(f"待执行action: {action}")
                try:
                    controller.execute_action_chunk(action, args.execute_steps,infer_controller)
                    # rate.sleep()
                except Exception as e:
                    print(f"执行异常：{e}")
                
                t += 1
                rate.sleep()
                if last_action is not None:
                    save_actions(last_action, args.output_actions, args.output_actions_json)
    except Exception as e:
            print(f"\n错误：{e}")
    finally:
        _cleanup()

def _action_producer_loop(ros_operator,infer_controller,model,processor,prompt,provider,context,context_mask,args):
    """后台线程：采帧→请求server→写入历史缓存"""
    global _control_t, _action_chunks
    rate = rospy.Rate(50)  
    while not _action_stop_event.is_set() and not rospy.is_shutdown()and not stop_signal.is_set():
        with _action_lock:
            cursor = _control_t
            already_inferred = cursor in _action_chunks

        if already_inferred:
            rate.sleep()
            continue
        try:
            observation = provider.get_observation(ros_operator,infer_controller)
            actions = predict_action_chunk(
                model=model,
                processor=processor,
                observation=observation,
                prompt=prompt,
                context=context,
                context_mask=context_mask,
                args=args,
            )
            # action_chunk = actions[:args.execute_steps]
            action_chunk = actions.detach().cpu().numpy().astype(np.float32)
            _enqueue_chunk_to_expected_queues(action_chunk, cursor)
        except Exception as e:
            print(f"Inference error: {e}")
        rate.sleep()

def _cleanup():
    """程序退出时的清理工作（统一管理）"""
    print("\n正在退出...")
    _action_stop_event.set()
    
    if _action_prod_thread is not None and _action_prod_thread.is_alive():
        _action_prod_thread.join(timeout=2.0)
        print("producer线程已停止")
    
    print("Inference stopped.")
if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s,f: sys.exit(0))
    run()