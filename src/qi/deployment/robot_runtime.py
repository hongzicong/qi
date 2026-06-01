"""RTC-style Piper robot runtime for WAM policy deployment.

Supports three arm modes, selected by ``runtime.arm_mode`` in the config
(overridable from the CLI in deploy_real_rtc_wam.py):

  - ``"dual"``  : 14-dim policy driving both arms.
                  state/action layout = [L j0..j5, L grip, R j0..j5, R grip].
  - ``"right"`` : 7-dim policy driving the right arm only ([R j0..j5, R grip]).
                  The left arm is still connected/enabled and just parks at
                  ``left_init_position`` (it carries the left-wrist camera the
                  model still consumes) and is never commanded by the policy.
  - ``"left"``  : 7-dim policy driving the left arm only ([L j0..j5, L grip]).
                  Mirror of "right": the right arm parks and is never commanded.

Both Piper arms and all three RGB cameras are always connected regardless of
mode; only which arm(s) the policy reads state from and commands changes.

Gripper unit: all modes read the gripper as ``.value`` (meters, 0-0.1), which
matches the training-data gripper unit and the reference openpi right-only
deploy. The pyAgxArm parser already scales the raw CAN value (µm) to meters by
1e-6 in width mode, so ``.value`` is directly in meters. (Earlier pyAgxArm
exposed a ``.width`` alias; current versions only provide ``.value``.)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from qi.deployment.wam_remote_policy_client import WAMRemotePolicyClient
from qi.deployment.rtc_buffer import RealTimeChunkingBuffer


ARM_MODES = ("dual", "left", "right")
ACTION_DIM_DUAL = 14
ACTION_DIM_SINGLE = 7


def action_dim_for(arm_mode: str) -> int:
    """14-dim for dual arms, 7-dim for a single arm."""
    return ACTION_DIM_DUAL if arm_mode == "dual" else ACTION_DIM_SINGLE


def normalize_arm_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode not in ARM_MODES:
        raise ValueError(f"runtime.arm_mode must be one of {ARM_MODES}, got {value!r}")
    return mode


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"Config section `{name}` must be a mapping.")
    return value


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class DualArmRosOperator:
    """Synchronize the three RGB ROS image topics used by WAM.

    Identical for every arm mode: the model always consumes front + both wrist
    cameras, even when only one arm is commanded.
    """

    def __init__(self, config: Mapping[str, Any]):
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image

        self.rospy = rospy
        self.Image = Image
        self.bridge = CvBridge()
        self.cameras = _section(config, "cameras")
        self.topics = _section(config, "ros_topics_dual")
        self.deques = {
            "front": deque(maxlen=2000),
            "left_wrist": deque(maxlen=2000),
            "right_wrist": deque(maxlen=2000),
        }
        self.camera_names = {
            "front": self.cameras.get("front_camera_name", "cam_high"),
            "left_wrist": self.cameras.get("left_wrist_camera_name", "cam_left_wrist"),
            "right_wrist": self.cameras.get("right_wrist_camera_name", "cam_right_wrist"),
        }

        if not all(
            _bool(self.cameras.get(flag, True))
            for flag in ["use_front", "use_wrist", "use_left_wrist", "use_right_wrist"]
        ):
            raise ValueError("WAM deployment currently requires front, left wrist, and right wrist RGB cameras.")

        self.init_ros()

    def init_ros(self) -> None:
        self.rospy.init_node("wam_rtc_robot_runtime", anonymous=True)
        self.rospy.Subscriber(
            self.topics.get("img_front_topic", "/camera_h/color/image_raw"),
            self.Image,
            lambda msg: self.deques["front"].append(msg),
            queue_size=1000,
            tcp_nodelay=True,
        )
        self.rospy.Subscriber(
            self.topics.get("img_left_topic", "/camera_l/color/image_raw"),
            self.Image,
            lambda msg: self.deques["left_wrist"].append(msg),
            queue_size=1000,
            tcp_nodelay=True,
        )
        self.rospy.Subscriber(
            self.topics.get("img_right_topic", "/camera_r/color/image_raw"),
            self.Image,
            lambda msg: self.deques["right_wrist"].append(msg),
            queue_size=1000,
            tcp_nodelay=True,
        )

    def get_frame(self) -> dict[str, np.ndarray] | None:
        if any(len(image_deque) == 0 for image_deque in self.deques.values()):
            return None

        frame_time = min(image_deque[-1].header.stamp.to_sec() for image_deque in self.deques.values())
        if any(image_deque[-1].header.stamp.to_sec() < frame_time for image_deque in self.deques.values()):
            return None

        images = {}
        for camera_key, image_deque in self.deques.items():
            while image_deque and image_deque[0].header.stamp.to_sec() < frame_time:
                image_deque.popleft()
            if not image_deque:
                return None
            images[self.camera_names[camera_key]] = self.bridge.imgmsg_to_cv2(image_deque.popleft(), "passthrough")
        return images


class DualPiperController:
    """Read state and optionally execute Piper actions through pyAgxArm.

    Drives both arms in ``arm_mode="dual"`` (14-dim) or a single arm in
    ``arm_mode="left"`` / ``"right"`` (7-dim). Both arms are always connected
    and enabled; in single-arm mode the unused arm only parks at its
    ``*_init_position`` and is never commanded by the policy.
    """

    def __init__(self, config: Mapping[str, Any], arm_mode: str = "dual"):
        from pyAgxArm import AgxArmFactory, create_agx_arm_config

        self.AgxArmFactory = AgxArmFactory
        self.create_agx_arm_config = create_agx_arm_config
        self.robot = _section(config, "robot")
        self.arm_mode = normalize_arm_mode(arm_mode)
        self.action_dim = action_dim_for(self.arm_mode)
        self.left_arm = None
        self.right_arm = None
        self.left_gripper = None
        self.right_gripper = None
        self.left_init_position = np.asarray(
            self.robot.get("left_init_position", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05]),
            dtype=np.float32,
        )
        self.right_init_position = np.asarray(
            self.robot.get("right_init_position", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05]),
            dtype=np.float32,
        )
        # Policy state/action layout depends on the mode. In single-arm mode the
        # other arm just parks at its init pose and is never part of state/action.
        if self.arm_mode == "dual":
            self.initial_position = np.concatenate(
                [self.left_init_position, self.right_init_position]
            ).astype(np.float32)
        elif self.arm_mode == "left":
            self.initial_position = self.left_init_position.astype(np.float32).copy()
        else:  # right
            self.initial_position = self.right_init_position.astype(np.float32).copy()
        self.action_safety_threshold = float(self.robot.get("action_safety_threshold", 1.5))
        self.state_safety_threshold = float(self.robot.get("state_safety_threshold", 0.3))

    def connect(self) -> bool:
        try:
            bitrate = int(self.robot.get("bitrate", 1000000))
            cfg_l = self.create_agx_arm_config(
                robot="piper",
                comm="can",
                channel=self.robot.get("left_channel", "can_left"),
                bitrate=bitrate,
            )
            cfg_r = self.create_agx_arm_config(
                robot="piper",
                comm="can",
                channel=self.robot.get("right_channel", "can_right"),
                bitrate=bitrate,
            )
            self.left_arm = self.AgxArmFactory.create_arm(cfg_l)
            self.right_arm = self.AgxArmFactory.create_arm(cfg_r)
            self.left_gripper = self.left_arm.init_effector(self.left_arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
            self.right_gripper = self.right_arm.init_effector(self.right_arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
            self.left_arm.connect()
            self.right_arm.connect()
            time.sleep(0.5)
            if not (self.left_arm.is_ok() and self.right_arm.is_ok()):
                raise RuntimeError("Robot connection status check failed.")

            for arm in [self.left_arm, self.right_arm]:
                arm.set_flange_vel_acc_limits(
                    max_linear_vel=float(self.robot.get("max_linear_vel", 0.5)),
                    max_angular_vel=float(self.robot.get("max_angular_vel", 0.1)),
                    max_linear_acc=float(self.robot.get("max_linear_acc", 0.1)),
                    max_angular_acc=float(self.robot.get("max_angular_acc", 0.05)),
                    timeout=1.0,
                )
                arm.set_speed_percent(int(self.robot.get("speed_pct", 15)))
                if not self._enable_arm(arm):
                    raise RuntimeError("Robot arm enable timeout.")
            return True
        except Exception as exc:
            print(f"[robot] connect failed: {exc}")
            return False

    def _enable_arm(self, arm) -> bool:
        for _ in range(5):
            if arm.enable():
                return True
            time.sleep(0.5)
        return False

    def _read_arm_into(self, arm, gripper, state: np.ndarray, base: int) -> None:
        """Fill state[base:base+6] with joint angles and state[base+6] with gripper.

        Gripper is read as ``.value`` (meters, 0-0.1; the parser already scales
        raw µm by 1e-6 in width mode) to match the training-data unit and the
        validated right-only deploy.
        """
        ja = arm.get_joint_angles()
        if ja is not None:
            state[base:base + 6] = ja.msg
        if gripper is not None:
            gs = gripper.get_gripper_status()
            if gs is not None:
                state[base + 6] = gs.msg.value

    def get_status_and_state(self) -> np.ndarray:
        state = np.zeros(self.action_dim, dtype=np.float32)
        try:
            if self.arm_mode == "dual":
                # 14-dim: [L j0..j5, L grip, R j0..j5, R grip]
                self._read_arm_into(self.left_arm, self.left_gripper, state, 0)
                self._read_arm_into(self.right_arm, self.right_gripper, state, 7)
            elif self.arm_mode == "left":
                # 7-dim left-arm state: [L j0..j5, L grip]
                self._read_arm_into(self.left_arm, self.left_gripper, state, 0)
            else:  # right
                # 7-dim right-arm state: [R j0..j5, R grip]
                self._read_arm_into(self.right_arm, self.right_gripper, state, 0)
        except Exception as exc:
            print(f"[robot] state read failed: {exc}")
        return state

    def move_initial(self) -> None:
        """Park both arms at their init poses (done in every mode).

        Reads each arm's joint angles directly (not via get_status_and_state,
        which is arm_mode-shaped) so the parking trajectory is mode-independent.
        """
        n = 50

        def _cur(arm, fallback):
            ja = arm.get_joint_angles()
            return np.asarray(ja.msg, dtype=np.float32) if ja is not None else np.asarray(fallback, dtype=np.float32)

        left_traj = np.linspace(_cur(self.left_arm, self.left_init_position[0:6]), self.left_init_position[0:6], n)
        right_traj = np.linspace(_cur(self.right_arm, self.right_init_position[0:6]), self.right_init_position[0:6], n)
        left_gripper = float(np.clip(self.left_init_position[6], 0.0, 0.1))
        right_gripper = float(np.clip(self.right_init_position[6], 0.0, 0.1))
        for i in range(n):
            self.left_arm.move_js(left_traj[i].tolist())
            self.left_gripper.move_gripper_m(value=left_gripper, force=1.0)
            self.right_arm.move_js(right_traj[i].tolist())
            self.right_gripper.move_gripper_m(value=right_gripper, force=1.0)
            time.sleep(0.02)

    def _move_arm(self, arm, gripper, joints: np.ndarray, gripper_cmd: float) -> None:
        arm.move_js(joints.tolist())
        gripper.move_gripper_m(value=float(np.clip(gripper_cmd, 0.0, 0.1)), force=1.0)

    def move(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        if self.arm_mode == "dual":
            # 14-dim layout [L j0..j5, L grip, R j0..j5, R grip]:
            # left = joints action[0:6] + grip action[6]; right = joints action[7:13] + grip action[13].
            self._move_arm(self.left_arm, self.left_gripper, action[0:6], action[6])
            self._move_arm(self.right_arm, self.right_gripper, action[7:13], action[13])
        elif self.arm_mode == "left":
            # 7-dim left action; right arm left untouched at its park pose.
            self._move_arm(self.left_arm, self.left_gripper, action[0:6], action[6])
        else:  # right
            # 7-dim right action; left arm left untouched at its park pose.
            self._move_arm(self.right_arm, self.right_gripper, action[0:6], action[6])
        time.sleep(0.02)


class RTCWAMRobotRuntime:
    """Robot-side runtime that talks to a persistent WAM policy server."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        self.wam = _section(config, "wam")
        self.runtime = _section(config, "runtime")
        self.policy_server = _section(config, "policy_server")
        self.arm_mode = normalize_arm_mode(self.runtime.get("arm_mode", "dual"))
        self.action_dim = action_dim_for(self.arm_mode)
        print(f"[runtime] arm_mode={self.arm_mode} (action_dim={self.action_dim})")
        chunk_size = int(self.runtime.get("action_chunk_size", self.wam.get("action_horizon", 32)))
        self.rtc = RealTimeChunkingBuffer(
            chunk_size=chunk_size,
            exp_weight_factor=float(self.runtime.get("exp_weight_factor", 0.5)),
            debug=_bool(self.runtime.get("rtc_debug", False)),
        )
        self.execute_actions = _bool(self.runtime.get("execute_actions", False))
        self.max_steps = self.runtime.get("max_steps")
        self.max_steps = None if self.max_steps is None else int(self.max_steps)
        self.output_dir = Path(str(self.runtime.get("output_dir", "./output_actions"))).expanduser()
        self.stop_event = threading.Event()
        self.producer_thread: threading.Thread | None = None
        self.selected_actions: list[np.ndarray] = []

    def run(self) -> bool:
        ros_operator = DualArmRosOperator(self.config)
        robot = DualPiperController(self.config, arm_mode=self.arm_mode)
        if not robot.connect():
            return False

        policy = WAMRemotePolicyClient(
            host=str(self.policy_server.get("host", "127.0.0.1")),
            port=int(self.policy_server.get("port", 8765)),
            authkey=self.policy_server.get("authkey", "wam"),
        )
        try:
            if _bool(self.runtime.get("move_to_initial", False)):
                robot.move_initial()

            self.rtc.clear()
            self.stop_event.clear()
            self.selected_actions = []
            input("Press Enter to start RTC WAM robot runtime...")

            self.producer_thread = threading.Thread(
                target=self._producer_loop,
                args=(ros_operator, robot, policy),
                daemon=True,
            )
            self.producer_thread.start()
            self._control_loop(ros_operator, robot)
            return True
        finally:
            self.stop_event.set()
            if self.producer_thread is not None and self.producer_thread.is_alive():
                self.producer_thread.join(timeout=2.0)
            try:
                policy.reset()
            finally:
                policy.close()
            self._save_selected_actions()

    def _producer_loop(self, ros_operator: DualArmRosOperator, robot: DualPiperController, policy: WAMRemotePolicyClient) -> None:
        rate = ros_operator.rospy.Rate(int(self.runtime.get("rospy_rate", 50)))
        printed_wait = False
        while not self.stop_event.is_set() and not ros_operator.rospy.is_shutdown():
            cursor = self.rtc.get_control_time()
            generation = self.rtc.get_generation()
            if self.rtc.has_chunk(cursor):
                rate.sleep()
                continue

            images = ros_operator.get_frame()
            if images is None:
                if not printed_wait:
                    print("[producer] waiting for synchronized ROS frames...")
                    printed_wait = True
                rate.sleep()
                continue
            printed_wait = False

            obs = {
                "images": images,
                "state": robot.get_status_and_state(),
                "prompt": self.wam.get("prompt", "clean the table"),
            }
            try:
                t0 = time.perf_counter()
                action_chunk = np.asarray(policy.infer(obs), dtype=np.float32)
                infer_dt = time.perf_counter() - t0
                rate_hz = max(int(self.runtime.get("rospy_rate", 50)), 1)
                budget = self.rtc.chunk_size / rate_hz
                tag = "WARN starve" if infer_dt > budget else "ok"
                print(f"[producer] infer {infer_dt * 1000:.0f}ms budget {budget * 1000:.0f}ms ({tag})")
                if action_chunk.ndim != 2 or action_chunk.shape[1] != self.action_dim:
                    raise ValueError(f"Expected action chunk [T,{self.action_dim}], got {action_chunk.shape}")
                action_chunk = action_chunk[: self.rtc.chunk_size]
                self.rtc.enqueue(action_chunk, cursor, generation=generation)
            except Exception as exc:
                print(f"[producer] inference failed: {exc}")
                self.stop_event.set()
                break
            rate.sleep()

    def _control_loop(self, ros_operator: DualArmRosOperator, robot: DualPiperController) -> None:
        rate = ros_operator.rospy.Rate(int(self.runtime.get("rospy_rate", 50)))
        last_action = robot.get_status_and_state()
        print(f"[runtime] start_state={last_action}")
        t = 0
        while not self.stop_event.is_set() and not ros_operator.rospy.is_shutdown():
            if self.max_steps is not None and t >= self.max_steps:
                print(f"[runtime] reached max_steps={self.max_steps}")
                break

            self.rtc.set_control_time(t)
            action = self.rtc.get_action(t)
            if action is None:
                print("[runtime] waiting for action...")
                rate.sleep()
                continue

            action = np.asarray(action, dtype=np.float32)
            if action.shape != (self.action_dim,):
                print(f"[runtime] expected action [{self.action_dim}], got {action.shape}")
                break

            prev_action_l1 = float(np.mean(np.abs(action - last_action)))
            if prev_action_l1 > robot.action_safety_threshold:
                print(f"[runtime] safety stop: action jump too large {prev_action_l1:.3f}")
                break

            current_state = robot.get_status_and_state()
            state_tracking_l1 = float(np.mean(np.abs(current_state - last_action)))
            if state_tracking_l1 > robot.state_safety_threshold:
                print(f"[runtime] safety stop: current state too far from last action {state_tracking_l1:.3f}")
                break

            target_state_l1 = float(np.mean(np.abs(current_state - action)))
            print(
                f"[runtime] t={t} action_l1={prev_action_l1:.3f} "
                f"state_tracking_l1={state_tracking_l1:.3f} target_state_l1={target_state_l1:.3f} "
                f"action={action}"
            )
            self.selected_actions.append(action.copy())
            if self.execute_actions:
                robot.move(action)
            last_action = action
            t += 1
            rate.sleep()

    def _save_selected_actions(self) -> None:
        if not self.selected_actions:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "rtc_selected_actions.npy"
        np.save(output_path, np.asarray(self.selected_actions, dtype=np.float32))
        print(f"[runtime] saved selected actions to {output_path}")