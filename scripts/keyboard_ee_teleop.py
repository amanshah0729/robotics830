#!/usr/bin/env python
"""Game-style keyboard teleop for the SO-101 follower. Two modes, TAB to switch.

  CLAW MODE (default) - drive the claw tip through space, IK does the rest
    W/S  forward/back   A/D  left/right   SPACE up   C  down
    arrows  aim claw (pitch/yaw)   Q/E  roll
  JOINT MODE - drive each motor directly, full manual control
    A/D  base rotate    W/S  shoulder up/down   arrows up/down  elbow
    arrows left/right  wrist bend   Q/E  wrist roll
  BOTH MODES
    F close claw (hold)   R open claw (hold)
    SHIFT sprint x3   ALT/OPTION precision x0.25
    H (hold) glide back to the calibrated middle pose
    P pause/resume — freezes the arm and ignores keys so you can type elsewhere
    TAB switch mode   ESC quit (arm holds pose)

Claw-mode W/A/S/D are TABLE directions (fixed), not claw-facing directions.

Keys are global hotkeys (terminal app needs Accessibility permission) - they
fire even when another window has focus.

Run:
    /Users/rickys/Documents/Dev/lerobot/.venv/bin/python \
        /Users/rickys/Documents/Dev/robotics830/scripts/keyboard_ee_teleop.py
"""

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

import numpy as np
from pynput import keyboard as pk

from lerobot.model.kinematics import RobotKinematics
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.processor import (
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import InverseKinematicsEEToJoints
from lerobot.utils.rotation import Rotation

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from musclememory.glasses_jog import GlassesJog  # noqa: E402
DEFAULT_URDF = REPO_ROOT / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"

# ----- KEYMAP (macOS ANSI virtual keycodes: letters tracked by keycode so
# ----- shift/option can't morph them into other characters mid-hold) ---------
VK_A, VK_S, VK_D, VK_F, VK_H, VK_C, VK_Q, VK_W, VK_E, VK_R = 0, 1, 2, 3, 4, 8, 12, 13, 14, 15

FWD, BACK = ("vk", VK_W), ("vk", VK_S)
LEFT, RIGHT = ("vk", VK_A), ("vk", VK_D)
UP, DOWN = pk.Key.space, ("vk", VK_C)
PITCH_UP, PITCH_DOWN = pk.Key.up, pk.Key.down
YAW_LEFT, YAW_RIGHT = pk.Key.left, pk.Key.right
ROLL_CCW, ROLL_CW = ("vk", VK_Q), ("vk", VK_E)
GRIP_CLOSE, GRIP_OPEN = ("vk", VK_F), ("vk", VK_R)
HOME = ("vk", VK_H)
PAUSE = ("vk", 35)  # P — freeze the arm and ignore keys (type safely elsewhere)
MODE_TOGGLE = pk.Key.tab
SPRINT = {pk.Key.shift, pk.Key.shift_l, pk.Key.shift_r}
PRECISION = {pk.Key.alt, pk.Key.alt_l, pk.Key.alt_r}

# JOINT mode: motor -> (positive key, negative key). Flip a pair if a joint
# moves opposite to what feels right.
JOINT_AXES = [
    ("shoulder_pan", LEFT, RIGHT),
    ("shoulder_lift", FWD, BACK),
    ("elbow_flex", PITCH_UP, PITCH_DOWN),
    ("wrist_flex", YAW_LEFT, YAW_RIGHT),
    ("wrist_roll", ROLL_CCW, ROLL_CW),
]
# Meta Ray-Ban Display sends four directions across two modes -- exactly the
# four degrees of freedom claw mode consumes. Orientation is left out on
# purpose: at --orient-weight 0.1 the IK is effectively position-only, so it
# picks better wrist angles than an operator could at 4 fps over a relay.
GLASSES_AXES = {
    "+X": FWD,  "-X": BACK,
    "+Y": RIGHT, "-Y": LEFT,
    "+Z": UP,   "-Z": DOWN,
    "GRIP+": GRIP_OPEN, "GRIP-": GRIP_CLOSE,
}

HOME_POSE = {  # calibrated middle pose: 0 deg on every joint, claw half-open
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 50.0,
}
# -----------------------------------------------------------------------------


@contextlib.contextmanager
def _quiet_native_output():
    """placo prints harmless self-collision notes for the SO-101 model on load."""
    saved = [os.dup(1), os.dup(2)]
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, devnull):
            os.close(fd)


class Keys:
    """Global key-state tracker; normalizes events so modifiers can't corrupt them."""

    def __init__(self) -> None:
        self.pressed: set = set()
        self.stop = False
        self._listener = pk.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    @staticmethod
    def _norm(key):
        if isinstance(key, pk.KeyCode) and key.vk is not None:
            return ("vk", key.vk)
        return key

    def _on_press(self, key) -> None:
        if key == pk.Key.esc:
            self.stop = True
            return
        self.pressed.add(self._norm(key))

    def _on_release(self, key) -> None:
        self.pressed.discard(self._norm(key))

    def axis(self, positive, negative) -> float:
        return float(positive in self.pressed) - float(negative in self.pressed)

    def any_held(self, keys) -> bool:
        return bool(self.pressed & keys)

    def close(self) -> None:
        self._listener.stop()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/tty.usbmodem5AE60798061", help="Follower serial port")
    parser.add_argument("--id", default="hack_follower", help="Calibration id used with lerobot-calibrate")
    parser.add_argument("--urdf", default=str(DEFAULT_URDF), help="Path to so101_new_calib.urdf")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mode", choices=["claw", "joint"], default="claw", help="Starting control mode")
    parser.add_argument("--step", type=float, default=0.003, help="Claw-mode metres per tick at base speed")
    parser.add_argument("--joint-step", type=float, default=1.2, help="Joint-mode degrees per tick at base speed")
    parser.add_argument("--orient-step", type=float, default=0.015, help="Claw-mode wrist radians per tick")
    parser.add_argument("--orient-weight", type=float, default=0.1, help="IK orientation weight (0 = position only)")
    parser.add_argument("--sprint", type=float, default=3.0, help="Speed multiplier while SHIFT held")
    parser.add_argument("--precision", type=float, default=0.25, help="Speed multiplier while ALT held")
    parser.add_argument("--gripper-step", type=float, default=2.0, help="Gripper units per tick")
    parser.add_argument("--leash", type=float, default=15.0, help="Max degrees a joint target may lead the real joint")
    parser.add_argument("--ee-leash", type=float, default=0.06, help="Max metres the claw target may lead the real claw")
    parser.add_argument("--glasses", default=None,
                        help="Muscle Memory server URL to take Ray-Ban Display jogs from, "
                             "e.g. https://<name>.trycloudflare.com")
    parser.add_argument("--glasses-hold", type=float, default=0.25,
                        help="Seconds a glasses swipe holds its key (one swipe = one nudge)")
    args = parser.parse_args()

    if not Path(args.urdf).exists():
        raise SystemExit(
            f"URDF not found at {args.urdf}\n"
            "Fetch it with:\n"
            "  git clone --depth 1 --filter=blob:none --sparse "
            f"https://github.com/TheRobotStudio/SO-ARM100 {REPO_ROOT / 'SO-ARM100'} && "
            f"git -C {REPO_ROOT / 'SO-ARM100'} sparse-checkout set Simulation/SO101"
        )

    robot = SO101Follower(
        SO101FollowerConfig(port=args.port, id=args.id, cameras={}, use_degrees=True)
    )
    motor_names = list(robot.bus.motors.keys())
    arm_joints = [n for n in motor_names if n != "gripper"]

    with _quiet_native_output():
        kinematics = RobotKinematics(
            urdf_path=args.urdf,
            target_frame_name="gripper_frame_link",
            joint_names=motor_names,
        )

    ee_to_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=motor_names,
                initial_guess_current_joints=True,
                orientation_weight=args.orient_weight,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    keys = Keys()

    # Glasses jogs land in the same pressed-key set the listener writes to, so
    # they travel the existing code path: speed scaling, leashing, IK and mode
    # all apply, and the keyboard keeps working alongside as a manual override
    # -- worth having when the demo depends on a tunnel staying up.
    glasses = None
    if args.glasses:
        glasses = GlassesJog(args.glasses, GLASSES_AXES, keys.pressed,
                             hold_s=args.glasses_hold,
                             on_event=lambda m: print(f"  [glasses] {m}", flush=True)).start()
        print(f"Taking Ray-Ban Display jogs from {args.glasses}")
    robot.connect()
    print(__doc__)
    mode = args.mode
    print(f"Connected in {mode.upper()} mode. TAB switches mode, ESC quits.")

    # Persistent targets: keys move these; the robot chases them. Targets are
    # leashed to the measured pose so gravity sag can't swallow key input and
    # end-stops can't cause wind-up.
    joint_targets: dict[str, float] = {}
    ee_pose: np.ndarray | None = None
    last_ctrl = None

    dt = 1.0 / args.fps
    tab_was_down = False
    pause_was_down = False
    paused = False
    last_action = None
    try:
        while not keys.stop:
            t0 = time.perf_counter()

            obs = robot.get_observation()
            meas = {n: float(obs[f"{n}.pos"]) for n in motor_names}

            pause_down = PAUSE in keys.pressed
            if pause_down and not pause_was_down:
                paused = not paused
                print("PAUSED — keys ignored, arm holding. P to resume." if paused else "RESUMED")
                if not paused:
                    last_ctrl = None  # re-sync targets to wherever the arm is now
            pause_was_down = pause_down

            if paused:
                robot.send_action(last_action or {f"{n}.pos": meas[n] for n in motor_names})
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                continue

            tab_down = MODE_TOGGLE in keys.pressed
            if tab_down and not tab_was_down:
                mode = "joint" if mode == "claw" else "claw"
                print(f"MODE: {mode.upper()}")
            tab_was_down = tab_down

            ctrl = "home" if HOME in keys.pressed else mode
            if ctrl != last_ctrl:  # entering a controller: sync its target to reality
                joint_targets = dict(meas)
                q = np.array([meas[n] for n in motor_names])
                ee_pose = kinematics.forward_kinematics(q)
                last_ctrl = ctrl

            speed = 1.0
            if keys.any_held(SPRINT):
                speed *= args.sprint
            if keys.any_held(PRECISION):
                speed *= args.precision

            grip_dir = keys.axis(GRIP_OPEN, GRIP_CLOSE)  # +1 open, -1 close
            joint_targets["gripper"] = clamp(
                joint_targets["gripper"] + grip_dir * args.gripper_step * speed, 0.0, 100.0
            )

            if ctrl == "home":
                for name, target in HOME_POSE.items():
                    step = (args.joint_step if name != "gripper" else args.gripper_step) * speed
                    cur = joint_targets[name]
                    joint_targets[name] = cur + clamp(target - cur, -step, step)
                for name in arm_joints:
                    joint_targets[name] = clamp(
                        joint_targets[name], meas[name] - args.leash, meas[name] + args.leash
                    )
                action = {f"{n}.pos": joint_targets[n] for n in motor_names}
            elif ctrl == "joint":
                for name, pos_key, neg_key in JOINT_AXES:
                    joint_targets[name] += keys.axis(pos_key, neg_key) * args.joint_step * speed
                    joint_targets[name] = clamp(
                        joint_targets[name], meas[name] - args.leash, meas[name] + args.leash
                    )
                action = {f"{n}.pos": joint_targets[n] for n in motor_names}
            else:  # claw mode: move a persistent EE pose target, IK chases it
                tx = keys.axis(LEFT, RIGHT)
                ty = keys.axis(BACK, FWD)
                tz = keys.axis(UP, DOWN)
                ee_pose[:3, 3] += np.array([tx, ty, tz]) * args.step * speed

                wx = keys.axis(PITCH_UP, PITCH_DOWN) * args.orient_step * speed
                wy = keys.axis(YAW_LEFT, YAW_RIGHT) * args.orient_step * speed
                wz = keys.axis(ROLL_CCW, ROLL_CW) * args.orient_step * speed
                if wx or wy or wz:
                    ee_pose[:3, :3] = ee_pose[:3, :3] @ Rotation.from_rotvec([wx, wy, wz]).as_matrix()
                    # round-trip through rotvec to re-orthonormalize (accumulated
                    # matrix products drift numerically and eventually corrupt IK)
                    ee_pose[:3, :3] = Rotation.from_rotvec(
                        Rotation.from_matrix(ee_pose[:3, :3]).as_rotvec()
                    ).as_matrix()

                # leash the position target to the claw's real position
                q = np.array([meas[n] for n in motor_names])
                real = kinematics.forward_kinematics(q)[:3, 3]
                offset = ee_pose[:3, 3] - real
                dist = float(np.linalg.norm(offset))
                if dist > args.ee_leash:
                    ee_pose[:3, 3] = real + offset * (args.ee_leash / dist)

                rotvec = Rotation.from_matrix(ee_pose[:3, :3]).as_rotvec()
                ee_action = {
                    "ee.x": float(ee_pose[0, 3]),
                    "ee.y": float(ee_pose[1, 3]),
                    "ee.z": float(ee_pose[2, 3]),
                    "ee.wx": float(rotvec[0]),
                    "ee.wy": float(rotvec[1]),
                    "ee.wz": float(rotvec[2]),
                    "ee.gripper_pos": joint_targets["gripper"],
                }
                action = ee_to_joints((ee_action, obs))
                # Rate-leash the IK output: near singularities the solver can
                # jump to a distant joint configuration — never let one tick
                # command more than the leash away from where the arm really is.
                for n in arm_joints:
                    key = f"{n}.pos"
                    if key in action:
                        action[key] = clamp(
                            float(action[key]), meas[n] - args.leash, meas[n] + args.leash
                        )

            robot.send_action(action)
            last_action = action

            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        if glasses is not None:
            glasses.stop()
        keys.close()
        robot.disconnect()
        print("Disconnected cleanly.")


if __name__ == "__main__":
    main()
