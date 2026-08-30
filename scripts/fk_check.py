#!/usr/bin/env python
"""Live kinematics truth-meter for the SO-101.

Connects with torque OFF (arm stays limp — move it by hand) and prints each
joint's angle in degrees plus the model's claw position (x, y, z metres)
twice a second. Ctrl+C to quit.

What to check:
  1. Hold the arm in the calibration video's zigzag pose
     (upper arm vertical, forearm horizontal) -> every joint should read ~0.
     A joint reading tens of degrees there means the calibration zero pose
     was off for that joint -> recalibrate holding the video pose.
  2. Raise the claw by hand -> z must increase. Move it away from the base
     -> x or y should change smoothly. If the numbers fight what your hands
     are doing, the model/calibration mismatch is on that axis.

    /Users/rickys/Documents/Dev/lerobot/.venv/bin/python scripts/fk_check.py
"""

import argparse
import time
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URDF = REPO_ROOT / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/tty.usbmodem5AE60798061")
    ap.add_argument("--id", default="hack_follower")
    ap.add_argument("--urdf", default=str(DEFAULT_URDF))
    args = ap.parse_args()

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, cameras={}, use_degrees=True))
    robot.connect()
    robot.bus.disable_torque()
    print("Torque OFF — move the arm by hand. Ctrl+C to quit.")

    kin = RobotKinematics(
        urdf_path=args.urdf,
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )

    names = list(robot.bus.motors.keys())
    print("  ".join(f"{n[:9]:>9s}" for n in names) + "  |      x       y       z")
    try:
        while True:
            obs = robot.get_observation()
            q = np.array([float(obs[f"{n}.pos"]) for n in names])
            xyz = kin.forward_kinematics(q)[:3, 3]
            print(
                "  ".join(f"{v:9.1f}" for v in q)
                + f"  | {xyz[0]:6.3f}  {xyz[1]:6.3f}  {xyz[2]:6.3f}",
                flush=True,
            )
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
