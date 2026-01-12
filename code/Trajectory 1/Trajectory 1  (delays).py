import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R, Slerp
import numpy as np
import time

import panda_py
from panda_py import controllers


class PandaCmdSub(Node):
    """
    ROS2 Node to move the Panda robot along a trajectory with waypoints,
    using Cartesian impedance control and SLERP orientation interpolation.
    Includes 5-second pauses at each waypoint.
    """

    def __init__(self, desk: panda_py.Desk, panda: panda_py.Panda):
        super().__init__('panda_cmd_sub_trajectory')
        self.desk = desk
        self.panda = panda

        # ----------------------------
        # Cartesian impedance controller
        # ----------------------------
        self.ctrl = controllers.CartesianImpedance()
        # Set safe stiffness and damping
        self.ctrl.set_stiffness([300, 300, 300, 20, 20, 20])
        self.ctrl.set_damping_ratio(1.0)
        self.panda.start_controller(self.ctrl)

        # ----------------------------
        # Home pose
        # ----------------------------
        self.home_pos = self.panda.get_position()
        self.home_quat = self.panda.get_orientation()  # [x, y, z, w]
        print("Home pose:", self.home_pos)

        # ----------------------------
        # Trajectory waypoints
        # ----------------------------
        self.point_A = self.home_pos + np.array([0.1, -0.3, -0.3])
        self.point_B = self.point_A + np.array([0.0, 0.3, 0.0])
        self.waypoints = [self.home_pos, self.point_A, self.point_B, self.home_pos]

        # Orientations (keep same orientation)
        self.orientations = [self.home_quat] * len(self.waypoints)

        # Trajectory parameters
        self.steps_per_segment = 50

        # ----------------------------
        # Run the trajectory
        # ----------------------------
        self.run_trajectory()

    def run_trajectory(self):
        """
        Move the robot through all waypoints using linear interpolation for position
        and SLERP for orientation. Pause 5 seconds at each waypoint.
        """
        for i in range(len(self.waypoints) - 1):
            pos_start = self.waypoints[i]
            pos_end = self.waypoints[i + 1]

            quat_start = R.from_quat(self.orientations[i])
            quat_end = R.from_quat(self.orientations[i + 1])

            # Create SLERP object
            key_times = [0, 1]
            key_rots = R.from_quat([quat_start.as_quat(), quat_end.as_quat()])
            slerp = Slerp(key_times, key_rots)

            # Interpolate positions and orientations
            for alpha in np.linspace(0, 1, self.steps_per_segment):
                pos = (1 - alpha) * pos_start + alpha * pos_end
                quat = slerp([alpha])[0].as_quat()  # quaternion

                # Send to Cartesian impedance controller
                self.ctrl.set_control(pos, quat)
                time.sleep(0.08)  # slower motion (~12.5 Hz)

            # Pause 5 seconds at the waypoint
            print(f"Reached waypoint {i + 1}, pausing 5 seconds...")
            time.sleep(5.0)

        # ----------------------------
        # Return to exact home position safely
        # ----------------------------
        home_joint_positions = self.panda.get_joint_angles()  # use joint angles for exact home
        self.panda.move_to_joint_position(home_joint_positions, speed_factor=0.2)
        print("Returned to home position, pausing 5 seconds...")
        time.sleep(5.0)

        print("Trajectory complete!")


# ----------------------------
# Main
# ----------------------------
def main(args=None):
    rclpy.init(args=args)

    # Connect to Panda
    desk = panda_py.Desk("192.168.1.11", "BRL", "IITADVRBRL")
    panda = panda_py.Panda("192.168.1.11")

    desk.unlock()
    desk.activate_fci()

    try:
        panda_cmd_sub = PandaCmdSub(desk, panda)
    except KeyboardInterrupt:
        print("Stopping robot...")

    # Shutdown safely
    desk.deactivate_fci()
    desk.lock()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
