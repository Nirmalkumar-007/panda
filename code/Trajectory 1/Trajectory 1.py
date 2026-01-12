import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R, Slerp
import numpy as np
import time

import panda_py
from panda_py import controllers

# PandaCmdSub Node

class PandaCmdSub(Node):

    def __init__(self, desk: panda_py.Desk, panda: panda_py.Panda):
        super().__init__('panda_cmd_sub_trajectory')
        self.desk = desk
        self.panda = panda

        # Cartesian impedance controller
        self.ctrl = controllers.CartesianImpedance()
        self.panda.start_controller(self.ctrl)

        # Home pose
        self.home_pos = self.panda.get_position()
        self.home_quat = self.panda.get_orientation()  # [x, y, z, w]
        print("Home pose:", self.home_pos)

        # Define trajectory waypoints
        self.point_A = self.home_pos + np.array([0.2, 0.0, 0.0])  # 20 cm forward
        self.point_B = self.point_A + np.array([0.0, 0.2, 0.0])   # 20 cm sideways
        self.waypoints = [self.home_pos, self.point_A, self.point_B, self.home_pos]

        # Orientation for each waypoint (can rotate if needed)
        self.orientations = [self.home_quat, self.home_quat, self.home_quat, self.home_quat]

        # Trajectory parameters
        self.steps_per_segment = 50
        self.run_trajectory()
    
    # Linear interpolation between positions
    
    def linear_interpolation(self, start, end, steps=50):
        return [(1 - t/(steps-1))*start + (t/(steps-1))*end for t in range(steps)]

    # Run trajectory with SLERP for orientation
 
    def run_trajectory(self):
        for i in range(len(self.waypoints)-1):
            pos_start = self.waypoints[i]
            pos_end   = self.waypoints[i+1]

            quat_start = R.from_quat(self.orientations[i])
            quat_end   = R.from_quat(self.orientations[i+1])

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
                time.sleep(0.05)  # ~20 Hz update

        print("Trajectory complete!")

# Main

def main(args=None):
    rclpy.init(args=args)

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

if __name__ == '__main__':
    main()
