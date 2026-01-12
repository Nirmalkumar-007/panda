import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R, Slerp
import numpy as np
import time

import panda_py
from panda_py import controllers


class PandaCmdSub(Node):

    def __init__(self, desk: panda_py.Desk, panda: panda_py.Panda):
        super().__init__('panda_cmd_sub_trajectory')

        self.desk = desk
        self.panda = panda

        # Save TRUE robot home
        self.home_pos  = self.panda.get_position()
        self.home_quat = self.panda.get_orientation()
        self.home_q    = self.panda.get_state().q.copy()   # ← correct API

        print("True Home Cartesian:", self.home_pos)
        print("True Home Joints:", self.home_q)

        # Start Cartesian Impedance
        self.ctrl = controllers.CartesianImpedance()
        self.panda.start_controller(self.ctrl)

        # Waypoints
        self.point_A = self.home_pos + np.array([0.1, -0.3, -0.3])
        self.point_B = self.point_A + np.array([0.0, 0.3, 0.0])

        self.waypoints = [
            self.home_pos,
            self.point_A,
            self.point_B,
            self.home_pos
        ]

        self.orientations = [
            self.home_quat,
            self.home_quat,
            self.home_quat,
            self.home_quat
        ]

        self.steps = 60

        self.run_trajectory()

  
    # Trajectory execution
   
    def run_trajectory(self):

        for i in range(len(self.waypoints) - 1):
            p0 = self.waypoints[i]
            p1 = self.waypoints[i + 1]

            q0 = R.from_quat(self.orientations[i])
            q1 = R.from_quat(self.orientations[i + 1])

            slerp = Slerp([0, 1], R.from_quat([q0.as_quat(), q1.as_quat()]))

            for a in np.linspace(0, 1, self.steps):
                pos  = (1 - a) * p0 + a * p1
                quat = slerp([a])[0].as_quat()

                self.ctrl.set_control(pos, quat)
                time.sleep(0.05)   # 50 Hz

        print("Cartesian path finished")

        # Hard correction to true home
       
        print("Returning to exact joint home...")

        joint_ctrl = controllers.JointPosition()
        self.panda.start_controller(joint_ctrl)
        self.panda.move_to_joint_position(self.home_q)

        print("Robot is now at exact original home.")

# Main

def main(args=None):

    rclpy.init(args=args)

    desk = panda_py.Desk("192.168.1.11", "BRL", "IITADVRBRL")
    panda = panda_py.Panda("192.168.1.11")

    desk.unlock()
    desk.activate_fci()

    try:
        PandaCmdSub(desk, panda)
    except KeyboardInterrupt:
        print("Stopped")

    desk.deactivate_fci()
    desk.lock()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
