import rclpy
from rclpy.node import Node
from rclpy import executors

from panda_interfaces.msg import PandaCommand
from panda_interfaces.msg import PandaPose
from panda_interfaces.msg import PandaStatus
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R

import panda_py
from panda_py import constants
from panda_py import controllers
import numpy as np
import time

class PandaCmdSub(Node):
    """Listens to commands and moves the Panda robot"""
    
    def __init__(self, desk: panda_py.Desk, panda: panda_py.Panda):
        super().__init__('panda_status_subscriber')
        self.absolute_pose = True
        self.motion_type = 1
        self.desk: panda_py.Desk = desk
        self.panda: panda_py.Panda = panda

        self.panda_pose_pub = PandaPosePub()

        self.status_subscription = self.create_subscription(
            PandaStatus,
            'panda_status_topic',
            self.status_callback,
            10
        )
        self.cmd_subscription = self.create_subscription(
            Pose,
            'panda_cmd_topic',
            self.cmd_callback,
            10
        )

        print("Starting pose:\n", panda.get_pose())
        self.pub_actual_pose()

    def status_callback(self, msg: PandaStatus):
        if not msg.run:
            rclpy.shutdown()
            return
        elif not msg.unlocked:
            self.desk.deactivate_fci()
            self.desk.lock()
            return
        else:
            self.desk.unlock()
            self.desk.activate_fci()

        self.absolute_pose = msg.absolute_pose
        self.motion_type = msg.motion_type
        if self.motion_type == 0:
            self.panda.move_to_start()
            print("Destination pose:\n", self.panda.get_pose())
            self.pub_actual_pose()
            return

    def cmd_callback(self, msg: Pose):
        cmd_rotation = R.from_quat([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        cmd_position = np.array([msg.position.x, msg.position.y, msg.position.z])

        position = self.panda.get_position()
        tmp_orientation = R.from_quat(self.panda.get_orientation())

        if self.absolute_pose:
            position = cmd_position
            orientation = (cmd_rotation.inv() * tmp_orientation) * tmp_orientation
            print("Moving to absolute pose")
        else:
            orientation = cmd_rotation * tmp_orientation
            position += cmd_position
            print("Moving to relative pose")

        pose = np.eye(4)
        pose[0:3, 0:3] = orientation.as_matrix()
        pose[0:3, 3] = position
        print("Moving to pose:\n", pose)

        if self.motion_type == 1:
            ctrl = controllers.CartesianImpedance()
            self.panda.start_controller(ctrl)
            ctrl.set_control(position, orientation.as_quat())
        elif self.motion_type == 2:
            q = panda_py.ik(pose)
            print("IK solution:\n", q)
            ctrl = controllers.JointPosition()
            self.panda.start_controller(ctrl)
            self.panda.move_to_joint_position(q)

        time.sleep(0.5)
        print("Destination pose:\n", self.panda.get_pose())
        self.pub_actual_pose()
        return

    def pub_actual_pose(self):
        actual_orientation = self.panda.get_orientation()
        self.panda_pose_pub.msg.position.x = self.panda.get_position()[0]
        self.panda_pose_pub.msg.position.y = self.panda.get_position()[1]
        self.panda_pose_pub.msg.position.z = self.panda.get_position()[2]
        self.panda_pose_pub.msg.orientation.x = actual_orientation[0]
        self.panda_pose_pub.msg.orientation.y = actual_orientation[1]
        self.panda_pose_pub.msg.orientation.z = actual_orientation[2]
        self.panda_pose_pub.msg.orientation.w = actual_orientation[3]
        self.panda_pose_pub.publish_wrapper()

    # --------------------------
    # New Trajectory Feature
    # --------------------------
    def linear_interpolate(self, start, end, steps=50):
        """Returns a list of linearly interpolated positions"""
        return [start + (end - start) * i / steps for i in range(1, steps + 1)]

    def move_trajectory(self, points: list, steps_per_segment: int = 50):
        """
        points: list of numpy arrays of shape (3,) representing positions
        Moves the robot along straight lines in Cartesian space
        """
        ctrl = controllers.CartesianImpedance()
        self.panda.start_controller(ctrl)

        for i in range(len(points)-1):
            start = points[i]
            end = points[i+1]
            trajectory = self.linear_interpolate(start, end, steps=steps_per_segment)
            
            for pos in trajectory:
                orientation = R.from_quat(self.panda.get_orientation())
                ctrl.set_control(pos, orientation.as_quat())
                time.sleep(0.01)  # small delay for smooth motion

        self.pub_actual_pose()
        print("Trajectory execution complete!")


class PandaPosePub(Node):
    """Publishes the current Panda pose"""
    
    def __init__(self):
        super().__init__('panda_pose_publisher')
        self.msg: Pose = Pose()
        self.msg.orientation.w = 1.0
        self.publisher_ = self.create_publisher(Pose, 'panda_pose_topic', 10)

    def publish_wrapper(self):
        self.publisher_.publish(self.msg)


# --------------------------
# Main Function
# --------------------------
def main(args=None):
    rclpy.init(args=args)
    
    desk = panda_py.Desk("192.168.1.11", "BRL", "IITADVRBRL")
    panda = panda_py.Panda("192.168.1.11")

    panda_cmd_sub = PandaCmdSub(desk, panda)

    # --------------------------
    # Example trajectory points
    # --------------------------
    home = np.array([0.5, 0.0, 0.5])
    point_a = np.array([0.6, 0.0, 0.5])
    point_b = np.array([0.6, 0.1, 0.5])
    trajectory_points = [home, point_a, point_b, home]

    # Uncomment to test trajectory immediately
    # panda_cmd_sub.move_trajectory(trajectory_points)

    rclpy.spin(panda_cmd_sub)
    panda_cmd_sub.destroy_node()
    desk.deactivate_fci()
    desk.lock()


if __name__ == '__main__':
    main()
