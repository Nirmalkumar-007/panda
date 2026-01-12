import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R  # For quaternion operations
import numpy as np
import time

# Define a ROS 2 node that will publish trajectory commands to the robot
class PandaTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('panda_trajectory_publisher')  # Initialize ROS 2 node with a name
        self.publisher_ = self.create_publisher(Pose, 'panda_cmd_topic', 10)  # Publisher to send Pose commands
        self.rate = 50  # Publishing rate in Hz
        self.dt = 1.0 / self.rate  # Time interval between publishing messages

    def publish_pose(self, pose_msg):
        """Publish a single pose to the robot"""
        self.publisher_.publish(pose_msg)  # Publish the Pose message
        time.sleep(self.dt)  # Wait for the specified time interval to control speed

    def move_straight_line(self, start_pose, end_pose, steps=50):
        """
        Move the robot from start_pose to end_pose in a straight line using linear interpolation
        """
        # Extract positions as numpy arrays for easy math
        start_pos = np.array([start_pose.position.x,
                              start_pose.position.y,
                              start_pose.position.z])
        end_pos = np.array([end_pose.position.x,
                            end_pose.position.y,
                            end_pose.position.z])

        # Convert orientations to Rotation objects (from quaternions) for smooth interpolation
        start_rot = R.from_quat([start_pose.orientation.x,
                                 start_pose.orientation.y,
                                 start_pose.orientation.z,
                                 start_pose.orientation.w])
        end_rot = R.from_quat([end_pose.orientation.x,
                               end_pose.orientation.y,
                               end_pose.orientation.z,
                               end_pose.orientation.w])
        
        # Create a SLERP object to interpolate orientation smoothly
        slerp = R.slerp(0, 1, [start_rot, end_rot])

        # Loop through interpolation steps
        for i, t in enumerate(np.linspace(0, 1, steps)):
            # Interpolate position linearly
            interp_pos = start_pos + t * (end_pos - start_pos)
            # Interpolate orientation using SLERP
            interp_rot = slerp([t])[0]

            # Create a Pose message for this intermediate step
            pose_msg = Pose()
            pose_msg.position.x = interp_pos[0]
            pose_msg.position.y = interp_pos[1]
            pose_msg.position.z = interp_pos[2]

            quat = interp_rot.as_quat()  # Convert interpolated rotation back to quaternion
            pose_msg.orientation.x = quat[0]
            pose_msg.orientation.y = quat[1]
            pose_msg.orientation.z = quat[2]
            pose_msg.orientation.w = quat[3]

            self.publish_pose(pose_msg)  # Publish the interpolated pose

# Main function to run the trajectory
def main(args=None):
    rclpy.init(args=args)  # Initialize ROS 2
    traj_pub = PandaTrajectoryPublisher()  # Create the trajectory publisher node

    # Define Home pose
    home = Pose()
    home.position.x, home.position.y, home.position.z = 0.4, 0.0, 0.4
    home.orientation.x, home.orientation.y, home.orientation.z, home.orientation.w = 0, 0, 0, 1

    # Define Point A
    point_a = Pose()
    point_a.position.x, point_a.position.y, point_a.position.z = 0.6, 0.2, 0.4
    point_a.orientation.x, point_a.orientation.y, point_a.orientation.z, point_a.orientation.w = 0, 0, 0, 1

    # Define Point B
    point_b = Pose()
    point_b.position.x, point_b.position.y, point_b.position.z = 0.6, -0.2, 0.4
    point_b.orientation.x, point_b.orientation.y, point_b.orientation.z, point_b.orientation.w = 0, 0, 0, 1

    # Move the robot from Home → Point A in a straight line
    traj_pub.move_straight_line(home, point_a, steps=100)

    # Move the robot from Point A → Point B in a straight line
    traj_pub.move_straight_line(point_a, point_b, steps=100)

    # Clean up the node and shutdown ROS 2
    traj_pub.destroy_node()
    rclpy.shutdown()

# Run the script
if __name__ == '__main__':
    main()
