#!/usr/bin/env python3

import rospy
import math
import tf
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

class MazeNavigator:
    def __init__(self):
        rospy.init_node('maze_navigator', anonymous=True)
        self.laser_data = None
        self.robot_position = [0, 0, 0]
        self.goal_reached = True
        self.min_front_distance = 0.6
        self.min_side_distance = 0.5
        self.current_goal = None
        rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.tf_listener = tf.TransformListener()
        rospy.loginfo("Maze Navigator đã được khởi tạo!")
        rospy.sleep(1)
        
    def laser_callback(self, data):
        self.laser_data = data
        
    def odom_callback(self, data):
        position = data.pose.pose.position
        orientation = data.pose.pose.orientation
        (_, _, yaw) = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w])
        self.robot_position = [position.x, position.y, yaw]
        
    def get_front_distance(self):
        if self.laser_data is None:
            return float('inf')
        front_range_size = len(self.laser_data.ranges)
        center_idx = front_range_size // 2
        angle_range = front_range_size // 12
        front_indices = range(center_idx - angle_range, center_idx + angle_range)
        front_ranges = [self.laser_data.ranges[i % front_range_size] for i in front_indices]
        valid_ranges = [r for r in front_ranges if r < self.laser_data.range_max and not math.isnan(r)]
        if len(valid_ranges) > 0:
            return min(valid_ranges)
        else:
            return float('inf')
            
    def get_left_distance(self):
        if self.laser_data is None:
            return float('inf')
        front_range_size = len(self.laser_data.ranges)
        left_center = (front_range_size * 3) // 4
        angle_range = front_range_size // 12
        left_indices = range(left_center - angle_range, left_center + angle_range)
        left_ranges = [self.laser_data.ranges[i % front_range_size] for i in left_indices]
        valid_ranges = [r for r in left_ranges if r < self.laser_data.range_max and not math.isnan(r)]
        if len(valid_ranges) > 0:
            return min(valid_ranges)
        else:
            return float('inf')
            
    def get_right_distance(self):
        if self.laser_data is None:
            return float('inf')
        front_range_size = len(self.laser_data.ranges)
        right_center = front_range_size // 4
        angle_range = front_range_size // 12
        right_indices = range(right_center - angle_range, right_center + angle_range)
        right_ranges = [self.laser_data.ranges[i % front_range_size] for i in right_indices]
        valid_ranges = [r for r in right_ranges if r < self.laser_data.range_max and not math.isnan(r)]
        if len(valid_ranges) > 0:
            return min(valid_ranges)
        else:
            return float('inf')
    
    def send_goal(self, x, y):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.w = 1.0
        rospy.loginfo("Gửi mục tiêu: (%s, %s)", x, y)
        self.move_base_client.send_goal(goal)
        self.current_goal = [x, y]
        self.goal_reached = False
    
    def wall_following(self):
        front_distance = self.get_front_distance()
        right_distance = self.get_right_distance()
        left_distance = self.get_left_distance()
        twist = Twist()
        rospy.loginfo("Khoảng cách - Trước: %.2f, Phải: %.2f, Trái: %.2f", 
                      front_distance, right_distance, left_distance)
        if front_distance < self.min_front_distance:
            rospy.loginfo("Vật cản phía trước, quay trái")
            twist.angular.z = 0.3
            twist.linear.x = 0.0
        elif right_distance > 1.0:
            rospy.loginfo("Không có tường bên phải, rẽ phải")
            twist.linear.x = 0.08
            twist.angular.z = -0.3
        elif right_distance < self.min_side_distance:
            rospy.loginfo("Quá gần tường bên phải, rẽ trái")
            twist.linear.x = 0.08
            twist.angular.z = 0.3
        else:
            rospy.loginfo("Đi thẳng dọc theo tường")
            twist.linear.x = 0.15
            error = right_distance - 0.7
            twist.angular.z = -0.1 * error
        self.cmd_vel_pub.publish(twist)
    
    def run(self):
        rate = rospy.Rate(10)
        rospy.loginfo("Bắt đầu thuật toán wall following cho mê cung cây bụi")
        while not rospy.is_shutdown():
            self.wall_following()
            rate.sleep()

if __name__ == '__main__':
    try:
        navigator = MazeNavigator()
        navigator.run()
    except rospy.ROSInterruptException:
        pass