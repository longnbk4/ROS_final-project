#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
import sys
import tty
import termios

def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

def arm_controller():
    rospy.init_node('arm_controller', anonymous=True)
    arm_1_pub = rospy.Publisher('/taymay1_joint_position_controller/command', Float64, queue_size=10)
    arm_2_pub = rospy.Publisher('/taymay2_joint_position_controller/command', Float64, queue_size=10)
    arm_1_pos = 0.0
    arm_2_pos = 0.0
    step = 0.5
    print("Điều khiển tay máy:")
    print("1: Tăng vị trí tay máy 1")
    print("2: Giảm vị trí tay máy 1")
    print("3: Tăng vị trí tay máy 2")
    print("4: Giảm vị trí tay máy 2")
    print("q: Thoát")
    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        key = get_key()
        if key == '1':
            arm_1_pos += step
            if arm_1_pos > 1.57:
                arm_1_pos = 1.57
        elif key == '2':
            arm_1_pos -= step
            if arm_1_pos < -1.57:
                arm_1_pos = -1.57
        elif key == '3':
            arm_2_pos += step
            if arm_2_pos > 1.57:
                arm_2_pos = 1.57
        elif key == '4':
            arm_2_pos -= step
            if arm_2_pos < -1.57:
                arm_2_pos = -1.57
        elif key == 'q':
            break
        arm_1_pub.publish(arm_1_pos)
        arm_2_pub.publish(arm_2_pos)
        rospy.loginfo("Tay máy 1: %.2f rad, Tay máy 2: %.2f rad", arm_1_pos, arm_2_pos)
        rate.sleep()

if __name__ == '__main__':
    try:
        arm_controller()
    except rospy.ROSInterruptException:
        pass
