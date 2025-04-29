#!/bin/bash

# Tắt tất cả các nút ROS đang chạy
echo "Đang tắt tất cả các nút ROS..."
rosnode kill -a

# Tắt các tiến trình Gazebo
echo "Đang tắt Gazebo..."
killall -9 gzserver gzclient rosmaster roscore

# Đợi 2 giây để đảm bảo tất cả đã tắt
sleep 2

# Khởi động lại Gazebo với bản đồ mê cung
echo "Đang khởi động lại Gazebo..."
roslaunch xerobotvisai2 gazebo.launch 