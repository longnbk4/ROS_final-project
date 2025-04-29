
# ROS Final Project: Differential Drive Robot with SLAM, Navigation, and YOLOv5 Human Detection

PDF: 
[ROS_GK-1.pdf](https://github.com/user-attachments/files/19553201/ROS_GK-1.pdf)

VIDEO:
https://byvn.net/IrCE


https://github.com/user-attachments/assets/59bf20da-7006-4f77-9fe3-57b78aea29d7

## Hướng dẫn mô phỏng và điều khiển Robot

## 🚀 Cài đặt

### 🔧 Bước 1: Tạo ROS Workspace

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
catkin_init_workspace   
```

### 📥 Bước 2: Clone Repository

```bash
git clone https://github.com/longnbk4/ROS_final-project.git
mv ROS_final-project xerobotvisai2
cd ~/catkin_ws
catkin_make
```

### 🧠 Bước 3: Source workspace

```bash
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```
## 🛰️ Chạy Gmapping 

### 🎯 Bước 4: Khởi chạy mô phỏng trong Gazebo 

``` bash
roslaunch xerobotvisai2 gazebo.launch
```

### ⚙️ Bước 5: Chạy rviz
``` bash
roslaunch xerobotvisai2 gmapping.launch 
```

### 🌐 Bước 6: Điểu khiển xe để quét map
```
rosrun teleop_twist_keyboard teleop_twist_keyboard.py 
```


Điều khiển bẳng các phím j, i, l k, .


## 📌Navigation 
⚠️ Để có thể chạy được navigation cần vào trong folder maps/mazemap1.yaml đổi đường dẫn image thành đường dẫn map quét được (mazemap1.pgm) tương ứng với máy của bạn


### 🎮 Bước 7:  Chay mô hình trong gazebo

```bash
roslaunch xerovotvisai2 gazebo.lauch
```
### 🕹️ Bước 8  Chạy navigation

```bash
roslaunch xerobotvisai2 navigation.launch 
```
### Bước 9 Điểu khiển xe dến vị trí cho trước

Trong rviz sử dụng 2D Nav Goal sau đó chọn vị trí để robot tìm đường đến vị trí đó 

## 🔧Detect Human Yolov5

#### Cài đặt thư viện cần thiết
```
pip install ultralytics
pip install opencv-python
pip install torch torchvision
```
### Thêm thư viện model người
```bash
nano ~/.bashrc
```
```bash
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/catkin_ws/src/xerobotvisai2/models
```

#### Chạy module phát hiện người
1. Vào file gazebo.launch và đổi đường dẫn sang human_map.world

2. Đảm bảo đã khởi động Gazebo và camera đang hoạt động:
   ```
   roslaunch xerobotvisai2 gazebo.launch
   ```

3. Chạy module phát hiện người:
   ```
   rosrun xerobotvisai2 detect_human.py
   ```

4. Module sẽ lắng nghe topic camera `/rrbot/camera1/image_raw` và hiển thị khung hình với các bounding box xung quanh người được phát hiện. 

⚠️ Nên điều khiển xe ra góc có thể thấy được toàn bộ khung hình người để có thể detect dễ hơn
