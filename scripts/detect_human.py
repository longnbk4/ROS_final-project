#!/usr/bin/env python3
import sys
sys.path.append('/home/longnb/.local/lib/python3.8/site-packages')
import ultralytics
print(ultralytics.__version__)
from ultralytics import YOLO

import rospy
import cv2
import numpy as np
import threading
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os

class YOLOv5PersonDetector:
    def __init__(self):
        rospy.init_node("yolov5_person_detector_node", anonymous=True)

        # ==== Configuration ====
        self.conf_threshold = 0.5  # Confidence threshold
        self.frame_skip = 2
        self.frame_counter = 0
        self.bridge = CvBridge()
        self.display_frame = None
        self.new_frame_available = False
        self.lock = threading.Lock()

        # ==== Load YOLOv5 Model ====
        current_path = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_path, "yolov5s.pt")  # Đường dẫn file .pt đã tải về
        self.model = YOLO(weights_path)
        
        # ==== ROS Subscriber ====
        rospy.Subscriber("/rrbot/camera1/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**18)
        threading.Thread(target=self.display_loop, daemon=True).start()

        rospy.loginfo("✅ YOLOv5 Person Detector đã sẵn sàng!")

    def image_callback(self, msg):
        self.frame_counter += 1
        if self.frame_counter % self.frame_skip != 0:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr("❌ Lỗi chuyển ảnh: %s", str(e))
            return

        frame = cv2.resize(frame, (600, 512), interpolation=cv2.INTER_LINEAR)
        frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)

        # Chỉ phát hiện lớp 'person' (ID=0) bằng cách lọc kết quả sau khi dự đoán
        results = self.model.predict(source=frame, imgsz=640, conf=self.conf_threshold)
        
        # Vẽ kết quả detection
        annotated_frame = frame.copy()
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Lấy class
                cls = int(box.cls[0])
                # Chỉ xử lý nếu là người (class 0)
                if cls == 0:  
                    # Lấy tọa độ
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Lấy confidence
                    conf = float(box.conf[0])
                    
                    # Vẽ bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(annotated_frame, f"person {conf:.2f}", (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        with self.lock:
            self.display_frame = annotated_frame
            self.new_frame_available = True

    def display_loop(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.new_frame_available:
                with self.lock:
                    if self.display_frame is None:
                        continue
                    frame = self.display_frame
                    self.new_frame_available = False
                cv2.imshow("YOLOv5 - Person Detection", frame)
            if cv2.waitKey(1) == 27:  # ESC để thoát
                rospy.signal_shutdown("ESC pressed")
            rate.sleep()

if __name__ == '__main__':
    try:
        detector = YOLOv5PersonDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
