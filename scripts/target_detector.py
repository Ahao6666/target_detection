import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray
from ultralytics import YOLO


class TargetDetector(Node):
    def __init__(self):
        super().__init__('target_detector')

        # 传感器话题用 BEST_EFFORT QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        # ---- 相机内参 ----
        self.fx = 4496.874694824219
        self.fy = 4496.875076293945
        self.cx = 640.0
        self.cy = 360.0
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        # ---- YOLO 模型 ----
        model_path = Path(__file__).resolve().parent / 'best.pt'
        self.model = YOLO(str(model_path))
        self.get_logger().info(f'Loaded model: {model_path}')

        # ---- ROS 2 通信 ----
        self.bridge = CvBridge()
        self.create_subscription(Image, '/world/waterdrop_runway/model/waterdrop/link/camera_link/sensor/camera/image',
                                 self.image_callback, qos)
        self.create_subscription(CameraInfo, '/world/waterdrop_runway/model/waterdrop/link/camera_link/sensor/camera/camera_info',
                                 self.info_callback, qos)
        self.pub_img = self.create_publisher(Image, '/target/annotated', 10)
        self.result_pub = self.create_publisher(Float32MultiArray, '/mavros/drone/detection_result_yolo', 10)

        # ---- 线程池与图像缓存 ----
        self.executor_pool = ThreadPoolExecutor(max_workers=2)
        self.latest_image = None
        self.lock = threading.Lock()
        self.timer = self.create_timer(0.01, self.inference_timer_cb)

        # ---- 频率统计 ----
        self.sub_count = 0
        self.pub_count = 0
        self.stat_lock = threading.Lock()
        self.last_stat_time = self.get_clock().now()

        self.get_logger().info('Target Detector Initialized')

    def info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.fx = self.K[0, 0]
            self.fy = self.K[1, 1]
            self.cx = self.K[0, 2]
            self.cy = self.K[1, 2]
            self.get_logger().info(f'Camera K:\n{self.K}')

    def image_callback(self, msg):
        # 订阅计数
        with self.stat_lock:
            self.sub_count += 1

        # 每隔 2 秒打印一次频率
        now = self.get_clock().now()
        interval = (now - self.last_stat_time).nanoseconds / 1e9
        if interval >= 2.0:
            with self.stat_lock:
                sub_fps = self.sub_count / interval
                pub_fps = self.pub_count / interval
                self.get_logger().info(
                    f'FPS Statistics >> Sub Rate: {sub_fps:.1f}Hz | Inference/Pub Rate: {pub_fps:.1f}Hz')
                self.sub_count = 0
                self.pub_count = 0
            self.last_stat_time = now

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self.lock:
                self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f'CvBridge Error: {e}')

    def inference_timer_cb(self):
        with self.lock:
            if self.latest_image is None:
                return
            img = self.latest_image.copy()
            self.latest_image = None

        self.executor_pool.submit(self.process_inference, img)

    def process_inference(self, img):
        try:
            start_t = time.time()
            results = self.model.predict(source=img, conf=0.5, imgsz=640, verbose=False)
            infer_dur = time.time() - start_t

            H, W = img.shape[:2]
            out = img.copy()

            # 绘制相机中心十字线
            if self.K is not None:
                cx0, cy0 = int(self.cx), int(self.cy)
            else:
                cx0, cy0 = W // 2, H // 2
            cv2.line(out, (cx0, 0), (cx0, H), (0, 0, 255), 2)
            cv2.line(out, (0, cy0), (W, cy0), (0, 0, 255), 2)
            cv2.circle(out, (cx0, cy0), 5, (0, 0, 255), 2)

            has_targets = False
            for res in results:
                boxes = res.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                has_targets = True
                box = max(boxes, key=lambda b: b.conf)
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                if self.K is not None:
                    # yaw: 左正右负；pitch: 上正下负
                    yaw_rad = math.atan2(self.cx - cx, self.fx)
                    pitch_rad = math.atan2(self.cy - cy, self.fy)
                    yaw = math.degrees(yaw_rad)
                    pitch = math.degrees(pitch_rad)

                    # 位置相对估计直接置 0
                    res_msg = Float32MultiArray()
                    res_msg.data = [0.0, 0.0, 0.0, float(yaw), float(pitch), float(W), float(H)]
                    self.result_pub.publish(res_msg)

                    with self.stat_lock:
                        self.pub_count += 1

                    label = f'yaw={yaw:.1f}deg pitch={pitch:.1f}deg'
                    cv2.putText(out, label, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    self.get_logger().info(
                        f'Angles: yaw={yaw:.1f}deg, pitch={pitch:.1f}deg, infer_time={infer_dur*1000:.1f}ms',
                        throttle_duration_sec=1.0)
                else:
                    cv2.putText(out, 'waiting camera_info', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            self.pub_img.publish(self.bridge.cv2_to_imgmsg(out, encoding='bgr8'))

        except Exception as e:
            self.get_logger().error(f'Inference Error: {e}')


def main():
    rclpy.init()
    node = TargetDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
