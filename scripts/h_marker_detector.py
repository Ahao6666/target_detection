import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path


class HMarkerDetector(Node):
    def __init__(self):
        super().__init__('h_marker_detector')

        # 关键：传感器话题用 BEST_EFFORT QoS，否则收不到数据
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        self.bridge = CvBridge()
        model_path = Path(__file__).resolve().parent / 'best.pt'
        self.model = YOLO(str(model_path))

        self.K = None                    # 相机内参 3x3
        self.marker_size = 1.0           # H 标实际边长(米)，改成你的真实值
        self.create_subscription(Image, '/zed/left_camera_link/image_raw',
                                 self.image_cb, qos)
        self.create_subscription(CameraInfo, '/zed/left_camera_link/camera_info',
                                 self.info_cb, qos)
        self.pub_img = self.create_publisher(Image, '/h_marker/annotated', 10)
        self.pub_pos = self.create_publisher(Point, '/h_marker/position', 10)

    def info_cb(self, msg):
        # 只需要取一次内参
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(f'Camera K:\n{self.K}')

    def image_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model.predict(frame, conf=0.5, verbose=False)[0]
        out = frame.copy()

        if len(results.boxes) > 0:
            # 取置信度最高的框
            box = max(results.boxes, key=lambda b: b.conf)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            h_px = y2 - y1

            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            if self.K is not None:
                fx, fy = self.K[0, 0], self.K[1, 1]
                u0, v0 = self.K[0, 2], self.K[1, 2]

                # 单目尺寸法反推相对位置（相机坐标系：z 前 x 右 y 下）
                z = fy * self.marker_size / h_px
                x = z * (cx - u0) / fx
                y = z * (cy - v0) / fy

                pos = Point(x=float(x), y=float(y), z=float(z))
                self.pub_pos.publish(pos)
                cv2.putText(out, f'x={x:.2f} y={y:.2f} z={z:.2f}m',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                self.get_logger().info(
                    f'Relative pose: x={x:.2f}, y={y:.2f}, z={z:.2f}',
                    throttle_duration_sec=1.0)

        self.pub_img.publish(self.bridge.cv2_to_imgmsg(out, encoding='bgr8'))


def main():
    rclpy.init()
    node = HMarkerDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
