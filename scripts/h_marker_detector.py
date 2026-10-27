import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
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
        model_path = Path(__file__).resolve().parent / 'best_H.pt'
        self.model = YOLO(str(model_path))

        self.K = None                    # 相机内参 3x3
        # H 标“最外侧正方形”的真实边长(米)，务必用尺子量准，这是唯一尺度来源
        self.marker_size = 1.0
        self.create_subscription(Image, '/zed/left_camera_link/image_raw',
                                 self.image_cb, qos)
        self.create_subscription(CameraInfo, '/zed/left_camera_link/camera_info',
                                 self.info_cb, qos)
        self.pub_img = self.create_publisher(Image, '/h_marker/annotated', 10)
        self.pub_pos = self.create_publisher(PointStamped, '/h_marker/position', 10)

    def info_cb(self, msg):
        # 只需要取一次内参
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(f'Camera K:\n{self.K}')

    @staticmethod
    def _order_corners(pts):
        """把 4 个角点排序为 左上/右上/右下/左下，便于与物体点一一对应。"""
        pts = pts.reshape(4, 2).astype(np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _detect_square(self, roi):
        """在 ROI 内提取 H 标最外侧正方形的 4 个角点(ROI 坐标系)。
        返回 (corners[4x2] 或 None, 最大轮廓 或 None)。"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # 白色方框通常比地面亮，用 Otsu 自动阈值分离
        _, th = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        c = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return self._order_corners(approx), c
        # 没能拟合出干净四边形，退化为旋转矩形（仍保留轮廓供后续用）
        return None, c

    def image_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model.predict(frame, conf=0.5, verbose=False)[0]
        out = frame.copy()
        H, W = frame.shape[:2]

        if len(results.boxes) > 0:
            # 取置信度最高的框
            box = max(results.boxes, key=lambda b: b.conf)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 1)

            # 外扩一点 ROI，保证方框边缘完整（否则角点会被裁掉）
            pad = 0.15
            rx1 = int(max(0, x1 - pad * (x2 - x1)))
            ry1 = int(max(0, y1 - pad * (y2 - y1)))
            rx2 = int(min(W, x2 + pad * (x2 - x1)))
            ry2 = int(min(H, y2 + pad * (y2 - y1)))
            roi = frame[ry1:ry2, rx1:rx2]

            if self.K is not None and roi.size > 0:
                fx, fy = self.K[0, 0], self.K[1, 1]
                u0, v0 = self.K[0, 2], self.K[1, 2]
                x = y = z = None
                method = ''

                corners_roi, contour = self._detect_square(roi)

                # —— 方案一(首选)：4 角点 + solvePnP，任意视角都准确 ——
                if corners_roi is not None:
                    corners = corners_roi + np.array([rx1, ry1], dtype=np.float32)
                    s = self.marker_size / 2.0
                    # 物体点：以方框中心为原点、z=0 的平面正方形(左上起顺时针)
                    obj_pts = np.array([[-s, -s, 0], [s, -s, 0],
                                        [s, s, 0], [-s, s, 0]], dtype=np.float32)
                    ok, rvec, tvec = cv2.solvePnP(
                        obj_pts, corners, self.K, None,
                        flags=cv2.SOLVEPNP_IPPE)
                    if ok:
                        x = float(tvec[0, 0])
                        y = float(tvec[1, 0])
                        z = float(tvec[2, 0])
                        method = 'P4P'
                        cv2.polylines(out,
                                      [corners.astype(np.int32).reshape(-1, 1, 2)],
                                      True, (255, 0, 0), 2)
                        for p in corners.astype(int):
                            cv2.circle(out, tuple(p), 4, (0, 0, 255), -1)
                        cv2.drawFrameAxes(out, self.K, None, rvec, tvec,
                                          0.5 * self.marker_size)

                # —— 方案二：最小外接旋转矩形，边长对画面内旋转不敏感 ——
                if z is None and contour is not None:
                    rect = cv2.minAreaRect(contour)
                    rw, rh = rect[1]
                    s_px = (rw + rh) / 2.0
                    if s_px > 1:
                        z = fy * self.marker_size / s_px
                        ccx = rx1 + rect[0][0]
                        ccy = ry1 + rect[0][1]
                        x = z * (ccx - u0) / fx
                        y = z * (ccy - v0) / fy
                        method = 'minAreaRect'

                # —— 方案三：退回旧的 AABB(仅正视时准，作为兜底) ——
                if z is None:
                    h_px = max(1.0, y2 - y1)
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    z = fy * self.marker_size / h_px
                    x = z * (cx - u0) / fx
                    y = z * (cy - v0) / fy
                    method = 'AABB'

                pos = PointStamped()
                pos.header.stamp = self.get_clock().now().to_msg()
                pos.header.frame_id = 'camera_link'
                pos.point.x = float(x)
                pos.point.y = float(y)
                pos.point.z = float(z)
                self.pub_pos.publish(pos)

                cv2.putText(out, f'x={x:.2f} y={y:.2f} z={z:.2f}m [{method}]',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)
                self.get_logger().info(
                    f'({method}) x={x:.2f}, y={y:.2f}, z={z:.2f}',
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
