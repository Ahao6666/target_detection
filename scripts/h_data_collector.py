"""
h_data_collector.py — H 标数据采集 + 自动标注

原理：H 标固定在世界原点 (0,0,0)，无人机位姿来自 /mavros/local_position/pose，
结合相机内参(camera_info)和外参(TF 或静态安装参数)，把 H 标地面占位(默认
1.0m x 1.0m)投影到图像上得到 2D 框，直接生成 YOLO 格式标注，无需人工 CVAT 标注。

输出结构（与 CVAT 导出 YOLO 1.1 一致）：
    h_dataset/
    ├── images/{train,val}/*.jpg
    └── labels/{train,val}/*.txt
并在数据集目录内生成 h_marker.yaml。

运行（先启动仿真）：
    source /opt/ros/$ROS_DISTRO/setup.bash
    source ~/target_detection/yolo_venv/bin/activate
    python3 h_data_collector.py

控制：
    ros2 service call /h_data_collector/set_collecting std_srvs/srv/SetBool "{data: true}"
    ros2 service call /h_data_collector/capture_once std_srvs/srv/Trigger "{}"
    rqt_image_view  # 选 /h_data_collector/preview 查看自动标注框
"""

import csv
import random
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseStamped, Point
from cv_bridge import CvBridge
from std_srvs.srv import SetBool, Trigger

try:
    import tf2_ros
except ImportError:
    tf2_ros = None


def quat_to_rot(qx, qy, qz, qw):
    """四元数 -> 3x3 旋转矩阵（不依赖 tf_transformations）"""
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)]])


def rpy_to_rot(roll, pitch, yaw):
    """弧度，ZYX 外旋：R = Rz(yaw) @ Ry(pitch) @ Rx(roll)"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


# 相机光轴系(z前 x右 y下) 到机体系(FLU: x前 y左 z上) 的常用安装预设
# TF 可用时优先用 TF，这里仅作静态回退
CAM_MOUNT_PRESETS = {
    'forward': np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
    'down':    np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
    'back':    np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
    'up':      np.eye(3),
    # Gazebo 相机约定(X前 Y左 Z 上)按 pitch=90° 朝下安装：视线朝下、机头在图像上方
    # 即 down 预设绕光轴转 90°，等价于 cam_mount:=down + cam_yaw:=90
    'down_gz': np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
}


class HDataCollector(Node):
    def __init__(self):
        super().__init__('h_data_collector')

        # ---- 参数 ----
        self.declare_parameter('image_topic', '/zed/left_camera_link/image_raw')
        self.declare_parameter('camera_info_topic', '/zed/left_camera_link/camera_info')
        self.declare_parameter('pose_topic', '/mavros/local_position/pose')
        self.declare_parameter('output_dir', '/home/ahao/target_detection/h_dataset')
        self.declare_parameter('val_ratio', 0.2)
        self.declare_parameter('seed', 42)

        # H 标：世界系中的中心位置与边长(米)
        self.declare_parameter('marker_size', 1.0)
        self.declare_parameter('marker_x', 0.0)
        self.declare_parameter('marker_y', 0.0)
        self.declare_parameter('marker_z', 0.0)  # 贴地一般为 0

        # 相机外参：优先 TF(base_frame -> 相机光轴系)，失败时用静态安装参数
        self.declare_parameter('use_tf', True)
        self.declare_parameter('base_frame', 'base_link')  # 机体系；TF 失败时按相机帧前缀自动补候选
        self.declare_parameter('cam_mount', 'down_gz')  # 本仿真: zed_camera include 姿态 pitch=90°(Gazebo 相机 X前Y左Z上)
        # 相机在体系下的安装位置(米)，来自模型 SDF:
        #   include 平移 (0,0,-0.124923) + imager 在相机模型内 (0.01,0.06,0)
        #   经安装旋转 Ry(90°) 后到体系 (0, 0.06, -0.134923)
        self.declare_parameter('cam_offset_x', 0.0)
        self.declare_parameter('cam_offset_y', 0.06)
        self.declare_parameter('cam_offset_z', -0.134923)
        self.declare_parameter('cam_roll', 0.0)   # 微调角(度)，在预设基础上叠加
        self.declare_parameter('cam_pitch', 0.0)
        self.declare_parameter('cam_yaw', 0.0)

        # 在线校准：向该话题发 geometry_msgs/Point 修正静态外参
        #   x = 绿框中心在真实标记右侧的像素偏差(右正)
        #   y = 绿框中心在真实标记下侧的像素偏差(下正)
        #   z = 额外偏航角(度)，排查镜像/轴向时用 90/180/270 试
        self.declare_parameter('nudge_topic', '/h_data_collector/nudge')

        # 采集策略
        self.declare_parameter('auto_start', False)      # 节点启动即开始自动采集
        self.declare_parameter('auto_interval', 1.0)     # 自动采集最小间隔(秒)
        self.declare_parameter('min_shift_px', 25.0)     # 框中心最小位移，过滤悬停重复帧
        self.declare_parameter('min_depth', 0.5)         # 相机到 H 标中心距离下限(米)
        self.declare_parameter('max_depth', 60.0)        # 上限(米)
        self.declare_parameter('min_box_px', 10.0)       # 框最小边长(像素)

        # 内参回退（收不到 camera_info 时使用，全部 >0 才生效）
        self.declare_parameter('fallback_fx', 0.0)
        self.declare_parameter('fallback_fy', 0.0)
        self.declare_parameter('fallback_cx', 0.0)
        self.declare_parameter('fallback_cy', 0.0)

        self.declare_parameter('jpeg_quality', 95)
        self.declare_parameter('publish_preview', True)
        self.declare_parameter('write_yaml', True)
        self.declare_parameter('yaml_path', '')  # 空则为 output_dir/h_marker.yaml

        p = self.get_parameter
        self.output_dir = Path(p('output_dir').value)
        self.val_ratio = p('val_ratio').value
        self.marker_size = p('marker_size').value
        self.marker_xyz = (p('marker_x').value, p('marker_y').value, p('marker_z').value)
        self.use_tf = p('use_tf').value
        self.base_frame = p('base_frame').value
        self.cam_mount = p('cam_mount').value
        self.cam_offset = np.array([p('cam_offset_x').value,
                                    p('cam_offset_y').value,
                                    p('cam_offset_z').value])
        self.cam_extra_rpy = np.radians([p('cam_roll').value,
                                         p('cam_pitch').value,
                                         p('cam_yaw').value])
        self.auto_interval_ns = int(p('auto_interval').value * 1e9)
        self.min_shift_px = p('min_shift_px').value
        self.min_depth = p('min_depth').value
        self.max_depth = p('max_depth').value
        self.min_box_px = p('min_box_px').value
        self.fallback_K = (p('fallback_fx').value, p('fallback_fy').value,
                           p('fallback_cx').value, p('fallback_cy').value)
        self.jpeg_quality = p('jpeg_quality').value
        self.publish_preview = p('publish_preview').value

        # ---- 状态 ----
        self.K = None
        self.bridge = CvBridge()
        self.rng = random.Random(p('seed').value)
        self.collecting = p('auto_start').value
        self.capture_once_req = False
        self.last_frame = None
        self.last_pose = None
        self.last_pose_time = None
        self.last_capture_time_ns = None
        self.last_saved_center = None
        self.counts = {'train': 0, 'val': 0}
        self.skip_counts = {}
        self._tf_warned = False
        self._tf_active = False
        self.last_dist = None
        # 记录参数原始值，供 reset_calibration 恢复
        self._base_cam_offset = self.cam_offset.copy()
        self._base_extra_rpy = self.cam_extra_rpy.copy()

        # ---- 数据集目录与 yaml ----
        for split in ('train', 'val'):
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        if p('write_yaml').value:
            self.write_yaml(p('yaml_path').value)

        self.log_csv = self.output_dir / 'log.csv'

        # ---- 通信 ----
        qos = QoSProfile(  # 传感器话题用 BEST_EFFORT，否则收不到数据
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)
        self.create_subscription(Image, p('image_topic').value, self.image_cb, qos)
        self.create_subscription(CameraInfo, p('camera_info_topic').value,
                                 self.info_cb, qos)
        self.create_subscription(PoseStamped, p('pose_topic').value,
                                 self.pose_cb, qos)

        self.create_service(SetBool, '~/set_collecting', self.set_collecting_cb)
        self.create_service(Trigger, '~/capture_once', self.capture_once_cb)
        self.create_service(Trigger, '~/reset_calibration', self.reset_calibration_cb)
        self.create_subscription(Point, p('nudge_topic').value, self.nudge_cb, 10)
        if self.publish_preview:
            self.pub_preview = self.create_publisher(Image, '~/preview', 10)

        self.tf_buffer = None
        if self.use_tf:
            if tf2_ros is not None:
                self.tf_buffer = tf2_ros.Buffer()
                tf2_ros.TransformListener(self.tf_buffer, self)
            else:
                self.get_logger().warn('tf2_ros 未安装，改用静态外参')

        self.get_logger().info(
            f'数据集目录: {self.output_dir}  auto_start={self.collecting}  '
            f'用 set_collecting 服务开始/停止采集')

    # ---------------- 参数/初始化 ----------------

    def write_yaml(self, yaml_path):
        yaml_path = Path(yaml_path) if yaml_path else self.output_dir / 'h_marker.yaml'
        content = (f'path: {self.output_dir.resolve()}\n'
                   'train: images/train\n'
                   'val: images/val\n'
                   "names: ['h_marker']\n"
                   'nc: 1\n')
        yaml_path.write_text(content)
        self.get_logger().info(f'已生成 {yaml_path}')

    def try_fallback_intrinsics(self):
        fx, fy, cx, cy = self.fallback_K
        if fx > 0 and fy > 0 and cx > 0 and cy > 0:
            self.K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
            self.get_logger().warn('使用 fallback 内参，未收到 camera_info')

    # ---------------- 回调 ----------------

    def info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(f'相机 K:\n{self.K}')

    def pose_cb(self, msg):
        self.last_pose = msg
        self.last_pose_time = self.get_clock().now()

    def image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {e}', throttle_duration_sec=2.0)
            return
        self.last_frame = frame

        # 计算当前帧的自动标注框（同时用于预览和采集）
        box, dist, reason = None, None, None
        if self.K is None:
            self.try_fallback_intrinsics()
        if self.K is None:
            reason = '等待 camera_info'
        elif self.last_pose is None:
            reason = '等待位姿'
        elif (self.get_clock().now() - self.last_pose_time).nanoseconds > 5e8:
            reason = '位姿数据过期(>0.5s)'
        else:
            T_world_cam = self.get_T_world_cam(msg.header.frame_id)
            if T_world_cam is None:
                reason = '相机外参不可用(TF/静态参数)'
            else:
                box, dist, reason = self.compute_box(T_world_cam, frame.shape)
        self.last_dist = dist

        if self.publish_preview:
            self.publish_preview_img(frame, box, dist)

        if not (self.collecting or self.capture_once_req):
            return
        manual = self.capture_once_req
        self.capture_once_req = False

        if box is None:
            self.skip(reason)
            if manual:
                self.get_logger().info(f'手动采集失败: {reason}')
            return
        if not manual:
            now_ns = self.get_clock().now().nanoseconds
            if (self.last_capture_time_ns is not None and
                    now_ns - self.last_capture_time_ns < self.auto_interval_ns):
                self.skip('间隔未到')
                return
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            if (self.last_saved_center is not None and
                    np.hypot(cx - self.last_saved_center[0],
                             cy - self.last_saved_center[1]) < self.min_shift_px):
                self.skip('位移太小(悬停重复帧)')
                return

        self.save_sample(frame, box, dist)

    def set_collecting_cb(self, req, res):
        self.collecting = req.data
        if self.collecting:
            self.last_capture_time_ns = None  # 启动后立刻采一帧
            res.message = f'开始自动采集，间隔 {self.auto_interval_ns / 1e9:.1f}s'
        else:
            res.message = '已停止自动采集'
        res.success = True
        self.get_logger().info(res.message)
        return res

    def capture_once_cb(self, req, res):
        self.capture_once_req = True
        res.success = True
        res.message = '将在下一帧图像到来时采集'
        return res

    def reset_calibration_cb(self, req, res):
        self.cam_offset = self._base_cam_offset.copy()
        self.cam_extra_rpy = self._base_extra_rpy.copy()
        res.success = True
        res.message = '已恢复启动参数中的外参'
        self.get_logger().info(res.message)
        return res

    def nudge_cb(self, msg):
        """在线校准静态外参。

        msg.x/msg.y: 绿框中心相对真实标记的像素偏差(右/下为正)，
                     即把框移到标记上需要的像素位移；
        msg.z: 额外偏航角(度)，只在排查轴向/镜像时非零。
        原理: 杠杆臂修正量(相机系,米) = 像素偏差 * 距离 / 焦距。
        """
        if self._tf_active:
            self.get_logger().warn('TF 外参生效中，无需校准')
            return
        if self.K is None or self.last_dist is None:
            self.get_logger().warn('校准失败：画面中还没有可用的 H 标投影，'
                                   '先飞到能看见 H 标的位置')
            return
        fx, fy = self.K[0, 0], self.K[1, 1]
        d_cam = np.array([msg.x * self.last_dist / fx,
                          msg.y * self.last_dist / fy,
                          0.0])
        T_bc = self.static_T_body_cam()
        if T_bc is None:
            return
        self.cam_offset = self.cam_offset + T_bc[:3, :3] @ d_cam
        if abs(msg.z) > 1e-6:
            self.cam_extra_rpy[2] += np.radians(msg.z)
        self.get_logger().info(
            f'校准: cam_offset=(x:{self.cam_offset[0]:.3f} y:{self.cam_offset[1]:.3f} '
            f'z:{self.cam_offset[2]:.3f}) cam_yaw_extra={np.degrees(self.cam_extra_rpy[2]):.1f}°，'
            '换高度/朝向验证，准确后写入启动参数固化')

    # ---------------- 核心：位姿 -> 2D 框 ----------------

    def get_T_world_cam(self, cam_frame):
        """世界系(pose.header.frame_id) 到相机光轴系的 4x4 变换"""
        pose = self.last_pose
        q = pose.pose.orientation
        R_wb = quat_to_rot(q.x, q.y, q.z, q.w)
        t_wb = np.array([pose.pose.position.x,
                         pose.pose.position.y,
                         pose.pose.position.z])
        T_wb = np.eye(4)
        T_wb[:3, :3] = R_wb
        T_wb[:3, 3] = t_wb

        T_bc = self.get_T_body_cam(cam_frame)
        if T_bc is None:
            return None
        return T_wb @ T_bc

    def get_T_body_cam(self, cam_frame):
        """体系到相机光轴系：优先 TF，失败用静态安装参数

        PoseStamped 不带 child_frame_id，机体系名称用 base_frame 参数；
        仿真 frame 常带命名空间前缀(如 alti_transition_quad/...)，
        自动把相机帧的第一级前缀拼到 base_frame 上作为候选。
        """
        candidates = [self.base_frame]
        ns = cam_frame.split('/')[0] if '/' in cam_frame else ''
        if ns:
            prefixed = f'{ns}/{self.base_frame}'
            if prefixed not in candidates:
                candidates.append(prefixed)
        if self.tf_buffer is not None:
            for body_frame in candidates:
                try:
                    tf = self.tf_buffer.lookup_transform(body_frame, cam_frame, Time())
                    t = tf.transform.translation
                    q = tf.transform.rotation
                    T = np.eye(4)
                    T[:3, :3] = quat_to_rot(q.x, q.y, q.z, q.w)
                    T[:3, 3] = [t.x, t.y, t.z]
                    self._tf_active = True
                    if body_frame != candidates[0] and not self._tf_warned:
                        self._tf_warned = True
                        self.get_logger().info(
                            f'TF 使用机体系 {body_frame}(由相机帧前缀推断)')
                    return T
                except Exception:
                    continue
            if not self._tf_warned:
                self._tf_warned = True
                self.get_logger().warn(
                    f'TF 查询失败({" / ".join(candidates)} -> {cam_frame})，'
                    f'回退到静态外参 cam_mount={self.cam_mount}；'
                    '若长期出现请检查 TF 或设置 cam_mount/cam_offset 参数，'
                    '或悬停在 H 标上方用 nudge 话题在线校准')
        return self.static_T_body_cam()

    def static_T_body_cam(self):
        if self.cam_mount not in CAM_MOUNT_PRESETS:
            self.get_logger().error(f'未知 cam_mount={self.cam_mount}，可选 {list(CAM_MOUNT_PRESETS)}')
            return None
        R = CAM_MOUNT_PRESETS[self.cam_mount] @ rpy_to_rot(*self.cam_extra_rpy)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.cam_offset
        return T

    def compute_box(self, T_world_cam, shape):
        """H 标地面占位 5x5 网格投影到图像，取有效投影的 min/max 作为包围框"""
        H, W = shape[:2]
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        T_cam_world = np.linalg.inv(T_world_cam)
        half = self.marker_size / 2.0
        gx, gy = np.meshgrid(
            np.linspace(self.marker_xyz[0] - half, self.marker_xyz[0] + half, 5),
            np.linspace(self.marker_xyz[1] - half, self.marker_xyz[1] + half, 5))
        pts = np.stack([gx.ravel(), gy.ravel(),
                        np.full(25, self.marker_xyz[2]), np.ones(25)])  # 4x25 世界系
        cam = (T_cam_world @ pts).T[:, :3]  # 25x3 相机系
        z = cam[:, 2]
        valid = z > 0.05
        if valid.sum() < 9:
            return None, None, 'H 标在相机后方或太近'

        uv = np.full((25, 2), np.nan)
        uv[valid, 0] = fx * cam[valid, 0] / z[valid] + cx
        uv[valid, 1] = fy * cam[valid, 1] / z[valid] + cy

        center_cam = cam[12]  # 网格中心即 H 标中心
        dist = float(np.linalg.norm(center_cam))
        if dist < self.min_depth:
            return None, None, f'距离过近({dist:.1f}m)'
        if dist > self.max_depth:
            return None, None, f'距离过远({dist:.1f}m)'

        cu = fx * center_cam[0] / center_cam[2] + cx
        cv_ = fy * center_cam[1] / center_cam[2] + cy
        if not (0 <= cu < W and 0 <= cv_ < H):
            return None, None, 'H 标中心不在画面内'

        u1, u2 = np.nanmin(uv[:, 0]), np.nanmax(uv[:, 0])
        v1, v2 = np.nanmin(uv[:, 1]), np.nanmax(uv[:, 1])
        x1, x2 = max(u1, 0.0), min(u2, W - 1.0)
        y1, y2 = max(v1, 0.0), min(v2, H - 1.0)
        if x2 - x1 < self.min_box_px or y2 - y1 < self.min_box_px:
            return None, None, '框太小(<min_box_px)'
        return (float(x1), float(y1), float(x2), float(y2)), dist, None

    # ---------------- 保存 ----------------

    def save_sample(self, frame, box, dist):
        split = 'val' if self.rng.random() < self.val_ratio else 'train'
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = box

        stem = f'h_{self.get_clock().now().nanoseconds}'
        img_path = self.output_dir / 'images' / split / f'{stem}.jpg'
        cv2.imwrite(str(img_path), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])

        # YOLO: class cx cy w h（归一化，裁剪到[0,1]）
        ncx = min(max(((x1 + x2) / 2.0) / W, 0.0), 1.0)
        ncy = min(max(((y1 + y2) / 2.0) / H, 0.0), 1.0)
        nw = min((x2 - x1) / W, 1.0)
        nh = min((y2 - y1) / H, 1.0)
        lbl_path = self.output_dir / 'labels' / split / f'{stem}.txt'
        lbl_path.write_text(f'0 {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}\n')

        # 记录位姿与框信息，便于后续筛选/排查
        pose = self.last_pose.pose
        new_file = not self.log_csv.exists()
        with open(self.log_csv, 'a', newline='') as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(['file', 'split', 'uav_x', 'uav_y', 'uav_z',
                            'qw', 'qx', 'qy', 'qz', 'dist_m',
                            'box_x1', 'box_y1', 'box_x2', 'box_y2', 'img_w', 'img_h'])
            q = pose.orientation
            w.writerow([img_path.name, split,
                        f'{pose.position.x:.3f}', f'{pose.position.y:.3f}',
                        f'{pose.position.z:.3f}',
                        f'{q.w:.4f}', f'{q.x:.4f}', f'{q.y:.4f}', f'{q.z:.4f}',
                        f'{dist:.2f}',
                        f'{x1:.1f}', f'{y1:.1f}', f'{x2:.1f}', f'{y2:.1f}', W, H])

        self.counts[split] += 1
        self.last_capture_time_ns = self.get_clock().now().nanoseconds
        self.last_saved_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        self.get_logger().info(
            f'[{split}] {stem}.jpg  dist={dist:.1f}m  '
            f'框=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})  '
            f'总数 train={self.counts["train"]} val={self.counts["val"]}')

    def publish_preview_img(self, frame, box, dist):
        out = frame.copy()
        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'h_marker {dist:.1f}m'
            cv2.putText(out, label, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(out, 'marker not visible', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if self.collecting:
            cv2.putText(out, 'COLLECTING', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        self.pub_preview.publish(self.bridge.cv2_to_imgmsg(out, encoding='bgr8'))

    def skip(self, reason):
        self.skip_counts[reason] = self.skip_counts.get(reason, 0) + 1
        if reason in ('等待 camera_info', '等待位姿', '相机外参不可用(TF/静态参数)',
                      '位姿数据过期(>0.5s)'):
            self.get_logger().info(f'跳过: {reason}', throttle_duration_sec=2.0)

    def print_summary(self):
        total = self.counts['train'] + self.counts['val']
        self.get_logger().info(
            f'采集结束：共 {total} 张 (train={self.counts["train"]}, '
            f'val={self.counts["val"]})，保存在 {self.output_dir}')
        if self.skip_counts:
            skips = ', '.join(f'{k}x{v}' for k, v in
                              sorted(self.skip_counts.items(),
                                     key=lambda kv: -kv[1])[:5])
            self.get_logger().info(f'主要跳过原因: {skips}')


def main():
    rclpy.init()
    node = HDataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.print_summary()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
