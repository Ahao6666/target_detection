"""
target_auto_poser.py — Gazebo 自动摆位节点（甜甜圈采样）

用途：自动把飞行器传送到目标(target)周围的随机位姿（环形分布，距离/高度/朝向/
小倾角随机），并把模型"钉"在该位姿（按固定频率重复置位，抵抗重力下坠），同时
向采集脚本发布**真值位姿**（传送会破坏 EKF，绕开后标注依然精确）。

搭配使用（目标可以是 target或任何静止平面目标）：
    # 方式A：直接存图，用于 target 手动标注
    python3 target_auto_poser.py -p save_images:=true \
        -p output_dir:=target_manual_images

    # 方式B：与 H 标自动标注联动，target_auto_poser 提供真值位姿
    # 终端1：采集脚本改听真值位姿（其 marker_x/y/z 与本节点 target_x/y/z 必须一致）
    python3 h_data_collector.py -p pose_topic:=/target_auto_poser/model_pose

    # 终端2：摆位节点（同时触发 h_data_collector 采集）
    python3 target_auto_poser.py -p control_collector:=true

    # 终端3：开始/停止
    ros2 service call /target_auto_poser/start std_srvs/srv/Trigger "{}"
    ros2 service call /target_auto_poser/stop  std_srvs/srv/Trigger "{}"

位姿设置后端（参数 backend:=auto 自动探测，默认优先 gz CLI）：
    1. gz_cli   : gz transport 服务，本机执行
                  gz service -s /world/<world_name>/set_pose --reqtype gz.msgs.Pose
                             --reptype gz.msgs.Boolean --timeout 1000 --req '<protobuf text>'
                  需要本机装有 gz CLI 且能访问仿真（同 GZ_PARTITION/网络可达）
    2. classic  : ROS 服务 /gazebo/set_model_state (Gazebo classic)

目标位置由参数 target_x/y/z 手动输入（Gazebo 世界系），脚本不做自动识别。
"""

import math
import os
import random
import shutil
import subprocess
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool, Trigger

try:
    from gazebo_msgs.srv import SetModelState
    HAS_GAZEBO_MSGS = True
except ImportError:
    HAS_GAZEBO_MSGS = False


def rpy_to_quat(roll, pitch, yaw):
    """ZYX 外旋欧拉角(弧度) -> 四元数 (x, y, z, w)"""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


class TargetAutoPoser(Node):
    def __init__(self):
        super().__init__('target_auto_poser')

        # ---- 参数 ----
        self.declare_parameter('model_name', 'waterdrop')
        self.declare_parameter('world_name', 'waterdrop_and_iris')  # gz-sim 世界名
        # Gazebo 世界系中的目标位置：手动输入，必须与采集脚本的 target_x/y/z 一致
        self.declare_parameter('target_x', 0.0)
        self.declare_parameter('target_y', 0.0)
        self.declare_parameter('target_z', 150.0)

        # 甜甜圈采样参数
        self.declare_parameter('radius_min', 30.0)         # 到目标水平距离下限(米)
        self.declare_parameter('radius_max', 150.0)        # 上限(米)
        self.declare_parameter('radius_sampling', 'log')  # log=对数均匀(远近兼顾)/uniform
        self.declare_parameter('rel_alt_min', 2.0)        # 相对目标高度下限(米)
        self.declare_parameter('rel_alt_max', 15.0)       # 上限(米)
        self.declare_parameter('face_target_ratio', 0.9)  # 机头朝向目标的概率(其余随机)
        self.declare_parameter('yaw_jitter_deg', 1.0)    # 朝向目标时的偏航抖动(度)
        self.declare_parameter('tilt_max_deg', 1.0)       # 随机横滚/俯仰幅值(度)，数据增广用
        # 对“相机模型”特别有用：把模型基座直接朝向目标（而不是随机偏航）
        self.declare_parameter('look_at_target', False)   # 是否让模型一直看向目标
        self.declare_parameter('look_at_pitch_offset_deg', 0.0)  # 额外俯仰修正(度)

        # 采集节奏
        self.declare_parameter('settle_time', 0.5)        # 传送后稳定等待(秒)
        self.declare_parameter('captures_per_pose', 1)    # 每位姿抓几帧(静态场景 >1 近乎重复)
        self.declare_parameter('capture_spacing', 0.3)    # 连抓间隔(秒)
        self.declare_parameter('poses_total', 0)          # 总位姿数，0=无限
        self.declare_parameter('hold_rate_hz', 30.0)      # 重复置位频率(抑制自由落体)

        # 后端
        self.declare_parameter('backend', 'auto')         # auto/gz_cli/classic
        self.declare_parameter('gz_cmd', 'gz')            # gz CLI 命令名(harmonic 之前为 ign)
        self.declare_parameter('cli_timeout', 1.0)        # gz service 单次调用超时(秒)

        # 与采集脚本联动
        self.declare_parameter('pose_topic', '/target_auto_poser/model_pose')
        self.declare_parameter('control_collector', False)
        self.declare_parameter('collector_set_collecting', '/h_data_collector/set_collecting')
        self.declare_parameter('collector_capture_once', '/h_data_collector/capture_once')

        # 直接存图（用于 target 手动标注）
        self.declare_parameter('image_topic', '/world/waterdrop_and_iris/model/waterdrop/link/camera_link/sensor/camera/image')
        self.declare_parameter('save_images', True)
        self.declare_parameter('output_dir', 'target_manual_images')
        self.declare_parameter('val_ratio', 0.2)
        self.declare_parameter('jpeg_quality', 95)

        # 结束时的停放位姿（远离目标、贴地；停止钉持后模型就地停住）
        self.declare_parameter('park_offset_x', 3.0)
        self.declare_parameter('park_offset_y', 0.0)
        self.declare_parameter('park_rel_alt', 0.2)

        self.declare_parameter('auto_start', True)
        self.declare_parameter('seed', 42)

        p = self.get_parameter
        self.model_name = p('model_name').value
        self.world_name = p('world_name').value
        self.target = (p('target_x').value, p('target_y').value, p('target_z').value)
        self.r_min = p('radius_min').value
        self.r_max = p('radius_max').value
        self.log_r = p('radius_sampling').value == 'log'
        self.alt_min = p('rel_alt_min').value
        self.alt_max = p('rel_alt_max').value
        self.face_ratio = p('face_target_ratio').value
        self.yaw_jitter = math.radians(p('yaw_jitter_deg').value)
        self.tilt_max = math.radians(p('tilt_max_deg').value)
        self.look_at_target = p('look_at_target').value
        self.look_at_pitch_offset = math.radians(p('look_at_pitch_offset_deg').value)
        self.settle_time = p('settle_time').value
        self.cap_per_pose = max(1, p('captures_per_pose').value)
        self.cap_spacing = p('capture_spacing').value
        self.poses_total = p('poses_total').value
        self.hold_period = 1.0 / max(1.0, p('hold_rate_hz').value)
        self.control_collector = p('control_collector').value
        self.gz_cmd = p('gz_cmd').value
        self.cli_timeout = p('cli_timeout').value
        self.park = (self.target[0] + p('park_offset_x').value,
                     self.target[1] + p('park_offset_y').value,
                     self.target[2] + p('park_rel_alt').value)

        # 存图相关
        self.save_images = p('save_images').value
        self.output_dir = Path(p('output_dir').value)
        if not self.output_dir.is_absolute():
            self.output_dir = (Path(__file__).resolve().parent.parent / self.output_dir).resolve()
        self.val_ratio = p('val_ratio').value
        self.jpeg_quality = p('jpeg_quality').value
        self.bridge = CvBridge()
        self.latest_frame = None
        self._image_warned = False
        if self.save_images:
            for split in ('train', 'val'):
                (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1)
            self.create_subscription(Image, p('image_topic').value, self.image_cb, qos)

        self.rng = random.Random(p('seed').value)

        # ---- 状态 ----
        self.backend = None            # 'gz_cli' / 'classic'
        self.cli = None                # ROS 服务客户端(classic 时用)
        self.gz_service = None         # gz transport 服务名(gz_cli 时用)
        self._cap_cli = None
        self._set_cli = None
        self._pending = []             # 未回收的 ROS service future
        self.current_pose = None       # (x,y,z,qx,qy,qz,qw) 当前钉持位姿
        self.last_hold_time = 0.0
        self.state = 'idle'            # idle / settle / capture / done
        self.state_t = 0.0
        self.issued_caps = 0
        self.last_cap_time = 0.0
        self.poses_done = 0
        self.pose_failed = 0
        self._consec_fail = 0          # CLI 后端连续失败计数(熔断)
        self._cli_ms_ema = None        # CLI 调用耗时 EMA(一次性日志)
        self._timing_logged = False

        self.pub_pose = self.create_publisher(
            PoseStamped, p('pose_topic').value, 10)
        self.create_service(Trigger, '~/start', self.start_cb)
        self.create_service(Trigger, '~/stop', self.stop_cb)

        ok = self._detect_backend(p('backend').value)
        if ok:
            self.timer = self.create_timer(min(self.hold_period, 0.05), self.tick)
            self.get_logger().info(
                f'后端={self.backend} 模型={self.model_name} '
                f'目标=({self.target[0]:.1f},{self.target[1]:.1f},{self.target[2]:.1f}) '
                f'环带 r∈[{self.r_min},{self.r_max}]m 高∈[{self.alt_min},{self.alt_max}]m，'
                f'用 start/stop 服务控制')
        else:
            self.get_logger().error(
                '未找到可用的位姿设置后端，节点空闲。排查：\n'
                '  gz CLI: which gz && gz service list | grep set_pose\n'
                '  ROS:    ros2 service list | grep -iE "set_pose|model_state"\n'
                'gz-sim 若两者都不可见，需在本机安装 gz 或给 ros_gz_bridge 暴露 '
                '/world/<world>/set_pose')
        if p('auto_start').value and ok:
            self._begin()

    # ---------------- 后端探测 ----------------

    def _detect_backend(self, backend):
        if backend in ('auto', 'gz_cli') and shutil.which(self.gz_cmd):
            self.backend = 'gz_cli'
            self.gz_service = f'/world/{self.world_name}/set_pose'
            return True
        services = dict(self.get_service_names_and_types())
        if backend in ('auto', 'classic') and '/gazebo/set_model_state' in services \
                and HAS_GAZEBO_MSGS:
            self.cli = self.create_client(SetModelState, '/gazebo/set_model_state')
            self.backend = 'classic'
            return True
        return False

    # ---------------- 位姿发送 ----------------

    def _send_pose(self, pose):
        if self.backend == 'gz_cli':
            self._send_pose_cli(pose)
            return
        if self.cli is None or not self.cli.service_is_ready():
            return
        x, y, z, qx, qy, qz, qw = pose
        req = SetModelState.Request()
        req.model_state.model_name = self.model_name
        req.model_state.pose.position.x = x
        req.model_state.pose.position.y = y
        req.model_state.pose.position.z = z
        req.model_state.pose.orientation.x = qx
        req.model_state.pose.orientation.y = qy
        req.model_state.pose.orientation.z = qz
        req.model_state.pose.orientation.w = qw
        req.model_state.reference_frame = 'world'
        try:
            self._pending.append(self.cli.call_async(req))
        except Exception as e:
            self.pose_failed += 1
            self.get_logger().error(f'set_pose 调用失败: {e}', throttle_duration_sec=2.0)

    def _send_pose_cli(self, pose):
        """gz transport 服务调用(gz service 子进程)。连续失败 5 次熔断，避免超时拖垮主循环。"""
        if self._consec_fail >= 5:
            return
        x, y, z, qx, qy, qz, qw = pose
        req = (f'name: "{self.model_name}", '
               f'position: {{x: {x:.6f}, y: {y:.6f}, z: {z:.6f}}}, '
               f'orientation: {{x: {qx:.6f}, y: {qy:.6f}, z: {qz:.6f}, w: {qw:.6f}}}')
        cmd = [self.gz_cmd, 'service', '-s', self.gz_service,
               '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
               '--timeout', '1000', '--req', req]
        import os
        import time
        t0 = time.monotonic()
        try:
            res = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=self.cli_timeout)
            ms = (time.monotonic() - t0) * 1000.0
            self._cli_ms_ema = ms if self._cli_ms_ema is None else \
                0.8 * self._cli_ms_ema + 0.2 * ms
            if not self._timing_logged and self._cli_ms_ema is not None:
                self._timing_logged = True
                self.get_logger().info(f'gz service 调用耗时约 {ms:.0f} ms；'
                                       f'若持续高于 {self.hold_period * 1000:.0f} ms '
                                       f'请降低 hold_rate_hz')
            if res.returncode == 0 and 'true' in res.stdout.lower():
                self._consec_fail = 0
                return
            self.pose_failed += 1
            self._consec_fail += 1
            self.get_logger().error(
                f'set_pose 失败(rc={res.returncode}): stdout={res.stdout!r} '
                f'stderr={res.stderr!r}',
                throttle_duration_sec=2.0)
        except subprocess.TimeoutExpired:
            self.pose_failed += 1
            self._consec_fail += 1
            self.get_logger().error(
                f'gz service 超时({self.cli_timeout}s)：CLI 无法访问仿真？'
                f'检查 gz 安装、GZ_PARTITION 与网络。连续失败 {self._consec_fail}/5',
                throttle_duration_sec=2.0)
        except Exception as e:
            self.pose_failed += 1
            self._consec_fail += 1
            self.get_logger().error(f'gz service 异常: {e}', throttle_duration_sec=2.0)
        if self._consec_fail >= 5:
            self.get_logger().error(
                'gz service 连续失败 5 次，已停止钉持。修复后请重启节点。')

    def _reap_pending(self):
        still = []
        for f in self._pending:
            if not f.done():
                still.append(f)
                continue
            try:
                res = f.result()
                ok = getattr(res, 'success', None)
                if ok is False:
                    self.pose_failed += 1
                    msg = getattr(res, 'status_message', '')
                    self.get_logger().error(
                        f'set_pose 被拒绝：{msg or "模型名/世界名是否正确？"}',
                        throttle_duration_sec=2.0)
            except Exception as e:
                self.pose_failed += 1
                self.get_logger().error(f'set_pose 结果异常: {e}',
                                        throttle_duration_sec=2.0)
        self._pending = still

    def image_cb(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            if not self._image_warned:
                self.get_logger().warn(f'图像转换失败: {e}')
                self._image_warned = True

    def _publish_pose(self, pose):
        x, y, z, qx, qy, qz, qw = pose
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'  # Gazebo 世界系（与 target_x/y/z 同系）
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = x, y, z
        msg.pose.orientation.x, msg.pose.orientation.y = qx, qy
        msg.pose.orientation.z, msg.pose.orientation.w = qz, qw
        self.pub_pose.publish(msg)

    # ---------------- 采样 ----------------

    def sample_pose(self):
        """甜甜圈采样：水平距离+方位角+高度+朝向+小倾角"""
        if self.log_r:
            r = math.exp(self.rng.uniform(math.log(self.r_min), math.log(self.r_max)))
        else:
            r = self.rng.uniform(self.r_min, self.r_max)
        th = self.rng.uniform(0.0, 2.0 * math.pi)
        x = self.target[0] + r * math.cos(th)
        y = self.target[1] + r * math.sin(th)
        z = self.target[2] + self.rng.uniform(self.alt_min, self.alt_max)

        if self.look_at_target:
            dx = self.target[0] - x
            dy = self.target[1] - y
            dz = self.target[2] - z
            yaw = math.atan2(dy, dx)
            yaw += self.rng.uniform(-self.yaw_jitter, self.yaw_jitter)
            horizontal = math.hypot(dx, dy)
            pitch = -math.atan2(dz, horizontal)
            pitch += self.look_at_pitch_offset
            pitch += self.rng.uniform(-self.tilt_max, self.tilt_max)
            roll = self.rng.uniform(-self.tilt_max, self.tilt_max)
        elif self.rng.random() < self.face_ratio:
            yaw = math.atan2(self.target[1] - y, self.target[0] - x)
            yaw += self.rng.uniform(-self.yaw_jitter, self.yaw_jitter)
            roll = self.rng.uniform(-self.tilt_max, self.tilt_max)
            pitch = self.rng.uniform(-self.tilt_max, self.tilt_max)
        else:
            yaw = self.rng.uniform(0.0, 2.0 * math.pi)
            roll = self.rng.uniform(-self.tilt_max, self.tilt_max)
            pitch = self.rng.uniform(-self.tilt_max, self.tilt_max)
        return self._quat_pose(x, y, z, roll, pitch, yaw)

    @staticmethod
    def _quat_pose(x, y, z, roll, pitch, yaw):
        qx, qy, qz, qw = rpy_to_quat(roll, pitch, yaw)
        return (x, y, z, qx, qy, qz, qw)

    # ---------------- 主循环 ----------------

    def tick(self):
        now = self.get_clock().now().nanoseconds / 1e9
        self._reap_pending()

        # 熔断(CLI 后端连续失败)时停止一切动作
        if self._consec_fail >= 5:
            return

        # 钉持：按固定频率重复发送当前位姿，抑制自由落体
        if self.current_pose is not None and now - self.last_hold_time >= self.hold_period:
            self._send_pose(self.current_pose)
            self._publish_pose(self.current_pose)
            self.last_hold_time = now

        if self.state in ('idle', 'done'):
            return

        if self.state == 'settle':
            if now - self.state_t >= self.settle_time:
                self.state = 'capture'
                self.issued_caps = 0
                self.last_cap_time = 0.0
            return

        if self.state == 'capture':
            if self.issued_caps < self.cap_per_pose and \
                    now - self.last_cap_time >= self.cap_spacing:
                if self.control_collector and self._cap_cli is not None and \
                        self._cap_cli.service_is_ready():
                    try:
                        self._cap_cli.call_async(Trigger.Request())
                    except Exception:
                        pass
                if self.save_images:
                    self._save_current_frame()
                self.issued_caps += 1
                self.last_cap_time = now
            if self.issued_caps >= self.cap_per_pose:
                self.poses_done += 1
                if 0 < self.poses_total <= self.poses_done:
                    self._finish()
                else:
                    self._teleport_next()
            return

    def _teleport_next(self):
        self.current_pose = self.sample_pose()
        self.last_hold_time = 0.0  # 下一 tick 立即发送
        self.state = 'settle'
        self.state_t = self.get_clock().now().nanoseconds / 1e9
        x, y, z = self.current_pose[:3]
        self.get_logger().info(f'[{self.poses_done + 1}] 传送至 ({x:.1f}, {y:.1f}, {z:.1f})')

    def _begin(self):
        if self.backend is None:
            return
        if self.control_collector:
            self._cap_cli = self.create_client(
                Trigger, self.get_parameter('collector_capture_once').value)
            self._set_cli = self.create_client(
                SetBool, self.get_parameter('collector_set_collecting').value)
            if self._set_cli.wait_for_service(timeout_sec=2.0):
                self._set_cli.call_async(SetBool.Request(data=True))
            else:
                self.get_logger().warn('采集脚本服务不可见，将只摆位不触发采集')
                self.control_collector = False
        self.get_logger().info('开始自动摆位采集')
        self._teleport_next()

    def _save_current_frame(self):
        if self.latest_frame is None:
            self.get_logger().warn('尚无图像，跳过保存', throttle_duration_sec=2.0)
            return
        split = 'val' if self.rng.random() < self.val_ratio else 'train'
        now_ns = self.get_clock().now().nanoseconds
        fname = f'frame_{self.poses_done}_{self.issued_caps}_{now_ns}.jpg'
        out_path = self.output_dir / 'images' / split / fname
        cv2.imwrite(str(out_path), self.latest_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        self.get_logger().info(f'保存 {split} 图片: {out_path}')

    def _finish(self):
        self.state = 'done'
        self.current_pose = self._quat_pose(*self.park, 0.0, 0.0, 0.0)
        if self.control_collector and self._set_cli is not None and \
                self._set_cli.service_is_ready():
            self._set_cli.call_async(SetBool.Request(data=False))
        self.get_logger().info(
            f'完成：共 {self.poses_done} 个位姿，set_pose 失败 {self.pose_failed} 次，'
            f'模型停放在 ({self.park[0]:.1f}, {self.park[1]:.1f}, {self.park[2]:.1f})')

    def start_cb(self, req, res):
        if self.backend is None:
            res.success, res.message = False, '位姿后端不可用'
        elif self.state in ('idle', 'done'):
            self.poses_done = 0
            self.pose_failed = 0
            self._begin()
            res.success, res.message = True, '开始自动摆位'
        else:
            res.success, res.message = False, '已在运行中'
        return res

    def stop_cb(self, req, res):
        if self.state in ('settle', 'capture'):
            self._finish()
            res.success, res.message = True, '已停止并停放模型'
        else:
            res.success, res.message = False, '当前不在运行'
        return res


def main():
    rclpy.init()
    node = TargetAutoPoser()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
