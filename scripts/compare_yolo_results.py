#!/usr/bin/env python3
"""
compare_yolo_results.py — 对比两个无人机检测结果话题

订阅：
    /mavros/drone/detection_result_yolo
    /mavros/drone/detection_result

类型：std_msgs/Float32MultiArray

只关注数组第 4/5/6/7 位（即索引 3/4/5/6），随时间绘制对比曲线。
按 Ctrl-C 结束后自动生成并保存对比图 compare_yolo_results.png。
"""

import os
import threading
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# 尽量使用有 GUI 的后端；无显示器时会自动回退到 Agg 并只保存图片
try:
    matplotlib.use('TkAgg')
except Exception:
    pass


LABELS = ['yaw (deg)', 'pitch (deg)', 'field 6', 'field 7']
INDICES = [3, 4, 5, 6]
COLORS = {'yolo': 'tab:blue', 'det': 'tab:orange'}


class CompareYoloNode(Node):
    def __init__(self):
        super().__init__('compare_yolo_results')

        self.data = {
            'yolo': {'t': [], 'v': [[] for _ in INDICES]},
            'det':  {'t': [], 'v': [[] for _ in INDICES]},
        }
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.create_subscription(
            Float32MultiArray,
            '/mavros/drone/detection_result_yolo',
            self.make_callback('yolo'),
            10)
        self.create_subscription(
            Float32MultiArray,
            '/mavros/drone/detection_result',
            self.make_callback('det'),
            10)

        self.get_logger().info(
            'Subscribed to /mavros/drone/detection_result_yolo and '
            '/mavros/drone/detection_result. Press Ctrl-C to plot.')

    def make_callback(self, key):
        def cb(msg):
            arr = msg.data
            if len(arr) < max(INDICES) + 1:
                self.get_logger().warn(
                    f'{key}: received array length {len(arr)} < {max(INDICES) + 1}, skipping')
                return
            now = time.time() - self.start_time
            with self.lock:
                self.data[key]['t'].append(now)
                for i, idx in enumerate(INDICES):
                    self.data[key]['v'][i].append(float(arr[idx]))
        return cb

    def plot_results(self):
        with self.lock:
            yolo = self.data['yolo']
            det = self.data['det']

        if not yolo['t'] and not det['t']:
            self.get_logger().warn('No data received, nothing to plot.')
            return

        fig, axes = plt.subplots(len(INDICES), 1, figsize=(12, 10), sharex=True)
        if len(INDICES) == 1:
            axes = [axes]

        for i, (ax, label) in enumerate(zip(axes, LABELS)):
            if yolo['t']:
                ax.plot(yolo['t'], yolo['v'][i], label='detection_result_yolo',
                        color=COLORS['yolo'], linewidth=1.2,
                        marker='o', markersize=3)
            if det['t']:
                ax.plot(det['t'], det['v'][i], label='detection_result',
                        color=COLORS['det'], linewidth=1.2,
                        marker='o', markersize=3)
            ax.set_ylabel(label)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='upper right')

        axes[-1].set_xlabel('Time (s)')
        fig.suptitle('YOLO vs Detection Result Comparison (fields 4-7)')
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])

        save_path = os.path.join(os.getcwd(), 'compare_yolo_results.png')
        fig.savefig(save_path, dpi=150)
        self.get_logger().info(f'Saved comparison plot to: {save_path}')

        plt.show(block=True)


def main():
    rclpy.init()
    node = CompareYoloNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.plot_results()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
