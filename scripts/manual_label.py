"""
manual_label.py — 用鼠标在图像上框选目标，生成 YOLO 格式标注

用途：对 target_auto_poser.py 保存下来的图片手动画框，每张图对应一个 .txt：
    class_id cx cy w h
其中 cx,cy,w,h 均为相对于图像宽高的归一化值。

运行：
    python3 scripts/manual_label.py \
      --images target_manual_images/images/train \
      --output target_manual_images/labels/train

操作：
    鼠标左键拖拽：画框
    r / 退格键：撤销当前图的框
    n / 空格 / 回车：保存并切换下一张
    s：跳过当前图（不生成标签）
    q / Esc：退出
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='手动标注目标，生成 YOLO txt')
    parser.add_argument('--images', '-i', required=True,
                        help='图片目录，例如 .../images/train')
    parser.add_argument('--output', '-o', required=True,
                        help='标签输出目录，例如 .../labels/train')
    parser.add_argument('--class-id', '-c', type=int, default=0,
                        help='YOLO 类别编号，默认 0')
    parser.add_argument('--log', default='manual_label_log.csv',
                        help='标注进度/日志文件路径')
    return parser.parse_args()


class Labeler:
    def __init__(self, image_dir, label_dir, class_id=0, log_path='manual_label_log.csv'):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.class_id = class_id
        self.log_path = Path(log_path)

        self.images = sorted([p for p in self.image_dir.iterdir()
                              if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])
        if not self.images:
            print(f'未找到图片: {self.image_dir}')
            sys.exit(1)

        # 读取已标注进度
        self.done = set()
        if self.log_path.exists():
            with open(self.log_path, newline='') as f:
                for row in csv.reader(f):
                    if row:
                        self.done.add(row[0])

        self.drawing = False
        self.ix = self.iy = 0
        self.box = None  # (x1, y1, x2, y2)
        self.current_img = None
        self.current_path = None
        self.current_name = None

    def write_label(self, box, shape):
        H, W = shape[:2]
        x1, y1, x2, y2 = box
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1 = max(0, min(x1, W - 1))
        x2 = max(0, min(x2, W - 1))
        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H - 1))

        cx = ((x1 + x2) / 2.0) / W
        cy = ((y1 + y2) / 2.0) / H
        w = (x2 - x1) / W
        h = (y2 - y1) / H
        if w <= 0 or h <= 0:
            return False

        txt_path = self.label_dir / f'{self.current_name}.txt'
        txt_path.write_text(f'{self.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
        return True

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.box = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.box = (self.ix, self.iy, x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.box = (self.ix, self.iy, x, y)

    def draw_overlay(self):
        img = self.current_img.copy()
        if self.box is not None:
            x1, y1, x2, y2 = self.box
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img,
                    f'{self.current_name}  左键画框  r撤销  n保存下一张  s跳过  q退出',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return img

    def run(self):
        cv2.namedWindow('manual_label', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('manual_label', self.mouse_cb)

        for img_path in self.images:
            self.current_path = img_path
            self.current_name = img_path.stem
            if self.current_name in self.done:
                continue

            self.current_img = cv2.imread(str(img_path))
            if self.current_img is None:
                print(f'无法读取: {img_path}')
                continue
            self.box = None

            while True:
                disp = self.draw_overlay()
                cv2.imshow('manual_label', disp)
                key = cv2.waitKey(50) & 0xFF

                if key in (ord('q'), 27):  # q / Esc
                    cv2.destroyAllWindows()
                    return

                if key in (ord('r'), 8):  # r / Backspace
                    self.box = None

                if key == ord('s'):  # skip
                    self._log('skipped')
                    break

                if key in (ord('n'), 32, 13):  # n / Space / Enter
                    if self.box is None:
                        print('  尚未画框，按 s 跳过或画框后按 n')
                        continue
                    if self.write_label(self.box, self.current_img.shape):
                        self._log('labeled')
                        print(f'  已保存: {self.label_dir / (self.current_name + ".txt")}')
                    else:
                        self._log('invalid_box')
                    break

        cv2.destroyAllWindows()
        print('全部完成')

    def _log(self, action):
        new_file = not self.log_path.exists()
        with open(self.log_path, 'a', newline='') as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(['image', 'action'])
            w.writerow([self.current_name, action])
        self.done.add(self.current_name)


def main():
    args = parse_args()
    Labeler(args.images, args.output, args.class_id, args.log).run()


if __name__ == '__main__':
    main()
