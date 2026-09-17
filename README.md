# YOLO v8 用于 H 标检测

从仿真中采集 H 标图像、训练 YOLOv8、部署实时检测的完整流程。

## 目录结构

```
~/target_detection/
├── scripts/
│   ├── h_data_collector.py    # 机载相机：数据采集 + 自动标注（ROS2 节点）
│   ├── h_auto_poser.py        # 自动摆位：甜甜圈采样传送 + 可选存图（ROS2 节点）
│   ├── h_marker_detector.py   # 实时检测 + 相对位置解算（ROS2 节点）
│   ├── manual_label.py        # 手动画框，生成 YOLO txt
│   ├── yolov8n.pt             # 官方预训练权重
│   └── best.pt                # 训练得到的最佳权重（训练后生成）
├── h_dataset/                 # h_data_collector 自动生成的数据集
│   ├── images/{train,val}/*.jpg
│   ├── labels/{train,val}/*.txt
│   └── log.csv
├── h_manual_images/           # h_auto_poser 自动存图 + manual_label.py 手标
│   ├── images/{train,val}/*.jpg
│   ├── labels/{train,val}/*.txt
│   └── dataset.yaml
├── h_marker.yaml              # h_dataset 的 YOLO 配置文件
├── runs/detect/train*/        # 训练产物（best.pt / last.pt / results.png）
├── weights/                   # Ultralytics 导出/训练时可能自动创建
└── yolo_venv/                 # Python 虚拟环境
```

## 整体工作流

项目支持两种采集方式，按需选一种即可：

1. **机载相机自动标注**（推荐，无人机在 H 标上方悬停）
   - `h_data_collector.py`：根据 MAVROS 位姿 + 相机内外参自动投影生成 YOLO 标注
2. **自动摆位 + 手动标注**（高速/复杂姿态，自动摆位后人工画框）
   - `h_auto_poser.py -p save_images:=true`：自动摆位并保存每个位姿的相机图片
   - `manual_label.py`：用鼠标手动画框生成 YOLO 标签

训练统一用 `yolo detect train data=<yaml> model=scripts/yolov8n.pt ...`。

---

## 1. 环境准备

### 1.1 安装 cv_bridge（ROS2 自带则跳过）

```bash
sudo apt install ros-$ROS_DISTRO-cv-bridge ros-$ROS_DISTRO-vision-opencv
```

### 1.2 创建虚拟环境

```bash
python3 -m venv ~/target_detection/yolo_venv --system-site-packages
source ~/target_detection/yolo_venv/bin/activate
```

### 1.3 安装 GPU 版 PyTorch

驱动较旧或 CUDA 版本受限时，用与系统驱动兼容的 wheel（例如 cu118/cu121）：

```bash
pip install -U torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### 1.4 安装 Ultralytics 及绘图依赖

```bash
pip install ultralytics matplotlib scipy contourpy
pip install --force-reinstall matplotlib scipy contourpy
```

### 1.5 环境注意事项

**① numpy 必须保持 1.x**

ROS Humble 的 `cv_bridge` 按 NumPy 1.x 编译，venv 里 NumPy 升到 2.x 会报 `_ARRAY_API not found` 并段错误：

```bash
pip show numpy        # 必须是 1.26.x
pip install "numpy==1.26.4"   # 被升级了就拉回来
```

**② 确认 `yolo` 命令来自 venv**

```bash
which yolo    # 必须是 ~/target_detection/yolo_venv/bin/yolo
```

若指向 `~/.local/bin/yolo`：

```bash
pip install --force-reinstall --no-deps ultralytics
```

---

## 2. 启动仿真环境

```bash
sudo sh start_docker.sh -v v1.1.5
sudo sh start_docker_ardupilot_mavros.sh -v v1.1.5
```

或目标打击场景：

```bash
gz sim -v4 waterdrop_and_iris.sdf
```

若使用 waterdrop 相机，需把 Gazebo 图像/相机信息桥接到 ROS2：

```bash
ros2 run ros_gz_bridge parameter_bridge \
  /world/waterdrop_and_iris/model/waterdrop/link/camera_link/sensor/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo

ros2 run ros_gz_bridge parameter_bridge \
  /world/waterdrop_and_iris/model/waterdrop/link/camera_link/sensor/camera/image@sensor_msgs/msg/Image@gz.msgs.Image
```

---

## 3. 数据采集

### 3.1 机载相机自动标注（h_data_collector.py）

H 标位置由 `marker_x/y/z` 参数输入，脚本结合位姿和相机内外参自动投影生成标注。

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
source ~/target_detection/yolo_venv/bin/activate
cd ~/target_detection/scripts
python3 h_data_collector.py
```

控制采集：

```bash
ros2 service call /h_data_collector/set_collecting std_srvs/srv/SetBool "{data: true}"
ros2 service call /h_data_collector/set_collecting std_srvs/srv/SetBool "{data: false}"
ros2 service call /h_data_collector/capture_once std_srvs/srv/Trigger "{}"

rqt_image_view   # 选 /h_data_collector/preview 查看绿框
```

输出到 `h_dataset/images/{train,val}` + `h_dataset/labels/{train,val}`，并自动生成 `h_marker.yaml`。

### 3.2 自动摆位 + 手动标注（h_auto_poser.py + manual_label.py）

适合高速飞行或复杂视角：自动把模型传送到目标周围并保存图片，然后人工画框。

```bash
# 终端1：自动摆位并保存图片
python3 scripts/h_auto_poser.py \
  --ros-args \
  -p model_name:=waterdrop \
  -p world_name:=waterdrop_and_iris \
  -p target_x:=0.0 -p target_y:=0.0 -p target_z:=150.0 \
  -p look_at_target:=true \
  -p image_topic:=/world/waterdrop_and_iris/model/waterdrop/link/camera_link/sensor/camera/image \
  -p save_images:=true \
  -p output_dir:=/home/ahao/target_detection/h_manual_images \
  -p auto_start:=true
```

图片会保存到 `h_manual_images/images/{train,val}`。

然后手动标注：

```bash
python3 scripts/manual_label.py \
  --images /home/ahao/target_detection/h_manual_images/images/train \
  --output /home/ahao/target_detection/h_manual_images/labels/train

python3 scripts/manual_label.py \
  --images /home/ahao/target_detection/h_manual_images/images/val \
  --output /home/ahao/target_detection/h_manual_images/labels/val
```

操作：鼠标左键拖拽画框，`n` 保存下一张，`r` 撤销，`s` 跳过，`q` 退出。

最后写数据集配置 `h_manual_images/dataset.yaml`：

```yaml
path: /home/ahao/target_detection/h_manual_images
train: images/train
val: images/val
names: ['h_marker']
nc: 1
```

---

## 4. 训练

```bash
source ~/target_detection/yolo_venv/bin/activate
cd ~/target_detection

# 自动标注得到的数据
yolo detect train data=h_marker.yaml model=scripts/yolov8n.pt epochs=100 imgsz=640 device=0

# 或手动标注得到的数据
yolo detect train data=h_manual_images/dataset.yaml model=scripts/yolov8n.pt epochs=100 imgsz=640 device=0
```

- `device=0` 指定 GPU；否则默认 CPU
- 训练后验证：`yolo detect val data=h_marker.yaml model=runs/detect/train/weights/best.pt`

---

## 5. 部署检测

把训练好的权重放到 `scripts/` 下：

```bash
cp runs/detect/train/weights/best.pt ~/target_detection/scripts/
source /opt/ros/$ROS_DISTRO/setup.bash
source ~/target_detection/yolo_venv/bin/activate
cd ~/target_detection/scripts
python3 h_marker_detector.py
```

查看结果：

```bash
ros2 topic echo /h_marker/position
rqt_image_view   # 选 /h_marker/annotated
```

`/h_marker/position` 为 H 标中心在左目相机光轴系下的位置（z 前、x 右、y 下，单位米）。

---

## 6. 常见问题

- **`No module named 'torch'`**：`yolo` 命令不在 venv，重装 ultralytics。
- **`_ARRAY_API not found`** / 段错误：NumPy 被升到 2.x，降回 `1.26.4`。
- **CUDA out of memory**：加 `batch=8` 或 `imgsz=512`。


---

## 7. 当前验证通过的环境

```text
OS:          Ubuntu 22.04
GPU:         NVIDIA GeForce RTX 4070 Ti 12GB
Driver:      535.309.01 (CUDA 12.2)
Python:      3.10.12
PyTorch:     2.7.1+cu118 / 2.14.0+cu130
numpy:       1.26.4
Ultralytics: 8.3.236+
```
