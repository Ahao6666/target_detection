# YOLO v8 用于 H 标检测

从仿真中自动采集标注 H 标数据集、训练 YOLOv8、部署实时检测的完整流程。

## 目录结构

```
~/target_detection/
├── scripts/
│   ├── h_data_collector.py   # 数据采集 + 自动标注（ROS2 节点）
│   ├── h_marker_detector.py  # 实时检测 + 相对位置解算（ROS2 节点）
│   └── yolov8n.pt            # 官方预训练权重
├── h_dataset/                # 数据集（采集脚本自动生成）
│   ├── images/{train,val}/*.jpg
│   └── labels/{train,val}/*.txt
├── h_marker.yaml             # YOLO 数据集配置（采集脚本自动生成）
├── runs/detect/train*/       # 训练产物（best.pt / last.pt / results.png）
└── yolo_venv/                # Python 虚拟环境
```

## 整体工作流

1. **环境准备**（只需一次）：装 cv_bridge、建 venv、装 GPU 版 torch + ultralytics
2. **启动仿真**：docker + ArduPilot/MAVROS
3. **采集数据**：`h_data_collector.py` 自动投影生成 YOLO 标注，无需人工标注
4. **训练**：`yolo detect train`，GPU 约 5–10 分钟（358 张图）
5. **部署检测**：`h_marker_detector.py` 加载 best.pt，发布检测框和相对位置

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

驱动 535.309.01（CUDA 12.2 档）实测使用 cu118 构建（cu121 亦可）：

```bash
pip install -U torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118
```

验证（关键看 `True` 和 `+cu118` 后缀）：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### 1.4 安装 Ultralytics 及绘图依赖

```bash
pip install ultralytics matplotlib scipy contourpy
```

`matplotlib/scipy/contourpy` 必须装进 venv（原因见下节）。

### 1.5 ⚠️ 环境注意事项（踩坑记录，必读）

**① numpy 必须保持 1.x（当前固定 1.26.4）**

ROS Humble 的 `cv_bridge` C 扩展按 NumPy 1.x 编译，venv 里 NumPy 一旦升到 2.x，
ROS 脚本（采集/检测）就会报 `_ARRAY_API not found` 并段错误。**每次 pip 安装/升级
任何包之后，确认 numpy 没被带走**：

```bash
pip show numpy        # 必须是 1.26.4
pip install "numpy==1.26.4"   # 被升级了就拉回来
```

**② `--system-site-packages` 会"冒用"系统旧包**

该模式下 pip 认为系统包"已满足"而跳过安装，运行时却拿到按 NumPy 1.x 编译的
系统 matplotlib/scipy → `ImportError: numpy.core.multiarray failed to import`。
因此 matplotlib/scipy/contourpy 必须用 `--force-reinstall` 装进 venv：

```bash
pip install --force-reinstall matplotlib scipy contourpy
```

**③ 确认 `yolo` 命令来自 venv**

```bash
which yolo    # 必须是 ~/target_detection/yolo_venv/bin/yolo
```

若指向 `~/.local/bin/yolo`（shebang 是系统 python3），会报
`No module named 'torch'`。修复：

```bash
pip install --force-reinstall --no-deps ultralytics
```

---

## 2. 启动仿真环境

```bash
sudo sh start_docker.h -v v1.1.5
sudo sh start_docker_ardupilot_mavros.sh -v v1.1.5
```

## 3. 数据采集 + 自动标注

H 标固定在世界原点，采集脚本结合 MAVROS 位姿和相机内外参把 H 标投影到图像，
自动生成 YOLO 格式标注（目录结构与 CVAT 导出 YOLO 1.1 一致）。

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
source ~/target_detection/yolo_venv/bin/activate
cd ~/target_detection/scripts
python3 h_data_collector.py
```

启动日志出现 `TF 查询失败...回退到静态外参 cam_mount=down_gz` 是**预期行为**
（仿真不发 TF，静态外参来自 zed_camera 模型 SDF，就是真值）。

另开终端控制采集：

```bash
# 开始/停止自动采集（默认间隔 1s，框中心移动 <25px 的悬停重复帧自动过滤）
ros2 service call /h_data_collector/set_collecting std_srvs/srv/SetBool "{data: true}"
ros2 service call /h_data_collector/set_collecting std_srvs/srv/SetBool "{data: false}"

# 手动抓一帧
ros2 service call /h_data_collector/capture_once std_srvs/srv/Trigger "{}"

# 查看自动标注效果（绿框应套住 H 标，先验证再采集）
rqt_image_view   # 选 /h_data_collector/preview
```

输出到 `h_dataset/images/{train,val}` + `h_dataset/labels/{train,val}`
（默认按 8:2 随机划分），并自动生成 `h_marker.yaml`；每帧位姿和框记录在
`h_dataset/log.csv`。

常用参数（`-p name:=value` 传入）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| marker_size | 1.0 | H 标实际边长(米) |
| use_tf | true | 相机外参走 TF(base_link→相机)；失败时回退静态参数 |
| cam_mount | down_gz | 静态回退安装方式：down_gz(Gazebo 朝下)/down/forward/back/up |
| cam_offset_x/y/z | 0 / 0.06 / -0.135 | 相机在体系下的安装位置(米)，来自 zed_camera 模型 SDF |
| val_ratio | 0.2 | 验证集比例 |
| auto_interval | 1.0 | 自动采集间隔(秒) |

## 4. 外参校准（框和 H 标对不上时）

先飞到能看见 H 标的位置（任意高度），在 rqt_image_view 里量一下绿框中心与
真实 H 标中心的像素偏差，然后发消息修正（右/下为正）：

```bash
ros2 topic pub --once /h_data_collector/nudge geometry_msgs/msg/Point \
  "{x: 30.0, y: -10.0, z: 0.0}"
```

- x：绿框中心在真实标记右侧的像素数（框需向右移多少）
- y：绿框中心在真实标记下侧的像素数（框需向下移多少）
- z：额外偏航角（度），排查轴向/镜像问题时用 90/180/270 试

脚本按 `修正量(米) = 像素偏差 × 距离 / 焦距` 换算成机体系安装偏移，日志会打印
新的 cam_offset，换几个高度/朝向验证套准后，把它写进启动参数固化：

```bash
python3 h_data_collector.py -p cam_offset_x:=0.50 -p cam_offset_y:=-0.20
```

恢复参数默认值：`ros2 service call /h_data_collector/reset_calibration std_srvs/srv/Trigger "{}"`

## 5. 训练

```bash
source ~/target_detection/yolo_venv/bin/activate
cd ~/target_detection
yolo detect train data=h_marker.yaml model=scripts/yolov8n.pt epochs=100 imgsz=640 device=0
```

- `device=0` 显式指定 GPU；启动行应从 `CPU` 变为 `CUDA:0 (NVIDIA GeForce RTX 4070 Ti, ...)`
- 图像为 960×600，训练时自动缩放到 640 并 letterbox，无需预处理
- 358 张图在 RTX 4070 Ti 上约 5–10 分钟；CPU 约 1 小时
- 训练后验证并查看指标：

```bash
yolo detect val data=h_marker.yaml model=runs/detect/train/weights/best.pt
```

## 6. 部署检测

把训练好的权重放到 scripts/ 下（检测脚本默认从当前目录加载 `best.pt`）：

```bash
cp runs/detect/train/weights/best.pt ~/target_detection/scripts/
source /opt/ros/$ROS_DISTRO/setup.bash
source ~/target_detection/yolo_venv/bin/activate
cd ~/target_detection/scripts
python3 h_marker_detector.py

# 另开终端看结果
ros2 topic echo /h_marker/position
rqt_image_view   # 选 /h_marker/annotated 看带框画面
```

**输出 `/h_marker/position`（geometry_msgs/Point）的坐标约定**

H 标中心在**左目相机光轴坐标系**下的位置，单位米，**z 前、x 右、y 下**
（消息无 frame_id，以此约定为准）。计算方式：深度 `z = fy·marker_size/框高`
（单目尺寸法），横向按针孔模型折算。注意：

- 深度依赖"H 标平面 ∥ 成像平面"假设，**斜视（偏离正上方）时 z 系统性偏大**，
  x、y 随之偏大；正上方精准定位没问题；
- 需要机体系（FLU）坐标时：`x_body=-y_cam`，`y_body=-x_cam+0.06`，`z_body=-z_cam-0.135`；
  需要世界系时再用 `/mavros/local_position/pose` 叠乘。

## 7. GPU 训练监控与常见问题

```bash
watch -n 1 nvidia-smi   # 训练中应看到 GPU-Util 80%+、显存占用上升
```

**CUDA available 为 False**：确认 `nvidia-smi` 正常、`python -c "import torch; print(torch.__version__)"`
带 `+cu118`/`+cu121` 后缀（CPU 版 torch 无此后缀）。

**CUDA out of memory**：降低 batch：`batch=8` 或 `imgsz=512`。

## 8. 当前验证通过的环境

```text
OS:          Ubuntu 22.04
GPU:         NVIDIA GeForce RTX 4070 Ti 12GB
Driver:      535.309.01 (CUDA 12.2)
Python:      3.10.12
PyTorch:     2.7.1+cu118
numpy:       1.26.4   （必须 <2，见 1.5 节）
Ultralytics: 8.4.154
```
