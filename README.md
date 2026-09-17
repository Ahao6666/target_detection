# YOLO v8用于H标检测
## 安装 cv_bridge（大多数 ROS2 安装已自带，先确认）
```
sudo apt install ros-$ROS_DISTRO-cv-bridge ros-$ROS_DISTRO-vision-opencv
```
## 创建虚拟环境
```
python3 -m venv ~/target_detection/yolo_venv --system-site-packages
source ~/target_detection/yolo_venv/bin/activate
pip install ultralytics
```

## 启动仿真环境

```
sudo sh start_docker.h -v v1.1.5
sudo sh start_docker_ardupilot_mavros.sh -v v1.1.5
```



## 数据采集 + 自动标注

H 标固定在世界原点，结合 MAVROS 位姿和相机内外参把 H 标投影到图像自动生成
YOLO 标注（等价于 CVAT 导出 YOLO 1.1 的结构），无需人工标注。

```
source /opt/ros/$ROS_DISTRO/setup.bash
source ~/target_detection/yolo_venv/bin/activate
python3 h_data_collector.py
```

另开终端控制采集：

```
# 开始/停止自动采集（默认间隔 1s，框中心移动 <25px 的悬停重复帧自动过滤）
ros2 service call /h_data_collector/set_collecting std_srvs/srv/SetBool "{data: true}"
ros2 service call /h_data_collector/set_collecting std_srvs/srv/SetBool "{data: false}"

# 手动抓一帧
ros2 service call /h_data_collector/capture_once std_srvs/srv/Trigger "{}"

# 查看自动标注效果（绿框应套住 H 标）
rqt_image_view   # 选 /h_data_collector/preview
```

输出到 `h_dataset/images/{train,val}` + `h_dataset/labels/{train,val}`（默认按 8:2
随机划分），并自动生成 `h_marker.yaml`。每帧的位姿和框记录在 `h_dataset/log.csv`。

常用参数（`-p name:=value` 传入）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| marker_size | 1.0 | H 标实际边长(米) |
| use_tf | true | 相机外参走 TF(base_link→相机)；失败时回退静态参数 |
| cam_mount | down_gz | 静态回退安装方式：down_gz(Gazebo 朝下)/down/forward/back/up |
| cam_offset_x/y/z | 0 / 0.06 / -0.135 | 相机在体系下的安装位置(米)，默认值来自 zed_camera 模型 SDF |
| val_ratio | 0.2 | 验证集比例 |
| auto_interval | 1.0 | 自动采集间隔(秒) |

## 外参校准（框和 H 标对不上时）

先飞到能看见 H 标的位置（任意高度），在 rqt_image_view 里量一下绿框中心与
真实 H 标中心的像素偏差，然后发消息修正（右/下为正）：

```
ros2 topic pub --once /h_data_collector/nudge geometry_msgs/msg/Point \
  "{x: 30.0, y: -10.0, z: 0.0}"
```

- x：绿框中心在真实标记右侧的像素数（框需向右移多少）
- y：绿框中心在真实标记下侧的像素数（框需向下移多少）
- z：额外偏航角（度），排查轴向/镜像问题时用 90/180/270 试

脚本按 `修正量(米) = 像素偏差 × 距离 / 焦距` 换算成机体系安装偏移，日志会打印
新的 cam_offset，换几个高度/朝向验证套准后，把它写进启动参数固化：

```
python3 h_data_collector.py -p cam_offset_x:=0.50 -p cam_offset_y:=-0.20
```

恢复参数默认值：`ros2 service call /h_data_collector/reset_calibration std_srvs/srv/Trigger "{}"`

## 训练

```
yolo detect train data=h_marker.yaml model=scripts/yolov8n.pt epochs=100 imgsz=640
```

## 运行

```
source /opt/ros/$ROS_DISTRO/setup.bash
source ~/yolo_venv/bin/activate
python3 h_marker_detector.py

# 另开终端看结果
ros2 topic echo /h_marker/position
rqt_image_view   # 选 /h_marker/annotated 看带框画面
```

