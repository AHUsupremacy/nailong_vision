# nailong_vision

参考 sp_vision 工程结构实现的奶龙（Nailong）摄像头实时识别工程。

**检测方案**（默认）：YOLOX 检测模型，直接输出识别框，CPU 实时
- 模型来自开源项目 [nkxingxh/NailongDetection](https://github.com/nkxingxh/NailongDetection)（nailong_v2.6_tiny，AGPL-3.0），19.3MB，416x416 输入，CPU 单帧约 60ms
- 支持多类别（nailong/anime/human/emoji/long/other/htgt），本工程只用 nailong 类

**旧方案**（保留兼容）：ResNet50 二分类 + 网格定位，`detector: "grid"` 切换

支持两种相机（参考 sp_vision `io::Camera` 工厂，`standard_mpc` 同款）：

- **海康工业相机 MV-CS016-10UC**（`camera_name: "hikrobot"`，MVS SDK + 取图线程 + 守护重连）
- **USB 摄像头**（`camera_name: "video0"`，V4L + 取图线程 + 守护重连）

ResNet50 二分类模型（ONNX）推理，网格定位并画出识别框。

## 工程结构

```
nailong_vision/
├── CMakeLists.txt            # 主构建文件
├── configs/
│   └── nailong.yaml          # 相机 / 模型 / 网格 / 显示配置
├── io/
│   ├── camera.hpp / .cpp     # 相机工厂：按 camera_name 选 hikrobot 或 V4L USB
│   ├── usbcamera.hpp / .cpp  # USB 摄像头（参考 sp_vision io/usbcamera）
│   └── hikrobot/             # 海康工业相机（参考 sp_vision io/hikrobot）
│       ├── hikrobot.hpp / .cpp
│       ├── include/          # MVS SDK 头文件
│       └── lib/              # libMvCameraControl.so（amd64 / arm64）
├── tasks/nailong/
│   ├── yolox_detector.hpp/.cpp # YOLOX 检测器（默认，直接输出识别框）
│   └── detector.hpp / .cpp     # ResNet50(onnx) + 网格定位（旧方案）
├── tools/
│   ├── thread_safe_queue.hpp # 线程安全队列（同 sp_vision）
│   ├── exiter.hpp / .cpp     # Ctrl-C 退出（同 sp_vision）
│   └── export_onnx.py        # nailong.pth -> nailong.onnx 转换脚本
├── model/
│   ├── nailong_v2.6_tiny.onnx  # YOLOX 检测模型（默认，19.3 MB）
│   └── nailong.onnx            # ResNet50 分类模型（旧方案，89.6 MB）
└── src/
    └── nailong.cpp           # 主程序
```

## 依赖（虚拟机 / NUC, Linux）

- OpenCV >= 4.5（需 dnn 模块）
- yaml-cpp
- libusb-1.0（海康相机 USB reset 守护重连用）
- CMake >= 3.16, g++ (C++17)

```bash
sudo apt install libopencv-dev libyaml-cpp-dev libusb-1.0-0-dev cmake g++
```

## 编译

```bash
cd nailong_vision
cmake -B build
cmake --build build
```

## 运行

```bash
# 默认读取 configs/nailong.yaml
./build/nailong

# 指定配置
./build/nailong configs/nailong.yaml
```

- 弹出窗口实时显示，命中奶龙的格子画红框 + 置信度，按 `q` 或 Ctrl-C 退出。
- NUC / 无显示器环境：把 `configs/nailong.yaml` 里 `show` 改为 `false`，只在终端打印状态变化。

## 海康 MV-CS016-10UC 用法

1. NUC 上装海康 MVS 驱动（或直接用本工程自带的 `io/hikrobot/lib`），插上 USB3 线。
2. `configs/nailong.yaml` 里：
   ```yaml
   camera_name: "hikrobot"
   exposure_ms: 2      # 曝光（ms）
   gain: 16            # 增益
   vid_pid: "2bdf:0001"  # 海康 USB 相机的 vid:pid
   ```
3. `lsusb` 确认有 `2bdf:0001` 设备；如果枚举不到，检查 USB3 口和 MVS 驱动。
4. 断线自动重连：守护线程检测到采图中断会自动 USB reset + 重开相机（同 sp_vision）。

## 配置说明（configs/nailong.yaml）

| 参数 | 说明 |
|---|---|
| `camera_name` | `"hikrobot"` = 海康工业相机；`"video0"` = USB 相机（V4L，对应 `/dev/video0`） |
| `detector` | `"yolox"` = YOLOX 检测（默认推荐）；`"grid"` = ResNet50 网格分类 |
| `yolox_model_path` | YOLOX onnx 模型路径 |
| `yolox_confidence` | YOLOX 置信度阈值（默认 0.3） |
| `yolox_class_id` | 检测类别 id（1 = nailong） |
| `exposure_ms` | 海康相机曝光（ms） |
| `gain` | 海康相机增益 |
| `vid_pid` | 海康相机 USB vid:pid（默认 `2bdf:0001`） |
| `image_width/height` | USB 相机采集分辨率 |
| `usb_frame_rate` | USB 相机采集帧率 |
| `usb_exposure` | USB 相机曝光 |
| `model_path` | onnx 模型路径 |
| `min_confidence` | 置信度阈值（0~1） |
| `grid_rows/grid_cols` | 网格行/列数。`1x1` = 整图一个框（最快）；`2x2`/`3x3` = 局部框（更准更慢） |
| `infer_every` | 每 N 帧推理一次（CPU 建议 2~5） |
| `show` | `true` 显示窗口；`false` 无窗口（headless） |

## 模型来源

- 检测模型: [nkxingxh/NailongDetection](https://github.com/nkxingxh/NailongDetection) v2.6 tiny（YOLOX，AGPL-3.0）
- 旧分类模型: [spawner1145/NailongRecognize](https://github.com/spawner1145/NailongRecognize)（ResNet50 二分类，class 1 = 奶龙）
- 已用 `tools/export_onnx.py` 从 `nailong.pth` 导出为 `model/nailong.onnx`
- 如需重新导出: `python tools/export_onnx.py --pth <nailong.pth路径> --out model/nailong.onnx`（需 torch + torchvision + onnx）

## 常见问题

- **海康相机枚举不到**：`lsusb | grep 2bdf` 确认设备，检查 USB3 口；虚拟机里需把 USB 设备直通给虚拟机。
- **USB 相机打不开**：检查设备名 `ls /dev/video*`，或权限（把用户加进 `video` 组：`sudo usermod -aG video $USER`）。
- **帧率低**：CPU 推理 ResNet50 较重，降低 `grid_rows/grid_cols`（改 `1x1`）或调大 `infer_every`、降低 `image_width/height`。
- **中文路径**：Windows 下 cv2.dnn 读不了中文路径，Linux 无此问题；本项目主要在 Linux 虚拟机上运行。
