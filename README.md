# nailong_vision

参考 sp_vision 工程结构实现的奶龙（Nailong）摄像头实时识别工程。
USB 摄像头读图（V4L + 取图线程 + 守护重连 + 线程安全队列），ResNet50 二分类模型（ONNX）推理，网格定位并画出识别框。

## 工程结构

```
nailong_vision/
├── CMakeLists.txt            # 主构建文件
├── configs/
│   └── nailong.yaml          # 相机 / 模型 / 网格 / 显示配置
├── io/
│   ├── camera.hpp / .cpp     # USB 摄像头（参考 sp_vision io/usbcamera）
├── tasks/nailong/
│   ├── detector.hpp / .cpp   # ResNet50(onnx) + 网格定位，输出识别框
├── tools/
│   ├── thread_safe_queue.hpp # 线程安全队列（同 sp_vision）
│   ├── exiter.hpp / .cpp     # Ctrl-C 退出（同 sp_vision）
│   └── export_onnx.py        # nailong.pth -> nailong.onnx 转换脚本
├── model/
│   └── nailong.onnx          # 模型权重（已导出，89.6 MB）
└── src/
    └── nailong.cpp           # 主程序
```

## 依赖（虚拟机 / NUC, Linux）

- OpenCV >= 4.5（需 dnn 模块）
- yaml-cpp
- CMake >= 3.16, g++ (C++17)

```bash
sudo apt install libopencv-dev libyaml-cpp-dev cmake g++
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

## 配置说明（configs/nailong.yaml）

| 参数 | 说明 |
|---|---|
| `camera_name` | 摄像头设备名，对应 `/dev/<camera_name>`，默认 `video0` |
| `image_width/height` | 采集分辨率 |
| `usb_frame_rate` | 采集帧率 |
| `usb_exposure` | 曝光 |
| `model_path` | onnx 模型路径 |
| `min_confidence` | 置信度阈值（0~1） |
| `grid_rows/grid_cols` | 网格行/列数。`1x1` = 整图一个框（最快）；`2x2`/`3x3` = 局部框（更准更慢） |
| `infer_every` | 每 N 帧推理一次（CPU 建议 2~5） |
| `show` | `true` 显示窗口；`false` 无窗口（headless） |

## 模型来源

- 原始模型: [spawner1145/NailongRecognize](https://github.com/spawner1145/NailongRecognize)（ResNet50 二分类，class 1 = 奶龙）
- 已用 `tools/export_onnx.py` 从 `nailong.pth` 导出为 `model/nailong.onnx`
- 如需重新导出: `python tools/export_onnx.py --pth <nailong.pth路径> --out model/nailong.onnx`（需 torch + torchvision + onnx）

## 常见问题

- **打不开摄像头**：检查设备名 `ls /dev/video*`，或权限（把用户加进 `video` 组：`sudo usermod -aG video $USER`）。
- **帧率低**：CPU 推理 ResNet50 较重，降低 `grid_rows/grid_cols`（改 `1x1`）或调大 `infer_every`、降低 `image_width/height`。
- **中文路径**：Windows 下 cv2.dnn 读不了中文路径，Linux 无此问题；本项目主要在 Linux 虚拟机上运行。
