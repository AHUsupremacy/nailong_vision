# 将 nailong.pth (ResNet50 二分类 state_dict) 导出为 ONNX
# 用法: python tools/export_onnx.py --pth <路径>/nailong.pth --out model/nailong.onnx
import argparse
import os

import torch
import torch.nn as nn
import torchvision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pth",
        default=r"E:\RM\视觉\NailongRecognize\nailong.pth",
        help="nailong.pth 路径",
    )
    parser.add_argument("--out", default="model/nailong.onnx", help="输出 onnx 路径")
    args = parser.parse_args()

    model = torchvision.models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    state_dict = torch.load(args.pth, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
    )
    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"exported: {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()


