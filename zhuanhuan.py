import torch
from ultralytics import YOLO  # 使用 ultralytics 的 YOLO 模型

# 加载 YOLOv8 模型
model = YOLO('D:/ultralytics-main/runs/detect/train92/weights/best.pt')

# 设置模型为推理模式
model.model.eval()

# 创建虚拟输入
dummy_input = torch.randn(1, 3, 640, 640)  # 输入尺寸

# 导出为 ONNX
torch.onnx.export(model.model, dummy_input, "best.onnx", verbose=True, opset_version=11)
