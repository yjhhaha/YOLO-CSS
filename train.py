#coding:utf-8
from ultralytics import YOLO

# 加载预训练模型
# 添加注意力机制，SEAtt_yolov8.yaml 默认使用的是n。
# SEAtt_yolov8s.yaml，则使用的是s，模型。
model = YOLO("D:/ultralytics-main/ultralytics/cfg/models/v8/yolov8n.yaml")
# .load('D:/ultralytics-main/runs/detect/train8/weights/best.pt')+-

# Use the model
if __name__ == '__main__':
    # Use the model
    results = model.train(data='D:/ultralytics-main/data_aug.yaml', epochs=299, batch=32)  # 训练模型e
    # 将模型转为onnx格式
    # success = model.export(format='onnx')
