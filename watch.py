from ultralytics import YOLO
# 加载训练好的模型或者网络结构配置文件
model = YOLO('D:/ultralytics-main/runs/detect/train15/weights/best.pt')
# model = YOLO('ultralytics/cfg/models/v8/yolov8n.yaml')
# 打印模型参数信息
print(model.info())
