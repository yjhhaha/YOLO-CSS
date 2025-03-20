from ultralytics import RTDETR

if __name__ == '__main__':
    # 加载COCO预训练的RT-DETR-l模型
    model = RTDETR("D:/ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml")

    # 显示模型信息（可选）
    model.info()

    # 在COCO8示例数据集上训练模型，训练100个周期
    results = model.train(data="D:/ultralytics-main/biandian.yaml", epochs=299, imgsz=640, batch=16)

    # 使用RT-DETR-l模型对 'bus.jpg' 图像进行推理
    # results = model("path/to/bus.jpg")
