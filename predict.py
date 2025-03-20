from ultralytics import YOLO
import os

# 加载模型
model = YOLO('D:/ultralytics-main/runs/detect/train15/weights/best.pt')

# 图片路径列表
image_paths = [
    'D:/ultralytics-main/data/images/biandian_09899.jpg',
    'D:/ultralytics-main/data/images/biandian_00453.jpg',
    'D:/ultralytics-main/data/images/biandian_01068.jpg',
    'D:/ultralytics-main/data/images/biandian_05355.jpg',
    'D:/ultralytics-main/data/images/biandian_04288.jpg',
    'D:/ultralytics-main/data/images/biandian_01830.jpg',
    'D:/ultralytics-main/data/images/biandian_04454.jpg',
    'D:/ultralytics-main/data/images/biandian_01954.jpg',
    'D:/ultralytics-main/data/images/biandian_02361.jpg',
    'D:/ultralytics-main/data/images/biandian_04436.jpg',
    'D:/ultralytics-main/data/images/biandian_00758.jpg',
    'D:/ultralytics-main/data/images/biandian_03623.jpg',
    'D:/ultralytics-main/data/images/biandian_10154.jpg',
    'D:/ultralytics-main/data/images/biandian_03589.jpg',
    'D:/ultralytics-main/data/images/biandian_00056.jpg',
    'D:/ultralytics-main/data/images/biandian_01143.jpg',
    'D:/ultralytics-main/data/images/biandian_06268.jpg',
    'D:/ultralytics-main/data/images/biandian_08358.jpg',
    'D:/ultralytics-main/data/images/biandian_10157.jpg',
    'D:/ultralytics-main/data/images/biandian_10161.jpg',
    'D:/ultralytics-main/data/images/biandian_10188.jpg',
    'D:/ultralytics-main/data/images/biandian_01393.jpg',
    'D:/ultralytics-main/data/images/biandian_09644.jpg',
    'D:/ultralytics-main/data/images/biandian_07369.jpg',
    'D:/ultralytics-main/data/images/biandian_06239.jpg',
    'D:/ultralytics-main/data/images/biandian_06585.jpg',
    'D:/ultralytics-main/data/images/biandian_10050.jpg',
    'D:/ultralytics-main/data/images/biandian_10043.jpg',
    'D:/ultralytics-main/data/images/biandian_06579.jpg',
    'D:/ultralytics-main/data/images/biandian_04744.jpg',
    'D:/ultralytics-main/data/images/biandian_06535.jpg',
    'D:/ultralytics-main/data/images/biandian_06552.jpg',
    'D:/ultralytics-main/data/images/biandian_06564.jpg',
    'D:/ultralytics-main/data/images/biandian_06570.jpg',
    'D:/ultralytics-main/data/images/biandian_10151.jpg',
    'D:/ultralytics-main/data/images/biandian_10147.jpg',
    'D:/ultralytics-main/data/images/biandian_00154.jpg',
    # 'D:/ultralytics-main/data/images/biandian_00453.jpg',
    # 'D:/ultralytics-main/data/images/biandian_01830.jpg',
    # 'D:/ultralytics-main/data/images/biandian_02027.jpg',
    # 'D:/ultralytics-main/data/images/biandian_02051.jpg',
    # 'D:/ultralytics-main/data/images/biandian_02121.jpg',
    'D:/ultralytics-main/data/images/biandian_07714.jpg',
    'D:/ultralytics-main/data/images/biandian_07847.jpg',
    'D:/ultralytics-main/data/images/biandian_09803.jpg',
    'D:/ultralytics-main/data/images/biandian_08930.jpg',
    'D:/ultralytics-main/data/images/biandian_04216.jpg',
    'D:/ultralytics-main/data/images/biandian_04271.jpg',
    'D:/ultralytics-main/data/images/biandian_09502.jpg',
    'D:/ultralytics-main/data/images/biandian_02400.jpg',
    'D:/ultralytics-main/data/images/biandian_10187.jpg',
    'D:/ultralytics-main/data/images/biandian_02581.jpg',
    'D:/ultralytics-main/data/images/biandian_07365.jpg',
    'D:/ultralytics-main/data/images/biandian_07364.jpg',
    'D:/ultralytics-main/data/images/biandian_07576.jpg',
    'D:/ultralytics-main/data/images/biandian_02451.jpg',
    'D:/ultralytics-main/data/images/biandian_10050.jpg',
    'D:/ultralytics-main/data/images/biandian_06617.jpg',
    # 添加更多图片路径
]

# 设置保存预测结果的文件夹路径
save_folder = 'D:/ultralytics-main/predictions/YOLOv8n'
# save_folder = 'D:/ultralytics-main/predictions/YOLO-CSS'

# 如果保存文件夹不存在，创建它
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 预测多张图片
for img_path in image_paths:
    # 检查图片路径是否存在
    if not os.path.exists(img_path):
        print(f"Image path does not exist: {img_path}")
        continue

    # 进行预测
    results = model(img_path)

    # 打印预测结果并保存图像
    for result in results:
        # 获取保存的文件名，基于图片文件名
        base_name = os.path.basename(img_path)
        save_path = os.path.join(save_folder, base_name)

        # 保存预测结果图片到指定文件夹
        result.save(save_path)

        # 提取预测信息
        boxes = result.boxes  # 检测到的边界框
        for box in boxes:
            cls = box.cls  # 类别
            conf = box.conf  # 置信度
            xyxy = box.xyxy  # 边界框坐标
            print(f"Class: {cls}, Confidence: {conf}, BBox: {xyxy}")


