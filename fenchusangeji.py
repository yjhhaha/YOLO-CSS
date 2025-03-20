import os
import shutil
from collections import defaultdict

# 设置路径
train_txt = 'D:/ultralytics-main/data/train.txt'
val_txt = 'D:/ultralytics-main/data/val.txt'
test_txt = 'D:/ultralytics-main/data/test.txt'

image_dir = 'D:/ultralytics-main/data/images'  # 图片目录
label_dir = 'D:/ultralytics-main/data/labels'  # 标签文件目录
output_dir = 'sorted_images'  # 分类输出目录


# 读取数据集的类别标签
def get_image_category(label_path):
    categories = set()
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            category_id = line.split()[0]  # 获取类别ID
            categories.add(category_id)
    return categories


# 处理每个数据集
def process_images(txt_file, set_type):
    with open(txt_file, 'r') as f:
        lines = f.readlines()

        for line in lines:
            image_path = line.strip()
            label_path = os.path.join(label_dir, os.path.basename(image_path).replace('.jpg', '.txt'))

            if not os.path.exists(label_path):
                print(f"Label file not found: {label_path}")
                continue

            # 获取图片对应的类别
            categories = get_image_category(label_path)

            # 将图片按类别保存
            for category in categories:
                set_dir = os.path.join(output_dir, set_type, f"类别_{category}")
                os.makedirs(set_dir, exist_ok=True)

                src_path = os.path.join(image_dir, os.path.basename(image_path))
                dst_path = os.path.join(set_dir, os.path.basename(image_path))

                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_path)
                else:
                    print(f"Image not found: {src_path}")


# 对训练集、验证集和测试集进行分类整理
process_images(train_txt, 'train')
process_images(val_txt, 'val')
process_images(test_txt, 'test')

print("图片整理完成。")
