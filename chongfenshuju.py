import os
import shutil

# 数据集路径
train_txt = 'D:/ultralytics-main/data/train.txt'
val_txt = 'D:/ultralytics-main/data/val.txt'
test_txt = 'D:/ultralytics-main/data/test.txt'

label_dir = 'D:/ultralytics-main/data/labels'  # 标签文件目录
image_dir = 'D:/ultralytics-main/data/images'  # 图片文件目录
output_dir = 'D:/ultralytics-main/data_newest'  # 新的数据集输出目录

# 需要删除的类别 ID 列表
categories_to_remove = {}  # 替换为需要删除的类别 ID

# 需要合并的类别映射表 (合并为同一类别的新 ID)
categories_to_merge = {
    '0': '17',
    '1': '17',
    '2': '17',
    '3': '18',
    '6': '19',
    '7': '19',
    '9': '20',  # 将类别 9 合并为新类别 17
    '10': '20'  # 将类别 10 合并为新类别 17
}

# 重新编号的类别映射（动态生成）
def get_new_category_map(label_dir, categories_to_remove, categories_to_merge):
    all_categories = set()
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(label_dir, label_file), 'r') as f:
                for line in f:
                    category_id = line.split()[0]
                    # 跳过删除的类别
                    if category_id in categories_to_remove:
                        continue
                    # 替换为合并后的类别 ID
                    category_id = categories_to_merge.get(category_id, category_id)
                    all_categories.add(category_id)
    all_categories = sorted(all_categories, key=int)  # 确保类别 ID 有序
    return {old_id: new_id for new_id, old_id in enumerate(all_categories)}

# 检查是否需要保留图片，并更新标签
def process_label_file(label_path, categories_to_remove, categories_to_merge, category_map):
    if not os.path.exists(label_path):
        return False, []

    new_labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.split()
            category_id = parts[0]
            # 跳过需要删除的类别
            if category_id in categories_to_remove:
                continue
            # 替换为合并后的类别 ID
            category_id = categories_to_merge.get(category_id, category_id)
            # 更新类别 ID
            new_category_id = category_map[category_id]
            new_labels.append(f"{new_category_id} " + " ".join(parts[1:]))

    return len(new_labels) > 0, new_labels

# 处理数据集
def process_dataset(txt_file, set_type, category_map):
    output_txt_file = os.path.join(output_dir, f'{set_type}.txt')
    with open(txt_file, 'r') as f, open(output_txt_file, 'w') as out_f:
        lines = f.readlines()
        for line in lines:
            image_path = line.strip()
            label_path = os.path.join(label_dir, os.path.basename(image_path).replace('.jpg', '.txt'))

            # 检查是否保留并更新标签
            keep, new_labels = process_label_file(label_path, categories_to_remove, categories_to_merge, category_map)
            if keep:
                # 复制图片和更新后的标签
                new_image_path = os.path.join(output_dir, set_type, os.path.basename(image_path))
                new_label_path = os.path.join(output_dir, set_type, os.path.basename(label_path))

                os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
                os.makedirs(os.path.dirname(new_label_path), exist_ok=True)

                shutil.copy(os.path.join(image_dir, image_path), new_image_path)
                with open(new_label_path, 'w') as label_out:
                    label_out.write("\n".join(new_labels) + "\n")

                # 更新数据集划分文件，使用绝对路径
                out_f.write(f"{new_image_path.replace(os.sep, '/')}\n")

# 获取类别映射表
category_map = get_new_category_map(label_dir, categories_to_remove, categories_to_merge)
print(f"类别映射表: {category_map}")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 处理每个数据集划分
process_dataset(train_txt, 'train', category_map)
process_dataset(val_txt, 'val', category_map)
process_dataset(test_txt, 'test', category_map)

print("数据集过滤、合并并重新编号完成，新数据集已生成。")
