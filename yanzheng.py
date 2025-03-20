import torch

# 加载模型
model_dict = torch.load('runs/detect/train92/weights/best.pt', map_location='cpu')

# 如果模型是保存在 'model' 键中，需要提取出来
if 'model' in model_dict:
    model = model_dict['model']
else:
    model = model_dict  # 直接加载的模型

# 打印模型结构
print(model)


