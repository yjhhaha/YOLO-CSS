import onnxruntime as ort

# 加载模型
session = ort.InferenceSession('best.onnx')

# 打印模型的输入信息
for input_meta in session.get_inputs():
    print(f"Input name: {input_meta.name}, shape: {input_meta.shape}, type: {input_meta.type}")
