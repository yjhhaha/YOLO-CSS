import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


# 定义图像预处理函数
def preprocess_image(image, input_shape):
    h, w, _ = image.shape
    target_h, target_w = input_shape[2], input_shape[3]

    # 计算缩放比例
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # 调整图像大小，保持长宽比
    resized = cv2.resize(image, (new_w, new_h))

    # 创建填充背景（例如黑色填充）
    padded_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # 将调整后的图像放到中心
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    padded_image[top:top + new_h, left:left + new_w] = resized

    # 归一化图像像素值为 [0, 1]
    normalized = padded_image / 255.0
    # 转换为 NCHW 格式（批量大小，通道数，高度，宽度）
    transposed = np.transpose(normalized, (2, 0, 1)).astype(np.float32)
    # 增加批量维度
    batched = np.expand_dims(transposed, axis=0)
    # 确保图像数据是连续的
    return np.ascontiguousarray(batched)


# 定义 TensorRT 推理函数
def do_inference(engine, input_image):
    # 创建推理上下文
    context = engine.create_execution_context()

    # 获取输入和输出绑定的形状
    input_shape = engine.get_binding_shape(0)  # 获取输入绑定的形状
    output_shape = engine.get_binding_shape(1)  # 获取输出绑定的形状

    # 显式使用批量大小 1
    input_size = trt.volume(input_shape) * 1  # 使用 1 批量大小
    output_size = trt.volume(output_shape) * 1  # 使用 1 批量大小

    # 分配 GPU 内存
    dtype = trt.nptype(engine.get_binding_dtype(0))  # 获取输入的数据类型
    d_input = cuda.mem_alloc(input_image.nbytes)
    d_output = cuda.mem_alloc(output_size * np.dtype(dtype).itemsize)
    bindings = [int(d_input), int(d_output)]

    # 使用 CUDA 流进行推理
    stream = cuda.Stream()

    # 将数据复制到 GPU
    cuda.memcpy_htod_async(d_input, input_image, stream)

    # 执行推理
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # 从 GPU 复制输出结果
    output = np.empty([1, *output_shape], dtype=np.float32)  # batch_size = 1
    cuda.memcpy_dtoh_async(output, d_output, stream)

    # 同步流
    stream.synchronize()

    # 释放 GPU 内存
    cuda.free(d_input)
    cuda.free(d_output)

    return output


# 加载 TensorRT 引擎
def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine_data = f.read()
        if not engine_data:
            raise ValueError(f"无法加载引擎文件: {engine_file_path}")
        return runtime.deserialize_cuda_engine(engine_data)


# 主函数
if __name__ == "__main__":
    # 初始化摄像头
    cap = cv2.VideoCapture(0)

    # 加载 TensorRT 引擎
    engine = load_engine("best.engine")

    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头输入")
            break

        # 预处理图像
        input_image = preprocess_image(frame, (1, 3, 640, 640))
        print("输入图像的形状:", input_image.shape)

        # 执行推理
        try:
            output = do_inference(engine, input_image)
            print("推理结果：", output)
        except Exception as e:
            print("推理时出现错误:", str(e))

        # 显示原始图像
        cv2.imshow("Camera", frame)

        # 按下 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头
    cap.release()
    cv2.destroyAllWindows()
