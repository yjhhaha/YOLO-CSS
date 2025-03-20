import cv2
import numpy as np
import onnxruntime as ort

# 加载 ONNX 模型，使用可用的推理提供者
session = ort.InferenceSession('best.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# 定义类别标签
class_labels = ['bj_bpmh', 'bj_bpps', 'bj_wkps', 'bjdsyc', 'jyz_pl', 'sly_dmyw', 'hxq_gjbs', 'hxq_gjtps',
                'xmbhyc', 'yw_gkxfw', 'yw_nc', 'gbps', 'wcaqm', 'wcgz', 'xy', 'ywzt_yfyc', 'kgg_ybh']


# 预处理图像
def preprocess(image):
    image = cv2.resize(image, (640, 640))  # YOLOv8n 模型输入尺寸
    image = image[:, :, ::-1].transpose(2, 0, 1)  # 转换为 CHW 格式
    image = np.expand_dims(image, axis=0).astype(np.float32) / 255.0  # 归一化
    return image


# 非极大值抑制（NMS）函数
def non_max_suppression(boxes, scores, iou_threshold=0.4):
    """
    简单实现 NMS (非极大值抑制)
    """
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=iou_threshold)
    return indices


# 后处理函数
def postprocess(outputs, frame):
    # 假设 'outputs' 是包含检测结果的列表，提取预测框和置信度信息
    boxes, scores, class_ids = [], [], []

    # 假设输出包含多个张量：边界框、置信度、类别索引
    predictions = outputs[0]  # 通常，第一个输出是预测的边界框信息
    batch_size, grid_size, grid_size, num_anchors, _ = predictions.shape
    for i in range(batch_size):
        for y in range(grid_size):
            for x in range(grid_size):
                for anchor in range(num_anchors):
                    pred = predictions[i, y, x, anchor]  # 获取每个预测框

                    # 提取坐标和置信度
                    x_center, y_center, w, h, confidence = pred[:5]
                    if confidence > 0.5:  # 置信度筛选

                        # 计算边框的绝对坐标
                        x_min = int((x_center - w / 2) * frame.shape[1])
                        y_min = int((y_center - h / 2) * frame.shape[0])
                        x_max = int((x_center + w / 2) * frame.shape[1])
                        y_max = int((y_center + h / 2) * frame.shape[0])

                        boxes.append([x_min, y_min, x_max, y_max])
                        scores.append(confidence)
                        # 提取最大类别
                        class_id = np.argmax(pred[5:])
                        class_ids.append(class_id)

    # 应用非极大值抑制
    indices = non_max_suppression(boxes, scores, iou_threshold=0.4)

    # 绘制边界框和标签
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x_min, y_min, x_max, y_max = box
            label = class_labels[class_ids[i]]  # 获取类别标签
            score = scores[i]

            # 绘制边框
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # 绘制类别标签和置信度
            cv2.putText(frame, f'{label}: {score:.2f}', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示处理后的帧
    cv2.imshow("YOLOv8 Detection", frame)
    cv2.waitKey(1)


# 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 预处理帧
    input_image = preprocess(frame)

    # 获取模型输入输出名称 (仅需执行一次，确认输入输出名)
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    print(f"Input name: {input_name}, Output names: {output_names}")

    # 推理
    outputs = session.run(None, {input_name: input_image})

    # 输出调试信息，帮助确认输出内容是否符合预期
    print("Outputs:", outputs)

    # 后处理并标注检测结果
    postprocess(outputs, frame)

    # 显示结果
    cv2.imshow('YOLOv8 Detection', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
