import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt


def parse_voc_xml(xml_path):
    """
    解析 Pascal VOC 格式的 XML 文件，提取目标框和类别信息
    :param xml_path: VOC 标注的 XML 文件路径
    :return: (boxes, labels) -> 目标框列表和对应类别名称
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    labels = []

    for obj in root.findall("object"):
        name = obj.find("name").text  # 获取类别名称
        bndbox = obj.find("bndbox")

        # 解析目标框坐标（VOC 格式的框是 x_min, y_min, x_max, y_max）
        x_min = int(bndbox.find("xmin").text)
        y_min = int(bndbox.find("ymin").text)
        x_max = int(bndbox.find("xmax").text)
        y_max = int(bndbox.find("ymax").text)

        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(name)

    return boxes, labels


def show_pic(image_path, boxes, labels=None):
    """
    显示带有目标框的图像
    :param image_path: 图像文件路径
    :param boxes: 目标框列表，每个框格式 [x_min, y_min, x_max, y_max]
    :param labels: 类别标签列表（可选）
    """
    image = cv2.imread(image_path)  # 读取图像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV 读取的是 BGR 需要转换为 RGB

    for i, (x_min, y_min, x_max, y_max) in enumerate(boxes):
        # 绘制目标框
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        if labels:
            label = labels[i]
            cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 显示图片
    plt.imshow(image)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    image_path = "test.jpg"  # 替换为你的图像路径
    xml_path = "test.xml"  # 替换为你的 XML 文件路径

    boxes, labels = parse_voc_xml(xml_path)  # 解析 XML 获取目标框
    show_pic(image_path, boxes, labels)  # 可视化目标框

