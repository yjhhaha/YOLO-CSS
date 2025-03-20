from utils.utils import preproc, vis
from utils.utils import BaseEngine
import numpy as np
import cv2
import time
import os
import argparse

class Predictor(BaseEngine):
    def __init__(self, engine_path):
        super(Predictor, self).__init__(engine_path)
        self.n_classes = 80  # your model classes

    def detect_video(self, video_path, conf=0.1, end2end=False):
        # 判断 video_path 是否为摄像头设备索引或文件路径
        if video_path.isdigit():  # 使用摄像头
            video_path = int(video_path)
            # 使用设备路径直接打开摄像头
            cap = cv2.VideoCapture('/dev/video0')
        else:  # 使用视频文件
            cap = cv2.VideoCapture(video_path)

        # 检查是否成功打开
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 推理处理
            result_frame = self.inference(frame, conf=conf, end2end=end2end)

            # 显示推理结果
            cv2.imshow("Inference", result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="TRT engine Path")
    parser.add_argument("-i", "--image", help="image path")
    parser.add_argument("-o", "--output", help="image output path")
    parser.add_argument("-v", "--video", help="video path or camera index", default="0")
    parser.add_argument("--end2end", default=False, action="store_true",
                        help="use end-to-end engine")

    args = parser.parse_args()
    print(args)

    pred = Predictor(engine_path=args.engine)
    pred.get_fps()
    img_path = args.image
    video = args.video

    if img_path:
        # 进行图像推理
        origin_img = pred.inference(img_path, conf=0.1, end2end=args.end2end)
        cv2.imwrite("%s" % args.output, origin_img)
    elif video:
        # 进行视频或摄像头推理，video=0 调用默认摄像头
        pred.detect_video(video, conf=0.1, end2end=args.end2end)
