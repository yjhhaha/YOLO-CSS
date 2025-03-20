# -*- coding=utf-8 -*-
import time
import random
import copy
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from lxml import etree, objectify
import xml.etree.ElementTree as ET
import argparse
import shutil


# 显示图片（调试时可用）
def show_pic(img, bboxes=None):
    '''
    输入:
        img:图像array
        bboxes:图像所有bounding box列表, 格式为[[x_min, y_min, x_max, y_max], ...]
    '''
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 3)
    cv2.namedWindow('pic', 0)
    cv2.moveWindow('pic', 0, 0)
    cv2.resizeWindow('pic', 1200, 800)
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 图像均为cv2读取
class DataAugmentForObjectDetection():
    def __init__(self, rotation_rate=0.5, max_rotation_angle=5,
                 crop_rate=0.5, shift_rate=0.5, change_light_rate=0.5,
                 add_noise_rate=0.5, flip_rate=0.5,
                 cutout_rate=0.5, cut_out_length=40, cut_out_holes=1, cut_out_threshold=0.5,
                 is_addNoise=True, is_changeLight=True, is_cutout=True, is_rotate_img_bbox=True,
                 is_crop_img_bboxes=True, is_shift_pic_bboxes=True, is_filp_pic_bboxes=True):

        # 配置各个操作的属性
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.flip_rate = flip_rate
        self.cutout_rate = cutout_rate

        self.cut_out_length = cut_out_length
        self.cut_out_holes = cut_out_holes
        self.cut_out_threshold = cut_out_threshold

        # 是否使用某种增强方式
        self.is_addNoise = is_addNoise
        self.is_changeLight = is_changeLight
        self.is_cutout = is_cutout
        self.is_rotate_img_bbox = is_rotate_img_bbox
        self.is_crop_img_bboxes = is_crop_img_bboxes
        self.is_shift_pic_bboxes = is_shift_pic_bboxes
        self.is_filp_pic_bboxes = is_filp_pic_bboxes

    # 加噪声
    def _addNoise(self, img):
        # 输出浮点型，乘以255后转换为uint8
        # noisy = random_noise(img, mode='gaussian', seed=int(time.time()), clip=True) * 255
        noisy = random_noise(img, mode='gaussian', clip=True) * 255
        return noisy.astype(np.uint8)

    # 调整亮度
    def _changeLight(self, img):
        alpha = random.uniform(0.35, 1)
        blank = np.zeros(img.shape, img.dtype)
        return cv2.addWeighted(img, alpha, blank, 1 - alpha, 0)

    # cutout操作
    def _cutout(self, img, bboxes, length=100, n_holes=1, threshold=0.5):
        def cal_iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            if xB <= xA or yB <= yA:
                return 0.0
            interArea = (xB - xA + 1) * (yB - yA + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            return interArea / float(boxBArea)

        h, w = img.shape[:2]
        mask = np.ones(img.shape, np.float32)
        for n in range(n_holes):
            valid = False
            while not valid:
                y = np.random.randint(h)
                x = np.random.randint(w)
                y1 = np.clip(y - length // 2, 0, h)
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)
                valid = True
                for box in bboxes:
                    if cal_iou([x1, y1, x2, y2], box) > threshold:
                        valid = False
                        break
            mask[y1:y2, x1:x2, :] = 0.
        img = img * mask
        return img.astype(np.uint8)

    # 旋转操作，同时更新bbox
    def _rotate_img_bbox(self, img, bboxes, angle=5, scale=1.0):
        h, w = img.shape[:2]
        rangle = np.deg2rad(angle)
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        rot_bboxes = []
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox
            # 使用四个角点进行仿射变换
            points = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
            ones = np.ones((points.shape[0], 1))
            points_ = np.hstack([points, ones])
            points_rot = np.dot(rot_mat, points_.T).T
            x_coords = points_rot[:, 0]
            y_coords = points_rot[:, 1]
            rot_bboxes.append([int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))])
        return rot_img, rot_bboxes

    # 裁剪操作（保留所有目标）
    def _crop_img_bboxes(self, img, bboxes):
        h, w = img.shape[:2]
        x_min = min([bbox[0] for bbox in bboxes])
        y_min = min([bbox[1] for bbox in bboxes])
        x_max = max([bbox[2] for bbox in bboxes])
        y_max = max([bbox[3] for bbox in bboxes])
        d_to_left = x_min
        d_to_right = w - x_max
        d_to_top = y_min
        d_to_bottom = h - y_max
        crop_x_min = int(max(0, x_min - random.uniform(0, d_to_left)))
        crop_y_min = int(max(0, y_min - random.uniform(0, d_to_top)))
        crop_x_max = int(min(w, x_max + random.uniform(0, d_to_right)))
        crop_y_max = int(min(h, y_max + random.uniform(0, d_to_bottom)))
        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        crop_bboxes = []
        for bbox in bboxes:
            crop_bboxes.append([bbox[0] - crop_x_min, bbox[1] - crop_y_min, bbox[2] - crop_x_min, bbox[3] - crop_y_min])
        return crop_img, crop_bboxes

    # 平移操作
    def _shift_pic_bboxes(self, img, bboxes):
        h, w = img.shape[:2]
        x_min = min([bbox[0] for bbox in bboxes])
        y_min = min([bbox[1] for bbox in bboxes])
        x_max = max([bbox[2] for bbox in bboxes])
        y_max = max([bbox[3] for bbox in bboxes])
        d_to_left = x_min
        d_to_right = w - x_max
        d_to_top = y_min
        d_to_bottom = h - y_max
        x_shift = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
        y_shift = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)
        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        shift_img = cv2.warpAffine(img, M, (w, h))
        shift_bboxes = []
        for bbox in bboxes:
            shift_bboxes.append([bbox[0] + x_shift, bbox[1] + y_shift, bbox[2] + x_shift, bbox[3] + y_shift])
        return shift_img, shift_bboxes

    # 修正后的翻转操作（包括水平、垂直、对角翻转）
    def _filp_pic_bboxes(self, img, bboxes):
        h, w = img.shape[:2]
        sed = random.random()
        if sed < 0.33:
            # 垂直翻转（上下翻转）
            flip_img = cv2.flip(img, 0)
            flip_bboxes = []
            for bbox in bboxes:
                x_min, y_min, x_max, y_max = bbox
                new_y_min = h - y_max
                new_y_max = h - y_min
                flip_bboxes.append([x_min, new_y_min, x_max, new_y_max])
        elif sed < 0.66:
            # 水平翻转（左右翻转）
            flip_img = cv2.flip(img, 1)
            flip_bboxes = []
            for bbox in bboxes:
                x_min, y_min, x_max, y_max = bbox
                new_x_min = w - x_max
                new_x_max = w - x_min
                flip_bboxes.append([new_x_min, y_min, new_x_max, y_max])
        else:
            # 对角翻转（同时上下、左右翻转）
            flip_img = cv2.flip(img, -1)
            flip_bboxes = []
            for bbox in bboxes:
                x_min, y_min, x_max, y_max = bbox
                new_x_min = w - x_max
                new_y_min = h - y_max
                new_x_max = w - x_min
                new_y_max = h - y_min
                flip_bboxes.append([new_x_min, new_y_min, new_x_max, new_y_max])
        return flip_img, flip_bboxes

    # 综合数据增强
    def dataAugment(self, img, bboxes):
        change_num = 0  # 记录进行的增强操作数
        # 可按需求组合多种增强操作，这里每次随机触发部分操作
        if self.is_rotate_img_bbox and random.random() < self.rotation_rate:
            change_num += 1
            angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
            scale = random.uniform(0.7, 0.8)
            img, bboxes = self._rotate_img_bbox(img, bboxes, angle, scale)
        if self.is_shift_pic_bboxes and random.random() < self.shift_rate:
            change_num += 1
            img, bboxes = self._shift_pic_bboxes(img, bboxes)
        if self.is_changeLight and random.random() < self.change_light_rate:
            change_num += 1
            img = self._changeLight(img)
        if self.is_addNoise and random.random() < self.add_noise_rate:
            change_num += 1
            img = self._addNoise(img)
        if self.is_cutout and random.random() < self.cutout_rate:
            change_num += 1
            img = self._cutout(img, bboxes, length=self.cut_out_length, n_holes=self.cut_out_holes,
                               threshold=self.cut_out_threshold)
        if self.is_filp_pic_bboxes and random.random() < self.flip_rate:
            change_num += 1
            img, bboxes = self._filp_pic_bboxes(img, bboxes)
        # 如果没有任何操作发生，可考虑至少返回原图（或再次递归调用，避免无变化）
        if change_num == 0:
            # 这里直接返回原图，也可以递归调用dataAugment重新增强
            return img, bboxes
        return img, bboxes


# XML解析与保存工具
class ToolHelper():
    def parse_xml(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        objs = root.findall('object')
        coords = []
        for obj in objs:
            name = obj.find('name').text
            box = obj.find('bndbox')
            x_min = int(box[0].text)
            y_min = int(box[1].text)
            x_max = int(box[2].text)
            y_max = int(box[3].text)
            coords.append([x_min, y_min, x_max, y_max, name])
        return coords

    def save_img(self, file_name, save_folder, img):
        cv2.imwrite(os.path.join(save_folder, file_name), img)

    def save_xml(self, file_name, save_folder, img_info, height, width, channel, bboxs_info):
        folder_name, img_name = img_info
        E = objectify.ElementMaker(annotate=False)
        anno_tree = E.annotation(
            E.folder(folder_name),
            E.filename(img_name),
            E.path(os.path.join(folder_name, img_name)),
            E.source(
                E.database('Unknown'),
            ),
            E.size(
                E.width(width),
                E.height(height),
                E.depth(channel)
            ),
            E.segmented(0),
        )
        labels, bboxs = bboxs_info
        for label, box in zip(labels, bboxs):
            anno_tree.append(
                E.object(
                    E.name(label),
                    E.pose('Unspecified'),
                    E.truncated('0'),
                    E.difficult('0'),
                    E.bndbox(
                        E.xmin(box[0]),
                        E.ymin(box[1]),
                        E.xmax(box[2]),
                        E.ymax(box[3])
                    )
                ))
        etree.ElementTree(anno_tree).write(os.path.join(save_folder, file_name), pretty_print=True)


if __name__ == '__main__':
    # 每张图片的增强倍数，取值范围0.5～1.0
    # 例如，设置为0.5时，约50%的图片生成1张增强图，总量约为1.5倍原始；设置为1时，每张都生成1张，约2倍原始
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_img_path', type=str,
                        default='F:/Dataset/train_images')
    parser.add_argument('--source_xml_path', type=str,
                        default='F:/Dataset/train_xml')
    parser.add_argument('--save_img_path', type=str,
                        default='F:/Dataset/train_img_aug')
    parser.add_argument('--save_xml_path', type=str,
                        default='F:/Dataset/train_xml_aug')
    parser.add_argument('--aug_multiplier', type=float, default=1,
                        help='生成增强图的概率（0.5~1），最终数据集大小为原始*(1+aug_multiplier)')
    args = parser.parse_args()

    source_img_path = args.source_img_path
    source_xml_path = args.source_xml_path
    save_img_path = args.save_img_path
    save_xml_path = args.save_xml_path
    aug_multiplier = args.aug_multiplier

    # 如果保存文件夹不存在则创建
    os.makedirs(save_img_path, exist_ok=True)
    os.makedirs(save_xml_path, exist_ok=True)

    dataAug = DataAugmentForObjectDetection()
    toolhelper = ToolHelper()

    # 遍历图片文件夹
    for parent, _, files in os.walk(source_img_path):
        files.sort()
        for file in files:
            # 仅处理常见后缀文件
            if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            pic_path = os.path.join(parent, file)
            # xml文件名假定与图片同名
            xml_path = os.path.join(source_xml_path, os.path.splitext(file)[0] + '.xml')
            if not os.path.exists(xml_path):
                print("xml文件不存在：", xml_path)
                continue

            values = toolhelper.parse_xml(xml_path)  # [[x_min, y_min, x_max, y_max, name], ...]
            coords = [v[:4] for v in values]
            labels = [v[-1] for v in values]

            # 复制原图及对应XML到目标目录
            shutil.copy(pic_path, os.path.join(save_img_path, file))
            shutil.copy(xml_path, os.path.join(save_xml_path, os.path.basename(xml_path)))

            # 根据aug_multiplier决定是否生成增强图（每张最多1张）
            if random.random() < aug_multiplier:
                img = cv2.imread(pic_path)
                auged_img, auged_bboxes = dataAug.dataAugment(img, coords)
                auged_bboxes_int = np.array(auged_bboxes).astype(np.int32)
                height, width, channel = auged_img.shape
                aug_img_name = '{}_aug{}'.format(os.path.splitext(file)[0], '')
                ext = os.path.splitext(file)[1]
                aug_img_name_full = aug_img_name + ext
                toolhelper.save_img(aug_img_name_full, save_img_path, auged_img)
                toolhelper.save_xml(aug_img_name + '.xml', save_xml_path, (save_img_path, aug_img_name_full),
                                    height, width, channel, (labels, auged_bboxes_int))
                print("生成增强图：", aug_img_name_full)
