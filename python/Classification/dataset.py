# --
import re
from configparser import ConfigParser
import cv2 as cv
import numpy as np
import os, sys, string, random
from pathlib import Path
from PIL import Image
from multiprocessing.pool import Pool as Pool
from albumentations import DualTransform
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co


def process_image(img, min_side=416):
    '''

    :param img:
    :param min_side:
    :return:
    '''
    size = img.shape
    h, w = size[0], size[1]
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w / scale), int(h / scale)
    resize_img = cv.resize(img, (new_w, new_h))
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2, (min_side - new_h) / 2, (min_side - new_w) / 2 + 1, (
                min_side - new_w) / 2
    elif new_h % 2 != 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2 + 1, (min_side - new_h) / 2, (min_side - new_w) / 2, (
                min_side - new_w) / 2
    elif new_h % 2 == 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2, (min_side - new_h) / 2, (min_side - new_w) / 2, (
                min_side - new_w) / 2
    else:
        top, bottom, left, right = (min_side - new_h) / 2 + 1, (min_side - new_h) / 2, (min_side - new_w) / 2 + 1, (
                min_side - new_w) / 2
    pad_img = cv.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right), cv.BORDER_CONSTANT,
                                value=[0, 0, 0])
    return pad_img


def cutmix(img_a, img_b, n=2):
    '''

    :param img_a:
    :param img_b:
    :param n:
    :return:
    '''
    img_a = process_image(img_a)
    img_b = process_image(img_b)
    h, w, _ = img_a.shape
    for _ in range(n):
        cut_w = int(np.random.uniform(0.3 * w, 0.5 * w, 1)[0])
        cut_h = int(np.random.uniform(0.3 * h, 0.5 * h, 1)[0])
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_w // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        img_a[bbx1:bbx2, bby1:bby2, :] = img_b[bbx1:bbx2, bby1:bby2, :]
    return img_a


def mosaic(imgs: list, size=416):
    '''

    :param imgs:
    :param size:
    :return:
    '''
    # imgs = [process_image(img, size) for img in imgs]
    imgss = []
    for img in imgs:
        (h, w, _) = img.shape
        img = cv.resize(img, (0, 0), fx=1.4, fy=1.4, interpolation=cv.INTER_NEAREST)
        cropped = img[int(h * 0.2):int(h * 1.2), int(w * 0.2):int(w * 1.2)]
        img = cv.resize(cropped, (size, size), interpolation=cv.INTER_AREA)
        imgss.append(img)

    image = cv.vconcat([cv.hconcat([imgss[0], imgss[1]]), cv.hconcat([imgss[2], imgss[3]])])
    return image


def samplepairing(imga, imgb):
    '''


    :param imga:
    :param imgb:
    :return:
    '''
    imgss = []
    for img in [imga, imgb]:
        (h, w, _) = img.shape
        img = cv.resize(img, (0, 0), fx=1.2, fy=1.2, interpolation=cv.INTER_NEAREST)
        cropped = img[int(h * 0.1):int(h * 1.1), int(w * 0.1):int(w * 1.1)]
        img = cv.resize(cropped, (416, 416), interpolation=cv.INTER_AREA)
        imgss.append(img)
    imga, imgb = imgss

    img = (imga + imgb) // 2
    return img


def cuthalf(img_a, img_b):
    """

    :param img_a:
    :param img_b:
    :return:
    """
    imgss = []
    for img in [img_a, img_b]:
        (h, w, _) = img.shape
        img = cv.resize(img, (0, 0), fx=1.1, fy=1.1, interpolation=cv.INTER_NEAREST)
        cropped = img[int(h * 0.05):int(h * 1.05), int(w * 0.05):int(w * 1.05)]
        img = cv.resize(cropped, (416, 416), interpolation=cv.INTER_AREA)
        imgss.append(img)
    img_a, img_b = imgss

    h, w = 416, 416

    center = (w // 2, h // 2)
    c = random.randint(1, 360)

    M = cv.getRotationMatrix2D(center, c, 1.0)
    # print(c)
    img_a, img_b = cv.warpAffine(img_a, M, (w, h)), cv.warpAffine(img_b, M, (w, h))

    a, b = np.zeros((h, w, 3), np.uint8), np.zeros((h, w, 3), np.uint8)
    a[0:h, 0:w // 2], a[0:h, w // 2:w] = img_a[0:h, 0:w // 2], img_b[0:h, w // 2:w]
    b[0:h, 0:w // 2], b[0:h, w // 2:w] = img_b[0:h, 0:w // 2], img_a[0:h, w // 2:w]

    return a, b


# 自适应直方图均衡化
def hisEqul(img, k=7, clip_limit=1.0):
    ycrcb = cv.cvtColor(img, cv.COLOR_RGB2YCR_CB)
    channels = cv.split(ycrcb)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(k, k))
    clahe.apply(channels[0], channels[0])
    cv.merge(channels, ycrcb)
    cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2RGB, img)
    return img


# 自适应直方图均衡化
class HisEqul(DualTransform):
    def __init__(self, k=7, clip_limit=1.0, always_apply: bool = False, p: float = 0.5):
        super(HisEqul, self).__init__(always_apply, p)
        self.k = k
        self.clip_limit = clip_limit

    def apply(self, img, **params) -> np.ndarray:
        return hisEqul(img, self.k, self.clip_limit)


# 颜色反转
def color_Reversal(img):
    return 255 - img


def Flip_Rotation_img(img):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    # h_img = cv.flip(img, 1)  # 水平翻转
    v_img = cv.flip(img, 0)  # 垂直翻转

    M = cv.getRotationMatrix2D(center, -90, 1.0)  # 90度
    left_img = cv.warpAffine(img, M, (w, h))
    M = cv.getRotationMatrix2D(center, -180, 1.0)  # 180度
    up_img = cv.warpAffine(img, M, (w, h))
    M = cv.getRotationMatrix2D(center, -270, 1.0)  # 270度
    right_img = cv.warpAffine(img, M, (w, h))

    return {"_rotate90": left_img, "_rotate180": up_img, "_rotate270": right_img, "_flip_v": v_img}


# 椒盐噪声
def PepperandSalt(img, percetage=0.005):
    NoiseImg = img
    NoiseNum = int(percetage * img.shape[0] * img.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if random.randint(0, 2) <= 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg


class datasets(Dataset):
    def __init__(self, mode="train", files="path"):
        self.mode = mode
        self.files = Path(files)
        self.images_labels = self.read_files()
        self.transform_train = A.Compose([
            A.GaussianBlur(blur_limit=(5, 9), sigma_limit=(0.1, 5), p=1),
            A.VerticalFlip(p=.7),
            A.Rotate(180, p=1),
            HisEqul(k=7, clip_limit=1.0),
            A.Resize(224, 224),
            A.Normalize(mean=[0, 0, 0],
                        std=[1, 1, 1]),
            ToTensorV2(),
        ])
        self.transform_val = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0, 0, 0],
                        std=[1, 1, 1]),
            ToTensorV2(),
        ])

    def __getitem__(self, index) -> T_co:
        path, label = self.images_labels[index]
        img = cv.imread(str(path))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # img = Image.fromarray(img)
        img = self.transform_train(image=img)['image'] if self.mode == "train" else self.transform_val(image=img)[
            'image']
        return img, label

    def __len__(self):
        return len(self.images_labels)

    def read_files(self):
        content = [line for line in self.files.open(encoding="utf-8") if line]
        pattern = r"(.*) (\d){1,}"
        fn = lambda x: re.findall(pattern, x)[0]
        content = [(Path(p), int(i)) for (p, i) in [fn(con) for con in content]]
        # print(content)
        return content


if __name__ == '__main__':
    # main()
    data = datasets(files="./datasets/train.txt")
    loder = DataLoader(data, batch_size=1, shuffle=False)
    # print(type(data[0][0]))
    for img, index in loder:
        print(img, index)
        print(img.shape, index.shape)
        break
