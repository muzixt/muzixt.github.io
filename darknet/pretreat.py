import cv2 as cv
import numpy as np
import os, sys
from PIL import Image
import string
import random
from multiprocessing.pool import Pool as Pool
from albumentations import *
import albumentations as A


def process_image(img, min_side=416):
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


def SamplePairing(imga, imgb):
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
    # img_a = process_image(img_a)
    # img_b = process_image(img_b)
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


def generate_img_cutmix(val):
    imga, imgb = cv.imread(val[0]), cv.imread(val[1])
    # default n=2
    img = cutmix(imga, imgb, n=2)
    save_path = os.path.dirname(val[0])
    save_path = os.path.join(save_path, ran_str() + "_cutmix.jpg")
    print(save_path)
    cv.imwrite(save_path, img)


def generate_img_mosaic(val):
    imgs = []
    for img in val:
        imgs.append(cv.imread(img))
    img = mosaic(imgs)
    save_path = os.path.dirname(val[0])
    save_path = os.path.join(save_path, ran_str() + "_Mosaic.jpg")
    print(save_path)
    cv.imwrite(save_path, img)


def generate_img_SamplePairing(val):
    imga, imgb = cv.imread(val[0]), cv.imread(val[1])
    img = SamplePairing(imga, imgb)
    save_path = os.path.dirname(val[0])
    save_path = os.path.join(save_path, ran_str() + "_SamplePairing.jpg")
    print(save_path)
    cv.imwrite(save_path, img)


def generate_img_cuthalf(val):
    imga, imgb = cv.imread(val[0]), cv.imread(val[1])
    imga, imgb = cuthalf(imga, imgb)
    save_path = os.path.dirname(val[0])
    save_patha = os.path.join(save_path, ran_str() + "_cuthalf.jpg")
    save_pathb = os.path.join(save_path, ran_str() + "_cuthalf.jpg")
    print(save_patha)
    print(save_pathb)
    cv.imwrite(save_patha, imga)
    cv.imwrite(save_pathb, imgb)


def generate_sample_img(num):
    global path
    imgs = os.listdir(path)
    imgs_path = [os.path.join(path, img) for img in imgs]
    choice_imgs = np.random.choice(imgs_path, size=num)
    while set(choice_imgs) == 1 and num != 1:
        choice_imgs = np.random.choice(imgs_path, size=num)
    return choice_imgs


ran_str = lambda: ''.join(random.sample(string.ascii_letters + string.digits, 18))


def generate_iterable(num, n=1000):
    tmp = []
    print("prepare...")
    for i in range(n):
        tmp.append(generate_sample_img(num))
        print(f"\r{i + 1}/{n}", end="")
    print()
    return tmp


def albumen_lst_iter(size=1000):
    global path
    imgs = os.listdir(path)
    imgs_path = [os.path.join(path, img) for img in imgs]
    choice_imgs = np.random.choice(imgs_path, size=size)
    imgs_dic = [(k, v) for k in range(len(trans)) for v in choice_imgs]
    return imgs_dic


def generate_albumen(val):
    image = cv.imread(val[1])
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    transform = A.Compose(trans[val[0]])
    image = transform(image=image)["image"]
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    save_path = os.path.dirname(val[1])
    save_path = os.path.join(save_path, ran_str() + f"_{val[0]}_albumen.jpg")
    print(save_path)
    cv.imwrite(save_path, image)


# -----------------------------------------------------------------------


# 自适应直方图均衡化
def hisEqul(img):
    # 将RGB图像转换到YCrCb空间中
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
    channels = cv.split(ycrcb)
    k = random.randint(7, 14)
    clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(k, k))
    clahe.apply(channels[0], channels[0])
    cv.merge(channels, ycrcb)
    cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2BGR, img)
    return img


# 颜色反转
def color_Reversal(img):
    # cha = img.shape
    # height, width, deep = cha
    # dst = np.zeros((height, width, 3), np.uint8)
    # for i in range(height):  # 色彩反转
    #     for j in range(width):
    #         b, g, r = img[i, j]
    #         dst[i, j] = (255 - b, 255 - g, 255 - r)
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

    # return {"_flip_v": v_img}


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


def generate_img_PepperandSalt(val):
    img = cv.imread(val)
    img = PepperandSalt(img)
    f, suffix = os.path.splitext(val)
    save_path = f + "_PepperandSalt" + suffix
    print(save_path)
    cv.imwrite(save_path, img)


def generate_img_Flip_Rotation(val):
    img = cv.imread(val)
    imgs = Flip_Rotation_img(img)
    f, suffix = os.path.splitext(val)
    for name, im in imgs.items():
        save_path = f + name + suffix
        print(save_path)
        cv.imwrite(save_path, im)


def generate_img_hisEqul(val):
    img = cv.imread(val)
    img = hisEqul(img)
    f, suffix = os.path.splitext(val)
    save_path = f + "_hisEqul" + suffix
    print(save_path)
    cv.imwrite(save_path, img)


def generate_img_color_reversal(val):
    img = cv.imread(val)
    img = color_Reversal(img)
    f, suffix = os.path.splitext(val)
    save_path = f + "color_reversal" + suffix
    print(save_path)
    cv.imwrite(save_path, img)


def generate_img_lst(name: str = ""):
    print(f"{name} prepare...")
    global path
    imgs = os.listdir(path)
    imgs_path = [os.path.join(path, img) for img in imgs]
    print(f"{name} prepared...")
    return imgs_path


def main():
    global path
    path = r'./data/train2/level1'
    # -----------------------
    cutmix_lst, mosaic_lst, SamplePairing_lst, cuthalf_lst, = [], [], [], []
    # cutmix_lst = generate_iterable(2, 1000) # x
    # mosaic_lst = generate_iterable(4, 20000)  # *
    # SamplePairing_lst = generate_iterable(2, 15000)
    # cuthalf_lst = generate_iterable(2, 10000)  # default *2

    # ---------------------

    hisEqul_lst, color_reversal_lst, PepperandSalt_lst, Flip_Rotation_lst = [], [], [], []
    # ///////
    hisEqul_lst = generate_sample_img(1000)
    # ////////////

    # Flip_Rotation_lst = generate_img_lst("Flip_Rotation")

    # color_reversal_lst = generate_img_lst("color_reversal") # xx

    # PepperandSalt_lst = generate_img_lst("PepperandSalt")

    # -----------------
    global trans
    trans = [
        [
            SafeRotate(always_apply=True, limit=180)
        ],
        [
            ColorJitter(always_apply=True)
        ],
        [
            RandomBrightnessContrast(always_apply=True, brightness_limit=0.7, contrast_limit=0.7,
                                     brightness_by_max=False)
        ],
        [
            ElasticTransform(sigma=70, alpha_affine=100, always_apply=True)
        ],
        [
            GridDistortion(num_steps=7, distort_limit=0.3, always_apply=True)
        ],

        [
            RandomGamma(gamma_limit=(100, 300), always_apply=True)
        ],
        [
            RandomToneCurve(scale=0.7, always_apply=True)
        ],
        [
            RandomResizedCrop(320, 320, scale=(0.2, 1.0), ratio=(0.75, 1.5), always_apply=True)
        ],

        [
            RandomGridShuffle(grid=(3, 3), always_apply=True)
        ],
        [
            PiecewiseAffine(scale=(0.03, 0.07), nb_rows=4, nb_cols=4, always_apply=True)
        ]
    ]
    albumen_lst = []
    albumen_lst = albumen_lst_iter(3500)

    with Pool(8) as pool:
        # cutmix
        if cutmix_lst:
            pool.map(generate_img_cutmix, cutmix_lst)
        # mosaic
        if mosaic_lst:
            pool.map(generate_img_mosaic, mosaic_lst)
        # SamplePairing
        if SamplePairing_lst:
            pool.map(generate_img_SamplePairing, SamplePairing_lst)
        # cuthalf
        if cuthalf_lst:
            pool.map(generate_img_cuthalf, cuthalf_lst)

        # -------------------------------

        if hisEqul_lst:
            pool.map(generate_img_hisEqul, hisEqul_lst)
        if color_reversal_lst:
            pool.map(generate_img_color_reversal, color_reversal_lst)
        if PepperandSalt_lst:
            pool.map(generate_img_PepperandSalt, PepperandSalt_lst)
        # * 4
        if Flip_Rotation_lst:
            pool.map(generate_img_Flip_Rotation, Flip_Rotation_lst)

        #  --------------
        if albumen_lst:
            pool.map(generate_albumen, albumen_lst)


if __name__ == '__main__':
    main()
