import cv2 as cv
import numpy as np
import os, sys
from PIL import Image
import string
import random
from multiprocessing.pool import Pool


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
    imgs = [process_image(img, size) for img in imgs]
    image = cv.vconcat([cv.hconcat([imgs[0], imgs[1]]), cv.hconcat([imgs[2], imgs[3]])])
    return image


def SamplePairing(imga, imgb):
    imga, imgb = process_image(imga), process_image(imgb)
    img = (imga + imgb) // 2
    return img


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


def generate_sample_img(num):
    global path
    imgs = os.listdir(path)
    imgs_path = [os.path.join(path, img) for img in imgs]
    choice_imgs = np.random.choice(imgs_path, size=num)
    while set(choice_imgs) == 1:
        choice_imgs = np.random.choice(imgs_path, size=num)
    return choice_imgs


ran_str = lambda: ''.join(random.sample(string.ascii_letters + string.digits, 18))


def generate_iterable(num, n=1000):
    tmp = []
    for _ in range(n):
        tmp.append(generate_sample_img(num))
    return tmp


def main():
    global path
    path = r'./data/train/level4'
    # -----------------------
    cutmix_lst = generate_iterable(2, 1000)
    mosaic_lst = generate_iterable(4, 1000)
    SamplePairing_lst = generate_iterable(2, 1000)
    # ---------------------
    with Pool(10) as pool:
        # cutmix
        if cutmix_lst:
            pool.map(generate_img_cutmix, cutmix_lst)
        # mosaic
        if mosaic_lst:
            pool.map(generate_img_mosaic, mosaic_lst)
        # SamplePairing
        if SamplePairing_lst:
            pool.map(generate_img_SamplePairing, SamplePairing_lst)


if __name__ == '__main__':
    main()
