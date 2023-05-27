import math

import cv2
import numpy as np


def build_filters(ksize, sigma, theta, lambd, gamma):
    filters = []
    kmax = math.pi / 2
    for phi in np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
        # 构建Gabor滤波器
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta + phi, lambd, gamma, ktype=cv2.CV_32F)
        if kern.sum() != 0:
            kern /= 0.1 * kern.sum()  # 归一化滤波器的值
        filters.append(kern)
    return filters


def process(img, filters):
    accum = np.zeros_like(img)
    weights = [0.2, 0.2, 0.35, 0.25]  # 权重可调
    for kern, weight in zip(filters, weights):
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        accum = accum + weight * fimg
    return accum


def get_gabor_feature(img, ksize=3, sigma=10, theta=0, lambd=35,
                      gamma=0.5):  # theta是角度，lambd=35时感觉挺好（自我感觉）
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转为灰度图像
    filters = build_filters(ksize, sigma, theta, lambd, gamma)  # 构建Gabor滤波器
    res = process(img_gray, filters)  # 对灰度图像进行Gabor特征提取
    return res


img = cv2.imread('00199.jpg')
gabor_feature = get_gabor_feature(img)

cv2.imshow('Gabor feature', gabor_feature)
cv2.waitKey(0)
cv2.destroyAllWindows()
