import cv2
import os
import sys
import numpy as np
import math
import glob
import pyspng
import torch
import PIL.Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def calculate_psnr(img1, img2, max_value=255):
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))
# def calculate_ssim(img1, img2):
#     C1 = (0.01 * 255) ** 2
#     C2 = (0.03 * 255) ** 2
#
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())
#
#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1 ** 2
#     mu2_sq = mu2 ** 2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
#
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#
#     return ssim_map.mean()
def calculate_ssim(img1, img2, data_range=255, win_size=11, multichannel=True):
    return compare_ssim(img1, img2, data_range=data_range, multichannel=multichannel, win_size=win_size)
# def calculate_l1(img1, img2):
#     img1 = img1.astype(np.float64) / 255.0
#     img2 = img2.astype(np.float64) / 255.0
#     l1 = np.mean(np.abs(img1 - img2))
#
#     return l1
def calculate_l1(img1, img2):
    return np.mean(np.abs((np.mean(img1, 2) - np.mean(img2, 2)) / 255))


def read_image(image_path):
    with open(image_path, 'rb') as f:
        if pyspng is not None and image_path.endswith('.png'):
            image = pyspng.load(f.read())
        else:
            image = np.array(PIL.Image.open(f))
    if image.ndim == 2:
        image = image[:, :, np.newaxis]  # HW => HWC
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    # image = image.transpose(2, 0, 1) # HWC => CHW

    return image

# def postprocess(img):
#     print("Shape of img before transpose:", img.shape)
#     img = (img + 1) / 2 * 255
#     img = img.permute(0, 2, 3, 1)
#     img = img.int().cpu().numpy().astype(np.uint8)
#     return img

def postprocess(img):
    # print("Shape of img before transpose:", img.shape)
    img = (img + 1) / 2 * 255  # 将图像归一化到 [0, 255]
    img = img.astype(np.uint8)  # 转换为 uint8 类型，适用于图像处理
    return img

def calculate_metrics(folder1, folder2):
    l1 = sorted(glob.glob(folder1 + '/*.png') + glob.glob(folder1 + '/*.jpg'))
    l2 = sorted(glob.glob(folder2 + '/*.png') + glob.glob(folder2 + '/*.jpg'))
    assert (len(l1) == len(l2))
    # print('length:', len(l1))
    print(l1)
    print(l2)
    total_images = len(l1)
    psnr_sum = 0
    ssim_sum = 0
    l1_sum = 0
    # psnr_l, ssim_l, dl1_l = [], [], []
    with open('eval.txt', 'w') as f:
        for i, (fpath1, fpath2) in enumerate(zip(l1, l2)):
            # print('Processing image', i + 1, 'of', len(l1))
            # print(i)
            _, name1 = os.path.split(fpath1)
            _, name2 = os.path.split(fpath2)
            name1 = name1.split('.')[0]
            name2 = name2.split('.')[0]
            # assert name1 == name2, 'Illegal mapping: %s, %s' % (name1, name2)
            img1 = read_image(fpath1).astype(np.float64)
            img2 = read_image(fpath2).astype(np.float64)
            assert img1.shape == img2.shape, 'Illegal shape'
            img1 = postprocess(img1)# Add batch dimension
            img2 = postprocess(img2)
            psnr = (calculate_psnr(img1, img2))
            ssim = (calculate_ssim(img1, img2))
            dl1 = (calculate_l1(img1, img2))

            psnr_sum += psnr
            ssim_sum += ssim
            l1_sum += dl1

            f.write('Image {}/{}'.format(i, total_images))
            f.write('PSNR: {:.4f}'.format(psnr))
            f.write('SSIM: {:.4f}'.format(ssim))
            f.write('L1 Distance: {:.4f}\n'.format(dl1))
    psnr_mean = psnr_sum / total_images
    ssim_mean = ssim_sum / total_images
    l1_mean = l1_sum / total_images

    return psnr_mean, ssim_mean, l1_mean


if __name__ == '__main__':
    folder1 = '/home/li/gt/'
    folder2 = '//home/li/our/'


    psnr_mean, ssim_mean, l1_mean = calculate_metrics(folder1, folder2)
    print('psnr: %.4f, ssim: %.4f, l1: %.4f' % (psnr_mean, ssim_mean, l1_mean))
    with open('psnr_ssim_l1.txt', 'w') as f:
        f.write('psnr: %.4f, ssim: %.4f, l1: %.4f' % (psnr_mean, ssim_mean, l1_mean))
