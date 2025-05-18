import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import torchvision
# import RandomErasing
import os
import math
import argparse
import scipy as sp
import scipy.stats
import pickle
import random
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
import torchnet as tnt
import torchvision.transforms.functional as F
import torch.nn.functional
# 自定义测试数据集
import torch.utils.data as data
import torchvision.transforms as transforms


def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):  # 0。9 1。1
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + beta * noise


def flip_augmentation(data):  # arrays tuple 0:(7, 7, 103) 1=(7, 7)
    horizontal = np.random.random() > 0.5  # True
    vertical = np.random.random() > 0.5  # False
    if horizontal:
        data = np.fliplr(data)
    if vertical:
        data = np.flipud(data)
    return data


class hsidataset_target(data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        # self.trans=RandomErasing.RandomErasing()

    def __getitem__(self, index):
        img = self.data[index]
        img1 = img
        '''
        img1=random_resized_crop_hs(img,(0.7, 1.0))
        img1 = img1[:, :].copy()
        '''
        return img, img1

    def __len__(self):
        return len(self.data)


def twist_loss(p1, p2, alpha=1, beta=1):
    eps = 1e-7  # ensure calculate
    # eps=0
    kl_div = ((p2 * p2.log()).sum(dim=1) - (p2 * p1.log()).sum(dim=1)).mean()
    mean_entropy = -(p1 * (p1.log() + eps)).sum(dim=1).mean()
    mean_prob = p1.mean(dim=0)
    entropy_mean = -(mean_prob * (mean_prob.log() + eps)).sum()

    return kl_div + alpha * mean_entropy - beta * entropy_mean


def sanity_check(all_set):
    nclass = 0
    nsamples = 0
    all_good = {}
    for class_ in all_set:
        if len(all_set[class_]) >= 400:
            all_good[class_] = all_set[class_][:400]
            nclass += 1
            nsamples += len(all_good[class_])
    print('the number of class:', nclass)
    print('the number of sample:', nsamples)
    return all_good


def ssl_data_maker(img):
    img1 = hsi_patch_augmentation(img)
    img2 = hsi_patch_augmentation(img)
    return img1, img2


def twist_loss(p1, p2, alpha=1, beta=1):
    eps = 1e-7  # ensure calculate
    # eps=0
    kl_div = ((p2 * p2.log()).sum(dim=1) - (p2 * p1.log()).sum(dim=1)).mean()
    mean_entropy = -(p1 * (p1.log() + eps)).sum(dim=1).mean()
    mean_prob = p1.mean(dim=0)
    entropy_mean = -(mean_prob * (mean_prob.log() + eps)).sum()

    return kl_div + alpha * mean_entropy - beta * entropy_mean





# 4. 随机旋转（自定义旋转）
def custom_rotate(patches, angle):
    if angle == 90:
        return patches.transpose(-2, -1).flip(-2)
    elif angle == 180:
        return patches.flip(-2).flip(-1)
    elif angle == 270:
        return patches.transpose(-2, -1).flip(-1)
    return patches



def hsi_patch_augmentation(hsi_patches):
    """
    高光谱图像批量 patch 数据增强函数，适用于 (batchsize, C, 33, 33) 的 patch。
    输入：hsi_patches (torch.Tensor) - 形状为 (batchsize, C, 33, 33) 的高光谱 patch 批量
    输出：增强后的 patch 批量，空间尺寸保持为 (batchsize, C, 33, 33)
    """
    batchsize, C, H, W = hsi_patches.shape
    assert H == 33 and W == 33, "Patch size must be 33x33"

    # 1. 随机裁剪和缩放（对整个batch使用相同的裁剪参数）
    if np.random.rand() < 0.5:
        scale_range = (0.7, 1.0)  # 裁剪面积占原图面积的比例范围
        ratio_range = (3.0 / 4.0, 4.0 / 3.0)  # 宽高比范围
        size = (33, 33)  # 目标大小

        # 为整个batch生成一次随机裁剪参数
        scale = np.random.uniform(*scale_range)
        aspect_ratio = np.random.uniform(*ratio_range)
        area = H * W
        target_area = scale * area
        crop_h = int(np.sqrt(target_area / aspect_ratio))
        crop_w = int(np.sqrt(target_area * aspect_ratio))
        # 确保裁剪尺寸不超过原图
        crop_h = min(crop_h, H)
        crop_w = min(crop_w, W)
        # 随机选择裁剪位置
        i = np.random.randint(0, H - crop_h + 1)
        j = np.random.randint(0, W - crop_w + 1)

        # 对整个batch应用相同的裁剪
        cropped_patches = hsi_patches[:, :, i:i + crop_h, j:j + crop_w]
        # 调整大小到目标尺寸
        hsi_patches = F.resize(cropped_patches, size, interpolation=transforms.InterpolationMode.BICUBIC)
    # 5. 光谱噪声
    else:
        alpha_range = (0.9, 1.1)
        # 生成与x相同设备/dtype的随机alpha
        alpha = (alpha_range[1] - alpha_range[0]) * torch.rand(
            1, device=hsi_patches.device, dtype=hsi_patches.dtype
        ) + alpha_range[0]

        # 生成与x相同形状/设备/dtype的高斯噪声
        noise = torch.randn_like(hsi_patches)

        # 构造带类型和设备的beta系数
        beta = torch.tensor(1 / 25, device=hsi_patches.device, dtype=hsi_patches.dtype)

        return alpha * hsi_patches + beta * noise

    # 2. 随机水平翻转
    if np.random.rand() < 0.5:
        hsi_patches = torch.flip(hsi_patches, dims=[3])  # width 维度

    # 3. 随机垂直翻转
    if np.random.rand() < 0.5:
        hsi_patches = torch.flip(hsi_patches, dims=[2])  # height 维度

    # 4. 随机旋转
    if np.random.rand() < 0.5:
        angle = int(np.random.choice([90, 180, 270]))
        hsi_patches = F.rotate(hsi_patches, angle)

    return hsi_patches



def add_gaussian_noise(x: torch.Tensor) -> torch.Tensor:
    """
    为高光谱像素数据添加高斯噪声，每个样本独立生成噪声和 alpha 参数
    Args:
        x: 输入数据，形状为 (batch_size, bands)
    Returns:
        x_noisy: 添加噪声后的数据，形状与 x 相同
    """
    alpha_range = (0.9, 1.1)
    # 生成与x相同设备/dtype的随机alpha
    alpha = (alpha_range[1] - alpha_range[0]) * torch.rand(
        1, device=x.device, dtype=x.dtype
    ) + alpha_range[0]

    # 生成与x相同形状/设备/dtype的高斯噪声
    noise = torch.randn_like(x)

    # 构造带类型和设备的beta系数
    beta = torch.tensor(1/5, device=x.device, dtype=x.dtype)

    return x + beta * noise


def band_dropout(x: torch.Tensor, dropout_rate: float = 0.05) -> torch.Tensor:
    """
    随机丢弃高光谱像素的某些波段（置零）
    Args:

        x: 输入数据，形状为 (batch_size, bands)
        dropout_rate: 波段丢弃比例（建议值：0.1~0.3）
    Returns:
        x_masked: 掩码后的数据，形状与 x 相同
    """
    mask = torch.rand_like(x) > dropout_rate  # 生成伯努利掩码
    return x * mask





