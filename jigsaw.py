from __future__ import print_function
import torch
import torch.utils.data as data
import torchvision
import torchnet as tnt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from PIL import Image
import os
import errno
import sys
import csv
from PIL import ImageFilter
from PIL import Image
from pdb import set_trace as breakpoint

_CIFAR_DATASET_DIR = './datasets/CIFAR'
_IMAGENET_DATASET_DIR = 'datasets/miniImagenet'
_PLACES205_DATASET_DIR = './datasets/Places205'

def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)
    return label2inds

class GenericDataset(data.Dataset):
    def __init__(self, dataset_name, split, num_imgs_per_cat=None):
        self.split = split.lower()
        self.dataset_name = dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        self.num_imgs_per_cat = num_imgs_per_cat

        if self.dataset_name=='miniimagenet':
            assert (self.split=='train' or self.split=='val')
            self.mean_pix = [0.485, 0.456, 0.406]
            self.std_pix = [0.229, 0.224, 0.225]
            transforms_list = [
                transforms.Resize(33),
                transforms.RandomResizedCrop(33),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x)
            ]
            self.transform = transforms.Compose(transforms_list)
            split_data_dir = _IMAGENET_DATASET_DIR + '/' + self.split
            self.data = datasets.ImageFolder(split_data_dir, self.transform)

    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label)

    def __len__(self):
        return len(self.data)

class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def create_jigsaw_patches(img, grid_size=11, target_size=33):
    """
    将图像切分成 grid_size×grid_size 的小块并随机打乱。
    每个小块将被缩放到 target_size × target_size。
    返回：
    - patches: 切分并打乱后的小块 (list of np.ndarray)，长度为 grid_size^2
    - permutation: 一个 list，表示打乱后小块对应该在的正确位置
    """
    h, w, c = img.shape
    patch_h = h // grid_size
    patch_w = w // grid_size

    patches = []
    for i in range(grid_size):
        for j in range(grid_size):
            top = i * patch_h
            left = j * patch_w
            patch = img[top:top+patch_h, left:left+patch_w, :]
            # 使用 Image.LANCZOS 进行高质量的图像缩放
            patch_image = Image.fromarray(patch)
            patch_resized = patch_image.resize((target_size, target_size), Image.LANCZOS)
            patches.append(np.array(patch_resized))

    indices = list(range(grid_size * grid_size))
    random.shuffle(indices)
    shuffled_patches = [patches[i] for i in indices]
    permutation = indices

    return shuffled_patches, permutation

class DataLoader(object):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 unsupervised=True,
                 epoch_size=None,
                 num_workers=0,
                 shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.num_workers = num_workers

        mean_pix = self.dataset.mean_pix
        std_pix = self.dataset.std_pix

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ])
        self.inv_transform = transforms.Compose([
            Denormalize(mean_pix, std_pix),
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1,2,0).astype(np.uint8),
        ])

    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)

        if self.unsupervised:
            # 在无监督模式下，定义一个“拼图”任务：
            # 1) 读入原图并进行初步 resize/变换
            # 2) 切分成 3×3 小块并随机打乱
            # 3) 返回打乱后的 block 列表，以及它们对应的原位置索引
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img0, _ = self.dataset[idx]

                # 生成拼图块和其随机顺序
                jigsaw_patches, patch_labels = create_jigsaw_patches(img0, grid_size=3)

                # 对每个小块做 transform
                # 最终希望返回形状 [9, C, patch_h, patch_w]
                jigsaw_patches_tensors = [self.transform(p) for p in jigsaw_patches]
                jigsaw_patches_tensors = torch.stack(jigsaw_patches_tensors, dim=0)

                # patch_labels 是长度为 9 的 list，比如 [2,5,1,0,7,8,3,6,4]
                patch_labels = torch.LongTensor(patch_labels)
                return jigsaw_patches_tensors, patch_labels

            def _collate_fun(batch):
                """
                batch 中的每个元素是 (jigsaw_patches, patch_labels),
                其中 jigsaw_patches 形状为 [9, C, ph, pw],
                patch_labels 形状为 [9].
                """
                batch = default_collate(batch)
                # batch[0]: [B, 9, C, ph, pw]
                # batch[1]: [B, 9]
                # 我们可以把 [B, 9, C, ph, pw] reshape 成 [B*9, C, ph, pw]
                # label 同样变为 [B*9]
                patches = batch[0]  # shape: (B, 9, C, ph, pw)
                labels = batch[1]   # shape: (B, 9)

                bsz, num_patches, c, ph, pw = patches.size()
                patches = patches.view(bsz * num_patches, c, ph, pw)
                labels = labels.view(bsz * num_patches)

                new_batch = (patches, labels)
                return new_batch

        else:
            # 有监督模式，则仅返回原图和原始分类标签
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img, categorical_label = self.dataset[idx]
                img = self.transform(img)
                return img, categorical_label

            _collate_fun = default_collate

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
                                              load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
                                           collate_fn=_collate_fun,
                                           num_workers=self.num_workers,
                                           shuffle=self.shuffle)
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size / self.batch_size


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    dataset = GenericDataset('miniImagenet', 'train')
    # 在无监督模式下，batch_size=2 仅做演示
    dataloader = DataLoader(dataset, batch_size=3, unsupervised=True)

    print("Dataset length:", len(dataloader.get_iterator(0)))

    batch_data = next(iter(dataloader(0)))
    data, label = batch_data
    print("Data shape:", data.shape)       # [batch_size*9, C, patch_h, patch_w]
    print("Label shape:", label.shape)     # [batch_size*9]

    inv_transform = dataloader.inv_transform
    # 可视化前 9 张拼图块，即第一个样本
    num_patches = 9
    plt.figure(figsize=(8, 8))
    for i in range(num_patches):
        plt.subplot(3, 3, i+1)
        # 因为 batch_size=2，所以前 9 个是第一个样本；后 9 个是第二个样本
        # 你也可以只看 data[i] 这样的单一块
        arr = inv_transform(data[i])
        plt.imshow(arr)
        plt.title(f"Pos={label[i].item()}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x