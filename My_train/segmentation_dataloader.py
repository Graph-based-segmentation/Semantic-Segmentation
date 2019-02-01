import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data

num_classes = 19
ignore_label = 255
root = '/home/mk/Semantic_Segmentation/Seg_dataset/Cityscapes_dataset'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

# def boundary_dataset(mode):
#     img_dir_name = 'leftImg8bit_trainvaltest'
#
#     boundary_path = os.path.join(root, 'boundary_image')
#     img_path = os.path.join(root, img_dir_name, 'leftImg8bit', mode)
#     boundary_items = []
#
#     categories = os.listdir(img_path)
#
#     for c in categories:
#         boundary_c_items = [boundary_name.split('_boundary.png')[0] for boundary_name in
#                             os.listdir(os.path.join(boundary_path, mode, c))]
#
#         for boundary_it in boundary_c_items:
#             boundary_item = os.path.join(boundary_path, mode, c, boundary_it + '_boundary.png')
#             boundary_items.append(boundary_item)
#
#     return boundary_items

def make_dataset(quality, mode):
    assert (quality == 'fine' and mode in ['train', 'val']) or \
           (quality == 'coarse' and mode in ['train', 'train_extra', 'val'])

    if quality == 'coarse':
        img_dir_name = 'leftImg8bit_trainextra' if mode == 'train_extra' else 'leftImg8bit_trainvaltest'
        mask_path = os.path.join(root, 'gtCoarse', 'gtCoarse', mode)
        mask_postfix = '_gtCoarse_labelIds.png'
    else:
        img_dir_name = 'leftImg8bit_trainvaltest'
        mask_path = os.path.join(root, 'gtFine_trainvaltest', 'gtFine', mode)
        mask_postfix = '_gtFine_labelIds.png'
    img_path = os.path.join(root, img_dir_name, 'leftImg8bit', mode)
    assert os.listdir(img_path) == os.listdir(mask_path)

    items = []

    categories = os.listdir(img_path)

    for c in categories:
        c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, c))]

        for it in c_items:
            item = (os.path.join(img_path, c, it + '_leftImg8bit.png'), os.path.join(mask_path, c, it + mask_postfix))
            items.append(item)

    return items


class CityScapes(data.Dataset):
    def __init__(self, quality, mode, joint_transform=None, transform=None, target_transform=None):
        self.imgs = make_dataset(quality, mode)
        # self.boundary_imgs = boundary_dataset(mode)

        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.quality = quality
        self.mode = mode
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']
        """7: road, 8: sidewalk, 11: building, ...."""

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        # boundary_path = self.boundary_imgs[index]

        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        # boundary = Image.open(boundary_path)

        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        if self.joint_transform is not None:
            # img, mask, boundary = self.joint_transform(img, mask, boundary)
            img, mask = self.joint_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
            # boundary = self.transform(boundary)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.imgs)