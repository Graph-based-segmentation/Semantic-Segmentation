import os
import random
import numpy as np
from numpy.core.fromnumeric import std

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

num_classes = 19
ignore_label = 255

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

def make_dataset(quality, args):
    assert (quality == 'fine' and args.mode in ['train', 'val']) or \
           (quality == 'coarse' and args.mode in ['train', 'train_extra', 'val'])

    if quality == 'coarse':
        img_dir_name = 'leftImg8bit_trainextra' if args.mode == 'train_extra' else 'leftImg8bit_trainvaltest'
        mask_path = os.path.join(args.data_path, 'gtCoarse', 'gtCoarse', args.mode)
        mask_postfix = '_gtCoarse_labelIds.png'
    else:
        img_dir_name = 'leftImg8bit_trainvaltest'
        mask_path = os.path.join(args.data_path, 'gtFine_trainvaltest', 'gtFine', args.mode)
        mask_postfix = '_gtFine_labelIds.png'
    img_path = os.path.join(args.data_path, img_dir_name, 'leftImg8bit', args.mode)
    assert os.listdir(img_path) == os.listdir(mask_path)

    items = []

    categories = os.listdir(img_path)

    for c in categories:
        c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, c))]

        for it in c_items:
            item = (os.path.join(img_path, c, it + '_leftImg8bit.png'), os.path.join(mask_path, c, it + mask_postfix))
            items.append(item)

    return items

def transform_preprocessing(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])

class msasppDataLoader(object):
    def __init__(self, args):
        if args.mode == 'train':
            self.training_samples = DataLoadPreprocess(quality='fine', args=args, transform=transform_preprocessing(args.mode))
            self.data = DataLoader(self.training_samples, args.train_batch_size,
                                   shuffle=True, num_workers=args.num_threads)

class DataLoadPreprocess(Dataset):
    def __init__(self, quality, args, transform=None):
        self.args = args
        self.pair_img = make_dataset(quality, args)

        if len(self.pair_img) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.quality = quality
        self.transform = transform
        self.to_tensor = ToTensor
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
        if self.args.mode == 'train':
            img_path, gt_path = self.pair_img[index]

            img, gt = Image.open(img_path).convert('RGB'), Image.open(gt_path)

            gt = np.array(gt)
            mask_copy = gt.copy()
            for k, v in self.id_to_trainid.items():
                mask_copy[gt == k] = v
            gt = Image.fromarray(mask_copy.astype(np.uint8))

            if self.args.random_rotate:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                img = self.rotate_image(img, random_angle)
                gt = self.rotate_image(gt, random_angle, flag=Image.NEAREST)

            img = np.asarray(img, dtype=np.float32) / 255.0
            gt = np.asarray(gt, dtype=np.float32)

            img, gt = self.random_crop(img, gt, self.args.input_height, self.args.input_width)
            img, gt = self.train_preprocess(img, gt)
            
            sample = {'image': img, 'gt': gt}
            
            if self.transform:
                sample = self.transform(sample)
                
            return sample

    def rotate_image(self, img, angle, flag=Image.BILINEAR):
        result = img.rotate(angle, resample=flag)

        return result

    def random_crop(self, img, gt, height, width):
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)

        img = img[y:y + height, x:x + width, :]
        gt = gt[y:y + height, x: x + width, :]

        return img, gt
    
    def train_preprocess(self, img, gt):
        do_flip = random.random()
        if do_flip > 0.5:
            img = (img[:, ::-1, :]).copy()
            gt = (gt[:, ::-1, :]).copy()
        
        do_augment = random.random()
        if do_augment > 0.5:
            img = self.augment_image(img)
            
        return img, gt

    def augment_image(self, img):
        # brightness augmentation
        brightness = random.uniform(0.9, 1.1)
        
        img_aug = img * brightness
        
        return img_aug
    
    def __len__(self):
        return len(self.pair_img)

class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])
        
    def __call__(self, sample):
        