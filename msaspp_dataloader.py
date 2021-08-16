import os
import random
import numpy as np
import torch
import math

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

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

def preprocessing_transform(mode):
    return transforms.Compose([ToTensor(mode=mode)])

class msasppDataLoader(object):
    def __init__(self, args):
        self.ignore_label = 255
        
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']
        
        if args.mode == 'train':
            self.training_samples = DataLoadPreprocess(self.ignore_label, quality='fine', args=args, transform=preprocessing_transform(args.mode))
            self.data = DataLoader(self.training_samples, args.train_batch_size,
                                   shuffle=True, num_workers=args.num_threads)

class DataLoadPreprocess(Dataset):
    def __init__(self, ignore_label, quality, args, transform=None):
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
        """7: road, 8: sidewalk, 11: building, ...."""

    def __getitem__(self, index):
        if self.args.mode == 'train':
            img_path, gt_path = self.pair_img[index]

            image, gt = Image.open(img_path), Image.open(gt_path)
            gt = np.array(gt)
            gt_copy = gt.copy()
            
            for key, value in self.id_to_trainid.items():
                gt_copy[gt == key] = value
            gt = Image.fromarray(gt_copy.astype(np.uint8))

            to_tensor = transforms.ToTensor()
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])
            
            image, gt = self.random_horizontally_flip(image, gt)
            image, gt = self.random_resized_crop(image, gt)
            # do_augment = random.random()
            # if do_augment > 0.5:
            #         image = self.augment_image(image)
                    
            image = to_tensor(image)
            gt = torch.from_numpy(np.array(gt, dtype=np.int32)).long()
            
            
            # image, gt = self.random_resized_crop(self.args, img, gt, scale=(0.5, 2.0))
            
            # if self.args.random_rotate:
            #     random_angle = (random.random() - 0.5) * 2 * self.args.degree
            #     image = self.rotate_image(img, random_angle)
            #     gt = self.rotate_image(gt, random_angle, flag=Image.NEAREST)

            # image = np.asarray(image, dtype=np.float32) / 255.0
            # gt = np.asarray(gt, dtype=np.float32)
            # gt = np.expand_dims(gt, axis=2)
            
            # image, gt = self.random_crop(image, gt, self.args.input_height, self.args.input_width)
            # image, gt = self.train_preprocess(image, gt)
            sample = {'image': image, 'gt': gt}
            
            # if self.transform:
            #     sample = self.transform(sample)
            return sample

    def random_horizontally_flip(self, image, gt, p=0.5):
        import random
        
        if random.random() < p:
            return image.transpose(Image.FLIP_LEFT_RIGHT), gt.transpose(Image.FLIP_LEFT_RIGHT)

        return image, gt
    
    def random_resized_crop(self, image, gt):
        assert image.size == gt.size
        for _ in range(10):
            area = image.size[0] * image.size[1]
            target_area = random.uniform(0.5, 2.0) * area
            aspect_ratio = random.uniform(3. / 4., 4. / 3.)


            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))


            if random.random() < 0.5:
                w, h = h, w

            if w <= image.size[0] and h <= image.size[1]:
                x1 = random.randint(0, image.size[0] - w)
                y1 = random.randint(0, image.size[1] - h)


                image = image.crop((x1, y1, x1 + w, y1 + h))
                gt = gt.crop((x1, y1, x1 + w, y1 + h))
                assert (image.size == (w, h))

                resized_image = image.resize((self.args.input_height, self.args.input_width), Image.BILINEAR)
                resized_gt = gt.resize((self.args.input_height, self.args.input_width),Image.NEAREST)

                return resized_image, resized_gt

        # Fallback
        scale = Scale(self.args.input_height)
        crop = CenterCrop(self.args.input_height)
        return crop(*scale(image, gt))

    def __len__(self):
        return len(self.pair_img)
    
class CenterCrop(object):
    def __init__(self, size):
        import numbers
        
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size


    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))
    
class Scale(object):
    def __init__(self, size):
        self.size = size


    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        # width, height = transforms.functional._get_image_size(image)
        # area = height * width
        
        # log_ratio = torch.log(torch.tensor(ratio))
        # for _ in range(10):
        #     target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        #     aspect_ratio = torch.exp(
        #         torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        #     ).item()

        #     w = int(round(math.sqrt(target_area * aspect_ratio)))
        #     h = int(round(math.sqrt(target_area / aspect_ratio)))

        #     if 0 < w <= width and 0 < h <= height:
        #         i = torch.randint(0, height - h + 1, size=(1,)).item()
        #         j = torch.randint(0, width - w + 1, size=(1,)).item()
                
        #         image = transforms.functional.resized_crop(image, i, j, h, w, (args.input_height, args.input_width), transforms.functional.InterpolationMode.BILINEAR)
        #         gt = transforms.functional.resized_crop(gt, i, j, h, w, (args.input_height, args.input_width), transforms.functional.InterpolationMode.BILINEAR)

        #         return image, gt
            
        # # Fallback to central crop
        # in_ratio = float(width) / float(height)
        # if in_ratio < min(ratio):
        #     w = width
        #     h = int(round(w / min(ratio)))
        # elif in_ratio > max(ratio):
        #     h = height
        #     w = int(round(h * max(ratio)))
        # else:  # whole image
        #     w = width
        #     h = height
        # i = (height - h) // 2
        # j = (width - w) // 2
        
        # image = transforms.functional.resized_crop(image, i, j, h, w, (self.args.input_height, self.args.input_width), transforms.functional.InterpolationMode.BILINEAR)
        # gt = transforms.functional.resized_crop(gt, i, j, h, w, (self.args.input_height, self.args.input_width), transforms.functional.InterpolationMode.BILINEAR)

        # return image, gt
    
    def augment_image(self, image):
        # gamma augmentation
        image = np.array(image) / 255.0
        
        gamma  = random.uniform(0.9, 1.1)
        image_aug = image ** gamma
        
        # brightness augmentation
        brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness
        
        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)
        
        return image_aug
    # def rotate_image(self, img, angle, flag=Image.BILINEAR):
    #     result = img.rotate(angle, resample=flag)

    #     return result

    # def random_crop(self, img, gt, height, width):
    #     x = random.randint(0, img.shape[1] - width)
    #     y = random.randint(0, img.shape[0] - height)

    #     img = img[y:y + height, x:x + width, :]
    #     gt = gt[y:y + height, x:x + width, :]

    #     return img, gt
    
    
    # def train_preprocess(self, image, gt):
    #     # Random horizontal flipping
    #     do_flip = random.random()
    #     if do_flip > 0.5:
    #         image = (image[:, ::-1, :]).copy()
    #         # gt = (gt[:, ::-1, :]).copy()
    #         gt = (gt[:, ::-1]).copy()
            
    #     # Random gamma, brightness, color augmentation
    #     do_augment = random.random()
    #     if do_augment > 0.5:
    #         image = self.augment_image(image)
            
    #     return image, gt

    
    def __len__(self):
        return len(self.pair_img)

class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])
        
    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image, gt = self.to_tensor(sample)
        # image = self.to_tensor(image)
        # gt = self.to_tensor(gt)
        
        image = self.normalize(image)
        
        return {'image': image, 'gt': gt}
    
    def to_tensor(self, pic):
        image, gt = pic['image'], pic['gt']
        
        if isinstance(image, np.ndarray) and isinstance(gt, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1))
            gt = torch.from_numpy(np.array(gt, dtype=np.int32)).long()
            # gt = np.expand_dims(np.array(gt, dtype=np.int32), -1).transpose((2, 0, 1))
            # gt = torch.from_numpy(gt).long()
            # gt[gt == 255] = 0
            return image, gt