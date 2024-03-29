import os
import random
import numpy as np
import torch

from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader, dataloader
from torchvision import transforms
from PIL import Image

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

zero_pad = 256 * 3 - len(palette)

for i in range(zero_pad):
    palette.append(0)

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def make_dataset(quality, args, mode):
    # assert (quality == 'fine' and args.mode in ['train', 'val']) or \
    #        (quality == 'coarse' and args.mode in ['train', 'train_extra', 'val'])

    if quality == 'coarse':
        img_dir_name = 'leftImg8bit_trainextra' if args.mode == 'train_extra' else 'leftImg8bit_trainvaltest'
        mask_path = os.path.join(args.data_path, "datasets", 'gtCoarse', 'gtCoarse', args.mode)
        mask_postfix = '_gtCoarse_labelIds.png'
        
    else:
        img_dir_name = 'leftImg8bit_trainvaltest'
        mask_postfix = '_gtFine_labelIds.png'
        
        if mode == 'train':
            mask_path = os.path.join(args.data_path, "datasets", "cityscapes", 'gtFine_trainvaltest', 'gtFine', mode)
            img_path = os.path.join(args.data_path, "datasets", "cityscapes",img_dir_name, 'leftImg8bit', mode)
        
        elif mode == 'val' or mode == 'eval':
            mask_path = os.path.join(args.data_path, "datasets", "cityscapes",'gtFine_trainvaltest', 'gtFine', 'val')
            img_path = os.path.join(args.data_path, "datasets", "cityscapes", img_dir_name, 'leftImg8bit', 'val')
        
        assert os.listdir(img_path) == os.listdir(mask_path)
        
        categories = os.listdir(img_path)
        items = []
        for category in categories:
            category_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, category))]
            for c_item in category_items:
                pair = (os.path.join(img_path, category, c_item + '_leftImg8bit.png'), os.path.join(mask_path, category, c_item + mask_postfix))
                items.append(pair)

    return items

def preprocessing_transform(mode):
    return transforms.Compose([ToTensor(mode=mode)])

class msasppDataLoader(object):
    def __init__(self, args, mode):
        self.ignore_label = 255
        self.class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']
        
        set_seeds(seed=args.num_seed)
        if mode == 'train':
            train_sampler = None
            self.training_samples = DataLoadPreprocess(self.ignore_label, 'fine', args, mode, transform=preprocessing_transform(mode))
            self.data = DataLoader(self.training_samples, args.batch_size, shuffle=(train_sampler is None), 
                                   num_workers=args.num_threads, 
                                   pin_memory=True, 
                                   sampler=train_sampler)
        
        elif mode == 'val':
            self.validation_samples = DataLoadPreprocess(self.ignore_label, 'fine', args, mode, transform=preprocessing_transform(mode))
            self.data = DataLoader(self.validation_samples, args.batch_size, shuffle=False, 
                                   num_workers=args.num_threads, 
                                   pin_memory=True, 
                                   sampler=None)
        elif mode == 'eval':
            self.validation_samples = DataLoadPreprocess(self.ignore_label, 'fine', args, mode, transform=preprocessing_transform(mode))
            self.data = DataLoader(self.validation_samples, 1, shuffle=False, num_workers=1)
            
        
class DataLoadPreprocess(Dataset):
    def __init__(self, ignore_label, quality, args, mode, transform=None):
        self.mode = mode
        self.args = args
        self.pair_img = make_dataset(quality, args, mode)
            
        if len(self.pair_img) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        
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

    def __getitem__(self, index):
        img_path, gt_path = self.pair_img[index]
        data_name = img_path.split('/')[-1].split('_leftImg8bit.png')[0]
        
        image, gt = Image.open(img_path), Image.open(gt_path)
        gt = np.array(gt)
        gt_copy = gt.copy()
        
        for key, value in self.id_to_trainid.items():
            gt_copy[gt == key] = value
        gt = Image.fromarray(gt_copy.astype(np.uint8))
        
        rescaled_image, rescaled_gt = self.resize_random_crop(image, gt, self.args.input_height, self.args.input_width)
        rescaled_image = self.rotate_image(rescaled_image, angle=self.args.degree)
        
        rescaled_image = np.array(rescaled_image, dtype=np.float32) / 255.0
        rescaled_gt = np.array(rescaled_gt, dtype=np.float32)
        image, gt = self.train_preprocess(rescaled_image, rescaled_gt)
        
        # sample = {'image': image, 'gt': gt}
        sample = {'img_data': image, 'seg_label': gt}
        sample = self.transform(sample)
        
        if self.mode == 'eval':
            return data_name, sample
        
        return sample
            
        # elif self.mode == 'val':
        #     image = np.array(image, dtype=np.float32) / 255.0
        #     gt = np.array(gt, dtype=np.float32)
            
        #     sample = {'image': image, 'gt': gt}
        #     sample = self.transform(sample)
        #     return sample
        
    def rotate_image(self, img, angle, flag=Image.BILINEAR):
        result = transforms.RandomRotation(angle, resample=flag)(img)
        # result = img.rotate(angle, resample=flag)

        return result

    def resize_random_crop(self, image, gt, height, width):
        scaling = random.uniform(0.5, 2.0)
        scale_w, scale_h = [int(i*scaling) for i in image.size]
        resized_image = image.resize((scale_w, scale_h), Image.CUBIC)
        resized_gt = gt.resize((scale_w, scale_h), Image.NEAREST)
        
        i, j, h, w = transforms.RandomCrop.get_params(resized_image, output_size=(height, width))
        crop_image = F.crop(resized_image, i, j, h, w)
        crop_gt = F.crop(resized_gt, i, j, h//8, w//8)

        return crop_image, crop_gt
    
    def train_preprocess(self, image, gt):
        # Random horizontal flipping
        hflip = random.random()
        if hflip > 0.5:
            image = (image[:, ::-1, :]).copy()
            gt = (gt[:, ::-1]).copy()
            
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)
            
        return image, gt

    def augment_image(self, image):
        # gamma augmentation
        # gamma  = random.uniform(0.5, 2.0)
        # image_aug = image ** gamma
        
        # brightness augmentation
        brightness = random.uniform(-10, 10)
        image_aug = image * brightness
        
        # color augmentation
        # colors = np.random.uniform(0.9, 1.1, size=3)
        # white = np.ones((image.shape[0], image.shape[1]))
        # color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        # image_aug *= color_image
        # image_aug = np.clip(image_aug, 0, 1)
        
        return image_aug
    
    def __len__(self):
        return len(self.pair_img)

class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])
        
    def __call__(self, sample):
        image, gt = self.to_tensor(sample)
        image = self.normalize(image)
        
        return {'img_data': image, 'seg_label': gt}
    
    def to_tensor(self, pic):
        image, gt = pic['img_data'], pic['seg_label']
        
        if isinstance(image, np.ndarray) and isinstance(gt, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1))
            gt = torch.from_numpy(np.array(gt, dtype=np.int32)).long()

            return image, gt