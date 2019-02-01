import types
import random
import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance
import torch
import numpy as np
import torchvision.transforms.functional as tf

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class Lambda(object):
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ColorJitter(object):
    def __init__(self, brightness=0):
        self.brightness = brightness

    @staticmethod
    def get_params(brightness):
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        transform = self.get_params(self.brightness)

        return transform(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        return format_string


# class AdjustBrightness(object):
#     def __init__(self, brightness=0):
#         self.brightness = brightness
#
#     def __call__(self, img):
        # img = np.asarray(img, dtype=np.float32)
        # # random_gamma = torch.FloatTensor(512, 512, 3).uniform_(0.8, 1.2)
        # # img_aug1 = img ** random_gamma
        #
        # # random_brightness = torch.FloatTensor(512, 512, 3).uniform_(1 - self.brightness, 1 + self.brightness)
        # random_brightness = random.uniform(-1 * self.brightness, 1 * self.brightness)
        # img_aug = img * random_brightness
        #
        # # img_aug = np.asarray(img_aug, dtype=np.float32)
        # return img_aug

class AdjustBrightness(object):
    def __init__(self, brightness):
        self.brightness = brightness

    def __call__(self, img):
        return tf.adjust_brightness(img, random.uniform(-1 * self.brightness, self.brightness))