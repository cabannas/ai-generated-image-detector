"""
This code is part of an adaptation/modification from the original project available at:
https://github.com/peterwang512/CNNDetection
The original code was created by Wang et al. and is used here under the terms of the license
specified in the original project's repository. Any use of this adapted/modified code
must respect the terms of such license.
Adaptations and modifications made by: Daniel Cabanas Gonzalez
Modification date: 08/04/2024
"""

import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CustomResizeTransform:
    def __init__(self, opt):
        self.opt = opt

    def __call__(self, img):
        return custom_resize(img, self.opt)

class IdentityTransform:
    def __call__(self, img):
        return img

class DataAugmentTransform:
    def __init__(self, opt):
        self.opt = opt

    def __call__(self, img):
        return data_augment(img, self.opt)

def dataset_folder(opt, root):
    if opt.mode == 'binary':
        return binary_dataset(opt, root)
    elif opt.mode == 'filename':
        return FileNameDataset(opt, root)
    else:
        raise ValueError('opt.mode needs to be binary or filename.')

def binary_dataset(opt, root):
    crop_func = transforms.RandomCrop(opt.cropSize) if opt.isTrain else transforms.CenterCrop(opt.cropSize)
    flip_func = transforms.RandomHorizontalFlip() if opt.isTrain and not opt.no_flip else IdentityTransform()
    rz_func = CustomResizeTransform(opt) if not opt.isTrain or not opt.no_resize else IdentityTransform()

    dset = datasets.ImageFolder(
        root,
        transforms.Compose([
            rz_func,
            DataAugmentTransform(opt),
            crop_func,
            flip_func,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )
    return dset

class FileNameDataset(datasets.ImageFolder):
    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root, transform=transforms.Compose([
            CustomResizeTransform(opt),
            DataAugmentTransform(opt),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

def data_augment(img, opt):
    img = np.array(img)
    if random() < opt.blur_prob:
        img = gaussian_blur(img, sample_continuous(opt.blur_sig))
    if random() < opt.jpg_prob:
        img = jpeg_from_key(img, sample_discrete(opt.jpg_qual), sample_discrete(opt.jpg_method))
    return Image.fromarray(img)

def gaussian_blur(img, sigma):
    return np.stack([gaussian_filter(img[:, :, channel], sigma=sigma) for channel in range(3)], axis=-1)

def cv2_jpg(img, compress_val):
    img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    if result:
        img = cv2.imdecode(encimg, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def pil_jpg(img, compress_val):
    img_pil = Image.fromarray(img)
    buffer = BytesIO()
    img_pil.save(buffer, format="JPEG", quality=compress_val)
    img = np.array(Image.open(buffer))
    return img

jpeg_dict = {
    'cv2': cv2_jpg,
    'pil': pil_jpg,
}

def jpeg_from_key(img, compress_val, key):
    return jpeg_dict[key](img, compress_val)

def custom_resize(img, opt):
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[sample_discrete(opt.rz_interp)])

rz_dict = {
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'lanczos': Image.LANCZOS,
    'nearest': Image.NEAREST,
}

def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        return random() * (s[1] - s[0]) + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")

def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)