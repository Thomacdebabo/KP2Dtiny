# Copyright 2020 Toyota Research Institute.  All rights reserved.

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


class PatchesDataset:
    """
    HPatches dataset class.
    Note: output_shape = (H, W)
    Note: this returns Pytorch tensors, resized to output_shape (if specified)
    Note: the homography will be adjusted according to output_shape.

    Parameters
    ----------
    root_dir : str
        Path to the dataset
    use_color : bool
        Return color images or convert to grayscale.
    data_transform : Function
        Transformations applied to the sample
    output_shape: tuple
        If specified, the images and homographies will be resized to the desired shape.
    type: str
        Dataset subset to return from ['i', 'v', 'all']: 
        i - illumination sequences
        v - viewpoint sequences
        all - all sequences
    """
    def __init__(self, root_dir, use_color=True, data_transform=None, output_shape=None, type='all',quantize=False, mode='quantized_default'):

        super().__init__()
        self.type = type
        self.root_dir = root_dir
        self.data_transform = data_transform
        self.output_shape = output_shape[::-1] # convert from H,W to W,H
        self.use_color = use_color
        self.quantize = quantize

        base_path = Path(root_dir)
        folder_paths = [x for x in base_path.iterdir() if x.is_dir()]
        image_paths = []
        warped_image_paths = []
        homographies = []
        for path in folder_paths:
            if self.type == 'i' and path.stem[0] != 'i':
                continue
            if self.type == 'v' and path.stem[0] != 'v':
                continue
            num_images = 5
            file_ext = '.ppm'
            for i in range(2, 2 + num_images):
                image_paths.append(str(Path(path, "1" + file_ext)))
                warped_image_paths.append(str(Path(path, str(i) + file_ext)))
                homographies.append(np.loadtxt(str(Path(path, "H_1_" + str(i)))))
        self.files = {'image_paths': image_paths, 'warped_image_paths': warped_image_paths, 'homography': homographies}

    @staticmethod
    def scale_homography(homography, original_scale, new_scale, pre):
        scales = np.divide(new_scale, original_scale)
        if pre:
            s = np.diag(np.append(scales, 1.))
            homography = np.matmul(s, homography)
        else:
            sinv = np.diag(np.append(1. / scales, 1.))
            homography = np.matmul(homography, sinv)
        return homography

    def __len__(self):
        return len(self.files['image_paths'])

    def __getitem__(self, idx):

        image = self._read_rgb_file(self.files['image_paths'][idx])
        warped_image = self._read_rgb_file(self.files['warped_image_paths'][idx])

        image_orig_shape = np.array(image).shape[:2][::-1]
        warped_image_orig_shape = np.array(warped_image).shape[:2][::-1]

        image = image.resize(self.output_shape)
        warped_image = warped_image.resize(self.output_shape)

        image = np.array(image, dtype='float32')/127.5 - 1.
        warped_image = np.array(warped_image, dtype='float32')/127.5 - 1.
        homography = np.array(self.files['homography'][idx])
        if self.quantize:
            image = image*127.5
            warped_image = warped_image*127.5
        image = np.expand_dims(image, 0)
        warped_image = np.expand_dims(warped_image, 0)
        homography = np.expand_dims(homography, 0)

        sample = {'image': image, 'image_aug': warped_image, 'homography': homography, 'index' : idx}

        # Apply transformations
        if self.output_shape is not None:
            sample['homography'] = self.scale_homography(sample['homography'],
                                                         image_orig_shape,
                                                         self.output_shape,
                                                         pre=False)
            sample['homography'] = self.scale_homography(sample['homography'],
                                                         warped_image_orig_shape,
                                                         self.output_shape,
                                                         pre=True)
        if self.data_transform:
            sample = self.data_transform(sample)
        return sample

    def _read_rgb_file(self, filename):
        return Image.open(filename)