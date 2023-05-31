from PIL import Image
import glob
import tensorflow as tf
import tensorflow_addons as tfa
import datasets.augmentations as aug
from utils.grid_sample import grid_sample_2d
import numpy as np
import random
from utils.image import image_grid
def random_augmentations(image,kernel_sizes = [0, 1, 3, 5]):
    image = image + tf.random.normal(image.shape, 0, 3) * tf.random.normal([1], 0, 1)
    image = tf.image.random_brightness(image,0.5)
    image = tf.image.random_contrast(image,0.5,1.5)
    image = tf.image.random_saturation(image,0.8,1.2)
    image = tf.image.random_hue(image, max_delta=0.05)
    rand_index = np.random.randint(4)
    kernel_size = kernel_sizes[rand_index]
    if kernel_size > 0:
        image = tfa.image.gaussian_filter2d(image, (kernel_size, kernel_size))
    image = tf.clip_by_value(image,0,255)
    image = tf.scalar_mul(1/127.5, image)
    image = tf.add(image, -1.)
    return image
def augment_sample(image, homography, shape):
    source_grid = image_grid(1, shape[0], shape[1], ones=False, normalized=True)
    source_warped = aug.warp_homography(source_grid, homography)
    source_img = grid_sample_2d(image, source_warped)

    target_img = image
    target_img = random_augmentations(target_img)
    source_img = random_augmentations(source_img)
    del source_grid, source_warped
    return target_img, source_img, tf.expand_dims(homography,0)


class SimpleDataLoader():
    def __init__(self, root_dir, quantize = False, batch_size=4, shape=(240, 320), patch_ratio=0.7, scaling_amplitude=0.2,
                 max_angle=3.14 / 2):
        self.root_dir = root_dir
        self.shape = shape
        self.files = []
        self.patch_ratio = patch_ratio
        self.scaling_amplitude = scaling_amplitude
        self.max_angle = max_angle
        self.jitter_paramters = [0.5, 0.5, 0.2, 0.05]
        self.batch_size = batch_size
        self.quantize = quantize

        for filename in glob.glob(root_dir + '/*.jpg'):
            self.files.append(filename)

    def __getitem__(self, batch_idx):
        homographies = []
        target_imgs = []
        source_imgs = []
        for idx in range(batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size):
            filename = self.files[idx]
            image = self._read_rgb_file(filename)
            if image.mode == 'L':
                image = image.convert('RGB')
            homography = tf.cast(aug.sample_homography(self.shape,
                                                       patch_ratio=self.patch_ratio,
                                                       scaling_amplitude=self.scaling_amplitude,
                                                       max_angle=self.max_angle), tf.float32)

            image = image.resize(self.shape[::-1])
            image = self.preprocessing(image)


            target_img, source_img, homography = augment_sample(image, homography, self.shape)
            if self.quantize:
                target_img = target_img*127.5
                source_img = source_img*127.5
            source_imgs.append(source_img)
            target_imgs.append(target_img)
            homographies.append(homography)
        homographies = tf.concat(homographies, axis=0)
        target_imgs = tf.concat(target_imgs, axis=0)
        source_imgs = tf.concat(source_imgs, axis=0)

        return {'image': target_imgs,
                'image_aug': source_imgs,
                'homography': homographies}

    def shuffle(self):
        random.shuffle(self.files)

    def __len__(self):
        return len(self.files) // self.batch_size

    def preprocessing(self, image):
        image = tf.expand_dims(tf.convert_to_tensor(image, dtype=tf.float32), 0)
        image = tf.image.random_flip_left_right(image)
        return tf.image.random_flip_up_down(image)

    def _read_rgb_file(self, filename):
        return Image.open(filename)

