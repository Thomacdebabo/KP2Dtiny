# Copyright 2020 Toyota Research Institute.  All rights reserved.

import glob
from utils.logging import timing
from PIL import Image
import tensorflow as tf
import datasets.augmentations as aug
from utils.utils import grid_sample_2d
from utils import image as img_utils
import tensorflow_addons as tfa
class COCOLoader(tf.keras.utils.Sequence):
    """
    Coco dataset class.

    Parameters
    ----------
    root_dir : str
        Path to the dataset
    data_transform : Function
        Transformations applied to the sample
    """
    def __init__(self, root_dir, batch_size=16, shape= (240,320), patch_ratio = 0.7, scaling_amplitude = 0.2, max_angle = 3.14 / 2):

        super().__init__()
        self.root_dir = root_dir
        self.shape = shape
        self.files=[]
        self.patch_ratio = patch_ratio
        self.scaling_amplitude = scaling_amplitude
        self.max_angle = max_angle
        self.jitter_paramters=[0.5, 0.5, 0.2, 0.05]
        self.batch_size = batch_size
        self.preprocessing = tf.keras.Sequential([
            tf.keras.layers.Resizing(self.shape[0], self.shape[1]),
            tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=None),
            tf.keras.layers.RandomRotation((-0.5, 0.8)),
            tf.keras.layers.RandomTranslation(0.2, 0.2)
        ])
        self.noise = tf.keras.layers.GaussianNoise(0.01)
        for filename in glob.glob(root_dir + '/*.jpg'):
            self.files.append(filename)

    def __len__(self):
        return len(self.files)//self.batch_size
    def _read_rgb_file(self, filename):
        return Image.open(filename)

    def __getitem__(self, batch_idx):
        homographies = []
        target_imgs = []
        source_imgs = []
        for idx in range(batch_idx*self.batch_size, (batch_idx+1)*self.batch_size):
            filename = self.files[idx]
            image = self._read_rgb_file(filename)
            if image.mode == 'L':
                image = image.convert('RGB')
            image = tf.keras.utils.img_to_array(image)
            image = self.preprocessing(image)
            target_img, source_img, homography = self.augment_sample(image)
            source_imgs.append(source_img)
            target_imgs.append(target_img)
            homographies.append(homography)

        homographies = tf.stack(homographies, axis = 0)
        target_imgs = tf.stack(target_imgs, axis = 0)
        source_imgs = tf.stack(source_imgs, axis = 0)

        return {'image': target_imgs,
                'image_aug': source_imgs,
                'homography': homographies}
    @staticmethod
    @tf.function
    def add_noise(img, mode="gaussian", percent=0.02):
        """Add image noise

        Parameters
        ----------
        image : np.array
            Input image
        mode: str
            Type of noise, from ['gaussian','salt','pepper','s&p']
        percent: float
            Percentage image points to add noise to.
        Returns
        -------
        image : np.array
            Image plus noise.
        """
        original_dtype = img.dtype

        mean = 0
        var = 0.01
        sigma = var * 0.5

        gauss = tf.random.normal(img.shape, mean, sigma)

        img = tf.clip_by_value(tf.cast(img, tf.float32) + gauss, -1., 1.)
        return img

    @tf.function
    def non_spatial_augmentation(self, img_warp_ori):
        """ Apply non-spatial augmentation to an image (jittering, color swap, convert to gray scale, Gaussian blur)."""


        # color_augmentation = transforms.ColorJitter()
        # augment_image = color_augmentation.get_params(brightness=[max(0, 1 - brightness), 1 + brightness],
        #                                               contrast=[max(0, 1 - contrast), 1 + contrast],
        #                                               saturation=[max(0, 1 - saturation), 1 + saturation],
        #                                               hue=[-hue, hue])

        kernel_sizes = tf.constant([0, 1, 3, 5])

        # img_warp_sub = torchvision.transforms.functional.to_pil_image(img_warp_sub)

        tf_img = tf.identity(img_warp_ori)

        rand_index = tf.random.uniform(shape=(), minval=0, maxval=len(kernel_sizes), dtype=tf.int32)
        kernel_size = kernel_sizes[rand_index]
        if kernel_size > 0:
            tf_img = tfa.image.gaussian_filter2d(tf_img, (kernel_size, kernel_size))

        # img_warp_sub = Image.fromarray(img_warp_sub_np)
        # img_warp_sub = color_augmentation(img_warp_sub)
        #
        # img_warp_sub = torchvision.transforms.functional.to_tensor(img_warp_sub).to(img_warp_ori.device)

        # if np.random.rand() > 0.5:
        #     tf_img = tf.image.random_saturation(tf_img, 0,1)
        #     tf_img = tf.image.random_brightness(tf_img, 0.1)
        #     tf_img = tf.image.random_contrast(tf_img, 0,0.1)

        tf_img = tf_img / 127.5 - 1.0
        #tf_img = self.add_noise(tf_img)
        return tf_img

    @tf.function
    def augment_sample(self, image):
        homography = tf.cast(aug.sample_homography(self.shape,
                                                   patch_ratio=self.patch_ratio,
                                                   scaling_amplitude=self.scaling_amplitude,
                                                   max_angle=self.max_angle), tf.float32)
        source_grid = tf.identity(img_utils.image_grid(1, self.shape[0], self.shape[1], ones=False, normalized=True))

        source_warped = aug.warp_homography(source_grid, homography)
        source_img = grid_sample_2d(tf.expand_dims(image, 0), source_warped)[0]

        target_img = image
        target_img = self.non_spatial_augmentation(target_img)
        source_img = self.non_spatial_augmentation(source_img)
        return target_img, source_img, homography


    def ha_augment_sample(self,image_batch):
        B,H,W,_ = image_batch.shape
        homographies = []
        target_imgs = []
        source_imgs = []
        for i in range(B):
            homography = tf.cast(aug.sample_homography([H, W],
                                                       patch_ratio=self.patch_ratio,
                                                       scaling_amplitude=self.scaling_amplitude,
                                                       max_angle=self.max_angle), tf.float32)
            source_grid = tf.identity(img_utils.image_grid(1, H, W, ones=False, normalized=True))



            source_warped = aug.warp_homography(source_grid, homography)
            source_img = grid_sample_2d(tf.expand_dims(image_batch[i],0), source_warped)[0]

            homographies.append(homography)


            to_gray = False
            # if np.random.rand() > 0.5:
            #     to_gray = True
            target_img = image_batch[i]
            target_img = self.non_spatial_augmentation(target_img)
            source_img = self.non_spatial_augmentation(source_img)
            source_imgs.append(source_img)
            target_imgs.append(target_img)
        homographies = tf.stack(homographies, axis = 0)
        target_imgs = tf.stack(target_imgs, axis = 0)
        source_imgs = tf.stack(source_imgs, axis = 0)