# Copyright 2020 Toyota Research Institute.  All rights reserved.

from math import pi
import numpy as np
import tensorflow as tf

def sample_homography(
        shape, perspective=True, scaling=True, rotation=True, translation=True,
        n_scales=100, n_angles=100, scaling_amplitude=0.1, perspective_amplitude=0.4,
        patch_ratio=0.8, max_angle=pi/4):
    """ Sample a random homography that includes perspective, scale, translation and rotation operations."""

    width = float(shape[1])
    hw_ratio = float(shape[0]) / float(shape[1])

    pts1 = np.stack([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], axis=0)
    pts2 = pts1.copy() * patch_ratio
    pts2[:,1] *= hw_ratio

    if perspective:

        perspective_amplitude_x = np.random.normal(0., perspective_amplitude/2, (2))
        perspective_amplitude_y = np.random.normal(0., hw_ratio * perspective_amplitude/2, (2))

        perspective_amplitude_x = np.clip(perspective_amplitude_x, -perspective_amplitude/2, perspective_amplitude/2)
        perspective_amplitude_y = np.clip(perspective_amplitude_y, hw_ratio * -perspective_amplitude/2, hw_ratio * perspective_amplitude/2)

        pts2[0,0] -= perspective_amplitude_x[1]
        pts2[0,1] -= perspective_amplitude_y[1]

        pts2[1,0] -= perspective_amplitude_x[0]
        pts2[1,1] += perspective_amplitude_y[1]

        pts2[2,0] += perspective_amplitude_x[1]
        pts2[2,1] -= perspective_amplitude_y[0]

        pts2[3,0] += perspective_amplitude_x[0]
        pts2[3,1] += perspective_amplitude_y[0]

    if scaling:

        random_scales = np.random.normal(1, scaling_amplitude/2, (n_scales))
        random_scales = np.clip(random_scales, 1-scaling_amplitude/2, 1+scaling_amplitude/2)

        scales = np.concatenate([[1.], random_scales], 0)
        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = np.expand_dims(pts2 - center, axis=0) * np.expand_dims(
                np.expand_dims(scales, 1), 1) + center
        valid = np.arange(n_scales)  # all scales are valid except scale=1
        idx = valid[np.random.randint(valid.shape[0])]
        pts2 = scaled[idx]

    if translation:
        t_min, t_max = np.min(pts2 - [-1., -hw_ratio], axis=0), np.min([1., hw_ratio] - pts2, axis=0)
        pts2 += np.expand_dims(np.stack([np.random.uniform(-t_min[0], t_max[0]),
                                         np.random.uniform(-t_min[1], t_max[1])]),
                               axis=0)

    if rotation:
        angles = np.linspace(-max_angle, max_angle, n_angles)
        angles = np.concatenate([[0.], angles], axis=0) 

        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul(
                np.tile(np.expand_dims(pts2 - center, axis=0), [n_angles+1, 1, 1]),
                rot_mat) + center

        valid = np.where(np.all((rotated >= [-1.,-hw_ratio]) & (rotated < [1.,hw_ratio]),
                                        axis=(1, 2)))[0]

        idx = valid[np.random.randint(valid.shape[0])]
        pts2 = rotated[idx]

    pts2[:,1] /= hw_ratio

    def ax(p, q): return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]
    def ay(p, q): return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = np.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    p_mat = np.transpose(np.stack(
        [[pts2[i][j] for i in range(4) for j in range(2)]], axis=0))

    homography = np.matmul(np.linalg.pinv(a_mat), p_mat).squeeze()
    homography = np.concatenate([homography, [1.]]).reshape(3,3)
    return homography
@tf.function
def warp_homography(sources, homography):
    """Warp features given a homography

    Parameters
    ----------
    sources: torch.tensor (1,H,W,2)
        Keypoint vector.
    homography: torch.Tensor (3,3)
        Homography.

    Returns
    -------
    warped_sources: torch.tensor (1,H,W,2)
        Warped feature vector.
    """
    _, H, W, _ = sources.shape
    warped_sources = tf.squeeze(tf.identity(sources))
    warped_sources = tf.reshape(warped_sources,(-1,2))
    warped_sources = tf.math.add(homography[:,2], tf.matmul(warped_sources, tf.transpose(homography[:,:2])))

    warped_sources = tf.multiply(warped_sources,1./tf.expand_dims(warped_sources[:,2],1))
    warped_sources = tf.reshape(warped_sources[:,:2],[1,H,W,2])
    return warped_sources