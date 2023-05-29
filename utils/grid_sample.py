import tensorflow as tf
import numpy as np
@tf.function
def grid_sample_2d(inp, grid):
    """
    Pytorch grid sample implementation in Tensorflow.
    :param inp: source from which we want to sample [B,H,W,C]
    :param grid: xy coordinates where to sample [B,Hc,Wc,2]
    :return: grid sampling from input
    """
    in_shape = tf.shape(inp)
    in_h = in_shape[1]
    in_w = in_shape[2]

    # Find interpolation sides
    i, j = grid[..., 1], grid[..., 0] # y-> H x-> W
    i = tf.cast(in_h - 1, grid.dtype) * (i + 1) / 2
    j = tf.cast(in_w - 1, grid.dtype) * (j + 1) / 2
    i_1 = tf.maximum(tf.cast(tf.floor(i), tf.int32), 0)
    i_2 = tf.minimum(i_1 + 1, in_h - 1)
    j_1 = tf.maximum(tf.cast(tf.floor(j), tf.int32), 0)
    j_2 = tf.minimum(j_1 + 1, in_w - 1)

    # Gather pixel values
    n_idx = tf.tile(tf.range(in_shape[0])[:, tf.newaxis, tf.newaxis], tf.concat([[1], tf.shape(i)[1:]], axis=0))
    q_11 = tf.gather_nd(inp, tf.stack([n_idx, i_1, j_1], axis=-1))
    q_12 = tf.gather_nd(inp, tf.stack([n_idx, i_1, j_2], axis=-1))
    q_21 = tf.gather_nd(inp, tf.stack([n_idx, i_2, j_1], axis=-1))
    q_22 = tf.gather_nd(inp, tf.stack([n_idx, i_2, j_2], axis=-1))

    # Interpolation coefficients
    di = tf.cast(i, inp.dtype) - tf.cast(i_1, inp.dtype)
    di = tf.expand_dims(di, -1)
    dj = tf.cast(j, inp.dtype) - tf.cast(j_1, inp.dtype)
    dj = tf.expand_dims(dj, -1)

    # Compute interpolations
    q_i1 = q_11 * (1 - di) + q_21 * di
    q_i2 = q_12 * (1 - di) + q_22 * di
    q_ij = q_i1 * (1 - dj) + q_i2 * dj

    return q_ij

def grid_sample_2d_np(inp, grid):
    """
    Pytorch grid sample implementation in Numpy.
    :param inp: source from which we want to sample [B,H,W,C]
    :param grid: xy coordinates where to sample [B,Hc,Wc,2]
    :return: grid sampling from input
    """
    in_shape = inp.shape
    in_h = in_shape[1]
    in_w = in_shape[2]

    # Find interpolation sides
    i, j = grid[..., 1].copy(), grid[..., 0].copy()
    i = (in_h - 1) * (i + 1) / 2
    j = (in_w - 1) * (j + 1) / 2
    i_1 = np.maximum(np.floor(i), 0).astype('int32')
    i_2 = np.minimum(i_1 + 1, in_h - 1)
    j_1 = np.maximum(np.floor(j), 0).astype('int32')
    j_2 = np.minimum(j_1 + 1, in_w - 1)

    q_11 = np.zeros([in_shape[0], grid.shape[1], grid.shape[2], in_shape[3]])
    q_12 = np.zeros([in_shape[0], grid.shape[1], grid.shape[2], in_shape[3]])
    q_21 = np.zeros([in_shape[0], grid.shape[1], grid.shape[2], in_shape[3]])
    q_22 = np.zeros([in_shape[0], grid.shape[1], grid.shape[2], in_shape[3]])

    for b in range(in_shape[0]):
        for x in range(grid.shape[2]):
            for y in range(grid.shape[1]):
                q_11[b, y, x, :] = inp[b, i_1[b, y, x], j_1[b, y, x], :]
                q_12[b, y, x, :] = inp[b, i_1[b, y, x], j_2[b, y, x], :]
                q_21[b, y, x, :] = inp[b, i_2[b, y, x], j_1[b, y, x], :]
                q_22[b, y, x, :] = inp[b, i_2[b, y, x], j_2[b, y, x], :]

    # Interpolation coefficients
    di = i - i_1
    di = np.expand_dims(di, -1)
    dj = j - j_1
    dj = np.expand_dims(dj, -1)

    # Compute interpolations
    q_i1 = q_11 * (1 - di) + q_21 * di
    q_i2 = q_12 * (1 - di) + q_22 * di
    q_ij = q_i1 * (1 - dj) + q_i2 * dj

    return q_ij.astype('float32')
