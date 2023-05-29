# Copyright 2020 Toyota Research Institute.  All rights reserved.

import tensorflow as tf

@tf.function
def meshgrid(B, H, W, normalized=False):
    """Create mesh-grid given batch size, height and width dimensions.

    Parameters
    ----------
    B: int
        Batch size
    H: int
        Grid Height
    W: int
        Batch size
    dtype: torch.dtype
        Tensor dtype
    device: str
        Tensor device
    normalized: bool
        Normalized image coordinates or integer-grid.

    Returns
    -------
    xs: torch.Tensor
        Batched mesh-grid x-coordinates (BHW).
    ys: torch.Tensor
        Batched mesh-grid y-coordinates (BHW).
    """
    if normalized:
        xs = tf.cast(tf.linspace(-1, 1, W), dtype = tf.float32)
        ys = tf.cast(tf.linspace(-1, 1, H), dtype = tf.float32)
    else:
        xs = tf.cast(tf.linspace(0, W - 1, W), dtype = tf.float32)
        ys = tf.cast(tf.linspace(0, H - 1, H), dtype = tf.float32)
    xs, ys = tf.meshgrid(xs, ys)
    return tf.repeat(tf.expand_dims(xs, 0),B,axis=0), tf.repeat(tf.expand_dims(ys, 0),B,axis=0)


@tf.function
def image_grid(B, H, W, ones=True, normalized=False):
    """Create an image mesh grid with shape B3HW given image shape BHW

    Parameters
    ----------
    B: int
        Batch size
    H: int
        Grid Height
    W: int
        Batch size
    dtype: str
        Tensor dtype
    device: str
        Tensor device
    ones : bool
        Use (x, y, 1) coordinates
    normalized: bool
        Normalized image coordinates or integer-grid.

    Returns
    -------
    grid: torch.Tensor
        Mesh-grid for the corresponding image shape (B3HW)
    """
    xs, ys = meshgrid(B, H, W, normalized=normalized)
    coords = [xs, ys]
    if ones:
        coords.append(tf.ones_like(xs))  # BHW
    grid = tf.stack(coords,  axis=-1)  # BHW3
    return grid
