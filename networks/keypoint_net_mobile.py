# Copyright 2020 Toyota Research Institute.  All rights reserved.
import keras.layers
import tensorflow as tf
from utils.grid_sample import grid_sample_2d
from utils.image import image_grid
from tensorflow.keras import layers

import tensorflow_model_optimization as tfmot
class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
  """QuantizeConfig which does not quantize any part of the layer."""
  def get_weights_and_quantizers(self, layer):
      return []

  def get_activations_and_quantizers(self, layer):
      return []

  def set_quantize_weights(self, layer, quantize_weights):
      pass
  def get_config(self):
      return {}
  def get_output_quantizers(self, layer):
      return []
  def set_quantize_activations(self, layer, quantize_activations):
      pass


class KeypointNetMobile(tf.keras.Model):
    """
    Keypoint detection network.

    Parameters
    ----------
    do_cross: bool
        Predict keypoints outside cell borders.
    downsample: int
        Amount of pooling layers
    shape: tuple
        Input resolution (height, width)
    QAT: bool
        Create model with quantize aware training enabled.
    legacy: bool
        Use legacy architecture, main difference is that channel dimension does not increase before pooling
    kwargs : dict
        Extra parameters to configure KeypointNetRaw
    """

    def __init__(self, do_cross=True, downsample=2, shape=(88,88), QAT = False, legacy = False, **kwargs):
        super().__init__()
        self.downsample = downsample
        self.cell = pow(2, self.downsample)
        self.training = False
        if downsample == 1:
            self.keypoint_net_raw = KeypointNetRawV0(downsample=downsample, **kwargs)
        elif downsample == 0:
            self.keypoint_net_raw = KeypointNetRawNano(**kwargs)
        else:
            if legacy and QAT:
                self.keypoint_net_raw = KeypointNetRawLegacy(downsample=downsample, **kwargs).quantize_model(shape)
            elif legacy:
                self.keypoint_net_raw = KeypointNetRawLegacy(downsample=downsample, **kwargs)
            elif QAT:
                self.keypoint_net_raw = KeypointNetRaw(downsample=downsample,**kwargs).quantize_model(shape)
            else:
                self.keypoint_net_raw = KeypointNetRaw(downsample=downsample,**kwargs)
        self.cross_ratio = 2.0
        self.do_cross = do_cross

        if self.do_cross is False:
            self.cross_ratio = 1.0

    def set_trainable(self, train):
        self.training = train
        self.trainable = train
        self.keypoint_net_raw.trainable = train

        self.compile()
        self.keypoint_net_raw.compile()

    @tf.function
    def call(self, x, training=False):
        """
        Processes a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input images (B, 3, H, W)

        Returns
        -------
        score : torch.Tensor
            Score map (B, 1, H_out, W_out)
        coord: torch.Tensor
            Keypoint coordinates (B, 2, H_out, W_out)
        feat: torch.Tensor
            Keypoint descriptors (B, 256, H_out, W_out)
        """
        B, H, W,_ = x.shape

        score, center_shift, feat = self.keypoint_net_raw(x,training=training)

        B,  Hc, Wc,_ = score.shape

        border_mask = self.create_border_mask(B, Hc, Wc,1)
        score = score * border_mask

        step = (self.cell-1) / 2.

        center_base = image_grid(B, Hc, Wc, ones=False, normalized=False) * self.cell + step
        coord_un = center_base + center_shift * self.cross_ratio * step

        coordx = tf.clip_by_value(coord_un[:, :, :, 0], 0, W - 1)
        coordy = tf.clip_by_value(coord_un[:, :, :, 1], 0, H - 1)
        coord = tf.stack([coordx, coordy], axis=-1)

        if self.training is False:
            coord_norm = tf.identity(coord)
            coordx_norm = (coord_norm[:, :, :, 0] / (float(W - 1) / 2.)) - 1.
            coordy_norm = (coord_norm[:, :, :, 1] / (float(H - 1) / 2.)) - 1.
            coord_norm = tf.stack([coordx_norm, coordy_norm], axis=-1)

            feat = grid_sample_2d(feat, coord_norm)

            dn = tf.norm(feat, ord='euclidean', axis=-1)  # Compute the norm.
            feat = tf.math.divide(feat, tf.expand_dims(dn, -1))  # Divide by norm to normalize.

        del border_mask, center_base, step
        return score, coord, feat
    def quantize(self, H, W):
        self.keypoint_net_raw = self.keypoint_net_raw.quantize_model(H,W)

    # @tf.function
    def create_border_mask(self, B,Hc, Wc,C):
        m = tf.ones([B, Hc-2, Wc-2,C])
        m_s = tf.zeros([B, Hc - 2, 1,C])
        m = tf.concat([m_s, m, m_s], axis=2)
        m_top = tf.zeros([B, 1, Wc,C])
        border_mask = tf.concat([m_top, m, m_top], axis=1)
        return border_mask

class SubPixel(tf.keras.layers.Layer):
    def __init__(self,n,**kwargs):
        super(SubPixel, self).__init__()
        self._config = {'n': n}
        self.up = layers.Lambda(lambda x: tf.nn.depth_to_space(x, n))

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config

    # @tf.function
    def call(self, inputs, training=False):
        return self.up(inputs,training=training)
    def sequential(self):
        return [self.up]

class Conv2dBNRelu(tf.keras.layers.Layer):
    def __init__(self,c, use_bias,bn_momentum, use_leaky_relu, _name,**kwargs):
        super(Conv2dBNRelu, self).__init__()
        self._config = {'c':c, 'use_bias': use_bias, 'bn_momentum':bn_momentum, 'use_leaky_relu': use_leaky_relu, '_name':_name}


        self.conv2d = layers.Conv2D(c, kernel_size=3, strides=(1, 1), padding="same", use_bias=use_bias, name=_name)
        self.bn = layers.BatchNormalization(momentum=bn_momentum)
        if use_leaky_relu:
            self.relu = layers.LeakyReLU()
        else:
            self.relu = layers.ReLU()

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config

    # @tf.function
    def call(self, inputs, training=False):
        x = self.conv2d(inputs,training=training)
        x = self.bn(x,training=training)
        return self.relu(x,training=training)

    def sequential(self):
        return [self.conv2d, self.bn, self.relu]

class SimpleConvBlock(tf.keras.layers.Layer):

    def __init__(self,ca,cb, use_bias,bn_momentum, use_leaky_relu, idx, dropout,**kwargs):
        super(SimpleConvBlock, self).__init__()
        self._config = {'ca': ca,'cb': cb, 'use_bias': use_bias, 'bn_momentum': bn_momentum, 'use_leaky_relu': use_leaky_relu,
                        'idx': idx,'dropout':dropout}

        self.conva = Conv2dBNRelu(ca,use_bias,bn_momentum,use_leaky_relu, "Conv{}a".format(idx))
        self.convb = Conv2dBNRelu(cb,use_bias,bn_momentum,use_leaky_relu, "Conv{}b".format(idx))
        self.use_dropout = dropout
        if self.use_dropout:
            self.dropout = layers.SpatialDropout2D(0.2)

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config
    # @tf.function
    def call(self, inputs, training=False):
        x = self.conva(inputs,training=training)
        x = self.convb(x,training=training)
        if self.use_dropout:
            x = self.dropout(x,training=training)
        return x
    def sequential(self):
        if self.use_dropout:
            return [*self.conva.sequential(), *self.convb.sequential(), self.dropout]
        else:
            return [*self.conva.sequential(), *self.convb.sequential()]

class SimpleConvBlockOut(tf.keras.layers.Layer):

    def __init__(self,ca,cb, use_bias,bn_momentum, use_leaky_relu, idx, dropout, activation,**kwargs):
        super(SimpleConvBlockOut, self).__init__()
        self._config = {'ca': ca,'cb': cb, 'use_bias': use_bias, 'bn_momentum': bn_momentum, 'use_leaky_relu': use_leaky_relu,
                        'idx': idx,'dropout':dropout, 'activation':activation}

        self.conva = Conv2dBNRelu(ca,use_bias,bn_momentum,use_leaky_relu,"Conv{}a".format(idx))
        self.convb = layers.Conv2D(cb, kernel_size=3, strides=(1, 1), padding="same", use_bias=True,name= "Conv{}b".format(idx), activation=activation)
        self.use_dropout = dropout
        if self.use_dropout:
            self.dropout = layers.SpatialDropout2D(0.2)

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config

    def call(self, inputs, training=False):
        x = self.conva(inputs,training=training)
        if self.use_dropout:
            x = self.dropout(x,training=training)
        x = self.convb(x,training=training)

        return x

    def sequential(self):
        if self.use_dropout:
            return [*self.conva.sequential(), self.convb, self.dropout]
        else:
            return [*self.conva.sequential(), self.convb]


class KeypointNetRaw(tf.keras.Model):
    """
    Keypoint detection network without post processing, designed to run on the coral edge tpu.

    Parameters
    ----------
    do_upsample: bool
        Upsample desnse descriptor map.
    with_drop : bool
        Use dropout.
    nfeatures: int
        Predict keypoints outside cell borders.
    model_params: tuple
        Tuple containing 5 values to set the channel dimensions (c1 to c5)
    downsample: int
        Amount of pooling layers, can be 1 to 3.
    use_leaky_relu: bool
        If true, leaky relu is used as activation, else normal relu.
    use_subpixel: bool
        Set if subpixel convolution is enabled
    large_feat: bool
        If true, the feature is configured to use c4 as channel dimension before concatenation, if false c3 is used.

    kwargs : dict
        Extra parameters
    """

    def __init__(self, do_upsample=True, with_drop=False, nfeatures=64, model_params=(16, 32, 32, 64, 64, 32), downsample=3, use_leaky_relu=False, use_subpixel=False, **kwargs):
        super(KeypointNetRaw, self).__init__()
        assert len(model_params) == 6, "model_params has to contain 6 parameters"
        print("KeypointNetRaw:")
        print("Dropout:",with_drop)
        print("Leaky Relu:", use_leaky_relu)
        print("model params:", model_params)
        self.with_drop = with_drop
        self.do_upsample = do_upsample
        self.n_features = nfeatures
        self.downsample = downsample
        self.bn_momentum = 0.9
        self.use_subpixel = use_subpixel

        c1, c2, c3, c4, c5, d1 = model_params

        if self.use_subpixel:
            d1 = d1*4

        use_bias=False

        self.conv1 = SimpleConvBlock(c1,c2,use_bias,self.bn_momentum,use_leaky_relu, 1, with_drop)
        self.conv2 = SimpleConvBlock(c2,c3,use_bias,self.bn_momentum,use_leaky_relu, 2, with_drop)
        self.conv3 = SimpleConvBlock(c3,c4,use_bias,self.bn_momentum,use_leaky_relu, 3, with_drop)
        self.conv4 = SimpleConvBlock(c4,c4,use_bias,self.bn_momentum,use_leaky_relu, 4, with_drop)

        self.convD = SimpleConvBlockOut(c4,1, use_bias,self.bn_momentum, use_leaky_relu, 'D', with_drop, 'sigmoid')
        self.convP = SimpleConvBlockOut(c4,2, use_bias,self.bn_momentum, use_leaky_relu, 'P', with_drop, 'tanh')

        self.convFa = Conv2dBNRelu(c4,use_bias,self.bn_momentum,use_leaky_relu, "convFa")
        self.convFb = Conv2dBNRelu(d1,use_bias,self.bn_momentum,use_leaky_relu, "convFb")

        self.convF = SimpleConvBlockOut(c5,nfeatures, use_bias,self.bn_momentum, use_leaky_relu, 'Fa', with_drop, None)

        self.pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
        self.pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
        self.pool_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")

        if self.do_upsample:
            if use_subpixel:
                self.subpixel = SubPixel(2)
            else:
                self.up = tf.keras.layers.UpSampling2D(size=(2,2))
                #self.convup = SimpleConvBlock(c3,c3, use_bias, self.bn_momentum, use_leaky_relu, "u", with_drop)
                self.convup = Conv2dBNRelu(d1,use_bias,self.bn_momentum,use_leaky_relu, "u")
                #self.convup = SimpleConvBlock(c4,c4, use_bias, self.bn_momentum, use_leaky_relu, "u", with_drop)

            self.concat = tf.keras.layers.Concatenate(axis=-1) # output is c4*2

    @tf.function
    def call(self, inputs, training=False):
        """
        Processes a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input images (B, 3, H, W)

        Returns
        -------
        score : torch.Tensor
            Score map (B, 1, H_out, W_out)
        coord: torch.Tensor
            Keypoint coordinates (B, 2, H_out, W_out)
        feat: torch.Tensor
            Keypoint descriptors (B, 256, H_out, W_out)
        """

        x = self.conv1(inputs)
        if self.downsample >= 2:
            x = self.pool_1(x, training=training)
        x = self.conv2(x)
        if self.downsample >= 3:
            x = self.pool_2(x, training=training)
        skip = self.conv3(x)
        if self.downsample >= 1:
            x = self.pool_3(skip, training=training)
        x = self.conv4(x)

        score = self.convD(x, training=training)
        center_shift = self.convP(x, training=training)

        feat = self.convFa(x, training=training)
        feat = self.convFb(feat, training=training)

        if self.do_upsample:
            if self.use_subpixel:
                feat = self.subpixel(feat, training=training)

            else:
                feat = self.up(feat, training=training)
                feat = self.convup(feat, training=training)
            feat = self.concat([feat, skip])
        feat = self.convF(feat, training=training)

        return score, center_shift, feat

    def create_functional_model_deprecated(self, H, W):
        training = False
        inputs = tf.keras.Input(shape=(H,W,3))

        x = self.conv1.sequential()(inputs)
        if self.downsample >= 2:
            x = self.pool_1(x, training=training)
        x = self.conv2.sequential()(x)
        if self.downsample >= 3:
            x = self.pool_2(x, training=training)
        skip = self.conv3.sequential()(x)
        if self.downsample >= 1:
            x = self.pool_3(skip, training=training)
        x = self.conv4.sequential()(x)

        score = self.convD.sequential()(x, training=training)
        center_shift = self.convP.sequential()(x, training=training)

        feat = self.convFa.sequential()(x, training=training)
        feat = self.convFb.sequential()(feat, training=training)
        if self.do_upsample:
            if self.use_subpixel:
                feat = self.subpixel.sequential()(feat, training = training)
            else:
                feat = self.up(feat, training=training)
                feat = self.convup.sequential()(feat, training=training)
            feat = self.concat([feat, skip])
        feat = self.convF.sequential()(feat, training=training)

        return tf.keras.Model(inputs=inputs, outputs =[score, center_shift, feat])
    def create_functional_model(self, shape):
        training = False
        inputs = tf.keras.Input(shape=(shape + (3,)))
        seq = []
        seq.extend(self.conv1.sequential())
        if self.downsample >= 2:
            seq.append(self.pool_1)
        seq.extend(self.conv2.sequential())
        if self.downsample >= 3:
            seq.append(self.pool_2)
        seq.extend(self.conv3.sequential())

        seq_2 = []
        if self.downsample >= 1:
            seq_2.append(self.pool_3)
        seq_2.extend(self.conv4.sequential())

        score_layers = self.convD.sequential()
        center_shift_layers = self.convP.sequential()

        feat_layers = self.convFa.sequential()
        feat_layers.extend(self.convFb.sequential())
        if self.do_upsample:
            if self.use_subpixel:
                feat_layers.extend(self.subpixel.sequential())
            else:
                feat_layers.append(self.up)
                feat_layers.extend(self.convup.sequential())

        feat_end = self.convF.sequential()
        x = inputs
        for s in seq:
            x = s(x)
        skip = x
        for s2 in seq_2:
            x = s2(x)
        score = x
        for s3 in score_layers:
            score = s3(score)
        center_shift = x
        for c in center_shift_layers:
            center_shift = c(center_shift)
        feat = x
        for f in feat_layers:
            feat = f(feat)

        feat = self.concat([skip,feat])
        for f2 in feat_end:
            feat = f2(feat)

        return tf.keras.Model(inputs=inputs, outputs =[score, center_shift, feat])


    def quantize_model(self, shape):
        with tfmot.quantization.keras.quantize_scope({'Conv2dBNRelu':Conv2dBNRelu, 'SimpleConvBlock':SimpleConvBlock, 'SimpleConvBlockOut': SimpleConvBlockOut, 'SubPixel':SubPixel}):
            def apply_quantization(layer):
                if not isinstance(layer, keras.layers.Lambda):
                    return tfmot.quantization.keras.quantize_annotate_layer(layer)
                else:
                    return tfmot.quantization.keras.quantize_annotate_layer(layer,
                                                                     quantize_config=NoOpQuantizeConfig())
                return layer

            annotated_model = tf.keras.models.clone_model(
                self.create_functional_model(shape),
                clone_function=apply_quantization,
            )

            return tfmot.quantization.keras.quantize_apply(annotated_model)

class KeypointNetRawLegacy(tf.keras.Model):
    """
    Keypoint detection network without post processing, designed to run on the coral edge tpu.
    Original design, where channel dimension is not increased before pooling

    Parameters
    ----------
    do_upsample: bool
        Upsample desnse descriptor map.
    with_drop : bool
        Use dropout.
    nfeatures: int
        Predict keypoints outside cell borders.
    model_params: tuple
        Tuple containing 5 values to set the channel dimensions (c1 to c5)
    downsample: int
        Amount of pooling layers, can be 1 to 3.
    use_leaky_relu: bool
        If true, leaky relu is used as activation, else normal relu.
    use_subpixel: bool
        Set if subpixel convolution is enabled
    kwargs : dict
        Extra parameters
    """

    def __init__(self, do_upsample=True, with_drop=False, nfeatures=64, model_params=(16, 32, 32, 64, 64),
                 downsample=3, use_leaky_relu=False, use_subpixel=False, **kwargs):
        super(KeypointNetRawLegacy, self).__init__()

        assert len(model_params) == 5, "In legacy mode, model_params has to contain 5 parameters"
        print("KeypointNetRawLegacy:")
        print("Dropout:", with_drop)
        print("Leaky Relu:", use_leaky_relu)
        self.with_drop = with_drop
        self.do_upsample = do_upsample
        self.n_features = nfeatures
        self.downsample = downsample
        self.bn_momentum = 0.9
        self.use_subpixel = use_subpixel

        c1, c2, c3, c4, c5 = model_params

        if self.use_subpixel:
            d1 = c3 * 4
        else:
            d1 = c3

        use_bias = False

        self.conv1 = SimpleConvBlock(c1, c1, use_bias, self.bn_momentum, use_leaky_relu, 1, with_drop)
        self.conv2 = SimpleConvBlock(c2, c2, use_bias, self.bn_momentum, use_leaky_relu, 1, with_drop)
        self.conv3 = SimpleConvBlock(c3, c3, use_bias, self.bn_momentum, use_leaky_relu, 1, with_drop)
        self.conv4 = SimpleConvBlock(c4, c4, use_bias, self.bn_momentum, use_leaky_relu, 1, with_drop)

        self.convD = SimpleConvBlockOut(c5, 1, use_bias, self.bn_momentum, use_leaky_relu, 'D', with_drop,
                                        'sigmoid')
        self.convP = SimpleConvBlockOut(c5, 2, use_bias, self.bn_momentum, use_leaky_relu, 'P', with_drop,
                                        'tanh')

        self.convFa = Conv2dBNRelu(c5, use_bias, self.bn_momentum, use_leaky_relu, "convFa")
        self.convFb = Conv2dBNRelu(d1, use_bias, self.bn_momentum, use_leaky_relu, "convFb")

        self.convF = SimpleConvBlockOut(c5, nfeatures, use_bias, self.bn_momentum, use_leaky_relu, 'Fa',
                                        with_drop, None)

        self.pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
        self.pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
        self.pool_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")

        if self.do_upsample:
            if use_subpixel:
                self.subpixel = SubPixel(2)
            else:
                self.up = tf.keras.layers.UpSampling2D(size=(2, 2))
                self.convup = SimpleConvBlock(c3, c3, use_bias, self.bn_momentum, use_leaky_relu, 1, with_drop)
            self.concat = tf.keras.layers.Concatenate(axis=-1)

    @tf.function
    def call(self, inputs, training=False):
        """
        Processes a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input images (B, 3, H, W)

        Returns
        -------
        score : torch.Tensor
            Score map (B, 1, H_out, W_out)
        coord: torch.Tensor
            Keypoint coordinates (B, 2, H_out, W_out)
        feat: torch.Tensor
            Keypoint descriptors (B, 256, H_out, W_out)
        """

        x = self.conv1(inputs)
        if self.downsample >= 2:
            x = self.pool_1(x, training=training)
        x = self.conv2(x)
        if self.downsample >= 3:
            x = self.pool_2(x, training=training)
        skip = self.conv3(x)
        if self.downsample >= 1:
            x = self.pool_3(skip, training=training)
        x = self.conv4(x)

        score = self.convD(x, training=training)
        center_shift = self.convP(x, training=training)

        feat = self.convFa(x, training=training)
        feat = self.convFb(feat, training=training)

        if self.do_upsample:
            if self.use_subpixel:
                feat = self.subpixel(feat, training=training)

            else:
                feat = self.up(feat, training=training)
                feat = self.convup(feat, training=training)
            feat = self.concat([feat, skip])
        feat = self.convF(feat, training=training)

        return score, center_shift, feat

    def create_functional_model(self, shape):
        training = False
        inputs = tf.keras.Input(shape=shape + (3,))

        x = self.conv1(inputs)
        if self.downsample >= 2:
            x = self.pool_1(x, training=training)
        x = self.conv2(x)
        if self.downsample >= 3:
            x = self.pool_2(x, training=training)
        skip = self.conv3(x)
        if self.downsample >= 1:
            x = self.pool_3(skip, training=training)
        x = self.conv4(x)

        score = self.convD(x, training=training)
        center_shift = self.convP(x, training=training)

        feat = self.convFa(x, training=training)
        feat = self.convFb(feat, training=training)
        if self.do_upsample:
            if self.use_subpixel:
                feat = self.subpixel(feat, training=training)
            else:
                feat = self.up(feat, training=training)
                feat = self.convup(feat, training=training)
            feat = self.concat([feat, skip])
        feat = self.convF(feat, training=training)

        return tf.keras.Model(inputs=inputs, outputs=[score, center_shift, feat])

    def quantize_model(self, shape):
        with tfmot.quantization.keras.quantize_scope(
                {'Conv2dBNRelu': Conv2dBNRelu, 'SimpleConvBlock': SimpleConvBlock,
                 'SimpleConvBlockOut': SimpleConvBlockOut, 'SubPixel': SubPixel}):
            def apply_quantization(layer):
                if not isinstance(layer, keras.layers.Lambda):
                    return tfmot.quantization.keras.quantize_annotate_layer(layer)
                else:
                    return tfmot.quantization.keras.quantize_annotate_layer(layer,
                                                                            quantize_config=NoOpQuantizeConfig())
                return layer

            annotated_model = tf.keras.models.clone_model(
                self.create_functional_model(shape),
                clone_function=apply_quantization,
            )

            return tfmot.quantization.keras.quantize_apply(annotated_model)


class KeypointNetRawV0(tf.keras.Model):
    """
    Keypoint detection network without post processing, designed to run on the coral edge tpu.

    Parameters
    ----------
    do_upsample: bool
        Upsample desnse descriptor map.
    with_drop : bool
        Use dropout.
    nfeatures: int
        Predict keypoints outside cell borders.
    model_params: tuple
        Tuple containing 5 values to set the channel dimensions (c1 to c5)
    downsample: int
        Amount of pooling layers, can be 1 to 3.
    use_leaky_relu: bool
        If true, leaky relu is used as activation, else normal relu.
    use_subpixel: bool
        Set if subpixel convolution is enabled
    large_feat: bool
        If true, the feature is configured to use c4 as channel dimension before concatenation, if false c3 is used.

    kwargs : dict
        Extra parameters
    """

    def __init__(self, do_upsample=True, with_drop=False, nfeatures=64, model_params=(16, 16, 32, 32, 32, 32), downsample=1, use_leaky_relu=False, use_subpixel=False, **kwargs):
        super(KeypointNetRawV0, self).__init__()

        assert len(model_params) == 6, "model_params has to contain 6 parameters"

        print("KeypointNetRawV0:")
        print("Dropout:",with_drop)
        print("Leaky Relu:", use_leaky_relu)
        print("model params:", model_params)
        self.with_drop = with_drop
        self.do_upsample = do_upsample
        self.n_features = nfeatures
        self.downsample = downsample
        self.bn_momentum = 0.9
        self.use_subpixel = use_subpixel

        c1, c2, c3, c4, c5, d1 = model_params

        if self.use_subpixel:
            d1 = d1*4

        use_bias=False

        self.conv1 = SimpleConvBlock(c1,c2,use_bias,self.bn_momentum,use_leaky_relu, 1, with_drop)
        self.conv2 = SimpleConvBlock(c2,c3,use_bias,self.bn_momentum,use_leaky_relu, 2, with_drop)
        self.conv3 = SimpleConvBlock(c3,c4,use_bias,self.bn_momentum,use_leaky_relu, 3, with_drop)
        self.conv4 = SimpleConvBlock(c4,c4,use_bias,self.bn_momentum,use_leaky_relu, 4, with_drop)

        self.convD = SimpleConvBlockOut(c4,1, use_bias,self.bn_momentum, use_leaky_relu, 'D', with_drop, 'sigmoid')
        self.convP = SimpleConvBlockOut(c4,2, use_bias,self.bn_momentum, use_leaky_relu, 'P', with_drop, 'tanh')

        self.convFa = Conv2dBNRelu(c4,use_bias,self.bn_momentum,use_leaky_relu, "convFa")
        self.convFb = Conv2dBNRelu(d1,use_bias,self.bn_momentum,use_leaky_relu, "convFb")

        self.convF = SimpleConvBlockOut(c5,nfeatures, use_bias,self.bn_momentum, use_leaky_relu, 'Fa', with_drop, None)

        self.pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")

        if self.do_upsample:
            if use_subpixel:
                self.subpixel = SubPixel(2)
            else:
                self.up = tf.keras.layers.UpSampling2D(size=(2,2))
                #self.convup = SimpleConvBlock(c3,c3, use_bias, self.bn_momentum, use_leaky_relu, "u", with_drop)
                self.convup = Conv2dBNRelu(d1,use_bias,self.bn_momentum,use_leaky_relu, "u")
                #self.convup = SimpleConvBlock(c4,c4, use_bias, self.bn_momentum, use_leaky_relu, "u", with_drop)

            self.concat = tf.keras.layers.Concatenate(axis=-1) # output is c4*2

    @tf.function
    def call(self, inputs, training=False):
        """
        Processes a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input images (B, 3, H, W)

        Returns
        -------
        score : torch.Tensor
            Score map (B, 1, H_out, W_out)
        coord: torch.Tensor
            Keypoint coordinates (B, 2, H_out, W_out)
        feat: torch.Tensor
            Keypoint descriptors (B, 256, H_out, W_out)
        """

        x = self.conv1(inputs)
        skip = self.conv2(x)
        x = self.pool_1(skip, training=training)
        x = self.conv3(x)
        x = self.conv4(x)
        score = self.convD(x, training=training)
        center_shift = self.convP(x, training=training)

        feat = self.convFa(x, training=training)
        feat = self.convFb(feat, training=training)

        if self.do_upsample:
            if self.use_subpixel:
                feat = self.subpixel(feat, training=training)
            else:
                feat = self.up(feat, training=training)
                feat = self.convup(feat, training=training)
            feat = self.concat([feat, skip])
        feat = self.convF(feat, training=training)

        return score, center_shift, feat

    def create_functional_model(self, shape):
        inputs = tf.keras.Input(shape=(shape + (3,)))
        seq = []
        seq.extend(self.conv1.sequential())
        seq.extend(self.conv2.sequential())

        seq.append(self.pool_1)
        seq_2 = []
        seq_2.extend(self.conv3.sequential())
        seq_2.extend(self.conv4.sequential())

        score_layers = self.convD.sequential()
        center_shift_layers = self.convP.sequential()

        feat_layers = self.convFa.sequential()
        feat_layers.extend(self.convFb.sequential())
        if self.do_upsample:
            if self.use_subpixel:
                feat_layers.extend(self.subpixel.sequential())
            else:
                feat_layers.append(self.up)
                feat_layers.extend(self.convup.sequential())

        feat_end = self.convF.sequential()
        x = inputs
        for s in seq:
            x = s(x)
        skip = x
        for s2 in seq_2:
            x = s2(x)
        score = x
        for s3 in score_layers:
            score = s3(score)
        center_shift = x
        for c in center_shift_layers:
            center_shift = c(center_shift)
        feat = x
        for f in feat_layers:
            feat = f(feat)

        feat = self.concat([skip,feat])
        for f2 in feat_end:
            feat = f2(feat)

        return tf.keras.Model(inputs=inputs, outputs =[score, center_shift, feat])


    def quantize_model(self, shape):
        with tfmot.quantization.keras.quantize_scope({'Conv2dBNRelu':Conv2dBNRelu, 'SimpleConvBlock':SimpleConvBlock, 'SimpleConvBlockOut': SimpleConvBlockOut, 'SubPixel':SubPixel}):
            def apply_quantization(layer):
                if not isinstance(layer, keras.layers.Lambda):
                    return tfmot.quantization.keras.quantize_annotate_layer(layer)
                else:
                    return tfmot.quantization.keras.quantize_annotate_layer(layer,
                                                                     quantize_config=NoOpQuantizeConfig())
                return layer

            annotated_model = tf.keras.models.clone_model(
                self.create_functional_model(shape),
                clone_function=apply_quantization,
            )

            return tfmot.quantization.keras.quantize_apply(annotated_model)


class KeypointNetRawNano(tf.keras.Model):
    """
    Keypoint detection network without post processing, designed to run on the coral edge tpu.

    Parameters
    ----------
    do_upsample: bool
        Upsample desnse descriptor map.
    with_drop : bool
        Use dropout.
    nfeatures: int
        Predict keypoints outside cell borders.
    model_params: tuple
        Tuple containing 5 values to set the channel dimensions (c1 to c5)
    downsample: int
        Amount of pooling layers, can be 1 to 3.
    use_leaky_relu: bool
        If true, leaky relu is used as activation, else normal relu.
    use_subpixel: bool
        Set if subpixel convolution is enabled
    large_feat: bool
        If true, the feature is configured to use c4 as channel dimension before concatenation, if false c3 is used.

    kwargs : dict
        Extra parameters
    """

    def __init__(self, with_drop=False, nfeatures=64, model_params=(16, 16, 32, 32, 32), use_leaky_relu=False, **kwargs):
        super(KeypointNetRawNano, self).__init__()

        assert len(model_params) == 5, "model_params has to contain 6 parameters"

        print("KeypointNetRawV0:")
        print("Dropout:",with_drop)
        print("Leaky Relu:", use_leaky_relu)
        print("model params:", model_params)
        self.with_drop = with_drop
        self.n_features = nfeatures
        self.bn_momentum = 0.9

        c1, c2, c3, c4, c5 = model_params
        use_bias=False

        self.conv1 = SimpleConvBlock(c1,c2,use_bias,self.bn_momentum,use_leaky_relu, 1, with_drop)
        self.conv2 = SimpleConvBlock(c3,c4,use_bias,self.bn_momentum,use_leaky_relu, 2, with_drop)

        self.convD = SimpleConvBlockOut(c5,3, use_bias,self.bn_momentum, use_leaky_relu, 'D', with_drop, 'sigmoid')

        self.convFa = Conv2dBNRelu(c5,use_bias,self.bn_momentum,use_leaky_relu, "convFa")
        self.convF = SimpleConvBlockOut(c5,nfeatures, use_bias,self.bn_momentum, use_leaky_relu, 'Fa', with_drop, None)


    @tf.function
    def call(self, inputs, training=False):
        """
        Processes a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input images (B, 3, H, W)

        Returns
        -------
        score : torch.Tensor
            Score map (B, 1, H_out, W_out)
        coord: torch.Tensor
            Keypoint coordinates (B, 2, H_out, W_out)
        feat: torch.Tensor
            Keypoint descriptors (B, 256, H_out, W_out)
        """

        x = self.conv1(inputs)
        x = self.conv2(x)
        detector = self.convD(x, training=training)

        score = tf.sigmoid(detector[:,:,:,:1])
        center_shift = tf.tanh(detector[:,:,:,1:])

        feat = self.convFa(x, training=training)
        feat = self.convF(feat, training=training)

        return score, center_shift, feat

    def quantize_model(self, shape):
        with tfmot.quantization.keras.quantize_scope({'Conv2dBNRelu':Conv2dBNRelu, 'SimpleConvBlock':SimpleConvBlock, 'SimpleConvBlockOut': SimpleConvBlockOut, 'SubPixel':SubPixel}):
            def apply_quantization(layer):
                if not isinstance(layer, keras.layers.Lambda):
                    return tfmot.quantization.keras.quantize_annotate_layer(layer)
                else:
                    return tfmot.quantization.keras.quantize_annotate_layer(layer,
                                                                     quantize_config=NoOpQuantizeConfig())
                return layer

            annotated_model = tf.keras.models.clone_model(
                self.create_functional_model(shape),
                clone_function=apply_quantization,
            )

            return tfmot.quantization.keras.quantize_apply(annotated_model)
