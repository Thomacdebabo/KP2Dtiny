# Copyright 2020 Toyota Research Institute.  All rights reserved.

import tensorflow as tf
import tensorflow_addons as tfa
# Slightly modified version of 1d-CNN from https://arxiv.org/abs/1905.04132.
# More details: https://github.com/vislearn/ngransac
# Code adapted from https://github.com/vislearn/ngransac/blob/master/network.py
from tensorflow.keras import layers

class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self,bn_momentum):
        super(ResNetBlock, self).__init__()
        self.conv1 = layers.Conv2D(128, kernel_size=1, strides=(1, 1), padding="valid", use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=bn_momentum)
        self.relu1 = tf.keras.layers.ReLU()
        self.instance_norm1 = tfa.layers.InstanceNormalization()

        self.conv2 = layers.Conv2D(128, kernel_size=1, strides=(1, 1), padding="valid", use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=bn_momentum)
        self.relu2 = tf.keras.layers.ReLU()
        self.instance_norm2 = tfa.layers.InstanceNormalization()

    def call(self, inputs, training=False):
        res = inputs
        x = self.conv1(inputs, training=training)
        x = self.instance_norm1(x, training=training)
        x = self.bn1(x,training=training)
        x = self.relu1(x,training=training)

        x = self.conv2(x, training=training)
        x = self.instance_norm2(x, training=training)
        x = self.bn2(x,training=training)
        x = self.relu2(x,training=training)
        return x+res

class InlierNet(tf.keras.Model):
    def __init__(self):
        super(InlierNet, self).__init__()


        self.bn_momentum = 0.9
        self.p_in = tf.keras.layers.Conv2D(128,  kernel_size=1, strides=(1,1), padding="valid", use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.1)

        self.r1 = ResNetBlock(self.bn_momentum)
        self.r2 = ResNetBlock(self.bn_momentum)
        self.r3 = ResNetBlock(self.bn_momentum)
        self.r4 = ResNetBlock(self.bn_momentum)

        self.p_out = tf.keras.layers.Conv2D(1,  kernel_size=1, strides=(1,1), padding="valid")
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, training = True):
        x = self.p_in(inputs, training = training)
        x = self.bn(x, training = training)
        x = self.relu(x, training = training)
        x = self.r1(x, training = training)
        x = self.r2(x, training = training)
        x = self.r3(x, training = training)
        x = self.r4(x, training = training)
        return self.p_out(x, training = training)

    def set_trainable(self, train):
        self.trainable = train
        self.compile()
