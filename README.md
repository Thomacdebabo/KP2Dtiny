# KP2DTiny

This is a tensorflow implementation of KP2D, the original work can be found [here](https://github.com/TRI-ML/KP2D). 
We introduce tools and adjusted model architectures to deploy KP2D on embedded platforms, in particular the [Coral USB Accelerator]().

While we only tested the coral ai device the conversion scripts exports a tflite model which should be compatible with a wide variety of devices. 
For more information please refer to our paper published at [AICAS 2023](https://ieeexplore.ieee.org/document/10168598)
## Introduction
KP2D is a fully unsupervised feature detector and descriptor model. It is based on the [superpoint](https://arxiv.org/abs/1712.07629) architecture and expands on the semi supervised learning methods proposed in [UnsuperPoint](https://arxiv.org/abs/1907.04011). 
- **Keypoint detection and description** or also known as [feature detection and description](https://docs.opencv.org/3.4/db/d27/tutorial_py_table_of_contents_feature2d.html), is a fundamental problem in computer vision where the goal is to find and describe points or regions in an image which are distinct and stable over multiple viewpoints, allowing us to triangulate their 3D position.
- **Unsupervised learning** refers to learning methods which do not require exact ground truth data, as is typical in supervised learning methods. In our case this means to train this model we only require RGB images to train our model. This makes training such a model on a custom dataset very cheap, since there is no need for manual annotation of data.

With KP2Dtiny we bring deep learning based keypoint detection and description to edge platforms. Our main contributions are:
- tensorflow implementation to export the model in the tflite format
- optimizing the model for inference on microcontrollers
- reducing the size from **20.4** MB to **0.37** MB
- significant improvement in accuracy on homography estimation (d1: 10%)

## Installation

1. set up conda environment
2. Install [tensorflow](https://www.tensorflow.org/install/pip)
3. Install requirements with ```pip3 install -r requirements.txt```
4. (Recommended) add environment variables:
   - `TRAIN_PATH`: path to training dataset (Coco)
   - `VAL_PATH`: path to validation dataset (HPatches)
### Datasets
Download the HPatches dataset for evaluation:

```bash
cd /data/datasets/kp2d/
wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz
tar -xvf hpatches-sequences-release.tar.gz
mv hpatches-sequences-release HPatches
```

Download the COCO dataset for training:
```bash
mkdir -p /data/datasets/kp2d/coco/ && cd /data/datasets/kp2d/coco/
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
```
(copied from the original [KP2D repo](https://github.com/TRI-ML/KP2D))

Note: Any directory containing RGB images can be used as a training dataset.
## Model architectures
![architecture](https://github.com/user-attachments/assets/53e9d891-ab85-435f-befe-d5d252825f56)
---
We have designed two model architectures to work at low (88x88) and high (240x320) resolutions.
They are based on the original KeypointNet found in the original repository.

 |Name | Channel sizes | nfeatures | downs. |  feature head | activation  | parameters  |  size [mb] | size quant. [mb]|
|---|---|---|---|---|---|---|---|---|
 Baseline | 32, 64, 128, 256, 256 | 256 | 3 | 128 | leakyReLu | 5,317k | 20.4 | **5.1**|
 KP2DtinyS (tinyV1)| 16, 32, 32, 64, 64 | 32 | 2 | 32 | ReLu | 387k | 1.48 |  **0.37**|
 KP2DtinyF (tinyV2) |  16, 32, 64, 128, 128 | 64 | 3 | 128 | ReLu | 1,849k | 7.6 | **1.9**|
## Code Overview

---
This repository is following the same structure as the pytorch version.
- **configs**: ``model_configs.py`` stores all configurations.
- **datasets**: Implements dataloaders for different datasets as well as augmentations for data augmentation. 
- **evaluation**: Implements evaluation functions. The code is split into descriptor and detector evaluation. From a user perspective only ```evaluate.py``` is important. This module takes a dataloader and a model and runs the evaluation.
- **models**: This folder contains the model called KeypointNetwithIOLoss. Don't confuse this with the network called KeypointNet. This model contains the KeypointNet and also handles the loss functions. The forward pass of this model returns the loss which we can then call the backward pass on. This model is only used during training. KeypointNetTFLite implements a wrapper to run .tflite models.
- **networks**: stores implementations for KeypointNet and IO-Net.
- **utils**: Various utility functions.
- **demo**: runs real time demo on coral


### Provided scripts:
- ```train.py```: Training script for KeypointNet
- ```evaluate_full_precision.py```: Evaluation script for KeypointNet, specify if model has been trained using QAT.
- ```quantize.py```: Converts and quantizes a .hdf5 model to tflite. Note: Make sure model name matches the config, resolution can be different for example: converting a KP2D88tinyV1 model with KP2D320tinyV1 config works but KP2D88tinyV1 with KP2D88tinyV2 does not.
- ```evaluate_quantized.py```: Evaluation script for Tflite models. Use --ues-tpu to run evaluation on edge tpu.

Logging with [wandb](https://wandb.ai/) is enabled by default. To disable use the ```--disable-wandb``` flag when running a script.

## Deploy on coral


Setup coral environment [link](https://coral.ai/docs/accelerator/get-started/#requirements)
- run ```python scripts/train.py``` to train the model
- run ```python scripts/quantize.py``` to get the general tflite model
- run ```edgetpu_compiler model.tflite``` to get the edge tpu model
- evaluate using ```python evaluate_quantized.py --m model.tflite --c MODEL_CONFIG_NAME --use-tpu```
## Results
![results](https://github.com/user-attachments/assets/6a8309ac-b7c0-4d66-a86d-da82ba1457ce)

---
### 320x240 1000 points
|Metric|Baseline|TinyV1|TinyV2|Baseline Q|TinyV1 Q|TinyV2 Q|
|---|---|---|---|---|---|---|
| Repeatability  | 0.670| 0.746| 0.652|  0.663| 0.735| 0.647| 
| Localization error  | 0.980| 0.848| 1.003| 1.315| 0.882| 1.217| 
| d1  | 0.541| 0.662| 0.438| 0.316| 0.648| 0.341| 
| d3  | 0.840| 0.864| 0.800| 0.778| 0.866| 0.721| 
| d5  | 0.898| 0.895| 0.859| 0.862| 0.905| 0.814| 
| MScore  | 0.499| 0.387| 0.454| 0.454| 0.389| 0.438| 


### Comparing results **320x240**:

| Model	| Repeatability |	Localization |	d1 |	d3 | 	d5 |	MScore |
|---|---|---|---|---|---|---|
|SIFT k300|0.451| 0.855| 0.622| 0.845| 0.878| 0.304|
| KP2D (previous work) k300|	0.687 |	0.892 |	0.593 |	0.867 |	0.91  |	0.546 |
| KP2D (ours) k300|	0.647 |	0.923 |	0.517 |	0.876|	0.853  |	0.493 |
| Q KP2D (ours) k300|	0.624 |	1.204 |	0.329 |	0.767 |	0.853  |	0.443 |
| Q TinyS k300|	 0.612 |	0.814|	0.562 |	0.821 |	0.872  |	0.362|
| Q KP2D (ours) k1000|	0.663|	1.315 |	0.316 |	0.778 |	0.862  |	 0.454 |
| Q TinyS k1000|	0.733 |	0.884|	0.633 |	0.862 |	0.902  |	0.387|



      
