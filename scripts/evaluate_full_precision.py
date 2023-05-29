import argparse
import numpy as np
import tensorflow as tf
from models import KeypointNetwithIOLoss as kp2dmodel

import configs.model_configs
import os
from evaluation.evaluate import evaluate_keypoint_net
from datasets.patches_dataset import PatchesDataset

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--config', type=str, default='KP2D88')
my_parser.add_argument('--weights-path', type=str)
my_parser.add_argument('--disable-wandb', action='store_false')
my_parser.add_argument('--QAT', action='store_true', help="Enable if model was trained using QAT")
my_parser.add_argument('--debug', action='store_true', help="Show debug plots")

args = my_parser.parse_args()
val_path = os.environ.get("VAL_PATH", '/data/datasets/kp2d/HPatches/')

wandb_enabled = args.disable_wandb

config = getattr(configs.model_configs, args.config)

model = kp2dmodel.KeypointNetwithIOLoss(debug=args.debug, QAT=args.QAT, **config)
model.keypoint_net.trainable = True
model.keypoint_net(np.random.rand(1, config['shape'][0], config['shape'][1], 3), training=True)
model.keypoint_net.load_weights(args.weights_path)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

eval_params = [{'shape':(88,88), 'top_k': 150}]
eval_params += [{'shape':(88,88), 'top_k': 300}]
eval_params += [{'shape':(240,320), 'top_k': 300}]
eval_params += [{'shape':(240,320), 'top_k': 1000}]

if wandb_enabled:
    from utils.wandb_utils import *
    init_wandb(args.config, project="TF Evaluation",
               config_dict={"model_config": config, "weights_path": args.weights_path, "QAT": args.QAT})


for param in eval_params:
    ds = PatchesDataset(val_path, output_shape=(param['shape']),type='a')

    param['top_k'] = min(param['top_k'], int(param['shape'][0]/model.keypoint_net.cell*
                         param['shape'][1]/model.keypoint_net.cell))
    print("running evaluation", str(param))
    result_dict = evaluate_keypoint_net(ds, model.keypoint_net, top_k=param['top_k'], debug=args.debug)

    result_dict['top_k'] = param['top_k']
    result_dict['shape'] = param['shape']
    print(result_dict)

    if wandb_enabled:
        wandb_log_panel(result_dict, prefix="Evaluation/")