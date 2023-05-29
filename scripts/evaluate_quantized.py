from datasets.patches_dataset import PatchesDataset
from evaluation.evaluate import evaluate_keypoint_net
import argparse
import configs.model_configs
import os
from models.KeypointNetTFLite import KP2DTFLite
import json

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--model-path', type=str)
my_parser.add_argument('--config', type=str, default='KP2D88')
my_parser.add_argument('--top-k', type=int, default=300)
my_parser.add_argument('--use-tpu', action='store_true')
my_parser.add_argument('--float-out', action='store_true')
my_parser.add_argument('--inf-test', action='store_true')

my_parser.add_argument('--disable-wandb', action='store_false')

args = my_parser.parse_args()
config = getattr(configs.model_configs, args.config)

data_dir = os.environ.get("VAL_PATH", '/home/thomas/PycharmProjects/KP2D/data/datasets/kp2d/HPatches')
debug = True
wandb_enabled = args.disable_wandb

keypoint_net = KP2DTFLite(args.model_path, downsample=config['downsample'],
                          use_tpu=args.use_tpu, use_upconv=config['do_upsample'], q_out = not args.float_out)
top_k = min(args.top_k, int(config['shape'][0]/keypoint_net.cell*
                                config['shape'][1]/keypoint_net.cell))
ds = PatchesDataset(data_dir, output_shape=(config['shape']))
if args.inf_test:
    keypoint_net.inference_test(100)
else:
    result_dict = evaluate_keypoint_net(ds, keypoint_net, top_k=top_k, debug=debug)
    json_path = args.model_path[:-6] +"json"
    print(result_dict)

    if os.path.exists(json_path):
        with open(json_path, "r") as fp:
            data = json.load(fp)
    else:
        data = {}

    data[str(args.top_k) + "_results"] = result_dict
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

    if wandb_enabled:
        from utils.wandb_utils import *

        config_dict = {'top_k': args.top_k}

        if 'tflite_args' in data.keys():
            config_dict['tflite_args'] = data['tflite_args']
        if 'config' in data.keys():
            config_dict['config'] = data['config']

        init_wandb(name=args.config + "_PTQ", project="TFlite Evaluation",
                   config_dict=config_dict)
        wandb_log_panel(result_dict, prefix="Evaluation/")




