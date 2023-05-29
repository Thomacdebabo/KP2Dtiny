import argparse
import numpy as np
import tensorflow as tf
from models import KeypointNetwithIOLoss as kp2dmodel
from datetime import datetime
import configs.model_configs
import os
from evaluation.evaluate import evaluate_keypoint_net
from datasets.patches_dataset import PatchesDataset
import random
import json
import gc
from datasets.dataloader import SimpleDataLoader

train_path = os.environ.get("TRAIN_PATH", None)
val_path = os.environ.get("VAL_PATH", None)

assert train_path, "No representative dataset specified. Make sure you have set the TRAIN_PATH environment variable."
assert os.path.exists(train_path), "This dataset path does not exist: {}".format(train_path)

assert val_path, "No representative dataset specified. Make sure you have set the VAL_PATH environment variable."
assert os.path.exists(val_path), "This dataset path does not exist: {}".format(val_path)


def parse_args():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--config', type=str, default='KP2D88', help="Specify which model configuration should be used to train")
    my_parser.add_argument('--wandb-project', type=str, default='KP2D-Tensorflow')
    my_parser.add_argument('--num-epochs', type=int, default=10)
    my_parser.add_argument('--reduce-lr', type=int, default=5, help="Set after which epoch the learning rate should be halfed.")
    my_parser.add_argument('--batch-size', type=int, default=32)
    my_parser.add_argument('--learning-rate', type=float, default=0.001)
    my_parser.add_argument('--pretrained', type=str, default=None, help="Set path of pretrained model weights")
    my_parser.add_argument('--QAT-pretrained', action='store_true', help="Set this to true if the model weights loaded have QAT enabled")
    my_parser.add_argument('--debug', action='store_true', help="Enable debug plots (only do this if there is a display to show the outputs)")
    my_parser.add_argument('--QAT', action='store_true', help="Enable quantize aware training")
    my_parser.add_argument('--disable-wandb', action='store_false')
    return my_parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def grad(model, inputs):
  with tf.GradientTape() as tape:
    loss_value, loss_dict = model.compute_loss(inputs, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables), loss_dict


def optimizer_step(model,data, opt):
    loss_value, grads, loss_dict = grad(model, data)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value, loss_dict


def train(ds, model, opt, log_freq = 100, wandb_enabled = True):
    model.keypoint_net.set_trainable(True)
    progbar = tf.keras.utils.Progbar(len(ds), stateful_metrics = ['total_loss', 'io_loss','usp_loss','metric_loss', 'loc_loss'])
    for i in range(len(ds)):
        data = ds.__getitem__(i)

        loss_value, loss_dict = optimizer_step(model, data, opt)
        loss_dict['total_loss'] = loss_value
        losses = list(loss_dict.items())

        progbar.update(i, values=losses)
        if i % log_freq == 0:
            if wandb_enabled:
                wandb_log_panel(loss_dict, prefix="Loss/")
        if i % 1000 == 10:
            gc.collect()
            tf.keras.backend.clear_session()


def wandb_log_panel(log_dict, prefix = ''):
    temp_dict = {}
    for k in log_dict.keys():
        temp_dict[prefix+k] = log_dict[k]
    wandb.log(temp_dict)
    del temp_dict

# init
set_seed()

args = parse_args()
pre = 'Final_'
name = args.config
wandb_enabled = args.disable_wandb

if args.QAT:
    pre += 'QAT_'
    name += '_QAT'


### Declaration of variables

current_time = datetime.now().strftime("_%Y_%m_%d__%H_%M_%S")
checkpoint_dir = './data/' +pre+args.config + current_time

os.makedirs(checkpoint_dir,exist_ok=True)

config = getattr(configs.model_configs, args.config)

num_epochs = args.num_epochs
batch_size = args.batch_size
shape = config['shape']

patch_ratio = 0.7
scaling_amplitude = 0.2
max_angle = 3.14 / 2

log_freq = 300

# log wandb config
if wandb_enabled:
    from utils.wandb_utils import *
    config_dict = {"model_config": config,
                   "QAT":args.QAT,
                   "batch_size": args.batch_size,
                   "num_epochs": args.num_epochs,
                   "lr": args.learning_rate,
                   "input_shape": shape}


    init_wandb(name=name, project=args.wandb_project,
               config_dict=config_dict)

ds = SimpleDataLoader(train_path, quantize=False, batch_size=batch_size, shape=shape, patch_ratio=patch_ratio, scaling_amplitude=scaling_amplitude,
                 max_angle=max_angle)

#halfing the learning rate after a set amount of epochs
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=args.learning_rate,
    decay_steps=int(args.reduce_lr*ds.__len__()/batch_size),
    decay_rate=0.5)

opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=1.0)


model = kp2dmodel.KeypointNetwithIOLoss(debug=args.debug, QAT=args.QAT, **config)

if config['with_io']:
    model.io_net.set_trainable(True)

_, _ = model.compute_loss(ds.__getitem__(1))

if args.pretrained:
    path, _ = os.path.split(args.pretrained)
    if config['with_io']:
        io_path = os.path.join(path, 'io_net.hdf5')
        if os.path.exists(io_path):
            model.io_net.load_weights(io_path)
            model.io_net.set_trainable(True)
            print("IO net loaded")

    if args.QAT:
        if args.QAT_pretrained:
            model.keypoint_net.load_weights(args.pretrained)
        else:
            model_temp = kp2dmodel.KeypointNetMobile(QAT=False, **config)

            _,_,_ = model_temp(ds.__getitem__(1)['image'])
            model_temp.load_weights(args.pretrained)
            raw = model_temp.keypoint_net_raw.quantize_model(config['shape'])
            raw.save_weights(os.path.join(checkpoint_dir,'QAT.hdf5'))
            print("Saving quantized file",os.path.join(checkpoint_dir,'QAT.hdf5'))
            model.keypoint_net.keypoint_net_raw.load_weights(os.path.join(checkpoint_dir,'QAT.hdf5'))
            del model_temp
    else:
        model.keypoint_net.load_weights(args.pretrained)

print(model.keypoint_net.keypoint_net_raw.summary())

gc.collect()
tf.keras.backend.clear_session()

if not args.debug:
    grad = tf.function(grad, autograph=False)
    optimizer_step = tf.function(optimizer_step, autograph=False)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Training loop
for i in range(num_epochs):

    print("epoch nr", i)
    train(ds, model, opt, log_freq=log_freq, wandb_enabled=wandb_enabled)
    print("done")

    kp_path = args.config + "_" + str(i) + ".hdf5"

    model.keypoint_net.save_weights(os.path.join(checkpoint_dir, kp_path))
    model.io_net.save_weights(os.path.join(checkpoint_dir, "io_net.hdf5"))

    print("Model saved:", os.path.join(checkpoint_dir, kp_path))

    print("running evaluation")


    val_ds = PatchesDataset(val_path, output_shape=config['shape'], quantize=False)

    model.keypoint_net.set_trainable(False)
    result_dict = evaluate_keypoint_net(val_ds, model.keypoint_net, top_k=config['top_k'], debug=args.debug, training=True)
    print("Results:", result_dict)

    json_path = args.config + "_" + str(i) + ".json"
    with open(os.path.join(checkpoint_dir, json_path), "w") as fp:
        json.dump(result_dict, fp)
    if wandb_enabled:
        wandb_log_panel(result_dict, prefix="Evaluation/")
    tf.keras.backend.clear_session()
    gc.collect()
    ds.shuffle()
