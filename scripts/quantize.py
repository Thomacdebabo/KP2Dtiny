import tensorflow as tf
from networks.keypoint_net_mobile import KeypointNetMobile
import numpy as np
import os
import argparse
import configs.model_configs
import json

def convert(model,config, output_type = tf.int8, n_samples = 100, train_path = None):
    if not train_path:
        train_path = os.environ.get("TRAIN_PATH", None)

    assert train_path, "No representative dataset specified. Make sure you have set the TRAIN_PATH environment " \
                       "variable or pass the path to the dataset as an input argument."
    assert os.path.exists(train_path), "This representative dataset path does not exist: {}".format(train_path)

    def create_dataset(batch_size, shape):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_path,
            labels=None,
            seed=123,
            image_size=shape,
            batch_size=batch_size)

        preprossesing = tf.keras.Sequential([
            tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=None),
            tf.keras.layers.RandomRotation((-0.5, 0.8)),
            tf.keras.layers.RandomTranslation(0.2, 0.2),
            tf.keras.layers.Rescaling(2. / 255, offset=-1.)
        ])
        return train_ds.map(lambda x: (preprossesing(x)))

    ds = create_dataset(config['batch_size'], config['shape'])

    def representative_data_gen():
        for input_value in ds.take(n_samples):
            yield [input_value]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    converter.representative_dataset = representative_data_gen
    converter.post_training_quantize = not args.QAT
    converter.inference_type = tf.int8
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = output_type  # or tf.uint8

    return converter.convert()



# Argument parser
def parse_args():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--weights-path', type=str, default=None)
    my_parser.add_argument('--dataset-path', type=str, default=None)
    my_parser.add_argument('--config', type=str, default='KP2D88')
    my_parser.add_argument('--QAT', action='store_true')
    my_parser.add_argument('--legacy', action='store_true')
    my_parser.add_argument('--to-coral', action='store_true')
    my_parser.add_argument('--batch-size', type=int, default=4)
    my_parser.add_argument('--n-samples', type=int, default=100)

    return my_parser.parse_args()

args = parse_args()
config = getattr(configs.model_configs, args.config)
config['batch_size'] = args.batch_size

# Configure output path
post = ""
path = ""

if args.QAT:
    post += "QAT"

if args.to_coral:
    print("Changing model config to coral (disabling subpixel convolution)")
    config['use_subpixel'] = False
    config['use_leaky_relu'] = False
    post += "_coral"

if args.weights_path:
    path, _ = os.path.split(args.weights_path)
else:
    print("No model loaded creating new model for inference time testing")
    post += "_inf"

model_path = os.path.join(path,args.config + post +'.tflite')

# Initialize and load model
keypoint_net = KeypointNetMobile(QAT=args.QAT,**config)
keypoint_net.trainable = True
keypoint_net.compile()
dummy_input = np.random.rand(1, config['shape'][0], config['shape'][1],3)

if args.QAT:
    keypoint_net(dummy_input, training=True)
else:
    keypoint_net(dummy_input, training=False)

if args.weights_path:
    keypoint_net.load_weights(args.weights_path)

keypoint_net.trainable=False
keypoint_net.compile()


print(keypoint_net.keypoint_net_raw.summary())
print("Model path:", model_path)

model = keypoint_net.keypoint_net_raw

tflite_model = convert(model, config, output_type=tf.int8, train_path=args.dataset_path, n_samples=args.n_samples)

with open(model_path, 'wb') as f:
    f.write(tflite_model)
    print("Model written to:" + model_path)

info = {'tflite_args':vars(args), 'config': config}
with open(os.path.join(path,args.config + post +'.json'), "w") as fp:
    json.dump(info, fp, indent=4)

