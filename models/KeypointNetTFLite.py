import numpy as np
from time import time
import tensorflow as tf
from utils.grid_sample import grid_sample_2d_np

def score_border_mask_np(score):
    B, Hc, Wc,_ = score.shape
    border_mask = np.ones([B, Hc, Wc])
    border_mask[:, 0] = 0
    border_mask[:, Hc - 1] = 0
    border_mask[:, :, 0] = 0
    border_mask[:, :, Wc - 1] = 0
    border_mask = np.expand_dims(border_mask, -1)
    return score * border_mask

def calculate_coordinates_np(center_shift, cell, cross_ratio, H, W):
    step = (cell - 1) / 2.

    B, Hc, Wc,_ = center_shift.shape

    xs = np.linspace(0, Wc - 1, Wc)
    ys = np.linspace(0, Hc - 1, Hc)

    xs, ys = np.meshgrid(xs, ys)

    xs = np.expand_dims(xs, 0).repeat(B, axis=0)
    ys = np.expand_dims(ys, 0).repeat(B, axis=0)

    center_base = np.stack([xs, ys], axis=-1) * cell + step

    coord_un = center_base + center_shift * cross_ratio * step
    coord = coord_un.copy()

    coord[:, 0] = np.clip(coord_un[:, 0], a_min=0, a_max=W - 1)
    coord[:, 1] = np.clip(coord_un[:, 1], a_min=0, a_max=H - 1)
    return coord

def sample_n_norm_feat_np(feat, coord, H, W):
    coord_np = coord[:, :,:,:2].copy()

    x = (coord_np[:,:,:, 0] / (float(W - 1) / 2.)) - 1.
    y = (coord_np[:,:,:, 1] / (float(H - 1) / 2.)) - 1.
    coord_norm = np.stack([x, y], axis=-1)
    feat_np = grid_sample_2d_np(feat.copy(),coord_norm)
    dn = np.linalg.norm(feat_np, ord=2, axis=-1)  # Compute the norm.
    return feat_np / np.expand_dims(dn, -1)  # Divide by norm to normalize.

class KP2DTFLite():
    """
    Wrapper for TFlite model to be used during evaluation

    """
    def __init__(self, tf_lite_model_path, downsample=3, use_tpu=True, use_upconv=True, q_out = True):
        if use_tpu:
            from pycoral.utils import edgetpu
            self.interpreter = edgetpu.make_interpreter(tf_lite_model_path)
        else:
            try:
                import tensorflow.lite as tflite
            except:
                print("Did not find tensorflow trying to import from tensorflow runtime")
                import tflite_runtime.interpreter as tflite

            self.interpreter = tflite.Interpreter(tf_lite_model_path, num_threads=8)
        self.interpreter.allocate_tensors()
        self.upconv = use_upconv
        self.q_out = q_out
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.cell = pow(2,downsample)
        self.cross_ratio = 2.0

    def __call__(self, x, *args, **kwargs):
        x = tf.cast(x*127.5, tf.int8)
        return self.inference(x)

    def get_outputs(self):
        for idx in range(3):

            if self.output_details[idx]['shape'][-1] == 1:
                score_idx = idx
            if self.output_details[idx]['shape'][-1] == 2:
                loc_idx = idx
            if self.output_details[idx]['shape'][-1] > 2:
                feat_idx = idx

        feat = self.interpreter.get_tensor(self.output_details[feat_idx]['index']).astype(np.float32)
        score = self.interpreter.get_tensor(self.output_details[score_idx]['index']).astype(np.float32)
        center_shift = self.interpreter.get_tensor(self.output_details[loc_idx]['index']).astype(np.float32)
        return score, center_shift, feat

    def inference(self, x, return_time=False):
        assert (self.input_details[0]['shape'] == x.shape).all(), f"Shapes don't match expected shape: {self.input_details[0]['shape']}, got: {x.shape}"
        B, H, W,C = x.shape

        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        start = time()
        self.interpreter.invoke()
        invoke_time = time()-start
        score, center_shift, feat = self.get_outputs()

        if self.q_out:
            center_shift = center_shift.astype(np.float32) / 127.5
            feat = feat.astype(np.float32) / 127.5
            score = score.astype(np.float32) / 255. + 0.5
        score = score_border_mask_np(score)
        coord = calculate_coordinates_np(center_shift, self.cell, self.cross_ratio, H, W)
        if self.upconv:
            feat = sample_n_norm_feat_np(feat, coord, H, W)
        if return_time:
            return score, coord, feat, invoke_time
        return tf.convert_to_tensor(score), tf.convert_to_tensor(coord), tf.convert_to_tensor(feat)

    def set_trainable(self, trainable):
        pass

    def compile(self):
        pass

    def inference_test(self, n=100):

        img = np.array(np.random.random_sample(self.input_details[0]['shape']), dtype=np.int8)
        t = []
        t_inv = []
        for i in range(n):
            start = time()
            score, coord, feat, inf_time = self.inference(img, return_time=True)
            dur = time() - start
            #print("inference took", dur, "s")
            #print("invoke took", inf_time, "s")
            t.append(dur)
            t_inv.append(inf_time)
        first_inference = 1/t[0]
        first_inference_inv = 1/t_inv[0]
        mean_fps = 1/np.array(t[1:]).mean()
        mean_fps_inv = 1/np.array(t_inv[1:]).mean()
        print("first inference: {:.3f} fps / {:.3f} ms".format(first_inference, t[0]*1000))
        print("first invoke: {:.3f} fps / {:.3f} ms".format(first_inference_inv, t_inv[0]*1000))
        print("mean inference: {:.3f} fps / {:.3f} ms".format(mean_fps.item(), np.array(t[1:]).mean().item()*1000))
        print("mean invoke: {:.3f} fps / {:.3f} ms".format(mean_fps_inv.item(), np.array(t_inv[1:]).mean().item()*1000))
        print("per pixel time",np.array(t_inv[1:]).mean().item()*1000000/img.shape[1]/img.shape[2],img.shape[1],img.shape[2])
        return score, coord, feat

    def set_trainable(self,trainable):
        pass
    def compile(self):
        pass