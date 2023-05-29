import tensorflow as tf
from evaluation.descriptor_evaluation import (compute_homography, compute_matching_score)
from evaluation.detector_evaluation import compute_repeatability
import numpy as np
from utils.keypoints import draw_keypoints
import cv2
from matplotlib.cm import get_cmap


def debug_plots(input_img, input_img_aug, target_score, target_uv_pred, source_score, source_uv_pred, W, H, top_k2=150):
    # Generate visualization data
    vis_ori = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    vis_ori = ((vis_ori * 127.5) + 127.5).astype(np.uint8)

    vis_aug = cv2.cvtColor(input_img_aug, cv2.COLOR_BGR2RGB)
    vis_aug = ((vis_aug * 127.5) + 127.5).astype(np.uint8)

    _, top_k = tf.math.top_k(tf.reshape(target_score, [-1]), k=top_k2)  # JT: Target frame keypoints
    points = np.reshape(target_uv_pred.numpy(), (-1, 2))[top_k.numpy(), :]
    vis_ori = draw_keypoints(vis_ori, points, (0, 0, 255))

    # _, top_k = tf.math.top_k(tf.reshape(source_score, [-1]), k=top_k)
    # points = np.reshape(source_uv_warped.numpy(), (-1, 2))[top_k.numpy(), :]
    # vis_ori = draw_keypoints(vis_ori, points, (255, 0, 255))

    _, top_k = tf.math.top_k(tf.reshape(source_score, [-1]), k=top_k2)
    points = np.reshape(source_uv_pred.numpy(), (-1, 2))[top_k.numpy(), :]
    vis_aug = draw_keypoints(vis_aug, points, (0, 0, 255))

    cm = get_cmap('plasma')
    heatmap = tf.squeeze(target_score).numpy()
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    heatmap = cv2.resize(heatmap, (W, H))
    heatmap = cm(heatmap)[:, :, :3]
    vis = {}
    vis['img_ori'] = np.clip(vis_ori, 0, 255) / 255.
    vis['heatmap'] = np.clip(heatmap * 255, 0, 255) / 255.
    vis['aug'] = np.clip(vis_aug, 0, 255) / 255.

    cv2.imshow('org', vis['img_ori'])
    cv2.imshow('heatmap', vis['heatmap'])
    cv2.imshow('aug', vis['aug'])
    cv2.waitKey(1)

def evaluate_keypoint_net(data_loader, keypoint_net, top_k=300, debug=False, training=False):
    """Keypoint net evaluation script. 

    Parameters
    ----------
    data_loader: torch.utils.data.DataLoader
        Dataset loader. 
    keypoint_net: torch.nn.module
        Keypoint network.
    output_shape: tuple [W,H]
        Original image shape.
    top_k: int
        Number of keypoints to use to compute metrics, selected based on probability.    
    use_color: bool
        Use color or grayscale images.
    """
    conf_threshold = 0.7
    keypoint_net.set_trainable(False)
    keypoint_net.compile()
    localization_err, repeatability = [], []
    correctness1, correctness3, correctness5, MScore, MSum = [], [], [], [],[]
    progbar = tf.keras.utils.Progbar(len(data_loader), verbose=1,stateful_metrics=["Repeatability",
                   "Localization Error",
                   "Correctness 1",
                   "Correctness 3",
                   "Correctness 5",
                   "MScore",
                   "MSum"])

    for i,sample in enumerate(iter(data_loader)):

        image = sample['image']
        warped_image = sample['image_aug']

        _, H, W, _ = image.shape

        score_1, coord_1, desc1 = keypoint_net(image, training= training)
        score_2, coord_2, desc2 = keypoint_net(warped_image, training=training)

        if debug:
            debug_plots(image[0], warped_image[0], score_1[0], coord_1[0],score_2[0],coord_2[0], W,H, top_k2=top_k)
        B, Hc, Wc, C = desc1.shape

        # Scores & Descriptors
        score_1 = tf.concat([coord_1, score_1], axis=-1)
        score_1 = tf.reshape(score_1, (-1, 3)).numpy()

        score_2 = tf.concat([coord_2, score_2], axis=-1)
        score_2 = tf.reshape(score_2, (-1, 3)).numpy()

        desc1 = tf.reshape(desc1, (-1, C)).numpy()
        desc2 = tf.reshape(desc2, (-1, C)).numpy()

        # Filter based on confidence threshold
        desc1 = desc1[score_1[:, 2] > conf_threshold, :]
        desc2 = desc2[score_2[:, 2] > conf_threshold, :]

        score_1 = score_1[score_1[:, 2] > conf_threshold, :]
        score_2 = score_2[score_2[:, 2] > conf_threshold, :]

        # Prepare data for eval
        data = {'image': sample['image'][0],
                'image_shape': (W, H),
                'image_aug': sample['image_aug'][0],
                'homography': sample['homography'][0],
                'prob': score_1,
                'warped_prob': score_2,
                'desc': desc1,
                'warped_desc': desc2}

        _, _, rep, loc_err = compute_repeatability(data, keep_k_points=top_k, distance_thresh=3)
        if (rep != -1) and (loc_err != -1):
            repeatability.append(rep)
            localization_err.append(loc_err)

        # Compute correctness
        c1, c2, c3, M_sum = compute_homography(data, keep_k_points=top_k, debug=debug)
        correctness1.append(c1)
        correctness3.append(c2)
        correctness5.append(c3)
        MSum.append(M_sum)

        # Compute matching score
        mscore = compute_matching_score(data, keep_k_points=top_k)
        MScore.append(mscore)
        metrics = [("Repeatability", rep),
                   ("Localization Error", loc_err),
                   ("Correctness 1", np.mean(correctness1)),
                   ("Correctness 3", np.mean(correctness3)),
                   ("Correctness 5", np.mean(correctness5)),
                   ("MScore", mscore),
                   ("MSum", M_sum)]
        progbar.update(i,metrics)
    print("\n")
    return {"Repeatability": np.mean(repeatability),
                   "Localization Error": np.mean(localization_err),
                   "Correctness 1": np.mean(correctness1),
                   "Correctness 3": np.mean(correctness3),
                   "Correctness 5": np.mean(correctness5),
                   "MScore": np.mean(MScore),
                   "MSum": np.mean(MSum)}

