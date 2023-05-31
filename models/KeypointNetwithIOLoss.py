# Copyright 2020 Toyota Research Institute.  All rights reserved.
import tensorflow as tf
import cv2
import numpy as np
from matplotlib.cm import get_cmap

from networks.inlier_net import InlierNet

from utils.keypoints import draw_keypoints
from networks.keypoint_net_mobile import KeypointNetMobile
from utils.grid_sample import grid_sample_2d

@tf.function
def build_descriptor_loss(source_des, target_des, source_points, tar_points, tar_points_un, keypoint_mask=None, relax_field=8,epsilon=1e-8, eval_only=False):
    """Desc Head Loss, per-pixel level triplet loss from https://arxiv.org/pdf/1902.11046.pdf..
    Parameters
    ----------
    source_des: torch.Tensor (B,256,H/8,W/8)
        Source image descriptors.
    target_des: torch.Tensor (B,256,H/8,W/8)
        Target image descriptors.
    source_points: torch.Tensor (B,H/8,W/8,2)
        Source image keypoints
    tar_points: torch.Tensor (B,H/8,W/8,2)
        Target image keypoints
    tar_points_un: torch.Tensor (B,2,H/8,W/8)
        Target image keypoints unnormalized
    eval_only: bool
        Computes only recall without the loss.
    Returns
    -------
    loss: torch.Tensor
        Descriptor loss.
    recall: torch.Tensor
        Descriptor match recall.
    """

    batch_size, _, _,C = source_des.shape
    loss, recall = 0., 0.
    margins = 0.2

    for cur_ind in range(batch_size):

        if keypoint_mask is None:
            ref_desc = tf.reshape(grid_sample_2d(tf.expand_dims(source_des[cur_ind],0), tf.expand_dims(source_points[cur_ind],0))[0],[-1,C])
            tar_desc = tf.reshape(grid_sample_2d(tf.expand_dims(target_des[cur_ind],0), tf.expand_dims(tar_points[cur_ind],0))[0],[-1,C])

            tar_points_raw = tar_points_un[cur_ind].view( -1,2)
        else:
            keypoint_mask_ind = tf.squeeze(keypoint_mask[cur_ind])

            n_feat = tf.reduce_sum(tf.cast(keypoint_mask_ind, tf.int64))

            ref_desc = tf.boolean_mask(tf.squeeze(grid_sample_2d(tf.expand_dims(source_des[cur_ind], 0), tf.expand_dims(source_points[cur_ind], 0))),keypoint_mask_ind )
            tar_desc = tf.boolean_mask(tf.squeeze(grid_sample_2d(tf.expand_dims(target_des[cur_ind], 0), tf.expand_dims(tar_points[cur_ind], 0))), keypoint_mask_ind )
            ref_desc = tf.transpose(ref_desc)
            tar_desc = tf.transpose(tar_desc)
            # ref_desc = torch.nn.functional.grid_sample(source_des[cur_ind].unsqueeze(0), source_points[cur_ind].unsqueeze(0), align_corners=True).squeeze()[:, keypoint_mask_ind]
            # tar_desc = torch.nn.functional.grid_sample(target_des[cur_ind].unsqueeze(0), tar_points[cur_ind].unsqueeze(0), align_corners=True).squeeze()[:, keypoint_mask_ind]
            tar_points_raw = tf.boolean_mask(tar_points_un[cur_ind],keypoint_mask_ind)
            tar_points_raw = tf.transpose(tar_points_raw)
            if n_feat < 20:
                continue
        # Compute dense descriptor distance matrix and find nearest neighbor

        ref_desc,_ = tf.linalg.normalize(ref_desc+epsilon, axis=0)
        tar_desc,_ = tf.linalg.normalize(tar_desc+epsilon, axis=0)
        ref_desc = tf.add(ref_desc, epsilon)
        tar_desc = tf.add(tar_desc, epsilon)

        # ref_desc = ref_desc.div(torch.norm(ref_desc+epsilon, p=2, dim=0)+epsilon)
        # tar_desc = tar_desc.div(torch.norm(tar_desc+epsilon, p=2, dim=0)+epsilon)
        dmat = tf.matmul(tf.transpose(ref_desc), tar_desc)
        dmat = tf.sqrt(2-2*tf.clip_by_value(dmat, -1, 1)+epsilon)

        # Sort distance matrix
        idx = tf.argsort(dmat, axis=1)
        dmat_sorted = tf.gather(dmat,idx, batch_dims=1)

        # Compute triplet loss and recall
        candidates = tf.transpose(idx) # Candidates, sorted by descriptor distance

        # Get corresponding keypoint positions for each candidate descriptor
        match_k_x = tf.gather(tar_points_raw[0], candidates)
        match_k_y = tf.gather(tar_points_raw[1], candidates)
        #match_k_x = tar_points_raw[candidates,0]
        #match_k_y = tar_points_raw[candidates,1]
        # True keypoint coordinates
        true_x = tar_points_raw[0]
        true_y = tar_points_raw[1]

        # Compute recall as the number of correct matches, i.e. the first match is the correct one
        #correct_matches = (abs(match_k_x[0]-true_x) == 0) & (abs(match_k_y[0]-true_y) == 0)
        #recall += float(1.0 / batch_size) * (float(tf.reduce_sum(tf.cast(correct_matches, dtype=tf.float32))) / float( ref_desc.shape[1]))

        if eval_only:
            continue

        # Compute correct matches, allowing for a few pixels tolerance (i.e. relax_field)
        correct_idx = tf.math.logical_and(tf.math.less_equal(tf.abs(match_k_x - true_x), relax_field),
                                          tf.math.less_equal(tf.abs(match_k_y - true_y), relax_field))
        # Get hardest negative example as an incorrect match and with the smallest descriptor distance
        incorrect_first = tf.transpose(dmat_sorted)


        incorrect_first = tf.where(correct_idx==False, incorrect_first, 2.0) # largest distance is at most 2
        incorrect_first = tf.cast(tf.argmin(incorrect_first, axis=0), tf.int32)

        idx = tf.stack([tf.range(tf.shape(incorrect_first)[0]), incorrect_first], axis=-1)

        incorrect_first_index = tf.gather_nd(candidates, idx)

        anchor_var = ref_desc
        pos_var    = tar_desc
        neg_var    = tf.transpose(tf.gather(tf.transpose(tar_desc), incorrect_first_index))

        loss += tf.scalar_mul(1 / batch_size, triplet_margin_loss(tf.transpose(anchor_var), tf.transpose(pos_var),
                                                                  tf.transpose(neg_var), margin=margins))

    return loss, recall
@tf.function
def triplet_margin_loss(anchor_output,positive_output,negative_output, margin):
    d_pos = tf.linalg.norm(anchor_output - positive_output, axis=1)
    d_neg = tf.linalg.norm(anchor_output - negative_output, axis=1)

    loss = tf.maximum(0., margin + d_pos - d_neg)
    return tf.reduce_mean(loss)

class KeypointNetwithIOLoss(tf.keras.Model):
    """
    Model class encapsulating the KeypointNet and the IONet.

    Parameters
    ----------
    keypoint_loss_weight: float
        Keypoint loss weight.
    descriptor_loss_weight: float
        Descriptor loss weight.
    score_loss_weight: float
        Score loss weight.
    keypoint_net_learning_rate: float
        Keypoint net learning rate.
    with_io:
        Use IONet.
    use_color : bool
        Use color or grayscale images.
    do_upsample: bool
        Upsample desnse descriptor map.
    do_cross: bool
        Predict keypoints outside cell borders.
    with_drop : bool
        Use dropout.
    descriptor_loss: bool
        Use descriptor loss.
    kwargs : dict
        Extra parameters
    """
    def __init__(
        self, keypoint_loss_weight=1.0, descriptor_loss_weight=2.0, score_loss_weight=1.0,
        keypoint_net_learning_rate=0.001, with_io=True, use_color=True, do_upsample=True,
        do_cross=True, descriptor_loss=True, with_drop=True,device='cpu', debug = 'False', top_k=300, QAT=False, **kwargs):

        super().__init__()

        print("loss weights:")
        print(keypoint_loss_weight,descriptor_loss_weight,score_loss_weight)
        self.device = device
        self.keypoint_loss_weight = keypoint_loss_weight
        self.descriptor_loss_weight = descriptor_loss_weight
        self.score_loss_weight = score_loss_weight
        self.keypoint_net_learning_rate = keypoint_net_learning_rate
        self.optim_params = []

        # self.cell = 8 # Size of each output cell. Keep this fixed.
        self.border_remove = 1 # Remove points this close to the border.
        self.top_k2 = top_k
        self.relax_field = 4

        self.debug = debug
        self.use_color = use_color
        self.descriptor_loss = descriptor_loss

        self.keypoint_net = KeypointNetMobile(use_color=use_color, do_upsample=do_upsample, with_drop=with_drop, do_cross=do_cross, QAT=QAT,**kwargs)
        self.mse = tf.keras.losses.MeanSquaredError()
        #self.add_optimizer_params('KeypointNet', self.keypoint_net.parameters(), keypoint_net_learning_rate)

        self.with_io = with_io
        self.io_net = None
        if self.with_io:
            self.io_net = InlierNet()
            #self.io_net = self.io_net.to(self.device)
           #self.add_optimizer_params('InlierNet', self.io_net.parameters(),  keypoint_net_learning_rate)

        self.train_metrics = {}
        self.vis = {}

        # if torch.cuda.current_device() == 0:
        #     print('KeypointNetwithIOLoss:: with io {} with descriptor loss {}'.format(self.with_io, self.descriptor_loss))

    def add_optimizer_params(self, name, params, lr):
        self.optim_params.append(
            {'name': name, 'lr': lr, 'original_lr': lr,
             'params': filter(lambda p: p.requires_grad, params)})

    # @tf.function
    def compute_loss(self, data, training = True, debug = False):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch.
        debug : bool
            True if to compute debug data (stored in self.vis).

        Returns
        -------
        output : dict
            Dictionary containing the output of depth and pose networks
        """
        loss_dict = {}
        loss_2d = 0.0
        B, H, W,_ = data['image'].shape

        input_img = tf.identity(data['image'])
        input_img_aug = tf.identity(data['image_aug'])

        homography = data['homography']

        # Get network outputs
        source_score, source_uv_pred, source_feat = self.keypoint_net(input_img_aug, training=training)
        target_score, target_uv_pred, target_feat = self.keypoint_net(input_img, training=training)

        _, Hc, Wc,_ = target_score.shape

        # Normalize uv coordinates
        target_uv_norm = _normalize_uv_coordinates(target_uv_pred, H, W)
        source_uv_norm = _normalize_uv_coordinates(source_uv_pred, H, W)

        source_uv_warped_norm = self._warp_homography_batch(source_uv_norm, homography)
        source_uv_warped = _denormalize_uv_coordinates(source_uv_warped_norm, H, W)

        # Border mask
        border_mask = _create_border_mask(B, Hc, Wc)
        border_mask = tf.greater(border_mask,1e-3)

        d_uv_l2_min, d_uv_l2_min_index = _min_l2_norm(source_uv_warped, target_uv_pred, B)
        dist_norm_valid_mask = tf.logical_and(tf.less(d_uv_l2_min, 4), tf.reshape(border_mask,[B,Hc*Wc]))

        # Keypoint loss
        loc_loss = tf.reduce_mean(d_uv_l2_min[dist_norm_valid_mask])
        loss_2d += self.keypoint_loss_weight * loc_loss


        loss_dict['loc_loss'] = loc_loss

       # Desc Head Loss, per-pixel level triplet loss from https://arxiv.org/pdf/1902.11046.pdf.
        if self.descriptor_loss:
            metric_loss, recall_2d = build_descriptor_loss(source_feat, target_feat, source_uv_norm, source_uv_warped_norm, source_uv_warped, keypoint_mask=border_mask, relax_field=self.relax_field)
            loss_2d += self.descriptor_loss_weight * metric_loss * 2

            loss_dict['metric_loss'] = metric_loss
        else:
            _, recall_2d = build_descriptor_loss(source_feat, target_feat, source_uv_norm, source_uv_warped_norm, source_uv_warped, keypoint_mask=border_mask, relax_field=self.relax_field, eval_only=True)

        #Score Head Loss
        ts = tf.reshape(target_score,[B,Hc*Wc])
        d_uv_l2_min_index = tf.cast(d_uv_l2_min_index, dtype=tf.int32)
        temp = []

        for i in range(ts.shape[0]):
            temp.append(tf.gather(ts[i], d_uv_l2_min_index[i]))

        ts = tf.stack(temp,axis=-1)
        target_score_associated = tf.expand_dims(tf.reshape(ts,[B,Hc,Wc]),-1)
        #target_score_associated = target_score.view(B,Hc*Wc).gather(1, d_uv_l2_min_index).view(B,Hc,Wc).unsqueeze(1)
        #dist_norm_valid_mask = dist_norm_valid_mask.view(B,Hc,Wc).unsqueeze(1) & border_mask.unsqueeze(1)

        dist_norm_valid_mask = tf.reshape(dist_norm_valid_mask,[B,Hc,Wc])
        dist_norm_valid_mask = tf.expand_dims(dist_norm_valid_mask,-1) & border_mask

        d_uv_l2_min = tf.expand_dims(tf.reshape(d_uv_l2_min, [B, Hc, Wc]),-1)
        # d_uv_l2_min = d_uv_l2_min.view(B,Hc,Wc).unsqueeze(1)
        loc_err = d_uv_l2_min[dist_norm_valid_mask]

        usp_loss = tf.reduce_mean((target_score_associated[dist_norm_valid_mask] + source_score[dist_norm_valid_mask]) * (loc_err - tf.reduce_mean(loc_err)))
        loss_2d += self.score_loss_weight * usp_loss

        loss_dict['usp_loss'] = usp_loss

        #target_score_resampled = torch.nn.functional.grid_sample(target_score, source_uv_warped_norm.detach(), mode='bilinear', align_corners=True)
        target_score_resampled = grid_sample_2d(target_score, source_uv_warped_norm)


        loss_2d += self.score_loss_weight * self.mse(target_score_resampled[border_mask],
                                                                            source_score[border_mask])* 2
        if self.with_io:
            # Compute IO loss
            io_loss = self._compute_io_loss(source_score,source_feat,target_feat, target_score,
                     B, Hc, Wc, H, W,
                     source_uv_norm, target_uv_norm, source_uv_warped_norm,
                     self.device, training=training)

            loss_2d += self.keypoint_loss_weight * io_loss
            loss_dict['io_loss'] = io_loss

        if debug or self.debug:
            idx = 0
            self.debug_plots(input_img[idx], input_img_aug[idx],target_score[idx],target_uv_pred[idx], source_score[idx],source_uv_pred[idx],source_uv_warped[idx],W,H)
        del data, input_img, input_img_aug
        return loss_2d, loss_dict


    def debug_plots(self, input_img, input_img_aug,target_score,target_uv_pred, source_score,source_uv_pred,source_uv_warped,W,H):
        # Generate visualization data
        vis_ori = input_img.numpy()
        vis_ori = ((vis_ori * 127.5) + 127.5).astype(np.uint8)

        vis_aug = input_img_aug.numpy()
        vis_aug = ((vis_aug * 127.5) + 127.5).astype(np.uint8)


        vis_ori = cv2.cvtColor(vis_ori, cv2.COLOR_BGR2RGB)
        vis_aug = cv2.cvtColor(vis_aug, cv2.COLOR_BGR2RGB)

        _, top_k = tf.math.top_k(tf.reshape(target_score, [-1]), k=self.top_k2)  # JT: Target frame keypoints
        points = np.reshape(target_uv_pred.numpy(), (-1, 2))[top_k.numpy(), :]
        vis_ori = draw_keypoints(vis_ori, points, (0, 0, 255))

        _, top_k = tf.math.top_k(tf.reshape(source_score, [-1]), k=self.top_k2)
        points = np.reshape(source_uv_warped.numpy(), (-1, 2))[top_k.numpy(), :]
        vis_ori = draw_keypoints(vis_ori, points, (255, 0, 255))

        _, top_k = tf.math.top_k(tf.reshape(source_score, [-1]), k=self.top_k2)
        points = np.reshape(source_uv_pred.numpy(), (-1, 2))[top_k.numpy(), :]
        vis_aug = draw_keypoints(vis_aug, points, (0, 0, 255))

        cm = get_cmap('plasma')
        heatmap = tf.squeeze(target_score).numpy()
        heatmap -= heatmap.min()
        heatmap /= heatmap.max()
        heatmap = cv2.resize(heatmap, (W, H))
        heatmap = cm(heatmap)[:, :, :3]

        self.vis['img_ori'] = np.clip(vis_ori, 0, 255) / 255.
        self.vis['heatmap'] = np.clip(heatmap * 255, 0, 255) / 255.
        self.vis['aug'] = np.clip(vis_aug, 0, 255) / 255.

        cv2.imshow('org', self.vis['img_ori'])
        cv2.imshow('heatmap', self.vis['heatmap'])
        cv2.imshow('aug', self.vis['aug'])
        cv2.waitKey(1)
    @tf.function
    def get_worst(self, score):
        vals, idxs = tf.math.top_k(-score, k=self.top_k2)

        return -vals, idxs

    @tf.function
    def get_uv(self, uv, topk_idxs, Hc, Wc):
        return tf.gather(tf.reshape(uv, [Hc * Wc, 2]), topk_idxs)

    def get_feat(self, feat, uv):
        return grid_sample_2d(tf.expand_dims(feat,0), tf.expand_dims(tf.expand_dims(uv,0),0))

    @tf.function
    def norm_feat(self, feat, epsilon):
        return tf.divide(feat, tf.expand_dims(tf.norm(feat, ord='euclidean', axis=-1), -1) + epsilon)

    @tf.function
    def match_feat(self,source_feat_topk, target_feat_topk, epsilon ):
        dmat = tf.linalg.matmul(tf.squeeze(source_feat_topk), tf.transpose(tf.squeeze(target_feat_topk), [1, 0]))
        dmat = tf.sqrt(2 - 2 * tf.clip_by_value(dmat, -1, 1) + epsilon)
        dmat_min_indice = tf.cast(tf.argmin(dmat, axis=0), dtype=tf.int32)
        dmat_idx = tf.stack([tf.range(tf.shape(dmat_min_indice)[0]), dmat_min_indice], axis=1)
        dmat_min = tf.gather_nd(dmat, dmat_idx)
        return dmat_min, dmat_min_indice

    @tf.function
    def build_point_pair(self, source_uv_norm_topk, target_uv_norm_topk_associated, dmat_min):
        return tf.expand_dims(tf.concat([source_uv_norm_topk, target_uv_norm_topk_associated, tf.expand_dims(dmat_min, 1)], 1),0)

    @tf.function
    def _compute_io_loss(self, source_score,source_feat,target_feat, target_score,
                         B, Hc, Wc, H, W,
                         source_uv_norm, target_uv_norm, source_uv_warped_norm,
                         device, epsilon = 1e-8, training = True):

        source_score = tf.reshape(source_score,[B, Hc * Wc])
        target_score = tf.reshape(target_score,[B, Hc * Wc])
        point_pairs = []
        source_uv_warped_norm_topk_batch = []
        target_uv_norm_topk_associated_raw_batch = []
        for i in range(B):
            vals, idxs= self.get_worst(source_score[i])
            vals_t, idxs_t= self.get_worst(target_score[i])

            source_uv_norm_topk = self.get_uv(source_uv_norm[i], idxs, Hc, Wc)
            target_uv_norm_topk = self.get_uv(target_uv_norm[i], idxs_t,Hc, Wc)
            source_uv_warped_norm_topk = self.get_uv(source_uv_warped_norm[i], idxs, Hc, Wc)

            source_feat_topk = self.get_feat(source_feat[i], source_uv_norm_topk)
            target_feat_topk = self.get_feat(target_feat[i], target_uv_norm_topk)


            source_feat_topk = self.norm_feat(source_feat_topk, epsilon=epsilon)
            target_feat_topk = self.norm_feat(target_feat_topk, epsilon=epsilon)

            dmat_min, dmat_min_indice = self.match_feat(source_feat_topk,target_feat_topk,epsilon)

            target_uv_norm_topk_associated = tf.gather(target_uv_norm_topk, dmat_min_indice,axis=0)

            point_pairs.append(self.build_point_pair(source_uv_norm_topk, target_uv_norm_topk_associated, dmat_min))

            source_uv_warped_norm_topk_batch.append(tf.expand_dims(source_uv_warped_norm_topk,0))
            target_uv_norm_topk_associated_raw_batch.append(tf.expand_dims(target_uv_norm_topk_associated,0))


        point_pairs = tf.stack(point_pairs)


        inlier_pred = self.io_net(point_pairs, training = training)

        target_uv_norm_topk_associated_raw = tf.stack(target_uv_norm_topk_associated_raw_batch)
        x = (target_uv_norm_topk_associated_raw[:, :,:, 0] + 1) * (float(W - 1) / 2.)
        y = (target_uv_norm_topk_associated_raw[:, :,:, 1] + 1) * (float(H - 1) / 2.)
        target_uv_norm_topk_associated_raw = tf.stack([x, y], axis=-1)

        source_uv_warped_norm_topk_raw = tf.stack(source_uv_warped_norm_topk_batch)
        x = (source_uv_warped_norm_topk_raw[:, :,:, 0] + 1) * (float(W - 1) / 2.)
        y = (source_uv_warped_norm_topk_raw[:, :,:, 1] + 1) * (float(H - 1) / 2.)
        source_uv_warped_norm_topk_raw = tf.stack([x, y], axis=-1)


        matching_score = tf.expand_dims(tf.norm(target_uv_norm_topk_associated_raw - source_uv_warped_norm_topk_raw, ord='euclidean', axis=1),1)
        inlier_mask = tf.cast(matching_score > 4, tf.float32)
        inlier_gt = 2 * inlier_mask - 1

        return float(tf.reduce_sum(inlier_mask) > 10 )*self.mse(inlier_pred, inlier_gt)

    @tf.function
    def _warp_homography_batch(self, sources, homographies):
        """Batch warp keypoints given homographies.

        Parameters
        ----------
        sources: torch.Tensor (B,H,W,C)
            Keypoints vector.
        homographies: torch.Tensor (B,3,3)
            Homographies.

        Returns
        -------
        warped_sources: torch.Tensor (B,H,W,C)
            Warped keypoints vector.
        """
        B, H, W, _ = sources.shape
        warped_sources = []

        for b in range(B):
            source = tf.identity(sources[b])
            source = tf.reshape(source,[-1,2])
            source = tf.add(homographies[b, :, 2], tf.matmul(source, tf.transpose(homographies[b,:, :2])))
            #source = torch.addmm(homographies[b, :, 2], source, homographies[b, :, :2].t())
            #source.mul_(1 / source[:, 2].unsqueeze(1))
            source = tf.multiply(source, 1/tf.expand_dims(source[:,2],1))

            source = tf.reshape(source[:, :2],[H, W, 2])
            warped_sources.append(source)
        return tf.stack(warped_sources, axis=0)
@tf.function
def _normalize_uv_coordinates(uv_pred, H, W):
    uv_norm = tf.identity(uv_pred)
    x = (uv_norm[:,:,:, 0] / (float(W - 1) / 2.)) - 1.
    y = (uv_norm[:,:,:, 1] / (float(H - 1) / 2.)) - 1.
    uv_norm = tf.stack([x,y], axis=-1)
    return uv_norm
@tf.function
def _denormalize_uv_coordinates(uv_norm, H, W):
    uv_pred = tf.identity(uv_norm)
    x = (uv_pred[:, :, :, 0] + 1) * (float(W - 1) / 2.)
    y = (uv_pred[:, :, :, 1] + 1) * (float(H - 1) / 2.)
    uv_pred = tf.stack([x,y], axis=-1)
    return uv_pred

@tf.function
def _create_border_mask(B, Hc, Wc):
    m = tf.ones([B, Hc - 2, Wc - 2])
    m_s = tf.zeros([B, Hc - 2, 1])
    m = tf.concat([m_s, m, m_s], axis=2)
    m_top = tf.zeros([B, 1, Wc])
    border_mask = tf.concat([m_top, m, m_top], axis=1)
    border_mask = tf.expand_dims(border_mask, -1)
    return border_mask

@tf.function
def _min_l2_norm(source_uv_warped, target_uv_pred, B, epsilon = 1e-8):
    d_uv_mat_abs = tf.math.abs(tf.expand_dims(tf.reshape(source_uv_warped,[B,-1,2]), 2) - tf.expand_dims(tf.reshape(target_uv_pred,[B, -1, 2]), 1))
    d_uv_l2_mat = tf.norm(d_uv_mat_abs+epsilon, ord='euclidean', axis=3)
    return tf.math.reduce_min(d_uv_l2_mat, axis=2),tf.argmin(d_uv_l2_mat, axis=2)
