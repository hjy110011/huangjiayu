# type: uploaded file
# fileName: uav_head.py
# fullContent:
# Copyright (c) Tencent Inc. All rights reserved.
import math
import copy
from typing import List, Optional, Tuple, Union, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.structures.bbox import bbox_overlaps
from mmengine.dist import get_dist_info
from mmengine.structures import InstanceData
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, InstanceList, OptInstanceList
from mmdet.models.utils import (multi_apply, unpack_gt_instances,
                                filter_scores_and_topk)
from mmyolo.registry import MODELS
from mmyolo.models.dense_heads import YOLOv8HeadModule, YOLOv8Head
from mmyolo.models.utils import gt_instances_preprocess
from mmcv.cnn.bricks import build_norm_layer


@MODELS.register_module()
class UavAHead(YOLOv8Head):
    """YOLO-World Head with NWD Loss for Small Object Detection"""

    def __init__(self, world_size=-1,
                 att_embeddings=None,
                 prev_intro_cls=0,
                 cur_intro_cls=0,
                 thr=0.8,
                 alpha=0.5,
                 use_sigmoid=True,
                 device='cuda',
                 prev_distribution=None,
                 distributions=None,
                 top_k=10,
                 # NWD 超参数
                 nwd_factor=1.0,
                 nwd_constant=12.8,
                 *args, **kwargs) -> None:
        self.positive_distributions = None
        self.negative_distributions = None
        super().__init__(*args, **kwargs)
        self.thr = thr
        self.world_size = world_size
        self.device = device
        self.alpha = alpha
        self.use_sigmoid = use_sigmoid
        self.distributions = distributions
        self.thrs = [thr]
        self.prev_intro_cls = prev_intro_cls
        self.cur_intro_cls = cur_intro_cls
        self.prev_distribution = prev_distribution
        self.top_k = top_k

        # NWD Loss Params
        self.nwd_factor = nwd_factor
        self.nwd_constant = nwd_constant

        self.load_att_embeddings(att_embeddings)
        self.register_buffer(
            'class_sample_ema',
            torch.zeros(self.num_classes)
        )
        self.class_ema_momentum = 0.999

    def disable_log(self):
        self.positive_distributions = None
        self.negative_distributions = None
        print('disable log')

    def enable_log(self):
        self.reset_log()
        print('enable log')

    def load_att_embeddings(self, att_embeddings):
        if att_embeddings is None:
            self.att_embeddings = None
            self.disable_log()
            return
        atts = torch.load(att_embeddings)
        self.texts = atts['att_text']
        self.all_atts = atts['att_embedding']
        if self.prev_distribution is not None:
            prev_atts_num = len(torch.load(self.prev_distribution, map_location='cuda')['positive_distributions'][
                                    self.thrs.index(self.thr)])
        else:
            prev_atts_num = 0
        self.att_embeddings = torch.nn.Parameter(atts['att_embedding'].float()[prev_atts_num:])

    def reset_log(self, intetval=0.0001):
        self.positive_distributions = [{att_i: torch.zeros(int((1) / intetval)).to(self.device)
                                        for att_i in range(self.att_embeddings.shape[0])} for _ in self.thrs]
        self.negative_distributions = [{att_i: torch.zeros(int((1) / intetval)).to(self.device)
                                        for att_i in range(self.att_embeddings.shape[0])} for _ in self.thrs]

    # [创新点] 计算 Normalized Wasserstein Distance
    # def compute_nwd(self, pred_bboxes, gt_bboxes):
    #     # pred_bboxes, gt_bboxes: [N, 4] (x1, y1, x2, y2)
    #     # Convert to (cx, cy, w, h)
    #     pred_cx = (pred_bboxes[:, 0] + pred_bboxes[:, 2]) / 2
    #     pred_cy = (pred_bboxes[:, 1] + pred_bboxes[:, 3]) / 2
    #     pred_w = (pred_bboxes[:, 2] - pred_bboxes[:, 0]).clamp(min=1e-6)
    #     pred_h = (pred_bboxes[:, 3] - pred_bboxes[:, 1]).clamp(min=1e-6)
    #
    #     gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2
    #     gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2
    #     gt_w = (gt_bboxes[:, 2] - gt_bboxes[:, 0]).clamp(min=1e-6)
    #     gt_h = (gt_bboxes[:, 3] - gt_bboxes[:, 1]).clamp(min=1e-6)
    #
    #     # Wasserstein Distance calculation for Gaussian modeled boxes
    #     w2 = torch.pow(pred_cx - gt_cx, 2) + torch.pow(pred_cy - gt_cy, 2) + \
    #          torch.pow(pred_w / 2 - gt_w / 2, 2) + torch.pow(pred_h / 2 - gt_h / 2, 2)
    #
    #     return torch.exp(-torch.sqrt(w2.clamp(min=1e-7)) / self.nwd_constant)
    def compute_nwd(self, pred_bboxes, gt_bboxes):
        # ... (前面的坐标转换代码保持不变) ...
            # pred_bboxes, gt_bboxes: [N, 4] (x1, y1, x2, y2)
            # Convert to (cx, cy, w, h)
        pred_cx = (pred_bboxes[:, 0] + pred_bboxes[:, 2]) / 2
        pred_cy = (pred_bboxes[:, 1] + pred_bboxes[:, 3]) / 2
        pred_w = (pred_bboxes[:, 2] - pred_bboxes[:, 0]).clamp(min=1e-6)
        pred_h = (pred_bboxes[:, 3] - pred_bboxes[:, 1]).clamp(min=1e-6)

        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2
        gt_w = (gt_bboxes[:, 2] - gt_bboxes[:, 0]).clamp(min=1e-6)
        gt_h = (gt_bboxes[:, 3] - gt_bboxes[:, 1]).clamp(min=1e-6)

        # Wasserstein Distance calculation
        w2 = torch.pow(pred_cx - gt_cx, 2) + torch.pow(pred_cy - gt_cy, 2) + \
             torch.pow(pred_w / 2 - gt_w / 2, 2) + torch.pow(pred_h / 2 - gt_h / 2, 2)

        # [核心修复]
        # 1. 使用 + 1e-7 代替 clamp，避免 clamp 导致的梯度截断问题
        # 2. 确保 w2 不为负 (虽然理论上不会，但在 float 精度误差下可能出现 -0.00000...)
        dist = torch.sqrt(torch.abs(w2) + 1e-7)

        # 3. 再次确保分母不为 0 (虽然 nwd_constant 是常数，但习惯性保护)
        nwd = torch.exp(-dist / (self.nwd_constant + 1e-7))

        return nwd


    def loss(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
             batch_data_samples: Union[list, dict], fusion_att: bool = False,
             epoch_info: Optional[dict] = None) -> dict:
        outs = self(img_feats, txt_feats)
        if self.att_embeddings is None:
            loss_inputs = outs + (None, batch_data_samples['bboxes_labels'],
                                  batch_data_samples['img_metas'])
            losses = self.loss_by_feat(*loss_inputs, epoch_info=epoch_info)
            return losses

        if fusion_att:
            num_att = self.att_embeddings.shape[0]
            att_feats = txt_feats[:, -num_att:, :]
            txt_feats = txt_feats[:, :-num_att, :]
        else:
            att_feats = self.att_embeddings[None].repeat(txt_feats.shape[0], 1, 1)

        with torch.no_grad():
            att_outs = self(img_feats, att_feats)[0]

        loss_inputs = outs + (att_outs, batch_data_samples['bboxes_labels'],
                              batch_data_samples['img_metas'])
        losses = self.loss_by_feat(*loss_inputs, epoch_info=epoch_info)

        return losses

    def loss_by_feat(
            self,
            cls_scores,
            bbox_preds,
            bbox_dist_preds,
            att_scores,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=None,
            epoch_info=None):

        num_imgs = len(batch_img_metas)

        # -------------------- priors --------------------
        current_featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes
            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)
            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(mlvl_priors_with_stride, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # -------------------- GT --------------------
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # -------------------- preds --------------------
        flatten_cls_preds = torch.cat([
            cls.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
            for cls in cls_scores
        ], dim=1)

        flatten_pred_bboxes = torch.cat([
            bbox.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox in bbox_preds
        ], dim=1)

        flatten_dist_preds = torch.cat([
            dist.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for dist in bbox_dist_preds
        ], dim=1)

        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2],
            flatten_pred_bboxes,
            self.stride_tensor[..., 0])

        # -------------------- assign --------------------
        assigned = self.assigner(
            flatten_pred_bboxes.detach().type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(),
            self.flatten_priors_train,
            gt_labels,
            gt_bboxes,
            pad_bbox_flag)

        assigned_bboxes = assigned['assigned_bboxes']
        assigned_scores = assigned['assigned_scores']
        fg_mask = assigned['fg_mask_pre_prior']

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        # -------------------- Class Balanced Loss --------------------
        # with torch.no_grad():
        #     pos_mask = assigned_scores > 0
        #     batch_class_count = pos_mask.sum(dim=(0, 1))
        #     self.class_sample_ema.mul_(self.class_ema_momentum)
        #     self.class_sample_ema.add_((1.0 - self.class_ema_momentum) * batch_class_count)
        #     beta = 0.999
        #     effective_num = 1.0 - torch.pow(beta, self.class_sample_ema + 1e-6)
        #     class_weights = (1.0 - beta) / effective_num
        #     class_weights = class_weights / class_weights.mean()
        #
        # cls_weight_map = class_weights.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            pos_mask = assigned_scores > 0
            batch_class_count = pos_mask.sum(dim=(0, 1))
            self.class_sample_ema.mul_(self.class_ema_momentum)
            self.class_sample_ema.add_((1.0 - self.class_ema_momentum) * batch_class_count)

            beta = 0.999
            # [修复] 增加 eps 防止 effective_num 为 0
            # 这里的 1e-6 在 FP16 下可能因精度丢失被忽略，建议加大保护逻辑
            term = torch.pow(beta, self.class_sample_ema)
            effective_num = 1.0 - term

            # 强制 effective_num 最小为 1e-4，防止除以 0 产生 Inf
            effective_num = effective_num.clamp(min=1e-4)

            class_weights = (1.0 - beta) / effective_num

            # [修复] 防止 mean 为 0
            mean_weight = class_weights.mean()
            if mean_weight > 0:
                class_weights = class_weights / mean_weight
            else:
                class_weights = torch.ones_like(class_weights)

        cls_weight_map = class_weights.unsqueeze(0).unsqueeze(0)

        # -------------------- cls loss --------------------
        loss_cls = self.loss_cls(
            flatten_cls_preds,
            assigned_scores,
            weight=cls_weight_map
        ).sum() / assigned_scores_sum

        progress = 0.0
        if epoch_info is not None:
            cur_e = epoch_info.get('current_epoch', 0)
            max_e = epoch_info.get('max_epochs', 1)
            progress = min(max(cur_e / float(max_e), 0.0), 1.0)

        cls_curriculum_weight = 1.5 - progress
        loss_cls = loss_cls * cls_curriculum_weight

        # -------------------- bbox / dfl / NWD loss --------------------
        assigned_bboxes = assigned_bboxes / self.stride_tensor
        flatten_pred_bboxes = flatten_pred_bboxes / self.stride_tensor

        if fg_mask.sum() > 0:
            prior_mask = fg_mask.unsqueeze(-1).repeat(1, 1, 4)
            pred_pos = torch.masked_select(flatten_pred_bboxes, prior_mask).reshape(-1, 4)
            gt_pos = torch.masked_select(assigned_bboxes, prior_mask).reshape(-1, 4)
            base_weight = torch.masked_select(assigned_scores.sum(-1), fg_mask).unsqueeze(-1)

            # --- ProgLoss Difficulty ---
            with torch.no_grad():
                ious = bbox_overlaps(pred_pos, gt_pos, is_aligned=True).clamp(0, 1)
            difficulty = 1.0 - ious
            with torch.no_grad():
                wh = gt_pos[:, 2:] - gt_pos[:, :2]
                area = (wh[:, 0] * wh[:, 1]).clamp(min=1e-6)
                area_weight = (1.0 / torch.sqrt(area)).clamp(max=5.0)
            difficulty = difficulty * area_weight

            alpha = 1.0
            gamma = 2.0
            prog_weight = 1.0 + alpha * (progress ** gamma) * difficulty
            prog_weight = prog_weight.detach()

            # [原有] IoU Loss
            loss_bbox_iou = self.loss_bbox(
                pred_pos,
                gt_pos,
                weight=base_weight * prog_weight.unsqueeze(-1)
            ) / assigned_scores_sum

            # [新增] NWD Loss: 1 - NWD
            # 对于微小目标，IoU Loss 梯度不好，NWD 补充梯度
            nwd_score = self.compute_nwd(pred_pos, gt_pos)
            loss_nwd = (1.0 - nwd_score) * (base_weight * prog_weight.unsqueeze(-1)).squeeze(-1)
            loss_nwd = loss_nwd.sum() / assigned_scores_sum

            # 融合 Loss: IoU + NWD
            loss_bbox = loss_bbox_iou + self.nwd_factor * loss_nwd

            # DFL Loss
            pred_dist_pos = flatten_dist_preds[fg_mask]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(assigned_ltrb, prior_mask).reshape(-1, 4)

            loss_dfl = self.loss_dfl(
                pred_dist_pos.reshape(-1, self.head_module.reg_max),
                assigned_ltrb_pos.reshape(-1),
                weight=(base_weight * prog_weight.unsqueeze(-1)).expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum
            )
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0

        if self.world_size == -1:
            _, world_size = get_dist_info()
        else:
            world_size = self.world_size

        return dict(
            loss_cls=loss_cls * num_imgs * world_size,
            loss_bbox=loss_bbox * num_imgs * world_size,
            loss_dfl=loss_dfl * num_imgs * world_size
        )

    # ... (Keep predict, predict_unknown, and other methods as they were in your file) ...
    # 为了节省空间，后续未修改的方法（predict_unknown, select_att 等）请保持原样
    # 只需要确保上面的 loss_by_feat 和 compute_nwd 被正确合入

    def predict(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False,
                fusion_att: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.
        """
        if self.att_embeddings.shape[0] != 25 * (self.num_classes):
            self.select_att()

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(img_feats, txt_feats)
        # outs = self.fomo_update_outs(outs)
        if self.att_embeddings is None:
            predictions = self.predict_by_feat(*outs,
                                               batch_img_metas=batch_img_metas,
                                               rescale=rescale)
            return predictions

        if fusion_att:
            num_att = self.att_embeddings.shape[0]
            att_feats = txt_feats[:, -num_att:, :]
            txt_feats = txt_feats[:, :-num_att, :]
        else:
            att_feats = self.att_embeddings[None].repeat(txt_feats.shape[0], 1, 1)

        if self.att_embeddings is not None:
            outs = self.predict_unknown(outs, img_feats, att_feats)
        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           rescale=rescale)
        return predictions

    def fomo_update_outs(self, outs):
        predictions = outs[0]
        ret_logits = []
        for prediction in predictions:
            known_logits = prediction.permute(0, 2, 3, 1)[..., :self.num_classes]
            unknown_logits = prediction.permute(0, 2, 3, 1)[..., :self.num_classes]
            unknown_logits = unknown_logits.max(-1, keepdim=True)[0]
            ret_logits.append(torch.cat([known_logits, unknown_logits], dim=-1).permute(0, 3, 1, 2))
        return (ret_logits, *outs[1:])

    def calculate_uncertainty(self, known_logits):
        known_logits = torch.clamp(known_logits, 1e-6, 1 - 1e-6)
        entropy = (-known_logits * torch.log(known_logits) - (1 - known_logits) * torch.log(1 - known_logits)).mean(
            dim=-1, keepdim=True)
        return entropy

    def select_top_k_attributes(self, adjusted_scores: Tensor, k: int = 3) -> Tensor:
        top_k_scores, _ = adjusted_scores.topk(k, dim=-1)
        top_k_average = top_k_scores.mean(dim=-1, keepdim=True)
        return top_k_average

    def compute_weighted_top_k_attributes(self, adjusted_scores: Tensor, k: int = 10) -> Tensor:
        top_k_scores, top_k_indices = adjusted_scores.topk(k, dim=-1)
        top_k_weights = F.softmax(top_k_scores, dim=-1)
        weighted_average = torch.sum(top_k_scores * top_k_weights, dim=-1, keepdim=True)
        return weighted_average

    def predict_unknown(self, outs, img_feats, att_embeddings):
        known_predictions = outs[0]
        unknown_predictions = self(img_feats, att_embeddings)[0]
        ret_logits = []

        for known_logits, unknown_logits in zip(known_predictions, unknown_predictions):
            known_logits = known_logits.sigmoid().permute(0, 2, 3, 1)
            unknown_logits = unknown_logits.sigmoid().permute(0, 2, 3, 1)

            # 1. 计算已知类别的不确定性 (Entropy)
            uncertainty = self.calculate_uncertainty(known_logits)

            # 2. 计算属性得分 (Visual Evidence)
            top_k_att_score = self.compute_weighted_top_k_attributes(unknown_logits, k=self.top_k)

            # ---------------- [创新点修改 START] ----------------
            # 策略：Attribute-Gated Uncertainty Calibration
            # 解释：在无人机图像中，背景噪声往往具有高不确定性。我们利用属性得分作为"物体性(Objectness)"的先验，
            #      来抑制背景的高不确定性，同时增强真实未知物体的得分。

            # 归一化不确定性到 [0, 1] 范围 (可选，防止数值爆炸)
            uncertainty_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-6)

            # 核心公式：S_unk = Att_Score * (1 + alpha * Uncertainty)
            # 只有当 Att_Score 高时，Uncertainty 才能显著贡献分数。
            # 如果 Att_Score 低（背景），Uncertainty 再高也会被抑制。
            calibration_factor = 0.5  # 这是一个超参数，可以调节不确定性的权重
            fused_score = top_k_att_score * (1 + calibration_factor * uncertainty_norm)

            # 最终分数：抑制已被识别为 Known 的区域
            max_known_score = known_logits.max(-1, keepdim=True)[0]
            unknown_logits_final = fused_score * (1 - max_known_score)
            # ---------------- [创新点修改 END] ------------------

            # 合并已知和未知类别的最终预测结果
            logits = torch.cat([known_logits, unknown_logits_final], dim=-1).permute(0, 3, 1, 2)
            ret_logits.append(logits)

        return (ret_logits, *outs[1:])

    def get_all_dis_sim(self, positive_dis, negative_dis):
        dis_sim = []
        for i in range(len(positive_dis)):
            positive = positive_dis[i]
            negative = negative_dis[i]
            positive = positive / positive.sum()
            negative = negative / negative.sum()
            dis_sim.append(self.get_sim(positive, negative))
        return torch.stack(dis_sim).to('cuda')

    def combine_distributions(self):
        if self.prev_distribution is None:
            return self.positive_distributions, self.negative_distributions
        prev_distributions = torch.load(self.prev_distribution, map_location='cuda')
        prev_positive_distributions, prev_negative_distributions = prev_distributions['positive_distributions'], \
            prev_distributions['negative_distributions']
        ret_pos, ret_neg = prev_positive_distributions, prev_negative_distributions
        for thr in self.thrs:
            thr_id = self.thrs.index(thr)
            if thr_id >= len(prev_positive_distributions) or prev_positive_distributions[thr_id] is None:
                continue
            if thr_id >= len(self.positive_distributions) or self.positive_distributions[thr_id] is None:
                continue
            cur_pos_dist = self.positive_distributions[thr_id]
            cur_neg_dist = self.negative_distributions[thr_id]
            prev_pos_dist = prev_positive_distributions[thr_id]
            prev_neg_dist = prev_negative_distributions[thr_id]
            prev_att = len(prev_pos_dist)
            prev_pos_dist.update({prev_att + k: v for k, v in cur_pos_dist.items()})
            prev_neg_dist.update({prev_att + k: v for k, v in cur_neg_dist.items()})
            ret_pos[thr_id] = prev_pos_dist
            ret_neg[thr_id] = prev_neg_dist
        return ret_pos, ret_neg

    def select_att(self, per_class=25):
        print(f'thr: {self.thr}')
        save_root = os.path.dirname(self.distributions)
        task_id = self.distributions[-5]
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        torch.save({'positive_distributions': self.positive_distributions,
                    'negative_distributions': self.negative_distributions},
                   os.path.join(save_root, f'current{task_id}.pth'))
        print('save current to {}'.format(os.path.join(save_root, f'current{task_id}.pth')))
        self.positive_distributions, self.negative_distributions = self.combine_distributions()
        torch.save({'positive_distributions': self.positive_distributions,
                    'negative_distributions': self.negative_distributions}, self.distributions)
        print('save distributions to {}'.format(self.distributions))
        distributions = torch.load(self.distributions, map_location='cuda')
        self.positive_distributions, self.negative_distributions = distributions['positive_distributions'], \
            distributions['negative_distributions']
        thr_id = self.thrs.index(self.thr)
        distribution_sim = self.get_all_dis_sim(self.positive_distributions[thr_id],
                                                self.negative_distributions[thr_id])
        all_atts = self.all_atts.to(self.att_embeddings.device)
        att_embeddings_norm = F.normalize(all_atts, p=2, dim=1)  # Normalize embeddings
        if self.use_sigmoid:
            cosine_sim_matrix = torch.matmul(att_embeddings_norm, att_embeddings_norm.T).sigmoid()
        else:
            cosine_sim_matrix = torch.matmul(att_embeddings_norm, att_embeddings_norm.T).abs()
        selected_indices = []
        for _ in range(per_class * self.num_classes):
            if len(selected_indices) == 0:
                _, idx = distribution_sim.min(dim=0)
            else:
                unselected_indices = list(set(range(len(self.texts))) - set(selected_indices))
                cosine_sim_with_selected = cosine_sim_matrix[unselected_indices][:, selected_indices].mean(
                    dim=1)
                distribution_sim_unselected = distribution_sim[unselected_indices]
                score = self.alpha * distribution_sim_unselected + (1 - self.alpha) * cosine_sim_with_selected
                idx = unselected_indices[score.argmin()]
            selected_indices.append(idx)
        selected_indices = torch.tensor(selected_indices).to(self.att_embeddings.device)
        self.att_embeddings = torch.nn.Parameter(all_atts[selected_indices]).to(self.att_embeddings.device)
        self.texts = [self.texts[i] for i in selected_indices]
        print('Selected attributes saved.')

    def get_sim(self, a, b):
        def jensen_shannon_divergence(p, q):
            m = 0.5 * (p + q)
            m = m.clamp(min=1e-6)
            js_div = 0.5 * (torch.sum(p * torch.log((p / m).clamp(min=1e-6))) +
                            torch.sum(q * torch.log((q / m).clamp(min=1e-6))))
            return js_div

        return jensen_shannon_divergence(a, b)

    def log_distribution(self, att_scores, assigned_scores):
        if not self.training or self.positive_distributions is None \
                or self.att_embeddings is None:
            return
        num_att = att_scores.shape[-1]
        num_known = assigned_scores.shape[-1]
        att_scores = att_scores.sigmoid().reshape(-1, num_att).float()
        assigned_scores = assigned_scores.reshape(-1, num_known)
        assigned_scores[:, 0: self.prev_intro_cls] = 0
        assigned_scores = assigned_scores.max(-1)[0]
        for idx, thr in enumerate(self.thrs):
            positive = (assigned_scores >= thr)
            positive_scores = att_scores[positive]
            negative_scores = att_scores[~positive]
            for att_i in range(num_att):
                self.positive_distributions[idx][att_i] += torch.histc(positive_scores[:, att_i], bins=int(1 / 0.0001),
                                                                       min=0, max=1)
                self.negative_distributions[idx][att_i] += torch.histc(negative_scores[:, att_i], bins=int(1 / 0.0001),
                                                                       min=0, max=1)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        assert len(cls_scores) == len(bbox_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors,), stride) for
            featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)
        num_classes = cls_scores[0].size(1)

        if self.att_embeddings is not None:
            flatten_cls_scores = [
                cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                      num_classes)
                for cls_score in cls_scores
            ]
        else:
            flatten_att_scores = [
                cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                      num_classes)
                for cls_score in cls_scores
            ]

        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]

        if self.att_embeddings is not None:
            flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1)
        else:
            flatten_cls_scores = torch.cat(flatten_att_scores, dim=1).sigmoid()

        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

        if with_objectnesses:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(num_imgs)]

        results_list = []
        for (bboxes, scores, objectness, img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                                                          flatten_objectness, batch_img_metas):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            if objectness is not None and score_thr > 0 and not cfg.get(
                    'yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            results = InstanceData(scores=scores,
                                   labels=labels,
                                   bboxes=bboxes[keep_idxs])

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(results=results,
                                              cfg=cfg,
                                              rescale=False,
                                              with_nms=with_nms,
                                              img_meta=img_meta)
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list


    def aug_test(self,
                 aug_batch_feats,
                 aug_batch_img_metas,
                 rescale=False,
                 with_ori_nms=False,
                 **kwargs):
        """Test function with our time augmentation."""
        raise NotImplementedError('aug_test is not implemented yet.')



    def loss_and_predict(
            self,
            img_feats: Tuple[Tensor],
            txt_feats: Tensor,
            batch_data_samples: SampleList,
            proposal_cfg: Optional[ConfigDict] = None
    ) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        outs = self(img_feats, txt_feats)

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           cfg=proposal_cfg)
        return losses, predictions

    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        return self.head_module(img_feats, txt_feats)
