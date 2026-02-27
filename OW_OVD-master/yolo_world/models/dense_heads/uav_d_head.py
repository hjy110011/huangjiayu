# Copyright (c) Tencent Inc. All rights reserved.
import math
import copy
import os
from typing import List, Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from mmengine.dist import get_dist_info, all_reduce, is_main_process
from mmengine.structures import InstanceData
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_overlaps
from mmdet.utils import OptConfigType, InstanceList, OptInstanceList
from mmdet.models.utils import multi_apply, unpack_gt_instances, filter_scores_and_topk

from mmyolo.registry import MODELS
from mmyolo.models.dense_heads import YOLOv8HeadModule, YOLOv8Head
from mmyolo.models.utils import gt_instances_preprocess


@MODELS.register_module()
class UavDHead(YOLOv8Head):
    """
    针对无人机 (UAV) 及微小目标检测优化的 YOLO-World 预测头。
    """

    def __init__(self,
                 # --- 消融实验控制开关 ---
                 use_prog_loss: bool = False,
                 use_inner_wiou: bool = False,
                 use_sad_nwd: bool = False,
                 use_bsu: bool = False,

                 # --- 核心模块超参数 ---
                 nwd_factor: float = 1.0,
                 nwd_constant: float = 36.9,
                 inner_ratio: float = 0.75,
                 tau_scale: float = 656,
                 wiou_alpha: float = 1.9,
                 wiou_delta: float = 3.0,

                 # --- 基础参数 ---
                 world_size=-1,
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
                 *args, **kwargs) -> None:

        self.positive_distributions = None
        self.negative_distributions = None
        super().__init__(*args, **kwargs)

        # 挂载开关与超参数
        self.use_prog_loss = use_prog_loss
        self.use_inner_wiou = use_inner_wiou
        self.use_sad_nwd = use_sad_nwd
        self.use_bsu = use_bsu

        self.nwd_factor = nwd_factor
        self.nwd_constant = nwd_constant
        self.inner_ratio = inner_ratio
        self.tau_scale = tau_scale
        self.wiou_alpha = wiou_alpha
        self.wiou_delta = wiou_delta

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

        # 注册用于长尾/类别平衡的全局缓冲变量
        self.register_buffer('class_sample_ema', torch.zeros(self.num_classes))
        self.register_buffer('iou_loss_ema', torch.tensor(1.0))
        self.class_ema_momentum = 0.999

        self.load_att_embeddings(att_embeddings)

    # -----------------------------------------------------------
    # 以下部分（非 loss_by_feat 及专用函数）尽量与 our_head.py 保持完全一致
    # -----------------------------------------------------------
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
        """Reset the log."""
        self.positive_distributions = [{att_i: torch.zeros(int((1) / intetval)).to(self.device)
                                        for att_i in range(self.att_embeddings.shape[0])} for _ in self.thrs]
        self.negative_distributions = [{att_i: torch.zeros(int((1) / intetval)).to(self.device)
                                        for att_i in range(self.att_embeddings.shape[0])} for _ in self.thrs]

    def loss(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
             batch_data_samples: Union[list, dict], fusion_att: bool = False, epoch_info=None) -> dict:
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

    def loss_and_predict(
            self,
            img_feats: Tuple[Tensor],
            txt_feats: Tensor,
            batch_data_samples: SampleList,
            proposal_cfg: Optional[ConfigDict] = None,
            epoch_info=None
    ) -> Tuple[dict, InstanceList]:
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        outs = self(img_feats, txt_feats)

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs, epoch_info=epoch_info)

        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           cfg=proposal_cfg)
        return losses, predictions

    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        return self.head_module(img_feats, txt_feats)

    def predict(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False,
                fusion_att: bool = False) -> InstanceList:
        if self.att_embeddings.shape[0] != 25 * (self.num_classes):
            self.select_att()

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(img_feats, txt_feats)
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

            uncertainty = self.calculate_uncertainty(known_logits)
            top_k_att_score = self.compute_weighted_top_k_attributes(unknown_logits, k=self.top_k)

            # --- UAV BSU 增强分支 ---
            if getattr(self, 'use_bsu', False):
                max_entropy = math.log(2.0)
                u_norm = (uncertainty / max_entropy).clamp(max=1.0)
                fused = (top_k_att_score + 0.5 * u_norm * top_k_att_score).clamp(max=1.0)
                unknown_logits_final = fused * (1 - known_logits.max(-1, keepdim=True)[0])
            else:
                # 原始逻辑
                unknown_logits_final = (top_k_att_score + uncertainty) / 2 * (1 - known_logits.max(-1, keepdim=True)[0])

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

        att_embeddings_norm = F.normalize(all_atts, p=2, dim=1)
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
                cosine_sim_with_selected = cosine_sim_matrix[unselected_indices][:, selected_indices].mean(dim=1)

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

    def aug_test(self,
                 aug_batch_feats,
                 aug_batch_img_metas,
                 rescale=False,
                 with_ori_nms=False,
                 **kwargs):
        raise NotImplementedError('aug_test is not implemented yet.')

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

        if featmap_sizes != getattr(self, 'featmap_sizes', None):
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

    # -----------------------------------------------------------
    # 以下为 UAV D 特有的 Loss 计算及其辅助函数 (保持不变)
    # -----------------------------------------------------------
    def compute_inner_wiou_loss(self, pred_bboxes, gt_bboxes):
        pred_cx = (pred_bboxes[:, 0] + pred_bboxes[:, 2]) / 2
        pred_cy = (pred_bboxes[:, 1] + pred_bboxes[:, 3]) / 2
        pred_w = (pred_bboxes[:, 2] - pred_bboxes[:, 0]).clamp(min=1e-6)
        pred_h = (pred_bboxes[:, 3] - pred_bboxes[:, 1]).clamp(min=1e-6)

        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2
        gt_w = (gt_bboxes[:, 2] - gt_bboxes[:, 0]).clamp(min=1e-6)
        gt_h = (gt_bboxes[:, 3] - gt_bboxes[:, 1]).clamp(min=1e-6)

        in_p_w = pred_w * self.inner_ratio
        in_p_h = pred_h * self.inner_ratio
        in_g_w = gt_w * self.inner_ratio
        in_g_h = gt_h * self.inner_ratio

        in_p_x1 = pred_cx - in_p_w / 2
        in_p_y1 = pred_cy - in_p_h / 2
        in_p_x2 = pred_cx + in_p_w / 2
        in_p_y2 = pred_cy + in_p_h / 2

        in_g_x1 = gt_cx - in_g_w / 2
        in_g_y1 = gt_cy - in_g_h / 2
        in_g_x2 = gt_cx + in_g_w / 2
        in_g_y2 = gt_cy + in_g_h / 2

        inter_x1 = torch.max(in_p_x1, in_g_x1)
        inter_y1 = torch.max(in_p_y1, in_g_y1)
        inter_x2 = torch.min(in_p_x2, in_g_x2)
        inter_y2 = torch.min(in_p_y2, in_g_y2)

        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        p_area = in_p_w * in_p_h
        g_area = in_g_w * in_g_h
        inner_iou = inter_area / (p_area + g_area - inter_area + 1e-6)

        enclose_x1 = torch.min(pred_bboxes[:, 0], gt_bboxes[:, 0])
        enclose_y1 = torch.min(pred_bboxes[:, 1], gt_bboxes[:, 1])
        enclose_x2 = torch.max(pred_bboxes[:, 2], gt_bboxes[:, 2])
        enclose_y2 = torch.max(pred_bboxes[:, 3], gt_bboxes[:, 3])
        cw = (enclose_x2 - enclose_x1).clamp(min=1e-6)
        ch = (enclose_y2 - enclose_y1).clamp(min=1e-6)

        dist_sq = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2
        diag_sq = cw ** 2 + ch ** 2

        R_WIoU = torch.exp((dist_sq / diag_sq).clamp(max=10.0))

        loss_iou = 1.0 - inner_iou
        wiou_loss_with_grad = R_WIoU * loss_iou

        with torch.no_grad():
            beta = (loss_iou / self.iou_loss_ema.clamp(min=1e-4)) * R_WIoU.detach()
            r = beta / (self.wiou_delta * torch.pow(self.wiou_alpha, beta - self.wiou_delta))

        return r * wiou_loss_with_grad, loss_iou.detach()

    def compute_nwd(self, pred_bboxes_abs, gt_bboxes_abs):
        pred_cx = (pred_bboxes_abs[:, 0] + pred_bboxes_abs[:, 2]) / 2
        pred_cy = (pred_bboxes_abs[:, 1] + pred_bboxes_abs[:, 3]) / 2
        pred_w = (pred_bboxes_abs[:, 2] - pred_bboxes_abs[:, 0]).clamp(min=1e-6)
        pred_h = (pred_bboxes_abs[:, 3] - pred_bboxes_abs[:, 1]).clamp(min=1e-6)

        gt_cx = (gt_bboxes_abs[:, 0] + gt_bboxes_abs[:, 2]) / 2
        gt_cy = (gt_bboxes_abs[:, 1] + gt_bboxes_abs[:, 3]) / 2
        gt_w = (gt_bboxes_abs[:, 2] - gt_bboxes_abs[:, 0]).clamp(min=1e-6)
        gt_h = (gt_bboxes_abs[:, 3] - gt_bboxes_abs[:, 1]).clamp(min=1e-6)

        w2 = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2 + \
             (pred_w / 2 - gt_w / 2) ** 2 + (pred_h / 2 - gt_h / 2) ** 2

        dist = torch.sqrt(torch.abs(w2) + 1e-7)
        return torch.exp(-dist / (self.nwd_constant + 1e-7))

    def loss_by_feat(self, cls_scores, bbox_preds, bbox_dist_preds, att_scores, batch_gt_instances, batch_img_metas,
                     batch_gt_instances_ignore=None, epoch_info=None):
        num_imgs = len(batch_img_metas)
        world_size = get_dist_info()[1] if self.world_size == -1 else self.world_size
        current_device = cls_scores[0].device

        featmap_sizes = [cls.shape[2:] for cls in cls_scores]
        if featmap_sizes != getattr(self, 'featmap_sizes_train', None):
            self.featmap_sizes_train = featmap_sizes
            priors = self.prior_generator.grid_priors(featmap_sizes, dtype=cls_scores[0].dtype,
                                                      device=current_device, with_stride=True)
            self.flatten_priors_train = torch.cat(priors, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels, gt_bboxes = gt_info[:, :, :1], gt_info[:, :, 1:]
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        flatten_cls = torch.cat([cls.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes) for cls in cls_scores],
                                dim=1)
        flatten_bbox = torch.cat([bbox.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bbox in bbox_preds], dim=1)
        flatten_dist = torch.cat([dist.reshape(num_imgs, -1, self.head_module.reg_max * 4) for dist in bbox_dist_preds],
                                 dim=1)

        if self.att_embeddings is not None and att_scores is not None:
            flatten_att_scores = [
                att_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                      att_score.shape[1])
                for att_score in att_scores
            ]
            flatten_att_scores = torch.cat(flatten_att_scores, dim=1)
        else:
            flatten_att_scores = None

        decoded_bboxes = self.bbox_coder.decode(self.flatten_priors_train[..., :2], flatten_bbox,
                                                self.stride_tensor[..., 0])

        assigned = self.assigner(decoded_bboxes.detach().type(gt_bboxes.dtype), flatten_cls.detach().sigmoid(),
                                 self.flatten_priors_train, gt_labels, gt_bboxes, pad_bbox_flag)
        assigned_bboxes = assigned['assigned_bboxes']
        assigned_scores = assigned['assigned_scores']
        fg_mask = assigned['fg_mask_pre_prior']
        assigned_scores_sum = assigned_scores.sum().clamp(min=1e-4)

        if flatten_att_scores is not None:
            self.log_distribution(flatten_att_scores, assigned_scores)

        self._update_class_sample_ema(assigned_scores, world_size)
        cls_weight_map = self._get_class_weight_map()
        loss_cls = self.loss_cls(flatten_cls, assigned_scores, weight=cls_weight_map).sum() / assigned_scores_sum

        progress = 0.0
        if epoch_info:
            progress = min(max(epoch_info.get('current_epoch', 0) / float(epoch_info.get('max_epochs', 1)), 0.0), 1.0)

        local_iou_sum = torch.tensor(0.0, device=current_device)
        local_pos_num = torch.tensor(0.0, device=current_device)

        if fg_mask.sum() > 0:
            prior_mask = fg_mask.unsqueeze(-1).repeat(1, 1, 4)
            pred_pos = torch.masked_select(decoded_bboxes / self.stride_tensor, prior_mask).reshape(-1, 4)
            gt_pos = torch.masked_select(assigned_bboxes / self.stride_tensor, prior_mask).reshape(-1, 4)

            base_weight = torch.masked_select(assigned_scores.sum(-1), fg_mask).unsqueeze(-1)

            pred_abs = torch.masked_select(decoded_bboxes, prior_mask).reshape(-1, 4)
            gt_abs = torch.masked_select(assigned_bboxes, prior_mask).reshape(-1, 4)

            loss_bbox_raw, local_iou_sum, local_pos_num, final_weight = self._compute_fused_bbox_loss(
                pred_pos, gt_pos, pred_abs, gt_abs, base_weight, progress
            )
            loss_bbox = loss_bbox_raw.sum() / assigned_scores_sum

            assigned_ltrb = self.bbox_coder.encode(self.flatten_priors_train[..., :2] / self.stride_tensor,
                                                   assigned_bboxes / self.stride_tensor,
                                                   max_dis=self.head_module.reg_max - 1, eps=0.01)
            loss_dfl = self.loss_dfl(
                flatten_dist[fg_mask].reshape(-1, self.head_module.reg_max),
                torch.masked_select(assigned_ltrb, prior_mask).reshape(-1),
                weight=final_weight.expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum)
        else:
            loss_bbox = flatten_bbox.sum() * 0.0
            loss_dfl = flatten_dist.sum() * 0.0

        self._update_global_iou_ema(local_iou_sum, local_pos_num, world_size)

        return dict(loss_cls=loss_cls * num_imgs * world_size,
                    loss_bbox=loss_bbox * num_imgs * world_size,
                    loss_dfl=loss_dfl * num_imgs * world_size)

    def _update_class_sample_ema(self, assigned_scores: Tensor, world_size: int) -> None:
        if not self.training:
            return

        with torch.no_grad():
            pos_mask = assigned_scores > 0
            batch_class_count = pos_mask.sum(dim=(0, 1)).float()

            if world_size > 1:
                all_reduce(batch_class_count, op='sum')

            new_ema = self.class_sample_ema * self.class_ema_momentum + (
                    1.0 - self.class_ema_momentum) * batch_class_count
            self.class_sample_ema.copy_(new_ema)

    def _get_class_weight_map(self) -> Tensor:
        with torch.no_grad():
            beta = 0.999
            term = torch.pow(beta, self.class_sample_ema)
            effective_num = (1.0 - term).clamp(min=1e-4)
            class_weights = (1.0 - beta) / effective_num

            mean_weight = class_weights.mean()
            if mean_weight > 0:
                class_weights = class_weights / mean_weight
            else:
                class_weights = torch.ones_like(class_weights)

        return class_weights.unsqueeze(0).unsqueeze(0)

    def _compute_fused_bbox_loss(self, pred_pos, gt_pos, pred_abs, gt_abs, base_weight, progress) -> Tuple[
        Tensor, Tensor, Tensor, Tensor]:
        current_device = pred_pos.device
        local_iou_sum = torch.tensor(0.0, device=current_device)
        local_pos_num = torch.tensor(0.0, device=current_device)

        if self.use_inner_wiou:
            bbox_loss_main, detached_loss_iou = self.compute_inner_wiou_loss(pred_pos, gt_pos)
            if self.training:
                local_iou_sum = detached_loss_iou.sum()
                local_pos_num = torch.tensor(detached_loss_iou.numel(), dtype=torch.float32, device=current_device)
        else:
            ious = bbox_overlaps(pred_pos, gt_pos, is_aligned=True).clamp(0, 1)
            bbox_loss_main = 1.0 - ious
            if self.training:
                local_iou_sum = bbox_loss_main.sum().detach()
                local_pos_num = torch.tensor(ious.numel(), dtype=torch.float32, device=current_device)

        prog_weight = torch.ones_like(base_weight).squeeze(-1)
        if self.use_prog_loss:
            with torch.no_grad():
                area_abs = ((gt_abs[:, 2] - gt_abs[:, 0]) * (gt_abs[:, 3] - gt_abs[:, 1])).clamp(min=1e-6)
                ious_current = bbox_overlaps(pred_pos, gt_pos, is_aligned=True).clamp(0, 1)
                relative_difficulty = ((1.0 - ious_current) / self.iou_loss_ema.clamp(min=1e-3)).clamp(max=10.0)

                difficulty = relative_difficulty * (1.0 / torch.sqrt(area_abs)).clamp(max=5.0)
                prog_weight = 1.0 + 1.0 * (progress ** 2.0) * difficulty
                prog_weight = prog_weight.detach()

        final_weight = base_weight * prog_weight.unsqueeze(-1)

        if self.use_sad_nwd:
            area_abs = ((gt_abs[:, 2] - gt_abs[:, 0]) * (gt_abs[:, 3] - gt_abs[:, 1])).clamp(min=1e-6)
            w_small = torch.exp(-area_abs / self.tau_scale)
            loss_nwd = 1.0 - self.compute_nwd(pred_abs, gt_abs)

            fused_bbox_loss = bbox_loss_main * (1.0 - w_small) + (loss_nwd * self.nwd_factor) * w_small
            loss_bbox_raw = fused_bbox_loss * final_weight.squeeze(-1)
        else:
            loss_bbox_raw = bbox_loss_main * final_weight.squeeze(-1)

        return loss_bbox_raw, local_iou_sum, local_pos_num, final_weight

    def _update_global_iou_ema(self, local_iou_sum: Tensor, local_pos_num: Tensor, world_size: int) -> None:
        if not self.training:
            return

        if world_size > 1:
            all_reduce(local_iou_sum, op='sum')
            all_reduce(local_pos_num, op='sum')

        if local_pos_num > 0:
            global_iou_mean = local_iou_sum / local_pos_num.clamp(min=1e-6)
            with torch.no_grad():
                new_iou_ema = self.iou_loss_ema * 0.999 + global_iou_mean * 0.001
                self.iou_loss_ema.copy_(new_iou_ema)