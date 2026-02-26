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

    核心特性:
    - Distributed-Safe Class Balanced Loss (EMA): 分布式安全的基于指数移动平均的类别平衡损失
    - Curriculum Progression Loss (ProgLoss): 渐进式课程学习损失，根据训练进度动态调整难易样本权重
    - Inner-WIoU & SAD-NWD: 专为极小目标设计的内部交并比与基于 Wasserstein 距离的度量
    - BSU (Batch-Safe Uncertainty): 用于未知目标检测的批次安全不确定性融合
    """

    def __init__(self,
                 # --- 消融实验控制开关 ---
                 use_prog_loss: bool = False,  # 是否启用渐进式损失 (ProgLoss)
                 use_inner_wiou: bool = False,  # 是否启用 Inner-WIoU 替代标准 IoU
                 use_sad_nwd: bool = False,  # 是否启用 SAD-NWD 处理极小目标
                 use_bsu: bool = False,  # 是否启用 BSU (Batch-Safe Uncertainty)

                 # --- 核心模块超参数 ---
                 nwd_factor: float = 1.0,  # NWD 损失的权重系数
                 nwd_constant: float = 12.8,  # NWD 计算中的常数 C (用于平滑指数)
                 inner_ratio: float = 0.75,  # Inner-WIoU 中缩小边界框的比例 (控制内部框的大小)
                 tau_scale: float = 1024.0,  # SAD-NWD 的绝对面积尺度阈值 (例如 32x32 = 1024)
                 wiou_alpha: float = 1.9,  # WIoU 非单调聚焦机制的超参数 alpha
                 wiou_delta: float = 3.0,  # WIoU 非单调聚焦机制的超参数 delta

                 # --- 基础参数 ---
                 world_size=-1,
                 att_embeddings=None,  # 属性文本特征的路径
                 prev_intro_cls=0,
                 cur_intro_cls=0,
                 thr=0.8,  # 伪标签或分布统计的置信度阈值
                 alpha=0.5,  # 特征选择中，分布相似度与特征多样性的平衡系数
                 use_sigmoid=True,
                 device='cuda',
                 prev_distribution=None,  # 预训练模型或上一阶段的特征分布
                 distributions=None,  # 当前保存特征分布的路径
                 top_k=10,  # 预测未知类别时取 Top-K 属性
                 *args, **kwargs) -> None:

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
        self.device = device  # 默认设备，实际计算时应优先从张量获取 device
        self.alpha = alpha
        self.use_sigmoid = use_sigmoid
        self.distributions = distributions
        self.thrs = [thr]
        self.prev_intro_cls = prev_intro_cls
        self.cur_intro_cls = cur_intro_cls
        self.prev_distribution = prev_distribution
        self.top_k = top_k

        # 注册用于长尾/类别平衡的全局缓冲变量 (Buffers不会被视为模型参数，但会随模型保存)
        self.register_buffer('class_sample_ema', torch.zeros(self.num_classes))
        self.register_buffer('iou_loss_ema', torch.tensor(1.0))
        self.class_ema_momentum = 0.999

        # 加载文本/属性嵌入特征
        self.load_att_embeddings(att_embeddings)

    def disable_log(self):
        """关闭特征分布直方图的记录"""
        self.positive_distributions, self.negative_distributions = None, None

    def enable_log(self):
        """开启特征分布直方图的记录"""
        self.reset_log()

    def load_att_embeddings(self, att_embeddings):
        """
        加载属性/文本嵌入特征 (Attribute Embeddings)。
        使用 CPU 映射加载，防止在多卡训练时，所有卡并发读取导致主卡 (cuda:0) 显存溢出 (OOM)。
        """
        if att_embeddings is None:
            self.att_embeddings = None
            self.disable_log()
            return

        atts = torch.load(att_embeddings, map_location='cpu')
        self.texts = atts['att_text']
        self.all_atts = atts['att_embedding']

        # 兼容增量学习：判断是否需要跳过上一阶段已引入的属性特征
        if self.prev_distribution is not None and os.path.exists(self.prev_distribution):
            prev_data = torch.load(self.prev_distribution, map_location='cpu')
            prev_atts_num = len(prev_data['positive_distributions'][self.thrs.index(self.thr)])
        else:
            prev_atts_num = 0

        # 初始化为可学习参数 (Parameter)
        self.att_embeddings = torch.nn.Parameter(atts['att_embedding'].float()[prev_atts_num:])

    def reset_log(self, interval=0.0001):
        """重置正负样本分布特征的直方图统计量"""
        bins = int(1 / interval)
        current_device = self.att_embeddings.device if self.att_embeddings is not None else self.device

        # 为每个阈值、每个属性维护一个直方图张量
        self.positive_distributions = [
            {att_i: torch.zeros(bins, device=current_device) for att_i in range(self.att_embeddings.shape[0])}
            for _ in self.thrs
        ]
        self.negative_distributions = [
            {att_i: torch.zeros(bins, device=current_device) for att_i in range(self.att_embeddings.shape[0])}
            for _ in self.thrs
        ]

    def log_distribution(self, att_scores, assigned_scores):
        """在训练过程中，根据分配的伪标签/GT统计正负样本的属性得分分布"""
        if not self.training or self.positive_distributions is None or self.att_embeddings is None:
            return

        num_att, num_known = att_scores.shape[-1], assigned_scores.shape[-1]
        att_scores = att_scores.sigmoid().reshape(-1, num_att).float()
        assigned_scores = assigned_scores.reshape(-1, num_known)

        # 忽略历史已知类别
        assigned_scores[:, 0: self.prev_intro_cls] = 0
        assigned_scores = assigned_scores.max(-1)[0]

        for idx, thr in enumerate(self.thrs):
            positive = (assigned_scores >= thr)
            pos_scores = att_scores[positive]
            neg_scores = att_scores[~positive]

            # 更新正样本分布直方图
            if pos_scores.numel() > 0:
                for att_i in range(num_att):
                    self.positive_distributions[idx][att_i] += torch.histc(pos_scores[:, att_i], bins=10000, min=0,
                                                                           max=1)
            # 更新负样本分布直方图
            if neg_scores.numel() > 0:
                for att_i in range(num_att):
                    self.negative_distributions[idx][att_i] += torch.histc(neg_scores[:, att_i], bins=10000, min=0,
                                                                           max=1)

    def get_sim(self, a, b):
        """计算两个分布之间的 Jensen-Shannon 散度 (JSD)，衡量分布相似性"""

        def jensen_shannon_divergence(p, q):
            m = 0.5 * (p + q).clamp(min=1e-6)
            return 0.5 * (torch.sum(p * torch.log((p / m).clamp(min=1e-6))) +
                          torch.sum(q * torch.log((q / m).clamp(min=1e-6))))

        return jensen_shannon_divergence(a, b)

    def get_all_dis_sim(self, positive_dis, negative_dis):
        """计算所有属性在正负样本上的分布相似度"""
        device = positive_dis[0].device if len(positive_dis) > 0 else 'cuda'
        return torch.stack([self.get_sim(p / p.sum().clamp(min=1e-6),
                                         n / n.sum().clamp(min=1e-6))
                            for p, n in zip(positive_dis, negative_dis)]).to(device)

    def combine_distributions(self):
        """合并历史阶段与当前阶段的特征分布统计量"""
        if self.prev_distribution is None or not os.path.exists(self.prev_distribution):
            return self.positive_distributions, self.negative_distributions

        prev = torch.load(self.prev_distribution, map_location='cpu')
        ret_pos, ret_neg = prev['positive_distributions'], prev['negative_distributions']

        for thr_id, thr in enumerate(self.thrs):
            if thr_id >= len(ret_pos) or ret_pos[thr_id] is None or self.positive_distributions[thr_id] is None:
                continue
            prev_att = len(ret_pos[thr_id])
            ret_pos[thr_id].update({prev_att + k: v for k, v in self.positive_distributions[thr_id].items()})
            ret_neg[thr_id].update({prev_att + k: v for k, v in self.negative_distributions[thr_id].items()})

        return ret_pos, ret_neg

    def select_att(self, per_class=25):
        """
        基于分布相似性与特征多样性进行属性特征选择 (Feature Selection)。
        旨在选出既能区分正负样本（分布差异大），又互不冗余（特征多样）的属性。
        """
        save_root = os.path.dirname(self.distributions) if self.distributions else './'
        os.makedirs(save_root, exist_ok=True)
        self.positive_distributions, self.negative_distributions = self.combine_distributions()

        # 仅在主进程保存分布统计，防止多进程写文件冲突
        if self.distributions and is_main_process():
            torch.save({'positive_distributions': self.positive_distributions,
                        'negative_distributions': self.negative_distributions}, self.distributions)

        thr_id = self.thrs.index(self.thr)
        # 获取基础分布差异评分 (越小说明散度越低，这里优先寻找辨识度相关的特征)
        distribution_sim = self.get_all_dis_sim(self.positive_distributions[thr_id],
                                                self.negative_distributions[thr_id])

        current_device = self.att_embeddings.device if self.att_embeddings is not None else self.device
        att_norm = F.normalize(self.all_atts.to(current_device), p=2, dim=1)

        selected_indices = []
        num_total = len(self.texts)
        target_num = min(per_class * self.num_classes, num_total)  # 安全限制，防止越界

        # 贪心选择策略
        for _ in range(target_num):
            if not selected_indices:
                idx = distribution_sim.argmin().item()
            else:
                unselected = list(set(range(num_total)) - set(selected_indices))
                unselected_feats = att_norm[unselected]  # [U, 512]
                selected_feats = att_norm[selected_indices]  # [S, 512]

                # 【显存优化】: 计算未选中特征与已选中特征的余弦相似度，避免全量 N^2 矩阵乘法
                sim_matrix = torch.matmul(unselected_feats, selected_feats.T)
                if self.use_sigmoid:
                    sim_matrix = sim_matrix.sigmoid()
                else:
                    sim_matrix = sim_matrix.abs()

                diversity_score = sim_matrix.mean(dim=1)
                # 综合评分：alpha 控制权重。得分越小越容易被选中。
                score = self.alpha * distribution_sim[unselected] + (1 - self.alpha) * diversity_score
                idx = unselected[score.argmin().item()]

            selected_indices.append(idx)

        # 更新特征矩阵
        selected_tensor = torch.tensor(selected_indices, device=current_device)
        new_att_tensor = self.all_atts[selected_tensor].to(current_device)

        if self.att_embeddings.shape == new_att_tensor.shape:
            self.att_embeddings.data.copy_(new_att_tensor)
        else:
            self.att_embeddings = torch.nn.Parameter(new_att_tensor)

        # 同步更新文本描述列表
        self.texts = [self.texts[i] for i in selected_indices]

    def compute_inner_wiou_loss(self, pred_bboxes, gt_bboxes):
        """
        【核心组件】Inner-WIoU 计算逻辑:
        用于解决微小目标对于轻微位移产生的剧烈 IoU 下降问题。
        通过 inner_ratio 在原框中心不变的基础上缩小边界框计算内部 IoU。
        同时结合 WIoU 的非单调聚焦机制惩罚。
        """
        # 获取预测框与真实框的中心点与宽高
        pred_cx = (pred_bboxes[:, 0] + pred_bboxes[:, 2]) / 2
        pred_cy = (pred_bboxes[:, 1] + pred_bboxes[:, 3]) / 2
        pred_w = (pred_bboxes[:, 2] - pred_bboxes[:, 0]).clamp(min=1e-6)
        pred_h = (pred_bboxes[:, 3] - pred_bboxes[:, 1]).clamp(min=1e-6)

        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2
        gt_w = (gt_bboxes[:, 2] - gt_bboxes[:, 0]).clamp(min=1e-6)
        gt_h = (gt_bboxes[:, 3] - gt_bboxes[:, 1]).clamp(min=1e-6)

        # 构建 Inner 缩小框
        in_p_w = pred_w * self.inner_ratio
        in_p_h = pred_h * self.inner_ratio
        in_g_w = gt_w * self.inner_ratio
        in_g_h = gt_h * self.inner_ratio

        # 缩小后预测框的四个顶点
        in_p_x1 = pred_cx - in_p_w / 2
        in_p_y1 = pred_cy - in_p_h / 2
        in_p_x2 = pred_cx + in_p_w / 2
        in_p_y2 = pred_cy + in_p_h / 2

        # 缩小后真实框的四个顶点
        in_g_x1 = gt_cx - in_g_w / 2
        in_g_y1 = gt_cy - in_g_h / 2
        in_g_x2 = gt_cx + in_g_w / 2
        in_g_y2 = gt_cy + in_g_h / 2

        # 计算 Inner 框的相交区域 (Intersection)
        inter_x1 = torch.max(in_p_x1, in_g_x1)
        inter_y1 = torch.max(in_p_y1, in_g_y1)
        inter_x2 = torch.min(in_p_x2, in_g_x2)
        inter_y2 = torch.min(in_p_y2, in_g_y2)

        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        p_area = in_p_w * in_p_h
        g_area = in_g_w * in_g_h
        inner_iou = inter_area / (p_area + g_area - inter_area + 1e-6)

        # 计算闭包区域 (Enclosing Box) 用于 WIoU 惩罚项
        enclose_x1 = torch.min(pred_bboxes[:, 0], gt_bboxes[:, 0])
        enclose_y1 = torch.min(pred_bboxes[:, 1], gt_bboxes[:, 1])
        enclose_x2 = torch.max(pred_bboxes[:, 2], gt_bboxes[:, 2])
        enclose_y2 = torch.max(pred_bboxes[:, 3], gt_bboxes[:, 3])
        cw = (enclose_x2 - enclose_x1).clamp(min=1e-6)
        ch = (enclose_y2 - enclose_y1).clamp(min=1e-6)

        # 欧式距离平方与对角线距离平方
        dist_sq = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2
        diag_sq = cw ** 2 + ch ** 2

        # WIoU 的距离惩罚因子 (R_WIoU)
        R_WIoU = torch.exp((dist_sq / diag_sq).clamp(max=10.0))

        loss_iou = 1.0 - inner_iou
        wiou_loss_with_grad = R_WIoU * loss_iou

        # 计算非单调聚焦系数 r，使用 no_grad 避免干涉反向传播的梯度流
        with torch.no_grad():
            beta = (loss_iou / self.iou_loss_ema.clamp(min=1e-4)) * R_WIoU.detach()
            r = beta / (self.wiou_delta * torch.pow(self.wiou_alpha, beta - self.wiou_delta))

        return r * wiou_loss_with_grad, loss_iou.detach()

    def compute_nwd(self, pred_bboxes_abs, gt_bboxes_abs):
        """
        【核心组件】NWD (Normalized Wasserstein Distance):
        将二维边界框建模为二维高斯分布，计算分布间的 Wasserstein 距离。
        极大缓解了在微小目标（如 10x10 像素）检测中，1 个像素偏差引发的 IoU 剧变。
        """
        pred_cx = (pred_bboxes_abs[:, 0] + pred_bboxes_abs[:, 2]) / 2
        pred_cy = (pred_bboxes_abs[:, 1] + pred_bboxes_abs[:, 3]) / 2
        pred_w = (pred_bboxes_abs[:, 2] - pred_bboxes_abs[:, 0]).clamp(min=1e-6)
        pred_h = (pred_bboxes_abs[:, 3] - pred_bboxes_abs[:, 1]).clamp(min=1e-6)

        gt_cx = (gt_bboxes_abs[:, 0] + gt_bboxes_abs[:, 2]) / 2
        gt_cy = (gt_bboxes_abs[:, 1] + gt_bboxes_abs[:, 3]) / 2
        gt_w = (gt_bboxes_abs[:, 2] - gt_bboxes_abs[:, 0]).clamp(min=1e-6)
        gt_h = (gt_bboxes_abs[:, 3] - gt_bboxes_abs[:, 1]).clamp(min=1e-6)

        # Wasserstein 距离的平方公式简化版 (针对对角协方差矩阵)
        w2 = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2 + \
             (pred_w / 2 - gt_w / 2) ** 2 + (pred_h / 2 - gt_h / 2) ** 2

        # 添加 abs 与 1e-7 防止因数值截断导致对负数开根号 (NaN 异常)
        dist = torch.sqrt(torch.abs(w2) + 1e-7)
        # 通过指数函数将距离归一化到 (0, 1] 类似 IoU 的区间
        return torch.exp(-dist / (self.nwd_constant + 1e-7))

    def loss(self, img_feats, txt_feats, batch_data_samples, fusion_att=False, epoch_info=None):
        """主框架调用的损失入口，处理特征提取及数据打包解包"""
        outs = self(img_feats, txt_feats)
        att_outs = None

        if self.att_embeddings is not None:
            if fusion_att:
                att_feats = txt_feats[:, -self.att_embeddings.shape[0]:, :]
            else:
                att_feats = self.att_embeddings[None].repeat(txt_feats.shape[0], 1, 1)

            # 脱离计算图提取属性输出
            with torch.no_grad():
                att_outs = self(img_feats, att_feats)[0]

        # 从 batch 数据中正确解包真实框 (GT) 和元信息
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = unpack_gt_instances(batch_data_samples)

        loss_inputs = outs + (att_outs, batch_gt_instances, batch_img_metas)
        return self.loss_by_feat(*loss_inputs, batch_gt_instances_ignore=batch_gt_instances_ignore,
                                 epoch_info=epoch_info)

    def loss_by_feat(self, cls_scores, bbox_preds, bbox_dist_preds, att_scores, batch_gt_instances, batch_img_metas,
                     batch_gt_instances_ignore=None, epoch_info=None):
        """具体执行正负样本分配、分类损失与回归损失计算的函数"""
        num_imgs = len(batch_img_metas)
        world_size = get_dist_info()[1] if self.world_size == -1 else self.world_size
        current_device = cls_scores[0].device

        # --- 1. 基础特征图预处理与先验框 (Priors) 生成 ---
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

        # 按照 (B, N, C) 展平预测特征
        flatten_cls = torch.cat([cls.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes) for cls in cls_scores],
                                dim=1)
        flatten_bbox = torch.cat([bbox.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bbox in bbox_preds], dim=1)
        flatten_dist = torch.cat([dist.reshape(num_imgs, -1, self.head_module.reg_max * 4) for dist in bbox_dist_preds],
                                 dim=1)

        # 解码预测的边界框
        decoded_bboxes = self.bbox_coder.decode(self.flatten_priors_train[..., :2], flatten_bbox,
                                                self.stride_tensor[..., 0])

        # --- 2. 标签分配 (Task-Aligned Label Assignment) ---
        assigned = self.assigner(decoded_bboxes.detach().type(gt_bboxes.dtype), flatten_cls.detach().sigmoid(),
                                 self.flatten_priors_train, gt_labels, gt_bboxes, pad_bbox_flag)
        assigned_bboxes = assigned['assigned_bboxes']
        assigned_scores = assigned['assigned_scores']
        fg_mask = assigned['fg_mask_pre_prior']
        assigned_scores_sum = assigned_scores.sum().clamp(min=1e-4)

        # --- 3. 计算分类损失 (带类别平衡权重) ---
        self._update_class_sample_ema(assigned_scores, world_size)
        cls_weight_map = self._get_class_weight_map()
        loss_cls = self.loss_cls(flatten_cls, assigned_scores, weight=cls_weight_map).sum() / assigned_scores_sum

        # --- 4. 计算边界框回归损失 (融合 Inner-WIoU, NWD, ProgLoss) ---
        progress = 0.0
        if epoch_info:
            # 获取当前训练进度 (0.0 ~ 1.0)
            progress = min(max(epoch_info.get('current_epoch', 0) / float(epoch_info.get('max_epochs', 1)), 0.0), 1.0)

        local_iou_sum = torch.tensor(0.0, device=current_device)
        local_pos_num = torch.tensor(0.0, device=current_device)

        if fg_mask.sum() > 0:
            prior_mask = fg_mask.unsqueeze(-1).repeat(1, 1, 4)
            # 获取正样本特征
            pred_pos = torch.masked_select(decoded_bboxes / self.stride_tensor, prior_mask).reshape(-1, 4)
            gt_pos = torch.masked_select(assigned_bboxes / self.stride_tensor, prior_mask).reshape(-1, 4)

            base_weight = torch.masked_select(assigned_scores.sum(-1), fg_mask).unsqueeze(-1)
            stride_pos = self.stride_tensor[fg_mask].unsqueeze(-1)

            pred_abs = pred_pos * stride_pos
            gt_abs = gt_pos * stride_pos

            # 调用综合回归损失计算模块
            loss_bbox_raw, local_iou_sum, local_pos_num, final_weight = self._compute_fused_bbox_loss(
                pred_pos, gt_pos, pred_abs, gt_abs, base_weight, progress
            )
            loss_bbox = loss_bbox_raw.sum() / assigned_scores_sum

            # DFL (Distribution Focal Loss) 损失计算
            assigned_ltrb = self.bbox_coder.encode(self.flatten_priors_train[..., :2] / self.stride_tensor,
                                                   assigned_bboxes / self.stride_tensor,
                                                   max_dis=self.head_module.reg_max - 1, eps=0.01)
            loss_dfl = self.loss_dfl(
                flatten_dist[fg_mask].reshape(-1, self.head_module.reg_max),
                torch.masked_select(assigned_ltrb, prior_mask).reshape(-1),
                weight=final_weight.expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum)
        else:
            # 处理无正样本的边界情况，避免梯度断裂
            loss_bbox = flatten_bbox.sum() * 0.0
            loss_dfl = flatten_dist.sum() * 0.0

        # --- 5. 更新全局 IoU 的指数移动平均状态 ---
        self._update_global_iou_ema(local_iou_sum, local_pos_num, world_size)

        return dict(loss_cls=loss_cls * num_imgs * world_size,
                    loss_bbox=loss_bbox * num_imgs * world_size,
                    loss_dfl=loss_dfl * num_imgs * world_size)

    def _update_class_sample_ema(self, assigned_scores: Tensor, world_size: int) -> None:
        """
        更新各类别样本数量的指数移动平均 (EMA)。
        主要用于长尾数据集中的 Class Balanced Weight 动态调整。
        """
        if not self.training:
            return

        with torch.no_grad():
            pos_mask = assigned_scores > 0
            batch_class_count = pos_mask.sum(dim=(0, 1)).float()

            if world_size > 1:
                all_reduce(batch_class_count, op='sum')

            # 使用 copy_ 进行原地赋值更新，避免破坏 Autograd 计算图
            new_ema = self.class_sample_ema * self.class_ema_momentum + (
                    1.0 - self.class_ema_momentum) * batch_class_count
            self.class_sample_ema.copy_(new_ema)

    def _get_class_weight_map(self) -> Tensor:
        """利用有效样本数理论 (Effective Number of Samples) 动态计算各类别的 Loss 权重。"""
        with torch.no_grad():
            beta = 0.999
            term = torch.pow(beta, self.class_sample_ema)
            effective_num = (1.0 - term).clamp(min=1e-4)
            class_weights = (1.0 - beta) / effective_num

            # 归一化，保证基础梯度的量级不发生剧烈突变
            mean_weight = class_weights.mean()
            if mean_weight > 0:
                class_weights = class_weights / mean_weight
            else:
                class_weights = torch.ones_like(class_weights)

        return class_weights.unsqueeze(0).unsqueeze(0)

    def _compute_fused_bbox_loss(self, pred_pos, gt_pos, pred_abs, gt_abs, base_weight, progress) -> Tuple[
        Tensor, Tensor, Tensor, Tensor]:
        """
        融合 Inner-WIoU, ProgLoss (课程学习) 和 SAD-NWD 的边界框联合计算逻辑。
        返回: 回归损失, 本地 IoU 累加, 正样本数, 最终融合权重
        """
        current_device = pred_pos.device
        local_iou_sum = torch.tensor(0.0, device=current_device)
        local_pos_num = torch.tensor(0.0, device=current_device)

        # --- 1. 基础回归 Loss (Inner-WIoU vs 标准 IoU) ---
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

        # --- 2. 渐进式困难样本挖掘权重 (ProgLoss) ---
        prog_weight = torch.ones_like(base_weight).squeeze(-1)
        if self.use_prog_loss:
            with torch.no_grad():
                # 依据面积的开根倒数，越小的物体分配越高的难度基底
                area_abs = ((gt_abs[:, 2] - gt_abs[:, 0]) * (gt_abs[:, 3] - gt_abs[:, 1])).clamp(min=1e-6)
                ious_current = bbox_overlaps(pred_pos, gt_pos, is_aligned=True).clamp(0, 1)
                # 与全局平均 IoU 对比，获取相对困难度
                relative_difficulty = ((1.0 - ious_current) / self.iou_loss_ema.clamp(min=1e-3)).clamp(max=10.0)

                difficulty = relative_difficulty * (1.0 / torch.sqrt(area_abs)).clamp(max=5.0)
                # 随着训练进度 progress (0->1) 增加，进一步激活困难样本惩罚
                prog_weight = 1.0 + 1.0 * (progress ** 2.0) * difficulty
                prog_weight = prog_weight.detach()

        final_weight = base_weight * prog_weight.unsqueeze(-1)

        # --- 3. SAD-NWD (极小目标 Wasserstein 融合) ---
        if self.use_sad_nwd:
            area_abs = ((gt_abs[:, 2] - gt_abs[:, 0]) * (gt_abs[:, 3] - gt_abs[:, 1])).clamp(min=1e-6)
            # 通过高斯函数将极小目标权重放大 (面积远小于 tau_scale 时 w_small 接近 1)
            w_small = torch.exp(-area_abs / self.tau_scale)
            loss_nwd = 1.0 - self.compute_nwd(pred_abs, gt_abs)

            # 动态融合：小目标偏向 NWD，大中目标偏向 IoU/WIoU
            fused_bbox_loss = bbox_loss_main * (1.0 - w_small) + (loss_nwd * self.nwd_factor) * w_small
            loss_bbox_raw = fused_bbox_loss * final_weight.squeeze(-1)
        else:
            loss_bbox_raw = bbox_loss_main * final_weight.squeeze(-1)

        return loss_bbox_raw, local_iou_sum, local_pos_num, final_weight

    def _update_global_iou_ema(self, local_iou_sum: Tensor, local_pos_num: Tensor, world_size: int) -> None:
        """安全更新全局平均 IoU 的 EMA (用于指导 ProgLoss 识别相对困难样本)"""
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

    def predict_unknown(self, outs, img_feats, att_embeddings):
        """
        基于已知类别的预测置信度与属性相似度，推断可能存在的未知类别 (Unknown Objects)。
        支持 BSU (Batch-Safe Uncertainty) 融合策略。
        """
        known_preds, unknown_preds = outs[0], self(img_feats, att_embeddings)[0]
        ret_logits = []
        max_entropy = math.log(2.0)

        for k_logits, u_logits in zip(known_preds, unknown_preds):
            k_logits = k_logits.sigmoid().permute(0, 2, 3, 1)
            u_logits = u_logits.sigmoid().permute(0, 2, 3, 1)

            # 限制极值，避免 log(0)
            k_clamped = torch.clamp(k_logits, 1e-6, 1 - 1e-6)
            # 计算已知预测的香农熵 (Uncertainty 不确定性)
            uncertainty = (-k_clamped * torch.log(k_clamped) - (1 - k_clamped) * torch.log(1 - k_clamped)).mean(dim=-1,
                                                                                                                keepdim=True)

            # 基于属性特征的 Top-K 加权得分
            top_k_scores, _ = u_logits.topk(self.top_k, dim=-1)
            att_score = torch.sum(top_k_scores * F.softmax(top_k_scores, dim=-1), dim=-1, keepdim=True)
            max_known = k_logits.max(-1, keepdim=True)[0]

            # 融合已知类别的不确定性(Uncertainty)与未知属性得分(att_score)
            if self.use_bsu:
                u_norm = (uncertainty / max_entropy).clamp(max=1.0)
                fused = (att_score + 0.5 * u_norm * att_score).clamp(max=1.0)
                # 抑制已知置信度高的区域 (max_known 越高，作为未知的概率越小)
                u_final = fused * (1 - max_known)
            else:
                u_final = ((att_score + uncertainty) / 2).clamp(max=1.0) * (1 - max_known)

            # 将预测拼接并恢复原有的张量排列 (B, C, H, W)
            ret_logits.append(torch.cat([k_logits, u_final], dim=-1).permute(0, 3, 1, 2))

        return (ret_logits, *outs[1:])

    def predict(self, img_feats, txt_feats, batch_data_samples, rescale=False, fusion_att=False):
        """推理入口函数，根据当前特征数量进行属性选择，并调用底层预测。"""
        # 懒加载策略：若当前保留属性不满足期望数，触发一轮特征选择
        if self.att_embeddings.shape[0] != 25 * self.num_classes:
            self.select_att()

        outs = self(img_feats, txt_feats)
        if self.att_embeddings is None:
            return self.predict_by_feat(outs[0], outs[1],
                                        batch_img_metas=[d.metainfo for d in batch_data_samples],
                                        rescale=rescale)

        if fusion_att:
            att_feats = txt_feats[:, -self.att_embeddings.shape[0]:, :]
        else:
            att_feats = self.att_embeddings[None].repeat(txt_feats.shape[0], 1, 1)

        # 对未知目标进行预测和特征重塑
        outs = self.predict_unknown(outs, img_feats, att_feats)
        return self.predict_by_feat(outs[0], outs[1], batch_img_metas=[d.metainfo for d in batch_data_samples],
                                    rescale=rescale)

    def predict_by_feat(self, cls_scores, bbox_preds, objectnesses=None, batch_img_metas=None, cfg=None, rescale=True,
                        with_nms=True):
        """
        特征解码与后处理逻辑：将预测特征图解码为最终的边界框，并应用 NMS 和缩放还原。
        """
        cfg = copy.deepcopy(self.test_cfg if cfg is None else cfg)
        cfg.multi_label = cfg.multi_label and self.num_classes > 1
        num_imgs = len(batch_img_metas)

        featmap_sizes = [cls.shape[2:] for cls in cls_scores]
        if featmap_sizes != getattr(self, 'featmap_sizes', None):
            self.featmap_sizes = featmap_sizes
            self.mlvl_priors = self.prior_generator.grid_priors(featmap_sizes, dtype=cls_scores[0].dtype,
                                                                device=cls_scores[0].device)

        flatten_priors = torch.cat(self.mlvl_priors)
        flatten_stride = torch.cat([flatten_priors.new_full((s.numel() * self.num_base_priors,), st)
                                    for s, st in zip(featmap_sizes, self.featmap_strides)])

        # 若存在文本属性特征，模型输出已包含未知信息融合，无需再接 sigmoid
        if self.att_embeddings is not None:
            flatten_cls = torch.cat(
                [cls.permute(0, 2, 3, 1).reshape(num_imgs, -1, cls_scores[0].size(1)) for cls in cls_scores], dim=1)
        else:
            flatten_cls = torch.cat(
                [cls.permute(0, 2, 3, 1).reshape(num_imgs, -1, cls_scores[0].size(1)) for cls in cls_scores],
                dim=1).sigmoid()

        flatten_bbox = torch.cat([bbox.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bbox in bbox_preds], dim=1)
        decoded_bboxes = self.bbox_coder.decode(flatten_priors[None], flatten_bbox, flatten_stride)

        if objectnesses:
            flatten_obj = torch.cat([obj.permute(0, 2, 3, 1).reshape(num_imgs, -1) for obj in objectnesses],
                                    dim=1).sigmoid()
        else:
            flatten_obj = [None] * num_imgs

        results_list = []
        for bboxes, scores, obj, img_meta in zip(decoded_bboxes, flatten_cls, flatten_obj, batch_img_metas):
            # 处理 YOLOX 风格的 objectness 分数过滤
            if obj is not None and cfg.get('score_thr', -1) > 0 and not cfg.get('yolox_style', False):
                conf_inds = obj > cfg.score_thr
                bboxes, scores, obj = bboxes[conf_inds], scores[conf_inds], obj[conf_inds]

            if obj is not None:
                scores *= obj[:, None]

            if scores.shape[0] == 0:
                results_list.append(InstanceData(bboxes=bboxes, scores=scores[:, 0], labels=scores[:, 0].int()))
                continue

            # NMS 前的 Top-K 及低分滤除
            if not cfg.multi_label:
                scores, labels, keep_idxs, results = filter_scores_and_topk(
                    scores, cfg.get('score_thr', -1), cfg.get('nms_pre', 100000), results=dict(labels=scores.max(1)[1]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, cfg.get('score_thr', -1), cfg.get('nms_pre', 100000))

            res = InstanceData(scores=scores, labels=labels, bboxes=bboxes[keep_idxs])

            # 将坐标从特征图缩放尺度还原到原图尺度
            if rescale:
                if 'pad_param' in img_meta:
                    res.bboxes -= res.bboxes.new_tensor([img_meta['pad_param'][2], img_meta['pad_param'][0]] * 2)
                res.bboxes /= res.bboxes.new_tensor(img_meta['scale_factor']).repeat((1, 2))

            if cfg.get('yolox_style', False):
                cfg.max_per_img = len(res)

            # 执行 NMS
            res = self._bbox_post_process(results=res, cfg=cfg, rescale=False, with_nms=with_nms, img_meta=img_meta)
            # 防止检测框越界
            res.bboxes[:, 0::2].clamp_(0, img_meta['ori_shape'][1])
            res.bboxes[:, 1::2].clamp_(0, img_meta['ori_shape'][0])
            results_list.append(res)

        return results_list

    def loss_and_predict(self, img_feats, txt_feats, batch_data_samples, proposal_cfg=None, epoch_info=None):
        """通常用于自训练或两阶段网络，同时返回计算损失与预测结果。"""
        outs = self(img_feats, txt_feats)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = unpack_gt_instances(batch_data_samples)
        losses = self.loss_by_feat(*outs, att_scores=None, batch_gt_instances=batch_gt_instances,
                                   batch_img_metas=batch_img_metas, batch_gt_instances_ignore=batch_gt_instances_ignore,
                                   epoch_info=epoch_info)

        predictions = self.predict_by_feat(outs[0], outs[1], batch_img_metas=batch_img_metas, cfg=proposal_cfg)
        return losses, predictions

    def forward(self, img_feats, txt_feats):
        """网络主干前向传播，通过基础 head_module 计算结果。"""
        return self.head_module(img_feats, txt_feats)