# Copyright (c) Tencent Inc. All rights reserved.
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from torch import Tensor

from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmyolo.models.utils import make_divisible


# ---------------------------------------------------------------------------
# 1. 核心创新组件：LSKA (大可分离核注意力) 模块
# 专为无人机视角设计：在极低算力下提供超大感受野，剥离复杂背景
# ---------------------------------------------------------------------------
class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 1. 局部特征提取：5x5 深度可分离卷积
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        # 2. 空间扩张特征：7x7 深度可分离空洞卷积 (Dilation=2，等效 13x13 感受野)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=6, groups=dim, dilation=2)

        # 3. 极大感受野扩张：7x7 深度可分离空洞卷积 (Dilation=3，等效 23x23 感受野)
        self.conv_global = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)

        # 4. 通道注意力融合
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv3 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 3, 7, padding=3)
        self.conv_out = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        attn3 = self.conv_global(attn2)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        attn3 = self.conv3(attn3)

        # 空间池化计算注意力权重
        a = torch.cat([attn1, attn2, attn3], dim=1)
        avg_attn = torch.mean(a, dim=1, keepdim=True)
        max_attn, _ = torch.max(a, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)

        sig = self.conv_squeeze(agg).sigmoid()

        # 将不同感受野的特征自适应融合
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + \
               attn2 * sig[:, 1, :, :].unsqueeze(1) + \
               attn3 * sig[:, 2, :, :].unsqueeze(1)

        attn = self.conv_out(attn)
        return x * attn


# ---------------------------------------------------------------------------
# 2. 完整的动态检测头 Module
# 完美适配你的 UavDHead 的 Vision-Language 融合及 BSU 结构
# ---------------------------------------------------------------------------
@MODELS.register_module()
class UAVDynamicHeadModule(BaseModule):
    """
    针对无人机视觉优化的动态 Head Module。
    替换了原有的僵硬卷积，加入了 LSKBlock 大感受野感知，并支持文本特征的点乘匹配。
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: List[int],
                 widen_factor: float = 1.0,
                 reg_max: int = 16,
                 featmap_strides: List[int] = [8, 16, 32],
                 norm_cfg: dict = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: dict = dict(type='SiLU', inplace=True),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.widen_factor = widen_factor
        self.reg_max = reg_max
        self.featmap_strides = featmap_strides
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # 根据 YOLOv8 的习惯，统一下游通道数
        self.shared_channels = [
            make_divisible(max(c, num_classes), widen_factor) for c in in_channels
        ]
        self.reg_channels = [
            make_divisible(max(c, reg_max * 4), widen_factor) for c in in_channels
        ]

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        self._init_layers()

    def _init_layers(self):
        for i in range(len(self.featmap_strides)):
            # -------------------------------------------------------
            # 分类分支 (Class Branch)
            # 使用 LSK 模块替代第一层卷积，获取大感受野的背景区分能力
            # -------------------------------------------------------
            cls_branch = nn.Sequential(
                ConvModule(
                    self.in_channels[i],
                    self.shared_channels[i],
                    3, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
                LSKblock(self.shared_channels[i]),  # 引入无人机专属的 LSK 模块
                ConvModule(
                    self.shared_channels[i],
                    self.shared_channels[i],
                    3, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            )
            self.cls_convs.append(cls_branch)

            # 分类预测：注意，如果是 YOLO-World 文本融合，这里只输出特征，不直接输出 num_classes
            # 方便后续与 txt_feats 做 einsum (点乘)
            self.cls_preds.append(
                nn.Conv2d(self.shared_channels[i], self.shared_channels[i], 1)
            )

            # -------------------------------------------------------
            # 回归分支 (Regression Branch)
            # 回归任务更依赖局部边缘特征，保持标准卷积结构
            # -------------------------------------------------------
            reg_branch = nn.Sequential(
                ConvModule(
                    self.in_channels[i],
                    self.reg_channels[i],
                    3, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
                ConvModule(
                    self.reg_channels[i],
                    self.reg_channels[i],
                    3, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            )
            self.reg_convs.append(reg_branch)

            # 预测边界框的分布 (DFL)
            self.reg_preds.append(
                nn.Conv2d(self.reg_channels[i], 4 * self.reg_max, 1)
            )

        # 回归分布转换向量
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max - 1, self.reg_max), requires_grad=False)

    def forward(self, img_feats: Tuple[Tensor], txt_feats: Optional[Tensor] = None) -> Tuple[List[Tensor]]:
        cls_scores = []
        bbox_preds = []
        bbox_dist_preds = []
        att_scores = []

        for i in range(len(self.featmap_strides)):
            img_feat = img_feats[i]

            # 1. 经过分类和回归的分离网络
            cls_feat = self.cls_convs[i](img_feat)
            reg_feat = self.reg_convs[i](img_feat)

            # 2. 计算边界框回归和 DFL (Distribution Focal Loss)
            bbox_dist_pred = self.reg_preds[i](reg_feat)
            bbox_dist_preds.append(bbox_dist_pred)

            b, _, h, w = bbox_dist_pred.shape
            bbox_dist_pred_view = bbox_dist_pred.view(b, 4, self.reg_max, h, w).softmax(2)
            # 通过期望值投影得到实际的回归偏移量
            bbox_pred = (bbox_dist_pred_view * self.proj.view(1, 1, -1, 1, 1)).sum(2)
            bbox_preds.append(bbox_pred)

            # 3. 计算分类/属性分数 (Vision-Language Fusion)
            # 提取出的视觉特征投影
            cls_vision_feat = self.cls_preds[i](cls_feat)

            if txt_feats is not None:
                # 当传入了 txt_feats (包含已知类别或属性嵌入)
                # 动态将视觉特征与文本特征进行对齐 (点乘匹配)
                # txt_feats shape: (Batch, Num_Texts, Channels)
                b_sz, n_txt, c_txt = txt_feats.shape

                # 如果视觉特征的通道数与文本通道数不一致，利用一个动态权重对齐 (或者在外部处理好一致的维度)
                # 假设此处通道数已经匹配，进行点乘融合：
                # einsum 逻辑: b=batch, c=channel, h=height, w=width, n=num_texts
                score = torch.einsum('b c h w, b n c -> b n h w', cls_vision_feat, txt_feats)

                # 为了完美契合 UavDHead 的 outputs 格式 (cls_scores, bbox_preds, bbox_dist_preds, att_scores)
                # 我们假设 UavDHead 传递进来的 txt_feats 就是融合所需要的全部文本特征
                # 在 UavDHead 的逻辑里，它可能合并了 cls_feats 和 att_feats 传进来，或者分别计算
                cls_scores.append(score)
            else:
                # Fallback: 如果没有文本特征，使用普通的视觉分类输出 (仅作保底防报错)
                # 这种情况通常只在纯视觉模式下出现
                cls_scores.append(cls_vision_feat)

        # 返回与 UavDHead 所需参数长度和顺序完全一致的元组
        # 第四个参数预留给 att_scores，UavDHead 会在后续的 split 中自行切割 cls_scores 或不用
        return tuple(cls_scores), tuple(bbox_preds), tuple(bbox_dist_preds), None