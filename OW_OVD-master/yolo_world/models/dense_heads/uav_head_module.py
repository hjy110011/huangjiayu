# Copyright (c) Tencent Inc. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=6, groups=dim, dilation=2)
        self.conv_global = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)

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

        a = torch.cat([attn1, attn2, attn3], dim=1)
        avg_attn = torch.mean(a, dim=1, keepdim=True)
        max_attn, _ = torch.max(a, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)

        sig = self.conv_squeeze(agg).sigmoid()

        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + \
               attn2 * sig[:, 1, :, :].unsqueeze(1) + \
               attn3 * sig[:, 2, :, :].unsqueeze(1)

        attn = self.conv_out(attn)
        return x * attn


# ---------------------------------------------------------------------------
# 2. 完整的动态检测头 Module
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
                 embed_dims: int = 512,  # <--- 新增：接收文本特征维度
                 use_bn_head: bool = False,  # <--- 新增：接收 BN 开关
                 freeze_all: bool = False,  # <--- 新增：接收冻结开关
                 widen_factor: float = 1.0,
                 reg_max: int = 16,
                 featmap_strides: List[int] = [8, 16, 32],
                 norm_cfg: dict = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: dict = dict(type='SiLU', inplace=True),
                 init_cfg: Optional[dict] = None,
                 **kwargs):  # <--- 新增：万能垃圾桶吸收多余参数
        super().__init__(init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = embed_dims  # 保存属性
        self.use_bn_head = use_bn_head
        self.freeze_all = freeze_all
        self.widen_factor = widen_factor
        self.reg_max = reg_max
        self.featmap_strides = featmap_strides
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

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

        # 如果启用 BN Head，初始化独立的 BN 层和对比学习温度系数
        if self.use_bn_head:
            self.cls_norms = nn.ModuleList()
            self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        self._init_layers()

        if self.freeze_all:
            self._freeze_all()

    def _init_layers(self):
        for i in range(len(self.featmap_strides)):
            # --- 分类分支 ---
            cls_branch = nn.Sequential(
                ConvModule(self.in_channels[i], self.shared_channels[i], 3, padding=1, norm_cfg=self.norm_cfg,
                           act_cfg=self.act_cfg),
                LSKblock(self.shared_channels[i]),
                ConvModule(self.shared_channels[i], self.shared_channels[i], 3, padding=1, norm_cfg=self.norm_cfg,
                           act_cfg=self.act_cfg)
            )
            self.cls_convs.append(cls_branch)

            # 关键修复：输出通道必须投影到 embed_dims (512)，才能与文本特征对齐！
            self.cls_preds.append(
                nn.Conv2d(self.shared_channels[i], self.embed_dims, 1)
            )

            if self.use_bn_head:
                self.cls_norms.append(nn.BatchNorm2d(self.embed_dims))

            # --- 回归分支 ---
            reg_branch = nn.Sequential(
                ConvModule(self.in_channels[i], self.reg_channels[i], 3, padding=1, norm_cfg=self.norm_cfg,
                           act_cfg=self.act_cfg),
                ConvModule(self.reg_channels[i], self.reg_channels[i], 3, padding=1, norm_cfg=self.norm_cfg,
                           act_cfg=self.act_cfg)
            )
            self.reg_convs.append(reg_branch)

            self.reg_preds.append(
                nn.Conv2d(self.reg_channels[i], 4 * self.reg_max, 1)
            )

        self.proj = nn.Parameter(torch.linspace(0, self.reg_max - 1, self.reg_max), requires_grad=False)

    def _freeze_all(self):
        """冻结模型参数"""
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, 'freeze_all', False):
            self._freeze_all()

    def forward(self, img_feats: Tuple[Tensor], txt_feats: Optional[Tensor] = None) -> Tuple[List[Tensor]]:
        cls_scores = []
        bbox_preds = []
        bbox_dist_preds = []

        for i in range(len(self.featmap_strides)):
            img_feat = img_feats[i]

            cls_feat = self.cls_convs[i](img_feat)
            reg_feat = self.reg_convs[i](img_feat)

            # 回归计算
            bbox_dist_pred = self.reg_preds[i](reg_feat)
            bbox_dist_preds.append(bbox_dist_pred)

            b, _, h, w = bbox_dist_pred.shape
            bbox_dist_pred_view = bbox_dist_pred.view(b, 4, self.reg_max, h, w).softmax(2)
            bbox_pred = (bbox_dist_pred_view * self.proj.view(1, 1, -1, 1, 1)).sum(2)
            bbox_preds.append(bbox_pred)

            # 视觉分类特征投影 (输出维度已对齐为 embed_dims)
            cls_vision_feat = self.cls_preds[i](cls_feat)

            # 图文融合 (Vision-Language Fusion)
            if txt_feats is not None:
                if getattr(self, 'use_bn_head', False):
                    # YOLO-World 标准对比学习逻辑
                    cls_vision_feat = self.cls_norms[i](cls_vision_feat)
                    cls_vision_feat = F.normalize(cls_vision_feat, dim=1, p=2)
                    txt_feats_norm = F.normalize(txt_feats, dim=-1, p=2)
                    # 点乘计算相似度矩阵
                    score = torch.einsum('b c h w, b n c -> b n h w', cls_vision_feat, txt_feats_norm)
                    # 乘以温度系数
                    score = score * self.logit_scale.exp()
                else:
                    # 普通点乘 (Fallback)
                    score = torch.einsum('b c h w, b n c -> b n h w', cls_vision_feat, txt_feats)

                cls_scores.append(score)
            else:
                cls_scores.append(cls_vision_feat)

        # 返回 4 个参数的元组，完美适配 UavDHead 外层的需求
        return tuple(cls_scores), tuple(bbox_preds), tuple(bbox_dist_preds), None