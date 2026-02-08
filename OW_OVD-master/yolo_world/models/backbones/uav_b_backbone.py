# Copyright (c) Tencent Inc. All rights reserved.
import itertools
from typing import List, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmdet.utils import OptMultiConfig, ConfigType


# ---------------------------------------------------------------
# [新增/保留] 基础组件定义
# ---------------------------------------------------------------

class CoordinateAttention(nn.Module):
    """
    保留 UAV-A 的 Coordinate Attention
    作用：用于浅层/中层特征，极致保留空间位置信息 (X/Y)，利于小目标定位。
    """

    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_h * a_w


class GlobalContextUnit(nn.Module):
    """
    [新增] 单层 Global Context Unit (简化版 SE-Block 逻辑)
    作用：用于深层特征，提取全局语义，忽略空间细节，计算量低。
    """

    def __init__(self, c, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(c, c // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // reduction, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Global Avg Pooling
        y = self.avg_pool(x).view(b, c)
        # Channel Weighting
        y = self.mlp(y).view(b, c, 1, 1)
        # Reweight
        return x * y


class HybridContextBlock(BaseModule):
    """
    [UAV-B 核心] 混合注意力模块
    策略：
    1. 浅层/中层 (e.g., P3, P4): 使用 CoordinateAttention -> 保留空间细节
    2. 深层 (e.g., P5): 使用 GlobalContextUnit -> 提取语义，节省算力
    """

    def __init__(self, in_channels_list):
        super().__init__()
        self.attentions = nn.ModuleList()

        for i, c in enumerate(in_channels_list):
            # 判断逻辑：如果是最后一层（最深层），使用 Global Context
            if i == len(in_channels_list) - 1:
                self.attentions.append(GlobalContextUnit(c))
            else:
                # 其他层（浅层）使用 Coordinate Attention
                self.attentions.append(CoordinateAttention(c))

    def forward(self, img_feats):
        new_feats = []
        for i, feat in enumerate(img_feats):
            enhanced_feat = self.attentions[i](feat)
            new_feats.append(enhanced_feat)
        return tuple(new_feats)


# ---------------------------------------------------------------
# 修改后的 Backbone
# ---------------------------------------------------------------

@MODELS.register_module()
class UAVBBackbone(BaseModule):
    # 建议重命名为 UAVBBackbone 以示区分，如果必须覆盖原名可改回 UAVABackbone

    def __init__(self,
                 image_model: ConfigType,
                 text_model: ConfigType,
                 frozen_stages: int = -1,
                 with_text_model: bool = True,
                 init_cfg: OptMultiConfig = None,
                 feat_channels=[256, 512, 1024]) -> None:
        super().__init__(init_cfg)
        self.with_text_model = with_text_model

        # Build the image backbone
        self.image_model = MODELS.build(image_model)

        if self.with_text_model:
            self.text_model = MODELS.build(text_model)
        else:
            self.text_model = None

        self.frozen_stages = frozen_stages
        self._freeze_stages()

        # [MODIFIED] Replacement: Use Hybrid Strategy
        # HybridContextBlock applies Coordinate Attention to shallow layers
        # and Global Context to deep layers.
        self.context_module = HybridContextBlock(feat_channels)

    def _freeze_stages(self):
        """Freeze the parameters of the specified stage."""
        if self.frozen_stages >= 0:
            if hasattr(self.image_model, 'layers'):
                for i in range(self.frozen_stages + 1):
                    m = getattr(self.image_model, self.image_model.layers[i])
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False
            elif hasattr(self.image_model, 'model'):
                pass

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()

    def forward(self, image: Tensor,
                text: List[List[str]] = None) -> Tuple[Tuple[Tensor], Tensor]:

        # 1. Extract raw features
        img_feats = self.image_model(image)

        # 2. Apply Hybrid Attention enhancement
        # (CA for spatial details on P3/P4, GC for semantics on P5)
        img_feats = self.context_module(img_feats)

        txt_feats = None
        if self.with_text_model and text is not None:
            txt_feats = self.text_model(text)
            return img_feats, txt_feats
        else:
            return img_feats, None

    def forward_text(self, text: List[List[str]]) -> Tensor:
        assert self.with_text_model, "forward_text() requires a text model"
        txt_feats = self.text_model(text)
        return txt_feats

    def forward_image(self, image: Tensor) -> Tuple[Tensor]:
        img_feats = self.image_model(image)
        return self.context_module(img_feats)