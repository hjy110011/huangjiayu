# Copyright (c) Tencent Inc. All rights reserved.
import torch
import torch.nn as nn
from typing import List, Tuple
from torch import Tensor
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmdet.utils import OptMultiConfig, ConfigType


class LSKLayer(BaseModule):
    """
    LSKLayer (Large Selective Kernel Layer)
    修正版：移除错误的通道降维，确保维度匹配
    """

    def __init__(self, dim):
        super().__init__()
        # 1. 大核卷积序列
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)

        # [修改 1] 删除之前的 conv1 和 conv2 (它们错误地将通道减半了)
        # self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        # self.conv2 = nn.Conv2d(dim, dim // 2, 1)

        # 2. 空间选择机制
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv_sigmoid = nn.Conv2d(2, 2, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        # [修改 2] 删除降维操作，保持维度为 dim
        # attn1 = self.conv1(attn1)
        # attn2 = self.conv2(attn2)

        # 堆叠两个特征用于生成权重
        attn = torch.cat([attn1, attn2], dim=1)

        # 生成空间注意力权重
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)

        sig = self.conv_squeeze(agg).sigmoid()

        # 加权融合
        # attn1/attn2 维度现在是 [B, dim, H, W]，与 x 一致
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)

        # 最终输出
        return x * attn


class LSKContextBlock(BaseModule):
    """
    将 LSKLayer 应用于多尺度特征图 (P3, P4, P5)
    替代原有的 GlobalContextBlock
    """

    def __init__(self, in_channels_list):
        super().__init__()
        self.lsk_layers = nn.ModuleList()
        # 对每一层特征图独立应用 LSK，不强制依赖最后一层
        for c in in_channels_list:
            self.lsk_layers.append(LSKLayer(c))

    def forward(self, img_feats):
        # img_feats 是一个 tuple (P3, P4, P5, ...)
        new_feats = []
        for i, feat in enumerate(img_feats):
            # 对每一层进行 LSK 空间选择性增强
            enhanced_feat = self.lsk_layers[i](feat)
            new_feats.append(enhanced_feat)

        return tuple(new_feats)


@MODELS.register_module()
class UAVCBackbone(BaseModule):

    def __init__(self,
                 image_model: ConfigType,
                 text_model: ConfigType,
                 frozen_stages: int = -1,
                 with_text_model: bool = True,
                 init_cfg: OptMultiConfig = None,
                 feat_channels=[256, 512, 512]) -> None:
        super().__init__(init_cfg)
        self.with_text_model = with_text_model

        # 构建基础视觉模型 (e.g. YOLOv8 CSPDarknet / ResNet)
        self.image_model = MODELS.build(image_model)

        if self.with_text_model:
            self.text_model = MODELS.build(text_model)
        else:
            self.text_model = None

        self.frozen_stages = frozen_stages
        self._freeze_stages()

        # [修改点]：替换 GlobalContextBlock 为 LSKContextBlock
        # 原逻辑：self.context_module = GlobalContextBlock(feat_channels)
        self.context_module = LSKContextBlock(feat_channels)

    def _freeze_stages(self):
        """Freeze the parameters of the specified stage."""
        if self.frozen_stages >= 0:
            # 兼容不同 Backbone 的层命名方式
            if hasattr(self.image_model, 'layers'):
                for i in range(self.frozen_stages + 1):
                    m = getattr(self.image_model, self.image_model.layers[i])
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False
            elif hasattr(self.image_model, 'model'):
                pass  # HuggingFace wrapper logic usually handled internally or separately

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()

    def forward(self, image: Tensor,
                text: List[List[str]]) -> Tuple[Tuple[Tensor], Tensor]:
        # 1. 提取基础特征
        img_feats = self.image_model(image)

        # 2. 应用 LSK 增强 (保持空间分辨率的同时增强语义)
        img_feats = self.context_module(img_feats)

        if self.with_text_model:
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