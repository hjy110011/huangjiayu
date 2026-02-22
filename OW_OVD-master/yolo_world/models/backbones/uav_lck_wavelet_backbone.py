# Copyright (c) Tencent Inc. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from torch import Tensor
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmdet.utils import OptMultiConfig, ConfigType


class LSKLayer(BaseModule):
    """
    LSKLayer (Large Selective Kernel Layer) + Wavelet Enhancement
    加入了基于小波变换 (DWT) 的高频特征提取机制与残差消融控制
    """

    def __init__(self, dim, use_wavelet=False, use_wavelet_residual=False):
        super().__init__()
        self.use_wavelet = use_wavelet
        self.use_wavelet_residual = use_wavelet_residual  # 新增：残差控制开关

        # 1. 大核卷积序列
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)

        # 2. 空间选择机制
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv_sigmoid = nn.Conv2d(2, 2, 1)

        # 3. [消融实验新增] 小波高频特征注意力模块
        if self.use_wavelet:
            # 输入为 3 个高频分量 (HL, LH, HH) 拼接，通道数为 dim * 3
            self.hf_attn = nn.Sequential(
                nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                nn.Sigmoid()  # 输出 [0, 1] 之间的权重
            )

    def forward(self, x):
        # --- 原始 LSK 空间特征提取 ---
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        # 堆叠两个特征用于生成权重
        attn = torch.cat([attn1, attn2], dim=1)

        # 生成空间注意力权重
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)

        sig = self.conv_squeeze(agg).sigmoid()

        # 加权融合 (基础版 LSK 空间注意力)
        spatial_attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)

        # --- [消融实验新增] 小波变换高频增强 ---
        if self.use_wavelet:
            # 1. 2D Haar 离散小波变换 (DWT) 的高效切片实现
            x00 = x[:, :, 0::2, 0::2]
            x01 = x[:, :, 0::2, 1::2]
            x10 = x[:, :, 1::2, 0::2]
            x11 = x[:, :, 1::2, 1::2]

            # 提取三个高频分量 (HF)
            HL = (x00 - x01 + x10 - x11) / 2.0  # 垂直高频
            LH = (x00 + x01 - x10 - x11) / 2.0  # 水平高频
            HH = (x00 - x01 - x10 + x11) / 2.0  # 对角高频

            # 2. 拼接高频分量
            hf_concat = torch.cat([HL, LH, HH], dim=1)

            # 3. 计算高频注意力权重 [B, dim, H/2, W/2]
            hf_weight = self.hf_attn(hf_concat)

            # 4. 上采样对齐原图分辨率
            hf_weight_up = F.interpolate(hf_weight, size=spatial_attn.shape[2:], mode='bilinear', align_corners=False)

            # 5. [消融实验核心] 残差机制控制
            if self.use_wavelet_residual:
                # 实验组 A: 带有残差调制 (安全模式，避免特征被抹杀)
                spatial_attn = spatial_attn * (1.0 + hf_weight_up)
            else:
                # 实验组 B: 纯注意力相乘 (激进模式，完全依赖高频模块的输出)
                spatial_attn = spatial_attn * hf_weight_up

        # 最终输出：将提取/增强后的注意力作用于原始输入
        return x * spatial_attn


class LSKContextBlock(BaseModule):
    """
    将 LSKLayer 应用于多尺度特征图
    """

    def __init__(self, in_channels_list, use_wavelet=False, use_wavelet_residual=False):
        super().__init__()
        self.lsk_layers = nn.ModuleList()
        for c in in_channels_list:
            self.lsk_layers.append(LSKLayer(
                dim=c,
                use_wavelet=use_wavelet,
                use_wavelet_residual=use_wavelet_residual
            ))

    def forward(self, img_feats):
        new_feats = []
        for i, feat in enumerate(img_feats):
            enhanced_feat = self.lsk_layers[i](feat)
            new_feats.append(enhanced_feat)
        return tuple(new_feats)


@MODELS.register_module()
class UAVDBackbone(BaseModule):

    def __init__(self,
                 image_model: ConfigType,
                 text_model: ConfigType,
                 frozen_stages: int = -1,
                 with_text_model: bool = True,
                 init_cfg: OptMultiConfig = None,
                 feat_channels=[256, 512, 512],
                 use_wavelet: bool = False,  # [开关 1] 是否启用小波
                 use_wavelet_residual: bool = False) -> None:  # [开关 2] 是否启用残差
        super().__init__(init_cfg)
        self.with_text_model = with_text_model

        self.use_wavelet = use_wavelet
        self.use_wavelet_residual = use_wavelet_residual

        # 构建基础视觉模型
        self.image_model = MODELS.build(image_model)

        if self.with_text_model:
            self.text_model = MODELS.build(text_model)
        else:
            self.text_model = None

        self.frozen_stages = frozen_stages
        self._freeze_stages()

        # 透传消融实验参数
        self.context_module = LSKContextBlock(
            in_channels_list=feat_channels,
            use_wavelet=self.use_wavelet,
            use_wavelet_residual=self.use_wavelet_residual
        )

    def _freeze_stages(self):
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
                text: List[List[str]]) -> Tuple[Tuple[Tensor], Tensor]:
        img_feats = self.image_model(image)
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