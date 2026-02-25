# Copyright (c) Tencent Inc. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from torch import Tensor
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmdet.utils import OptMultiConfig, ConfigType


# ---------------------------------------------------------------
# 1. 基础组件：坐标注意力 (用于浅层)
# ---------------------------------------------------------------
class CoordinateAttention(nn.Module):
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


# ---------------------------------------------------------------
# 2. 基础组件：全局上下文 (用于深层)
# ---------------------------------------------------------------
class GlobalContextUnit(nn.Module):
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
        y = self.avg_pool(x).view(b, c)
        y = self.mlp(y).view(b, c, 1, 1)
        return x * y


# ---------------------------------------------------------------
# 3. 新增组件：独立的小波高频注意力模块
# ---------------------------------------------------------------
class WaveletAttention(nn.Module):
    """
    基于 2D Haar 离散小波变换的高频特征提取器
    """

    def __init__(self, dim):
        super().__init__()
        # 输入为 3 个高频分量 (HL, LH, HH) 拼接
        self.hf_attn = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. DWT 高效切片
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        # 2. 提取三个高频分量
        HL = (x00 - x01 + x10 - x11) / 2.0  # 垂直高频
        LH = (x00 + x01 - x10 - x11) / 2.0  # 水平高频
        HH = (x00 - x01 - x10 + x11) / 2.0  # 对角高频

        # 3. 计算高频注意力权重
        hf_concat = torch.cat([HL, LH, HH], dim=1)
        hf_weight = self.hf_attn(hf_concat)

        # 4. 上采样对齐原图分辨率
        hf_weight_up = F.interpolate(hf_weight, size=x.shape[2:], mode='bilinear', align_corners=False)
        return hf_weight_up


# ---------------------------------------------------------------
# 4. 融合模块：小波 + 混合注意力
# ---------------------------------------------------------------
class WaveletHybridContextBlock(BaseModule):
    def __init__(self, in_channels_list, use_wavelet=True, use_wavelet_residual=True):
        super().__init__()
        self.use_wavelet = use_wavelet
        self.use_wavelet_residual = use_wavelet_residual

        self.spatial_attentions = nn.ModuleList()
        self.wavelet_attentions = nn.ModuleList()

        for i, c in enumerate(in_channels_list):
            if i == len(in_channels_list) - 1:
                # 深层：仅使用全局上下文提取语义，不使用小波 (深层缺乏高频细节)
                self.spatial_attentions.append(GlobalContextUnit(c))
                self.wavelet_attentions.append(None)
            else:
                # 浅层/中层：使用坐标注意力，并配置小波模块
                self.spatial_attentions.append(CoordinateAttention(c))
                if self.use_wavelet:
                    self.wavelet_attentions.append(WaveletAttention(c))
                else:
                    self.wavelet_attentions.append(None)

    def forward(self, img_feats):
        new_feats = []
        for i, feat in enumerate(img_feats):
            # 1. 基础空间/语义调制
            enhanced_feat = self.spatial_attentions[i](feat)

            # 2. 小波高频叠加 (仅针对浅层且开启小波时)
            wavelet_module = self.wavelet_attentions[i]
            if wavelet_module is not None:
                # 提取原图的高频权重
                hf_weight = wavelet_module(feat)
                if self.use_wavelet_residual:
                    # 残差模式：安全增强
                    enhanced_feat = enhanced_feat * (1.0 + hf_weight)
                else:
                    # 激进模式：直接相乘
                    enhanced_feat = enhanced_feat * hf_weight

            new_feats.append(enhanced_feat)
        return tuple(new_feats)


# ---------------------------------------------------------------
# 5. 最终的主干网络
# ---------------------------------------------------------------
@MODELS.register_module()
class UAVBFBackbone(BaseModule):
    def __init__(self,
                 image_model: ConfigType,
                 text_model: ConfigType,
                 frozen_stages: int = -1,
                 with_text_model: bool = True,
                 init_cfg: OptMultiConfig = None,
                 feat_channels=[256, 512, 1024],
                 use_wavelet: bool = True,
                 use_wavelet_residual: bool = True) -> None:
        super().__init__(init_cfg)
        self.with_text_model = with_text_model
        self.image_model = MODELS.build(image_model)

        if self.with_text_model:
            self.text_model = MODELS.build(text_model)
        else:
            self.text_model = None

        self.frozen_stages = frozen_stages
        self._freeze_stages()

        # [核心替换] 引入自带小波的高级混合模块
        self.context_module = WaveletHybridContextBlock(
            in_channels_list=feat_channels,
            use_wavelet=use_wavelet,
            use_wavelet_residual=use_wavelet_residual
        )

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if hasattr(self.image_model, 'layers'):
                for i in range(self.frozen_stages + 1):
                    m = getattr(self.image_model, self.image_model.layers[i])
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()

    def forward(self, image: Tensor,
                text: List[List[str]] = None) -> Tuple[Tuple[Tensor], Tensor]:
        img_feats = self.image_model(image)
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