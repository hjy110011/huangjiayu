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
# 3. 基础组件：小波高频注意力模块
# ---------------------------------------------------------------
class WaveletAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hf_attn = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        nn.init.constant_(self.hf_attn[3].weight, 0)

    def forward(self, x):
        x_float = x.float()

        x00 = x_float[:, :, 0::2, 0::2]
        x01 = x_float[:, :, 0::2, 1::2]
        x10 = x_float[:, :, 1::2, 0::2]
        x11 = x_float[:, :, 1::2, 1::2]

        HL = (x00 - x01 + x10 - x11) / 2.0
        LH = (x00 + x01 - x10 - x11) / 2.0
        HH = (x00 - x01 - x10 + x11) / 2.0

        hf_concat = torch.cat([HL, LH, HH], dim=1)
        hf_concat = hf_concat.to(x.dtype)
        hf_weight = self.hf_attn(hf_concat)

        hf_weight_up = F.interpolate(hf_weight, size=x.shape[2:], mode='bilinear', align_corners=False)
        return torch.clamp(hf_weight_up, min=1e-4, max=1.0 - 1e-4)


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
                self.spatial_attentions.append(GlobalContextUnit(c))
                self.wavelet_attentions.append(None)
            else:
                self.spatial_attentions.append(CoordinateAttention(c))
                if self.use_wavelet:
                    self.wavelet_attentions.append(WaveletAttention(c))
                else:
                    self.wavelet_attentions.append(None)

    def forward(self, img_feats):
        new_feats = []
        for i, feat in enumerate(img_feats):
            att_feat = self.spatial_attentions[i](feat)
            wavelet_module = self.wavelet_attentions[i]
            if wavelet_module is not None:
                hf_weight = wavelet_module(feat)
                if self.use_wavelet_residual:
                    att_feat = att_feat * (1.0 + hf_weight)
                else:
                    att_feat = att_feat * hf_weight

            enhanced_feat = feat + att_feat
            new_feats.append(enhanced_feat)
        return tuple(new_feats)


# ---------------------------------------------------------------
# 5. 【全新核心组件】：预测编码误差模块 (模拟“脑补失败”)
# ---------------------------------------------------------------
class PredictiveCodingErrorModule(nn.Module):
    """
    通过极其狭窄的信息瓶颈（Bottleneck）强迫网络只能“背下”面积巨大的常见背景。
    对于罕见的高频未知物体，网络无法重构，产生巨大的预测误差。
    我们将此误差作为“惊奇感（Surprise）”注意力，注入原特征。
    """

    def __init__(self, in_channels, bottleneck_ratio=8):
        super().__init__()
        # 故意制造一个极其狭窄的“脑容量瓶颈”，强迫网络只记大面积背景
        bottleneck_dim = max(in_channels // bottleneck_ratio, 16)

        # 编码器：极限压缩信息
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.SiLU(inplace=True)
        )

        # 解码器：试图凭常识脑补恢复画面
        self.decoder = nn.Sequential(
            nn.Conv2d(bottleneck_dim, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
            # 注意：不加激活函数，输出线性特征直接用于求误差
        )

    def forward(self, x):
        # 1. 网络凭借常识脑补出的特征图
        reconstructed = self.decoder(self.encoder(x))

        # 2. 计算脑补误差 (均方误差，并在通道维度求平均)
        # error_map 是一张形状为 [B, 1, H, W] 的二维空间热力图
        error_map = torch.mean((x - reconstructed) ** 2, dim=1, keepdim=True)

        # 3. 归一化误差（限制在 0~1 之间），防止极端特征爆炸
        normalized_error = torch.sigmoid(error_map)

        # 4. 核心逻辑：将“惊奇感”作为特征放大器注入原特征！
        # 对于脑补失败（误差大）的地方，特征响应最高可被放大 2 倍
        enhanced_feat = x * (1.0 + normalized_error)

        return enhanced_feat


# ---------------------------------------------------------------
# 6. 最终的主干网络 (集成了脑补模块)
# ---------------------------------------------------------------
@MODELS.register_module()
class UAVGBackbone(BaseModule):
    def __init__(self,
                 image_model: ConfigType,
                 text_model: ConfigType,
                 frozen_stages: int = -1,
                 with_text_model: bool = True,
                 init_cfg: OptMultiConfig = None,
                 feat_channels=[256, 512, 512],
                 use_wavelet: bool = True,
                 use_wavelet_residual: bool = True,
                 use_predictive_error: bool = True) -> None:  # <--- 新增开关
        super().__init__(init_cfg)
        self.with_text_model = with_text_model
        self.image_model = MODELS.build(image_model)

        if self.with_text_model:
            self.text_model = MODELS.build(text_model)
        else:
            self.text_model = None

        self.frozen_stages = frozen_stages
        self._freeze_stages()

        # 小波高级混合模块
        self.context_module = WaveletHybridContextBlock(
            in_channels_list=feat_channels,
            use_wavelet=use_wavelet,
            use_wavelet_residual=use_wavelet_residual
        )

        self.use_predictive_error = use_predictive_error
        # 实例化脑补误差模块 (每个特征金字塔层级配置一个)
        if self.use_predictive_error:
            self.predictive_error_modules = nn.ModuleList([
                PredictiveCodingErrorModule(c) for c in feat_channels
            ])

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
        # 1. 基础图像特征提取
        img_feats = self.image_model(image)

        # 2. 小波变换增强细节
        img_feats = self.context_module(img_feats)

        # 3. 【新加入的机制】：计算脑补误差，激发出未知区域的惊奇感
        if self.use_predictive_error:
            enhanced_feats = []
            for i, feat in enumerate(img_feats):
                # 利用并行的 Autoencoder 计算重构误差，并用作高光特效
                enhanced_feat = self.predictive_error_modules[i](feat)
                enhanced_feats.append(enhanced_feat)
            img_feats = tuple(enhanced_feats)

        # 4. 文本多模态特征
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
        img_feats = self.context_module(img_feats)

        if self.use_predictive_error:
            enhanced_feats = []
            for i, feat in enumerate(img_feats):
                enhanced_feat = self.predictive_error_modules[i](feat)
                enhanced_feats.append(enhanced_feat)
            img_feats = tuple(enhanced_feats)

        return img_feats