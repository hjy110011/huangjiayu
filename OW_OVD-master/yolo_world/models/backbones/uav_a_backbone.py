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


class SPDConv(BaseModule):
    """
    Space-to-Depth Convolution (SPD-Conv) for preventing information loss
    during downsampling. Critical for small object detection.
    Ref: YOLOv5-SPD: https://arxiv.org/abs/2208.03679
    """

    def __init__(self, in_channels, out_channels, dimension=1, init_cfg=None):
        super().__init__(init_cfg)
        self.d = dimension
        self.conv = nn.Conv2d(in_channels * (self.d ** 2), out_channels,
                              kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, 4C, H/2, W/2]
        return self.conv(self._space_to_depth(x))

    def _space_to_depth(self, x):
        # Equivalent to PixelUnshuffle
        B, C, H, W = x.shape
        x = x.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(B, C * 4, H // 2, W // 2)


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention for Efficient Mobile Network Design
    Paper: https://arxiv.org/abs/2103.02907
    Replaces GlobalContextBlock to preserve spatial information (X/Y location).
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

        # X-direction and Y-direction pooling
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

        out = identity * a_h * a_w
        return out


class MultiScaleContextBlock(BaseModule):
    """
    Improved Context Block: Uses Coordinate Attention on each scale independently
    instead of forcing the last layer's global context onto everything.
    """

    def __init__(self, in_channels_list):
        super().__init__()
        self.attentions = nn.ModuleList()
        for c in in_channels_list:
            self.attentions.append(CoordinateAttention(c))

    def forward(self, img_feats):
        # img_feats: Tuple(P3, P4, P5, ...)
        new_feats = []
        for i, feat in enumerate(img_feats):
            # Apply spatial-aware attention to each scale
            enhanced_feat = self.attentions[i](feat)
            new_feats.append(enhanced_feat)
        return tuple(new_feats)


@MODELS.register_module()
class UAVABackbone(BaseModule):

    def __init__(self,
                 image_model: ConfigType,
                 text_model: ConfigType,
                 frozen_stages: int = -1,
                 with_text_model: bool = True,
                 init_cfg: OptMultiConfig = None,
                 feat_channels=[256, 512, 512]) -> None:  # Adjusted channels usually resemble this
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

        # REPLACEMENT: Use Coordinate Attention instead of Global Context
        # This preserves spatial details critical for small UAV targets
        self.context_module = MultiScaleContextBlock(feat_channels)

    def _freeze_stages(self):
        """Freeze the parameters of the specified stage."""
        if self.frozen_stages >= 0:
            # Note: This assumes image_model has 'layers' attribute
            # or follows a standard structure. If using HuggingFace AutoModel,
            # you might need to adjust how layers are accessed (e.g., model.encoder.layer)
            if hasattr(self.image_model, 'layers'):
                for i in range(self.frozen_stages + 1):
                    m = getattr(self.image_model, self.image_model.layers[i])
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False
            elif hasattr(self.image_model, 'model'):  # For HuggingFace wrappers
                # Logic for freezing HF layers if needed
                pass

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()

    def forward(self, image: Tensor,
                text: List[List[str]] = None) -> Tuple[Tuple[Tensor], Tensor]:

        # 1. Extract raw features
        # img_feats is expected to be a tuple of multi-scale features (e.g., P3, P4, P5)
        img_feats = self.image_model(image)

        # 2. Apply Coordinate Attention enhancement
        # Unlike the previous GlobalContextBlock which smeared the last layer's
        # average over everything, this enhances features while keeping X/Y info.
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