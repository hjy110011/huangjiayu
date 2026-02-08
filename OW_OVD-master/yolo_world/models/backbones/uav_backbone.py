# Copyright (c) Tencent Inc. All rights reserved.
import itertools
from typing import List, Sequence, Tuple
import torch
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmdet.utils import OptMultiConfig, ConfigType
from transformers import (AutoTokenizer, AutoModel, CLIPTextConfig)
from transformers import CLIPTextModelWithProjection as CLIPTP

import torch.nn as nn


class GlobalContextBlock(nn.Module):
    """
    创新模块：利用全局特征来调制局部特征
    """

    def __init__(self, in_channels_list, reduction=16):
        super().__init__()
        # 假设我们用最后一层（语义最强）来提取全局上下文
        self.global_context_dim = in_channels_list[-1]

        # 定义一个共享的 MLP 来生成通道权重
        self.context_mlps = nn.ModuleList()
        for c in in_channels_list:
            self.context_mlps.append(nn.Sequential(
                nn.Linear(self.global_context_dim, c // reduction),
                nn.ReLU(),
                nn.Linear(c // reduction, c),
                nn.Sigmoid()
            ))

    def forward(self, img_feats):
        # img_feats 是一个 tuple (P3, P4, P5, ...)
        # 1. 提取全局上下文向量 (Batch, C_last, 1, 1) -> (Batch, C_last)
        last_feat = img_feats[-1]
        global_context = last_feat.mean(dim=(2, 3))  # Global Average Pooling

        new_feats = []
        # 2. 对每一层特征图进行调制
        for i, feat in enumerate(img_feats):
            # 生成该层的通道权重 (Batch, C)
            weight = self.context_mlps[i](global_context)
            # 扩展维度以便相乘 (Batch, C, 1, 1)
            weight = weight.view(feat.size(0), feat.size(1), 1, 1)
            # 调制特征：Feature * Weight
            new_feats.append(feat * weight)

        return tuple(new_feats)




@MODELS.register_module()
class UAVBackbone(BaseModule):

    def __init__(self,
                 image_model: ConfigType,
                 text_model: ConfigType,
                 frozen_stages: int = -1,
                 with_text_model: bool = True,
                 init_cfg: OptMultiConfig = None,
                 feat_channels=[256, 512, 512]) -> None:
        super().__init__(init_cfg)
        self.with_text_model = with_text_model
        self.image_model = MODELS.build(image_model)
        if self.with_text_model:
            self.text_model = MODELS.build(text_model)
        else:
            self.text_model = None
        self.frozen_stages = frozen_stages
        self._freeze_stages()
        self.context_module = GlobalContextBlock(feat_channels)

    def _freeze_stages(self):
        """Freeze the parameters of the specified stage so that they are no
        longer updated."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self.image_model, self.image_model.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
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
        return self.image_model(image)

