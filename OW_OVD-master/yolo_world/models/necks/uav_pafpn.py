# type: uploadedfile
# fileName: uav_pafpn.py
# fullContent:
# Copyright (c) Tencent Inc. All rights reserved.
import copy
from typing import List, Union

import torch
import torch.nn as nn
from torch import Tensor
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from mmyolo.models.utils import make_divisible, make_round
from mmyolo.models.necks.yolov8_pafpn import YOLOv8PAFPN


class SubpixelUpsample(nn.Module):
    """
    Subpixel Upsampling (PixelShuffle) replacing standard nn.Upsample.
    Preserves high-frequency details better than Bilinear Interpolation.
    Crucial for small object recovery in FPN top-down path.
    """

    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        # Expansion convolution: C -> C * r^2
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=1, stride=1, padding=0)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))


@MODELS.register_module()
class UAVPAFPN(YOLOv8PAFPN):
    """Path Aggregation Network used in YOLO World
    Following YOLOv8 PAFPN, including text to image fusion
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 guide_channels: int,
                 embed_channels: List[int],
                 num_heads: List[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 block_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg: OptMultiConfig = None) -> None:
        if act_cfg is None:
            act_cfg = dict(type='SiLU', inplace=True)
        if norm_cfg is None:
            norm_cfg = dict(type='BN',
                            momentum=0.03,
                            eps=0.001)
        if block_cfg is None:
            block_cfg = dict(type='CSPLayerWithTwoConv')
        self.guide_channels = guide_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.block_cfg = block_cfg
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         deepen_factor=deepen_factor,
                         widen_factor=widen_factor,
                         num_csp_blocks=num_csp_blocks,
                         freeze_all=freeze_all,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)

        # [核心修改] 替换默认的 nn.Upsample 为 SubpixelUpsample
        self.upsample_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            c_in = make_divisible(self.out_channels[len(in_channels) - 1 - idx], self.widen_factor)
            self.upsample_layers.append(SubpixelUpsample(c_in, scale_factor=2))

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer."""
        block_cfg = copy.deepcopy(self.block_cfg)
        block_cfg.update(
            dict(in_channels=make_divisible(
                (self.in_channels[idx - 1] + self.in_channels[idx]),
                self.widen_factor),
                out_channels=make_divisible(self.out_channels[idx - 1],
                                            self.widen_factor),
                guide_channels=self.guide_channels,
                embed_channels=make_round(self.embed_channels[idx - 1],
                                          self.widen_factor),
                num_heads=make_round(self.num_heads[idx - 1],
                                     self.widen_factor),
                num_blocks=make_round(self.num_csp_blocks,
                                      self.deepen_factor),
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        return MODELS.build(block_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer."""
        block_cfg = copy.deepcopy(self.block_cfg)
        block_cfg.update(
            dict(in_channels=make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
                out_channels=make_divisible(self.out_channels[idx + 1],
                                            self.widen_factor),
                guide_channels=self.guide_channels,
                embed_channels=make_round(self.embed_channels[idx + 1],
                                          self.widen_factor),
                num_heads=make_round(self.num_heads[idx + 1],
                                     self.widen_factor),
                num_blocks=make_round(self.num_csp_blocks,
                                      self.deepen_factor),
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        return MODELS.build(block_cfg)

    def forward(self, img_feats: List[Tensor], txt_feats: Tensor = None) -> tuple:
        """Forward function.
        including multi-level image features, text features: BxLxD
        """
        assert len(img_feats) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]

            # 使用 SubpixelUpsample
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 - idx](feat_high)

            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs, txt_feats)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](torch.cat(
                [downsample_feat, feat_high], 1), txt_feats)
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results)


@MODELS.register_module()
class UAVDualPAFPN(UAVPAFPN):
    """Path Aggregation Network used in YOLO World v8."""

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 guide_channels: int,
                 embed_channels: List[int],
                 num_heads: List[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 text_enhancder=None,
                 block_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         guide_channels=guide_channels,
                         embed_channels=embed_channels,
                         num_heads=num_heads,
                         deepen_factor=deepen_factor,
                         widen_factor=widen_factor,
                         num_csp_blocks=num_csp_blocks,
                         freeze_all=freeze_all,
                         block_cfg=block_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)

        if act_cfg is None:
            act_cfg = dict(type='SiLU', inplace=True)
        if norm_cfg is None:
            norm_cfg = dict(type='BN',
                            momentum=0.03,
                            eps=0.001)
        if block_cfg is None:
            block_cfg = dict(type='CSPLayerWithTwoConv')
        if text_enhancder is None:
            text_enhancder = dict(
                type='ImagePoolingAttentionModule',
                embed_channels=256,
                num_heads=8,
                pool_size=3)
        text_enhancder.update(
            dict(
                image_channels=[int(x * widen_factor) for x in out_channels],
                text_channels=guide_channels,
                num_feats=len(out_channels),
            ))
        print(text_enhancder)
        self.text_enhancer = MODELS.build(text_enhancder)

        # DualPAFPN 也需要替换上采样层
        self.upsample_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            c_in = make_divisible(self.out_channels[len(in_channels) - 1 - idx], self.widen_factor)
            self.upsample_layers.append(SubpixelUpsample(c_in, scale_factor=2))

    # [修复] 增加 = None 默认值，与父类签名保持一致
    def forward(self, img_feats: List[Tensor], txt_feats: Tensor = None) -> tuple:
        """Forward function."""
        assert len(img_feats) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]

            # 使用 SubpixelUpsample
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 - idx](feat_high)

            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs, txt_feats)
            inner_outs.insert(0, inner_out)

        # 如果 txt_feats 为 None，这里可能会出错，但为了满足签名一致性我们允许 None。
        # 实际上调用时应该保证有 txt_feats，或者在此处做 None 检查。
        if txt_feats is not None:
            txt_feats = self.text_enhancer(txt_feats, inner_outs)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](torch.cat(
                [downsample_feat, feat_high], 1), txt_feats)
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results)