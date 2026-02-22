# Copyright (c) Tencent Inc. All rights reserved.
import copy
from typing import List, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from mmyolo.models.utils import make_divisible, make_round
from mmyolo.models.necks.yolov8_pafpn import YOLOv8PAFPN


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    """
    DySample: Learning to Upsample by Learning to Sample (ICCV 2023)
    官方完整实现：支持动态点采样，极其适合无人机小目标的高频细节恢复。
    """

    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)

        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            normal_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h], indexing='ij')).transpose(1, 2).repeat(1, self.groups, 1).reshape(1,
                                                                                                                    -1,
                                                                                                                    1,
                                                                                                                    1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)

        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing='ij')).transpose(1, 2).unsqueeze(
            1).unsqueeze(0).type(x.dtype).to(x.device)

        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


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
class UAVAPAFPN(YOLOv8PAFPN):
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
            norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
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

        # 替换默认的 nn.Upsample 为 DySample
        self.upsample_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            c_in = make_divisible(self.out_channels[len(in_channels) - 1 - idx], self.widen_factor)
            self.upsample_layers.append(DySample(c_in, scale=2))

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer."""
        block_cfg = copy.deepcopy(self.block_cfg)
        block_cfg.update(
            dict(in_channels=make_divisible(
                (self.in_channels[idx - 1] + self.in_channels[idx]), self.widen_factor),
                out_channels=make_divisible(self.out_channels[idx - 1], self.widen_factor),
                guide_channels=self.guide_channels,
                embed_channels=make_round(self.embed_channels[idx - 1], self.widen_factor),
                num_heads=make_round(self.num_heads[idx - 1], self.widen_factor),
                num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        return MODELS.build(block_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer."""
        block_cfg = copy.deepcopy(self.block_cfg)
        block_cfg.update(
            dict(in_channels=make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]), self.widen_factor),
                out_channels=make_divisible(self.out_channels[idx + 1], self.widen_factor),
                guide_channels=self.guide_channels,
                embed_channels=make_round(self.embed_channels[idx + 1], self.widen_factor),
                num_heads=make_round(self.num_heads[idx + 1], self.widen_factor),
                num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        return MODELS.build(block_cfg)

    def forward(self, img_feats: List[Tensor], txt_feats: Optional[Tensor] = None) -> tuple:
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

            # 使用 DySample
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
class UAVADualPAFPN(YOLOv8PAFPN):
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
            norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
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

        # 统一使用 DySample，防止使用 SubpixelUpsample 时产生的棋盘效应
        self.upsample_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            c_in = make_divisible(self.out_channels[len(in_channels) - 1 - idx], self.widen_factor)
            self.upsample_layers.append(DySample(c_in, scale=2))

    def forward(self, img_feats: List[Tensor], txt_feats: Optional[Tensor] = None) -> tuple:
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

            # 使用 DySample
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 - idx](feat_high)

            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs, txt_feats)
            inner_outs.insert(0, inner_out)

        # 安全处理 txt_feats
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