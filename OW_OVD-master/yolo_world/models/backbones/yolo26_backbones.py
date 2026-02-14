# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmyolo.registry import MODELS


# ---------------------------------------------------------------------------- #
#  基础工具函数
# ---------------------------------------------------------------------------- #

def make_divisible(x: float, divisor: int = 8) -> int:
    """Ensure the channel number is divisible by the divisor."""
    return math.ceil(x / divisor) * divisor


# ---------------------------------------------------------------------------- #
#  核心组件: Bottleneck, C3k, C3k2, Attention, PSA
# ---------------------------------------------------------------------------- #

class Bottleneck(BaseModule):
    """Standard Bottleneck with optional shortcut."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, act_cfg=dict(type='SiLU'),
                 norm_cfg=dict(type='BN')):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvModule(c1, c_, k[0], 1, padding=(k[0] - 1) // 2, act_cfg=act_cfg, norm_cfg=norm_cfg)
        self.cv2 = ConvModule(c_, c2, k[1], 1, padding=(k[1] - 1) // 2, groups=g, act_cfg=act_cfg, norm_cfg=norm_cfg)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k(BaseModule):
    """C3k is a CSP Bottleneck with customizable kernel sizes."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3, act_cfg=dict(type='SiLU'),
                 norm_cfg=dict(type='BN')):
        super().__init__()
        self.m = nn.Sequential(*(
            Bottleneck(c1, c2, shortcut, g, k=(k, k), e=1.0, act_cfg=act_cfg, norm_cfg=norm_cfg)
            for _ in range(n)
        ))

    def forward(self, x):
        return self.m(x)


class C3k2(BaseModule):
    """
    YOLO11/26 核心模块: CSP 结构 + 可变卷积核
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3, act_cfg=dict(type='SiLU'),
                 norm_cfg=dict(type='BN')):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvModule(c1, 2 * self.c, 1, 1, act_cfg=act_cfg, norm_cfg=norm_cfg)
        self.cv2 = ConvModule((2 + n) * self.c, c2, 1, 1, act_cfg=act_cfg, norm_cfg=norm_cfg)
        self.m = nn.ModuleList(C3k(self.c, self.c, 2, shortcut, g, k=k, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Attention(BaseModule):
    """
    修正后的 Attention 机制，防止维度不匹配。
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # [Fix] 强制 key_dim 等于 head_dim，防止 Q @ K 转置时维度不匹配 (64 vs 32)
        # 如果你想保留 attn_ratio 压缩 Key，必须确保后续矩阵乘法逻辑兼容，
        # 但在 YOLO 通用实现中，保持维度一致是最稳健的。
        self.key_dim = self.head_dim

        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2

        self.qkv = ConvModule(dim, h, 1, act_cfg=None, norm_cfg=None)
        self.proj = ConvModule(dim, dim, 1, act_cfg=None, norm_cfg=None)
        self.pe = ConvModule(dim, dim, 3, 1, 1, groups=dim, act_cfg=None, norm_cfg=None)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)

        # Split qkv: Q=[B, Heads, C//Heads, N], K=[..., KeyDim, N], V=[..., KeyDim, N]
        q, k, v = qkv.view(B, self.num_heads, -1, N).split([C // self.num_heads, self.key_dim, self.key_dim], dim=2)

        # Attention Map: (B, Heads, KeyDim, N).T @ (B, Heads, KeyDim, N) -> (B, Heads, N, N)
        # 或者 (B, Heads, C_head, N) @ (B, Heads, KeyDim, N).T -> (B, Heads, C_head, KeyDim) -> 这里是通道注意力
        # 根据原始报错，这里做的是 Q.transpose @ K -> (C_head, N).T @ (KeyDim, N) -> (N, C) @ (C, N) -> (N, N) 空间注意力
        # 所以必须保证 Q 和 K 的通道维度 (dim=2) 一致。

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(BaseModule):
    """Position-Sensitive Attention Block."""

    def __init__(self, c, attn_ratio=0.5, num_heads=None, act_cfg=dict(type='SiLU'), norm_cfg=dict(type='BN')):
        super().__init__()
        self.scale = 4
        # 动态计算 num_heads，防止通道过小时为 0
        heads = c // 64 if num_heads is None else num_heads
        heads = max(heads, 1)

        self.attn = Attention(c, num_heads=heads, attn_ratio=attn_ratio)
        self.ffn = nn.Sequential(
            ConvModule(c, c * self.scale, 1, act_cfg=act_cfg, norm_cfg=norm_cfg),
            ConvModule(c * self.scale, c, 1, act_cfg=None, norm_cfg=norm_cfg)
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


class C2PSA(BaseModule):
    """C2 with Position-Sensitive Attention."""

    def __init__(self, c1, c2, n=1, e=0.5, act_cfg=dict(type='SiLU'), norm_cfg=dict(type='BN')):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = ConvModule(c1, 2 * self.c, 1, 1, act_cfg=act_cfg, norm_cfg=norm_cfg)
        self.cv2 = ConvModule(2 * self.c, c1, 1, 1, act_cfg=act_cfg, norm_cfg=norm_cfg)
        self.m = nn.Sequential(*(PSABlock(self.c, act_cfg=act_cfg, norm_cfg=norm_cfg) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), dim=1))


# ---------------------------------------------------------------------------- #
#  动态 Backbone 实现
# ---------------------------------------------------------------------------- #

@MODELS.register_module()
class YOLO26Backbone(BaseModule):
    """
    YOLO26 Backbone (Dynamic Scaling).

    自动根据 arch_settings, deepen_factor, widen_factor 构建网络。

    默认 arch_settings 对应 YOLO 系列的标准配置:
    [
       [64, 1, 2],    # P1/2: 64ch, 1 block, stride 2
       [128, 1, 2],   # P2/4: 128ch, 1 block, stride 2
       [256, 2, 2],   # P3/8: 256ch, 2 blocks, stride 2 (Output P3)
       [512, 2, 2],   # P4/16: 512ch, 2 blocks, stride 2 (Output P4)
       [1024, 2, 2]   # P5/32: 1024ch, 2 blocks, stride 2 (Output P5)
    ]
    """

    def __init__(self,
                 arch_settings: list = [
                     [64, 1, 2],  # Stage 0 (Stem/P1)
                     [128, 1, 2],  # Stage 1 (P2)
                     [256, 2, 2],  # Stage 2 (P3) - C3k2
                     [512, 2, 2],  # Stage 3 (P4) - C3k2
                     [1024, 2, 2]  # Stage 4 (P5) - C3k2 + C2PSA
                 ],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 out_indices: tuple = (2, 3, 4),
                 act_cfg: dict = dict(type='SiLU', inplace=True),
                 norm_cfg: dict = dict(type='BN', requires_grad=True),
                 init_cfg: dict = None):
        super().__init__(init_cfg)
        self.out_indices = out_indices
        self.layers = nn.ModuleList()

        # 输入通道 (RGB)
        in_channels = 3

        for i, (base_c, base_n, stride) in enumerate(arch_settings):
            # 1. 动态计算当前 Stage 的输出通道数和块数
            out_channels = make_divisible(base_c * widen_factor, 8)
            num_blocks = max(round(base_n * deepen_factor), 1) if base_n > 1 else base_n

            stage_layers = []

            # 2. 下采样层 (Conv s=2)
            # 如果是 Stage 0，通常作为 Stem
            conv_layer = ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg
            )
            stage_layers.append(conv_layer)

            # 3. 特征提取块 (C3k2 或 C2PSA)
            # Stage 0 和 Stage 1 通常只做卷积或简单处理，YOLO11中 P3, P4, P5 用 C3k2
            if i >= 2:
                # P3, P4, P5 使用 C3k2
                # 最后一个 Stage (P5) 额外添加 Attention (C2PSA)
                is_last_stage = (i == len(arch_settings) - 1)

                block = C3k2(
                    out_channels,
                    out_channels,
                    n=num_blocks,
                    shortcut=(i > 0),  # 除了极早期层，通常开启 shortcut
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg
                )
                stage_layers.append(block)

                if is_last_stage:
                    stage_layers.append(C2PSA(out_channels, out_channels, n=1, act_cfg=act_cfg, norm_cfg=norm_cfg))

            # 将构建好的 Stage 添加到 ModuleList
            self.layers.append(nn.Sequential(*stage_layers))

            # 更新输入通道为下一层准备
            in_channels = out_channels

    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)