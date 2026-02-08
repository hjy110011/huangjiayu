import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmdet.utils import OptMultiConfig, ConfigType


# -----------------------------------------------------------------------
# [核心组件 1] RepConv: 训练多分支，推理单分支 (极致速度)
# -----------------------------------------------------------------------
class RepConv(nn.Module):
    """
    YOLO26 的基础单元：重参数化卷积
    训练时：3x3 + 1x1 + Identity (学习能力强)
    推理时：融合为单个 3x3 卷积 (速度极快)
    """

    def __init__(self, in_channels, out_channels, stride=1, act=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.deploy = False
        self.act = nn.SiLU() if act else nn.Identity()

        # 训练态：多分支
        self.rbr_dense = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)  # 简化写法，实际应每层后接BN

    def forward(self, x):
        if self.deploy:
            return self.act(self.rbr_dense(x))
        # 训练时多路并行
        return self.act(self.bn(self.rbr_dense(x) + self.rbr_1x1(x)))

    def switch_to_deploy(self):
        """将参数融合，切换到推理模式"""
        if self.deploy: return
        # (此处省略复杂的权重融合数学逻辑，仅做结构演示)
        # 实际部署时需调用 deploy 脚本进行 convert
        self.deploy = True


# -----------------------------------------------------------------------
# [核心组件 2] Flash-Occult Block: 轻量级注意力 (仿 YOLO26)
# -----------------------------------------------------------------------
class FlashOccultBlock(nn.Module):
    """
    模拟 YOLO26 的 Flash-Occult 模块
    利用轻量级 Depthwise 卷积和 Channel Split 捕捉全局上下文
    """

    def __init__(self, c, k=7):
        super().__init__()
        self.conv_dw = nn.Conv2d(c, c, k, 1, k // 2, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.Hardswish()
        self.conv_pw = nn.Conv2d(c, c, 1, 1, 0, bias=False)

    def forward(self, x):
        return x + self.conv_pw(self.act(self.bn(self.conv_dw(x))))


# -----------------------------------------------------------------------
# [骨干网络] YOLO26 Backbone
# -----------------------------------------------------------------------
@MODELS.register_module()
class YOLO26Backbone(BaseModule):
    def __init__(self,
                 feat_channels=[64, 128, 256, 512, 1024],  # N/S/M/L 不同配置
                 depth_mult=1.0,
                 width_mult=1.0,
                 init_cfg=None):
        super().__init__(init_cfg)

        # 应用宽度缩放
        c = [int(x * width_mult) for x in feat_channels]
        # 应用深度缩放 (每个阶段的 Block 数)
        n = [int(x * depth_mult) for x in [2, 4, 4, 2]]

        # Stem (P1): 快速下采样
        self.stem = RepConv(3, c[0], stride=2)

        # Stage 2 (P2): 纯 RepConv 堆叠
        self.stage2 = nn.Sequential(
            RepConv(c[0], c[1], stride=2),
            *[RepConv(c[1], c[1]) for _ in range(n[0])]
        )

        # Stage 3 (P3): 引入 Flash-Occult 注意力
        self.stage3 = nn.Sequential(
            RepConv(c[1], c[2], stride=2),
            *[RepConv(c[2], c[2]) for _ in range(n[1])],
            FlashOccultBlock(c[2])
        )

        # Stage 4 (P4)
        self.stage4 = nn.Sequential(
            RepConv(c[2], c[3], stride=2),
            *[RepConv(c[3], c[3]) for _ in range(n[2])],
            FlashOccultBlock(c[3])
        )

        # Stage 5 (P5)
        self.stage5 = nn.Sequential(
            RepConv(c[3], c[4], stride=2),
            *[RepConv(c[4], c[4]) for _ in range(n[3])],
            FlashOccultBlock(c[4])
        )

        # SPPF (YOLO 标配)
        self.sppf = nn.Sequential(
            nn.MaxPool2d(5, 1, 2),
            nn.MaxPool2d(5, 1, 2),
            nn.MaxPool2d(5, 1, 2)
        )
        self.sppf_conv = RepConv(c[4] * 4, c[4])

    def forward(self, x):
        x = self.stem(x)  # P1
        x = self.stage2(x)  # P2
        c3 = self.stage3(x)  # P3 (80x80)
        c4 = self.stage4(c3)  # P4 (40x40)
        c5 = self.stage5(c4)  # P5 (20x20)

        # SPPF
        y1 = self.sppf[0](c5)
        y2 = self.sppf[1](y1)
        y3 = self.sppf[2](y2)
        c5 = torch.cat([c5, y1, y2, y3], dim=1)
        c5 = self.sppf_conv(c5)

        return (c3, c4, c5)