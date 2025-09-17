import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock
__all__ = (
    "MFAM",
    "IEMA",
    "DASI",
    "CBS"
)

# ---------- MFAM 模块 ----------
class MFAM(nn.Module):
    def __init__(self, c1, c2, *args, **kwargs):
        super().__init__()
        # 主分支：3x3 DWConv
        self.dw3 = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False)
        self.pw3 = nn.Conv2d(c1, c2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c2)

        # 分支1：1x7 -> 7x1
        self.branch7 = nn.Sequential(
            nn.Conv2d(c1, c1, (1,7), padding=(0,3), groups=c1, bias=False),
            nn.Conv2d(c1, c1, (7,1), padding=(3,0), groups=c1, bias=False),
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2)
        )

        # 分支2：1x9 -> 9x1
        self.branch9 = nn.Sequential(
            nn.Conv2d(c1, c1, (1,9), padding=(0,4), groups=c1, bias=False),
            nn.Conv2d(c1, c1, (9,1), padding=(4,0), groups=c1, bias=False),
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2)
        )

        # shortcut 投影
        self.proj = nn.Conv2d(c1, c2, 1, bias=False)

        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y1 = self.act(self.bn3(self.pw3(self.dw3(x))))
        y2 = self.act(self.branch7(x))
        y3 = self.act(self.branch9(x))
        shortcut = self.proj(x)
        return self.act(y1 + y2 + y3 + shortcut)


# ---------- IEMA 模块 ----------
class IEMA(nn.Module):
    def __init__(self, c1, *args, **kwargs):
        super().__init__()
        g = c1 // 4  # 分组

        self.b1 = nn.Conv2d(g, g, 3, 1, 1, groups=g, bias=False)
        self.b2 = nn.Conv2d(g, g, (1,5), padding=(0,2), groups=g, bias=False)
        self.b3 = nn.Conv2d(g, g, (5,1), padding=(2,0), groups=g, bias=False)
        self.id = nn.Identity()

        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU(inplace=True)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, c1 // 4, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1 // 4, c1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        g = x.shape[1] // 4
        chunks = torch.chunk(x, 4, dim=1)
        y1 = self.b1(chunks[0])
        y2 = self.b2(chunks[1])
        y3 = self.b3(chunks[2])
        y4 = self.id(chunks[3])
        out = torch.cat([y1,y2,y3,y4], dim=1)

        att = self.fc(self.global_pool(out))
        out = out * att
        return self.act(self.bn(out + x))


# ---------- DASI 模块 ----------
class DASI(nn.Module):
    def __init__(self, c1_low, c1_mid, c1_high, c2, *args, **kwargs):
        """
        DASI (三路特征融合)
        :param c1_low: 低层输入通道 (e.g., C/2)
        :param c1_mid: 中层输入通道 (e.g., C)
        :param c1_high: 高层输入通道 (e.g., 2C)
        :param c2: 输出通道 (C)
        """
        super().__init__()
        self.align_low = nn.Conv2d(c1_low, c2, 1, bias=False)
        self.align_mid = nn.Conv2d(c1_mid, c2, 1, bias=False)
        self.align_high = nn.Conv2d(c1_high, c2, 1, bias=False)

        self.branch1 = nn.Sequential(
            nn.Conv2d(c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        """
        x: list [low_feat, mid_feat, high_feat]
        - low_feat:  (B, c1_low, 2H, 2W)
        - mid_feat:  (B, c1_mid, H, W)
        - high_feat: (B, c1_high, H/2, W/2)
        return: (B, c2, H, W)
        """
        assert isinstance(x, (list, tuple)) and len(x) == 3, "DASI 需要三路输入"
        low_feat, mid_feat, high_feat = x

        B, _, H, W = mid_feat.shape

        # 尺度对齐
        low = F.adaptive_avg_pool2d(low_feat, (H, W))      # 下采样
        mid = mid_feat                                     # 尺度本来就对齐
        high = F.interpolate(high_feat, size=(H, W), mode="nearest")  # 上采样

        # 通道对齐
        low = self.align_low(low)
        mid = self.align_mid(mid)
        high = self.align_high(high)

        # 拼接 (3C, H, W)
        x = torch.cat([low, mid, high], dim=1)

        # 分成三份走分支
        x1, x2, x3 = torch.chunk(x, 3, dim=1)
        y1 = self.branch1(x1)
        y2 = self.branch2(x2)
        y3 = self.branch3(x3)

        # 自适应融合 (加权)
        gate12 = torch.sigmoid(y1 + y2)
        fused12 = y1 * gate12 + y2 * (1 - gate12)
        gate = torch.sigmoid(fused12 + y3)
        fused = fused12 * gate + y3 * (1 - gate)

        return self.out_conv(fused)


# ========== 基础 CBS ==========
class CBS(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k,p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def autopad(k, p=None):  # kernel, padding
    if p is None:
        p = k // 2
    return p