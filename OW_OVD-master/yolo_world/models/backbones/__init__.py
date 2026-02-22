# Copyright (c) Tencent Inc. All rights reserved.
# YOLO Multi-Modal Backbone (Vision Language)
# Vision: YOLOv8 CSPDarknet
# Language: CLIP Text Encoder (12-layer transformer)
from .mm_backbone import (
    MultiModalYOLOBackbone,
    HuggingVisionBackbone,
    HuggingCLIPLanguageBackbone,
    PseudoLanguageBackbone,
)
from .uav_backbone import UAVBackbone
from  .uav_a_backbone import UAVABackbone
from .uav_b_backbone import UAVBBackbone
from .yolo26_backbones import YOLO26Backbone
from .uav_lck_backbone import UAVCBackbone
from .uav_lck_wavelet_backbone import UAVDBackbone
__all__ = [
    'MultiModalYOLOBackbone',
    'HuggingVisionBackbone',
    'HuggingCLIPLanguageBackbone',
    'PseudoLanguageBackbone',
    'UAVBackbone',
    'UAVABackbone',
    'UAVBBackbone',
    'YOLO26Backbone',
    'UAVCBackbone',
    'UAVDBackbone'
]
