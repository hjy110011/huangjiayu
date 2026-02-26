# Copyright (c) Tencent Inc. All rights reserved.
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS
from mmengine.logging import MessageHub  # 导入 MessageHub 用于获取全局训练进度


@MODELS.register_module()
class UAVBDetector(YOLODetector):
    """Implementation of YOLO World Series for UAV/Small Object Detection"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 prompt_dim=512,
                 num_prompts=80,
                 embedding_path='',
                 reparameterized=False,
                 freeze_prompt=False,
                 use_mlp_adapter=False,
                 pipline=None,
                 fusion_att=False,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts
        self.reparameterized = reparameterized
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        self.pipline = pipline
        self.fusion_att = fusion_att
        super().__init__(*args, **kwargs)

        if not self.reparameterized:
            if len(embedding_path) > 0:
                import numpy as np
                self.embeddings = torch.nn.Parameter(
                    torch.from_numpy(np.load(embedding_path)).float())
            else:
                # random init
                embeddings = nn.functional.normalize(torch.randn(
                    (num_prompts, prompt_dim)),
                    dim=-1)
                self.embeddings = nn.Parameter(embeddings)

            if self.freeze_prompt:
                self.embeddings.requires_grad = False
            else:
                self.embeddings.requires_grad = True

            if use_mlp_adapter:
                self.adapter = nn.Sequential(
                    nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(True),
                    nn.Linear(prompt_dim * 2, prompt_dim))
            else:
                self.adapter = None

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_training_classes

        # 【修复 1：动态调整 EMA Buffer 尺寸】
        # 防止由于 num_classes 动态切换导致 Head 中的 class_sample_ema 张量形状不匹配而报错
        if hasattr(self.bbox_head, 'class_sample_ema') and self.bbox_head.class_sample_ema.shape[
            0] != self.num_training_classes:
            self.bbox_head.class_sample_ema = torch.zeros(self.num_training_classes, device=batch_inputs.device)

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        # 获取当前 epoch 和 max_epochs 以激活 Head 中的 ProgLoss
        message_hub = MessageHub.get_current_instance()
        epoch_info = {}
        if message_hub is not None:
            # 【修复 2：增加 has_info 校验】
            # 防止在特定 Hook 或极早期验证流程中因为日志板尚未注册 'epoch' 而引发 KeyError 崩溃
            epoch_info['current_epoch'] = message_hub.get_info('epoch') if message_hub.has_info('epoch') else 0
            epoch_info['max_epochs'] = message_hub.get_info('max_epochs') if message_hub.has_info('max_epochs') else 1

        if self.reparameterized:
            losses = self.bbox_head.loss(img_feats, batch_data_samples, epoch_info=epoch_info)
        else:
            losses = self.bbox_head.loss(img_feats, txt_feats,
                                         batch_data_samples, fusion_att=self.fusion_att,
                                         epoch_info=epoch_info)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        self.bbox_head.num_classes = self.num_test_classes
        if self.reparameterized:
            results_list = self.bbox_head.predict(img_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)
        else:
            results_list = self.bbox_head.predict(img_feats,
                                                  txt_feats,
                                                  batch_data_samples,
                                                  rescale=rescale,
                                                  fusion_att=self.fusion_att)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            results = self.bbox_head.forward(img_feats)
        else:
            results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # only image features
        img_feats, _ = self.backbone(batch_inputs, None)

        if not self.reparameterized:
            # use embeddings
            txt_feats = self.embeddings[None]
            if self.adapter is not None:
                txt_feats = self.adapter(txt_feats) + txt_feats
                txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)

            txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)

            if self.fusion_att:
                # 【修复 3：增加 NoneType 安全校验】
                # 防止由于未提供 attribute 嵌入文件导致 att_embeddings 为 None 时的 AttributeError 崩溃
                if hasattr(self.bbox_head, 'att_embeddings') and self.bbox_head.att_embeddings is not None:
                    att_feats = self.bbox_head.att_embeddings
                    att_feats = att_feats[None].repeat(img_feats[0].shape[0], 1, 1)
                    txt_feats = torch.cat([txt_feats, att_feats], dim=1)
        else:
            txt_feats = None

        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats