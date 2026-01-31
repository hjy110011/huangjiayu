_base_ = ('../../../../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)
print(type(_base_))

# --- VisDrone11 Hyper-parameters 调整 ---
# VisDrone11 2019 DET 的 10 个有效类别 (根据官方文档)
class_names = [
    "pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
    "awning-tricycle", "bus", "motor"
]

# VisDrone11 数据集根目录。
_base_data_root = '/home/gdut-627/huangjiayu/datasets/VisDrone11'

# Open World Setting (T3 阶段)
split = 5
prev_intro_cls = 0
cur_intro_cls = 5 # 当前引入的类别数量 (T3 阶段通常训练 0-4 索引的类别)

num_total_classes = len(class_names) # 10
# --- 适配 OWOD T3 阶段：训练 5 个类别 ---
num_training_classes = cur_intro_cls # 5
num_classes = num_total_classes  # 测试时检测所有 10 个类别

# VisDrone11 标注文件路径
train_json = 'annotations/instances_train.json'
val_json = 'annotations/instances_val.json'
test_json = 'annotations/instances_test-dev.json'

# Image 路径前缀
train_img_prefix = ''
val_test_img_prefix = ''

# YOLO World 特有设置
# --- 关键修改 1：设置为 None 以跳过文件加载 ---
embedding_path = f'/home/gdut-627/huangjiayu/datasets/VisDrone11/OWOD/gt_full.npy'
att_embeddings = None
pipline = []

max_epochs = 200  # Maximum training epochs (请根据实际情况调整)
close_mosaic_epochs = 10
save_epoch_intervals = 1
text_channels = 768
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4 / 4
weight_decay = 0.05
train_batch_size_per_gpu = 16
load_from = '/home/gdut-627/huangjiayu/pretrained_models/yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth'
persistent_workers = False

# --- Model Settings (适配 num_training_classes = 5) ---
model = dict(type='OurDetector',
             mm_neck=True,
             # 训练类别数量改为 5
             num_train_classes=num_training_classes,
             # 测试类别数量为 10
             num_test_classes=num_classes,
             # --- 关键修改 2：设置为 None ---
             embedding_path=embedding_path,
             prompt_dim=text_channels,
             num_prompts=80,
             pipline=pipline,
             data_preprocessor=dict(type='YOLOv5DetDataPreprocessor'),
             backbone=dict(_delete_=True,
                           type='MultiModalYOLOBackbone',
                           text_model=None,
                           image_model={{_base_.model.backbone}},
                           frozen_stages=4,
                           with_text_model=False),
             neck=dict(type='YOLOWorldPAFPN',
                       freeze_all=True,
                       guide_channels=text_channels,
                       embed_channels=neck_embed_channels,
                       num_heads=neck_num_heads,
                       block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
             bbox_head=dict(type='OurHead',
                            att_embeddings=att_embeddings,
                            prev_intro_cls=prev_intro_cls,
                            cur_intro_cls=cur_intro_cls,
                            head_module=dict(
                                type='OurHeadModule',
                                freeze_all=True,
                                use_bn_head=True,
                                embed_dims=text_channels,
                                # 训练类别数量改为 5
                                num_classes=num_training_classes,),),
             train_cfg=dict(assigner=dict(num_classes=num_training_classes)))


# --- Dataset Settings ---
# 建议将 class_text_path 改为 VisDrone11 专用的路径
_visdrone_class_text_path = f'/home/gdut-627/huangjiayu/datasets/VisDrone11/visdrone_class_texts_{num_total_classes}cls.json'

coco_train_dataset = dict(
        _delete_=True,
        type='MultiModalDataset',
        dataset=dict(
            type='YOLOv5CocoDataset',
            metainfo=dict(classes=class_names),
            data_root=_base_data_root,
            ann_file=train_json,
            data_prefix=dict(img=train_img_prefix),
            filter_cfg=dict(filter_empty_gt=False, min_size=32)),
        # --- 关键修改 3：更新为 VisDrone11 文本路径 ---
        class_text_path=_visdrone_class_text_path,
        pipeline=_base_.train_pipeline)

train_dataloader = dict(persistent_workers=persistent_workers,
                        batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=coco_train_dataset)

# --- Test/Val Dataloader 保持 VisDrone11 路径不变 ---
test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param'))
]

test_dataloader = dict(dataset=dict(type='YOLOv5CocoDataset',
                        metainfo=dict(classes=class_names),
                        data_root=_base_data_root,
                        ann_file=test_json,
                        data_prefix=dict(img=val_test_img_prefix),
                        filter_cfg=dict(filter_empty_gt=False, min_size=32),
                        pipeline=test_pipeline)
                       )

val_dataloader = dict(dataset=dict(type='YOLOv5CocoDataset',
                        metainfo=dict(classes=class_names),
                        data_root=_base_data_root,
                        ann_file=val_json,
                        data_prefix=dict(img=val_test_img_prefix),
                        filter_cfg=dict(filter_empty_gt=False, min_size=32),
                        pipeline=test_pipeline)
                       )

# --- Evaluator Settings (适配 num_training_classes = 5) ---
test_evaluator = dict(_delete_=True,
                     type='OWODEvaluator',
                     cfg=dict(
                        dataset_root='/home/gdut-627/huangjiayu/datasets/VisDrone11/VisDrone2019-DET-val',
                        file_name='annotations/test.txt',
                        prev_intro_cls=prev_intro_cls,
                        cur_intro_cls=cur_intro_cls,
                        # unknown_id 在所有已知类别 (10) 之后
                        unknown_id=num_total_classes,
                        class_names=class_names
                     )
                    )
val_evaluator = test_evaluator

# --- 其他设置保持不变 ---
custom_hooks = [
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=_base_.train_pipeline_stage2),
    dict(type='OurWorkPiplineHook')
]

default_hooks = dict(
    checkpoint=dict(
        interval=save_epoch_intervals, max_keep_ckpts=2, save_best='Current class AP50', rule='greater',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(
        lr_factor=0.01,
        max_epochs=500,
        scheduler_type='linear',
        type='YOLOv5ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))

train_cfg = dict(max_epochs=max_epochs,
                 val_interval=1,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])
optim_wrapper = dict(optimizer=dict(
    _delete_=True,
    type='AdamW',
    lr=base_lr,
    weight_decay=weight_decay,
    batch_size_per_gpu=train_batch_size_per_gpu),
                     paramwise_cfg=dict(bias_decay_mult=0.0,
                                        norm_decay_mult=0.0,
                                        custom_keys={
                                            'backbone.text_model':
                                            dict(lr_mult=0.01),
                                            'logit_scale':
                                            dict(weight_decay=0.0),
                                            'embeddings':
                                            dict(weight_decay=0.0)
                                        }),
                     constructor='YOLOWv5OptimizerConstructor')

find_unused_parameters = True