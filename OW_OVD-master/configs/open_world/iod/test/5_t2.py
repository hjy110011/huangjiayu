_base_ = ('../../../../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_syncbn_fast_8xb16-500e_coco.py')

custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

# ==============================================================================
# 0. 超参数定义和路径
# ==============================================================================

# --- 1. VisDrone11 类别定义 ---
visdrone_class_names = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]
class_names = visdrone_class_names

# --- 2. Step 2 IOD/开放世界设置 (5+5 划分) ---
VISDRONE_DATA_ROOT = '/home/gdut-627/huangjiayu/datasets/VisDrone11' # VisDrone11 根目录
split = 5
prev_intro_cls = 5 # 之前已学习 5 个类别 (旧类别)
cur_intro_cls = 5 # 当前引入 5 个新类别 (新类别)

# 假设 VisDrone11 转换后的 IOD 标注文件
# **注意：train_json 必须包含全部 10 个类别的标注**
train_json = 'OWOD/VisDrone_train_10_t2.json' # 全部 10 类的 JSON 标注
embedding_path = f'/home/gdut-627/huangjiayu/datasets/VisDrone11/OWOD/gt_10_t2.npy' # 全部 10 类的文本嵌入
att_embeddings = f'/home/gdut-627/huangjiayu/datasets/VisDrone11/OWOD/att_embeddings.pth'
pipline = []

# --- 3. 模型和训练参数 ---
num_classes = 10 # 总类别数
num_training_classes = 10 # 当前训练类别数 (5旧 + 5新)
max_epochs = 50  # 示例值，请根据实验需要调整
close_mosaic_epochs = 10
save_epoch_intervals = 1
text_channels = 768
# **【补全】** 颈部嵌入通道和头数，根据 YOLOv8-L 的惯例或原项目设置
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4 / 4
weight_decay = 0.05
train_batch_size_per_gpu = 4
# **【核心】** 加载 Step 1 训练得到的权重
load_from = '/home/gdut-627/huangjiayu/work_dirs/15_t1/best_Current class AP50_epoch_49.pth'
persistent_workers = False


# ==============================================================================
# 4. 模型配置
# ==============================================================================
model = dict(type='OurDetector',
             mm_neck=True,
             num_train_classes=num_training_classes, # 10
             num_test_classes=num_classes, # 10
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
                            prev_intro_cls=prev_intro_cls, # 5
                            cur_intro_cls=cur_intro_cls, # 5
                            head_module=dict(
                                type='OurHeadModule',
                                freeze_all=True,
                                use_bn_head=True,
                                embed_dims=text_channels,
                                num_classes=num_training_classes,),), # 10
             train_cfg=dict(assigner=dict(num_classes=num_training_classes))) # 10


# ==============================================================================
# 5. 数据集配置 (训练)
# ==============================================================================
coco_train_dataset = dict(
        _delete_=True,
        type='MultiModalDataset',
        dataset=dict(
            type='YOLOv5CocoDataset',
            metainfo=dict(classes=visdrone_class_names),
            data_root=VISDRONE_DATA_ROOT,
            ann_file=train_json, # 全部 10 类 JSON 标注
            data_prefix=dict(img='VisDrone2019-DET-train/images/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=4)),
        class_text_path=f'/home/gdut-627/huangjiayu/datasets/VisDrone11/visdrone_class_texts_10cls.json',
        pipeline=_base_.train_pipeline)

train_dataloader = dict(persistent_workers=persistent_workers,
                        batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=coco_train_dataset)

# ==============================================================================
# 6. 训练 Hooks 和优化器
# ==============================================================================
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


# ==============================================================================
# 7. 评估配置
# ==============================================================================
test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param'))
]
test_dataloader = dict(dataset=dict(type='YOLOv5CocoDataset',
                        data_root=VISDRONE_DATA_ROOT,
                        ann_file='annotations/val_coco.json',
                        data_prefix=dict(img='VisDrone2019-DET-val/images/'),
                        filter_cfg=dict(filter_empty_gt=False, min_size=4),
                        pipeline=test_pipeline)
                       )

test_evaluator = dict(_delete_=True,
                     type='OWODEvaluator',
                     cfg=dict(
                        dataset_root='/home/gdut-627/huangjiayu/datasets/VisDrone11/VisDrone2019-DET-val',
                        file_name='annotations/our.txt',
                        prev_intro_cls=prev_intro_cls, # 5
                        cur_intro_cls=cur_intro_cls, # 5
                        unknown_id=10,
                        class_names=visdrone_class_names
                     )
                    )
val_evaluator = test_evaluator
val_dataloader = test_dataloader
find_unused_parameters = True