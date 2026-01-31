_base_ = ('../../../../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

class_names = [
    # Task 1 (4 classes)
    "pedestrian", "people", "bicycle", "car",
    # Task 2 (3 classes)
    "van", "truck", "tricycle",
    # Task 3 (3 classes)
    "awning-tricycle", "bus", "motor"
]
# ===================== Hyper-parameters =====================
# 路径修改
# T2 需要包含前7类 (T1+T2) 的 Embedding
embedding_path = '/home/gdut-627/huangjiayu/t3_gt_embeddings.npy'
# Attribute Embeddings
att_embeddings = '/home/gdut-627/huangjiayu/task_att_embeddings_t3.pth'
train_json = '/home/gdut-627/huangjiayu/datasets/VisDrone/annotations/train/task3.json'
test_json = '/home/gdut-627/huangjiayu/datasets/VisDrone/annotations/val/task3.json'
save_epoch_intervals = 1
thr = 0.95
alpha = 0.2
use_sigmoid=True
# 增量设置
prev_intro_cls = 7  # 之前已有4类
cur_intro_cls = 3  # 新增3类
num_training_classes = 10  # 总共训练7类
num_classes = 10  # 验证时也是7类
pipline = [dict(type='att_select', log_start_epoch=1)]
# 分布传递
distributions = 'work_dirs/visdrone_t3/distribution_sim3.pth'
# 重要：指向 T1 训练生成的分布文件
previous_distribution = 'work_dirs/visdrone_t2/distribution_sim2.pth'
top_k = 10
# yolo world setting
num_classes = prev_intro_cls+cur_intro_cls
num_training_classes = num_classes
max_epochs = 50  # Maximum training epochs
close_mosaic_epochs = 10
save_epoch_intervals = 1
text_channels = 768
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4 / 4
weight_decay = 0.05
train_batch_size_per_gpu = 4
# load_from = '/home/gdut-627/huangjiayu/work_dirs/task1/best_Current class AP50_epoch_65.pth'
persistent_workers = False
# ===================== Model Updates =====================
model = dict(type='OurDetector',
             mm_neck=True,
             num_train_classes=num_training_classes,
             num_test_classes=num_classes,
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
                            thr=thr,
                            alpha=alpha,
                            use_sigmoid=use_sigmoid,
                            distributions=distributions,
                            prev_intro_cls=prev_intro_cls,
                            cur_intro_cls=cur_intro_cls,
                            prev_distribution=previous_distribution,
                            top_k=top_k,
                            head_module=dict(
                                type='OurHeadModule',
                                freeze_all=True,
                                use_bn_head=True,
                                embed_dims=text_channels,
                                num_classes=num_training_classes,),),
             train_cfg=dict(assigner=dict(num_classes=num_training_classes)))
# ===================== Data Updates =====================
coco_train_dataset = dict(
        _delete_=True,
        type='MultiModalDataset',
        dataset=dict(
            type='YOLOv5CocoDataset',
            metainfo=dict(classes=class_names),
            data_root='/home/gdut-627/huangjiayu/datasets/VisDrone',
            ann_file=train_json,
            data_prefix=dict(img='VisDrone2019-DET-train/images/'), # 需确认图片路径
            filter_cfg=dict(filter_empty_gt=False, min_size=32)),
        class_text_path=f'/home/gdut-627/huangjiayu/datasets/VisDrone/visdrone_class_texts.json', # 需确认文本路径
        pipeline=_base_.train_pipeline)

train_dataloader = dict(persistent_workers=False,
                        batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=coco_train_dataset)


# ===================== Hooks & Optimizer =====================
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

test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param'))
]
# 测试集更新
test_dataloader = dict(dataset=dict(type='YOLOv5CocoDataset',
                        data_root='/home/gdut-627/huangjiayu/datasets/VisDrone',
                        ann_file='annotations/val/task3.json',
                        data_prefix=dict(img='VisDrone2019-DET-val/images/'),
                        filter_cfg=dict(filter_empty_gt=False, min_size=32),
                        pipeline=test_pipeline))
test_evaluator = dict(_delete_=True, type='OWODEvaluator',
                     cfg=dict(dataset_root='/home/gdut-627/huangjiayu/datasets/VisDrone/VisDrone_VOC_tasks/task3', file_name='our.txt',
                        prev_intro_cls=prev_intro_cls, cur_intro_cls=cur_intro_cls,
                        unknown_id=10, class_names=class_names))
val_evaluator = test_evaluator
val_dataloader = test_dataloader
