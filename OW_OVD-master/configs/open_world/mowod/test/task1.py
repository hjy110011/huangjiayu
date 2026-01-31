_base_ = ('../../../../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

# ===================== Hyper-parameters =====================
# VisDrone 全集类别 (共10类)
class_names = [
    # Task 1 (4 classes)
    "pedestrian", "people", "bicycle", "car",
    # Task 2 (3 classes)
    "van", "truck", "tricycle",
    # Task 3 (3 classes)
    "awning-tricycle", "bus", "motor"
]
top_k = 10
# 路径设置 (请根据实际情况修改)
data_root = '/home/gdut-627/huangjiayu/datasets/VisDrone'
# Task 1 需要包含前4类的 Embedding
embedding_path = '/home/gdut-627/huangjiayu/t1_gt_embeddings.npy'
# Attribute Embeddings
att_embeddings = '/home/gdut-627/huangjiayu/task_att_embeddings_t1.pth'
# 训练集标注 (只包含前4类)
train_json = '/home/gdut-627/huangjiayu/datasets/VisDrone/annotations/train/task1.json'
test_json = '/home/gdut-627/huangjiayu/datasets/VisDrone/annotations/val/task1.json'

# Open World / 增量设置
prev_intro_cls = 0
cur_intro_cls = 4  # T1 训练4类
num_training_classes = 4
num_classes = 4    # 当前已知的总类别数

# 核心参数
thr = 0.55
alpha = 0.2
top_k = 10
distributions = 'work_dirs/visdrone_t1/distribution_sim1.pth' # 保存当前分布
prev_distribution = None # T1 没有之前的分布

# 训练设置
max_epochs = 50  # T1 通常训练久一点
close_mosaic_epochs = 10
text_channels = 768
base_lr = 2e-4 / 4
weight_decay = 0.05
train_batch_size_per_gpu = 4
# 加载官方预训练权重作为起点
load_from = 'pretrained_models/yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth'

# ===================== Model =====================
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
pipline = [dict(type='att_select', log_start_epoch=1)] # 注意这里拼写是 pipline 对应你的 hook

model = dict(type='OurDetector',
             mm_neck=True,
             num_train_classes=num_training_classes,
             num_test_classes=num_classes,
             embedding_path=embedding_path,
             prompt_dim=text_channels,
             num_prompts=80, # 这里的 num_prompts 实际上如果用 embedding_path 会被覆盖，但最好保持一致
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
                            use_sigmoid=True,
                            distributions=distributions,
                            prev_intro_cls=prev_intro_cls,
                            cur_intro_cls=cur_intro_cls,
                            prev_distribution=prev_distribution,
                            top_k=top_k,
                            head_module=dict(
                                type='OurHeadModule',
                                freeze_all=True,
                                use_bn_head=True,
                                embed_dims=text_channels,
                                num_classes=num_training_classes,),),
             train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

# ===================== Data =====================
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

# ===================== Hooks & Evaluation =====================
custom_hooks = [
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=_base_.train_pipeline_stage2),
    dict(type='OurWorkPiplineHook')
]

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='Current class AP50', rule='greater', type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='YOLOv5ParamSchedulerHook', scheduler_type='linear', lr_factor=0.01, max_epochs=max_epochs),
    visualization=dict(type='mmdet.DetVisualizationHook'))

train_cfg = dict(max_epochs=max_epochs, val_interval=5,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs), _base_.val_interval_stage2)])

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

test_pipeline = [*_base_.test_pipeline[:-1],
                 dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_param'))]

test_dataloader = dict(dataset=dict(type='YOLOv5CocoDataset',
                        data_root='/home/gdut-627/huangjiayu/datasets/VisDrone',
                        ann_file='annotations/val/task1.json',
                        data_prefix=dict(img='VisDrone2019-DET-val/images/'),
                        filter_cfg=dict(filter_empty_gt=False, min_size=32),
                        pipeline=test_pipeline))

test_evaluator = dict(_delete_=True, type='OWODEvaluator',
                     cfg=dict(dataset_root='/home/gdut-627/huangjiayu/datasets/VisDrone/VisDrone_VOC_tasks/task1', file_name='our.txt',
                        prev_intro_cls=prev_intro_cls, cur_intro_cls=cur_intro_cls,
                        unknown_id=10, class_names=class_names))
val_evaluator = test_evaluator
val_dataloader = test_dataloader
find_unused_parameters = True