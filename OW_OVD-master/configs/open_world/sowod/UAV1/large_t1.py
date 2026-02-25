_base_ = ('../../../../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

# hyper-parameters
class_names = [
    # Task 1 (14)
    'airplane', 'ship', 'car', 'truck', 'bus', 'van', 'bridge',
    'harbor', 'storage-tank', 'chimney', 'pedestrian', 'people',
    'bicycle', 'motor',
    # Task 2 (7)
    'baseball-field', 'basketball-court', 'tennis-court',
    'ground-track-field', 'soccer-ball-field', 'swimming-pool', 'stadium',
    # Task 3 (7)
    'airport', 'train-station', 'dam', 'overpass',
    'toll-station', 'service-area', 'roundabout',
    # Task 4 (5)
    'helicopter', 'windmill', 'golf-field',
    'tricycle', 'awning-tricycle'
]

# open world setting
prev_intro_cls = 0
cur_intro_cls = 14
data_root = '/home/gdut-627/106G/public-dataset/OWOD/UAV-OWD/SOWOD_CLEAN/'
train_json = 'COCO_JSONB/instances_train_t1.json'
test_json = 'COCO_JSONB/instances_test.json'
# embedding_path = 'data/VOC2007/SOWOD/t1_gt_embeddings.npy'
# att_embeddings = 'data/VOC2007/SOWOD/task_att_1_embeddings.pth'
# embedding_path = 'data/VOC2007/SOWOD/fomo_image_net_t1.npy'
# att_embeddings = None
embedding_path = '/home/gdut-627/106G/public-dataset/OWOD/UAV-OWD/SOWOD_CLEAN/SOWOD_Split/t1_gt_embeddings.npy'
att_embeddings = '/home/gdut-627/106G/public-dataset/OWOD/UAV-OWD/SOWOD_CLEAN/SOWOD_Split/task_att_1_embeddings.pth'
class_text_path=f'/home/gdut-627/106G/public-dataset/OWOD/UAV-OWD/SOWOD_CLEAN/SOWOD_Split/texts/class_texts.json'
file_name='all.txt'
pipline = [dict(type='att_select', log_start_epoch=1)]
thr = 0.6
alpha = 0.3 # 0.3
top_k =10
use_sigmoid=True
distributions = 'paper_repeat/sowod/previous_log/sowod_distribution_sim1.pth'

# yolo world setting
num_classes = prev_intro_cls+cur_intro_cls
num_training_classes = prev_intro_cls+cur_intro_cls
max_epochs = 70  # Maximum training epochs
close_mosaic_epochs = 10
save_epoch_intervals = 1
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4 / 4
weight_decay = 0.03
train_batch_size_per_gpu = 4
load_from = None
persistent_workers = False

# model settings
model = dict(type='OurDetector',
             mm_neck=True,
             num_train_classes=num_training_classes,
             num_test_classes=num_classes,
             embedding_path=embedding_path,
             prompt_dim=text_channels,
             num_prompts=33,
             pipline=pipline,
             data_preprocessor=dict(type='YOLOv5DetDataPreprocessor'),
             backbone=dict(_delete_=True,
                           type='MultiModalYOLOBackbone',
                           text_model=None,
                           image_model={{_base_.model.backbone}},
                           frozen_stages=-1,
                           with_text_model=False),
             neck=dict(type='YOLOWorldPAFPN',
                       freeze_all=False,
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
                            top_k=top_k,
                            head_module=dict(
                                type='OurHeadModule',
                                freeze_all=False,
                                use_bn_head=True,
                                embed_dims=text_channels,
                                num_classes=num_training_classes)),
             train_cfg=dict(assigner=dict(num_classes=num_training_classes)))



# dataset settings
coco_train_dataset = dict(
        _delete_=True,
        type='MultiModalDataset',
        dataset=dict(
            type='YOLOv5CocoDataset',
            metainfo=dict(classes=class_names),
            data_root=data_root,
            ann_file=train_json,
            data_prefix=dict(img='JPEGImages/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=1)),
        class_text_path=class_text_path,
        pipeline=_base_.train_pipeline)

train_dataloader = dict(persistent_workers=persistent_workers,
                        batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=coco_train_dataset)

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
# evaluation settings
test_dataloader = dict(dataset=dict(type='YOLOv5CocoDataset',
                        data_root=data_root,
                        ann_file=test_json,
                        data_prefix=dict(img='JPEGImages/'),
                        filter_cfg=dict(filter_empty_gt=False, min_size=1),
                        pipeline=test_pipeline)
                       )

test_evaluator = dict(_delete_=True,
                     type='OWODEvaluator',
                     cfg=dict(
                        dataset_root=data_root,
                        file_name=file_name,
                        prev_intro_cls=prev_intro_cls,
                        cur_intro_cls=cur_intro_cls,
                        unknown_id=33,
                        class_names=class_names
                     )
                    )
val_evaluator = test_evaluator
val_dataloader = test_dataloader
