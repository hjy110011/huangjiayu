# configs/open_world/visdrone/test/visdrone_owod_step1.py

# 继承自基础配置
_base_ = ('../../../../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)
# ======================== 1. 数据集配置 ========================
dataset_type = 'CocoDataset'
data_root = 'D:/huangjiayu/datasets/VisDrone/OWOD/'  # 你的 OWOD 输出目录

# 类别信息
class_names = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'unknown'  # 注意：将 'unknown' 加到类别列表末尾
]
num_classes = len(class_names)
unknown_id = 5  # 'unknown' 类别的 ID (索引从 0 开始)

# 训练集配置
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=2,  # 根据你的 GPU 显存调整
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'VisDrone_OWOD_step1_5classes.json',  # 你生成的 Step1 训练集标注
        img_prefix='D:/huangjiayu/datasets/VisDrone/VisDrone2019-DET-train/images/',  # 原始图片目录
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False)  # 不要过滤掉没有标注的图片
    )
)

# 验证集配置 (使用完整的验证集，包含所有类别)
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'VisDrone_OWOD_val_all.json',  # 你需要为验证集也生成一个类似的标注文件
        img_prefix='D:/huangjiayu/datasets/VisDrone/VisDrone2019-DET-val/images/',
        pipeline=val_pipeline
    )
)

# 评估器配置，使用 OWOD 专用评估器
val_evaluator = dict(
    type='OWODEvaluator',
    cfg=dict(
        dataset_root=data_root,
        file_name='visdrone_step1_results.txt',
        prev_intro_cls=0,
        cur_intro_cls=5,  # 当前步骤引入的已知类别数 (不含 unknown)
        unknown_id=unknown_id,
        class_names=class_names
    )
)
test_evaluator = val_evaluator

# ======================== 2. 模型配置 ========================
model = dict(
    type='OurDetector',  # 使用 OUR 方法
    num_train_classes=num_classes,
    num_test_classes=num_classes,
    # 加载你生成的类别嵌入
    embedding_path='data/visdrone/gt_embeddings.npy',
    backbone=dict(
        type='MultiModalYOLOBackbone',
        image_model=dict(
            type='YOLOv8CSPDarknet',
            deepen_factor=1.0,
            widen_factor=1.0,
            out_indices=(3, 4, 5),
            sppf=dict(kernel_size=5, padding=2, act=dict(type='SiLU'))),
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name='openai/clip-vit-base-patch32',
            frozen_modules=['all'])),
    neck=dict(
        type='YOLOWorldPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        act=dict(type='SiLU'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        guide_channels=512,
        embed_channels=256,
        num_heads=8,
        use_checkpoint=False),
    bbox_head=dict(
        type='OurHead',
        prev_intro_cls=0,
        cur_intro_cls=5,  # 与上面 val_evaluator 中的 cur_intro_cls 保持一致
        unknown_id=unknown_id,
        head_module=dict(
            type='YOLOWorldHead',
            num_classes=num_classes,
            in_channels=256,
            reg_channels=256,
            cls_channels=256,
            reg_decoded_bbox=True,
            sync_bn=True,
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='IoULoss',
                iou_mode='ciou',
                bbox_format='xyxy',
                loss_weight=5.0),
            loss_dfl=dict(
                type='DistributionFocalLoss', loss_weight=0.5)),
        # 其他 OUR 方法的特定参数
        lambda_u=0.1,
        lambda_contrast=0.01,
        temperature=0.5,
        queue_size=8192,
        momentum=0.999),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            topk=13,
            alpha=1.0,
            beta=6.0,
            eps=1e-9,
            use_ciou=True,
            iou_calculator=dict(type='BboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# ======================== 3. 训练策略配置 ========================
# 优化器
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.0005,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone.text_model': dict(lr_mult=0.01),
            'logit_scale': dict(weight_decay=0.0)
        }))

# 学习率调度
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-6,
        by_epoch=True,
        begin=1,
        T_max=30,
        convert_to_iter_based=True)
]

# 训练轮数
max_epochs = 30  # 根据你的数据量和需求调整
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5,  # 每 5 个 epoch 进行一次验证
    dynamic_intervals=[(max_epochs - 5, 1)])

# ======================== 4. 其他配置 ========================
# 工作目录，用于保存模型和日志
work_dir = './work_dirs/visdrone_owod_step1'

# 日志配置
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # 如果你安装了 wandb 或 tensorboard，也可以配置
        # dict(type='WandbLoggerHook')
    ])

# 可视化配置
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')

# 自定义钩子
custom_hooks = [
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - 5,
        switch_pipeline=train_pipeline)
]

# 种子设置
seed = 0
deterministic = True
cudnn_benchmark = False

# 运行选项
runner_type = 'EpochBasedRunner'