_base_ = '../third_party/mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'

classes = ('lie', 'sit', 'other', 'off_bed')
metainfo = dict(classes=classes)
data_root = 'mmdetection_data/'
max_epochs = 30
stage2_num_epochs = 5
base_lr = 0.004

model = dict(
    bbox_head=dict(num_classes=4),
    test_cfg=dict(max_per_img=100))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=(320, 320), pad_val=114.0),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(320, 320)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(320, 320), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(320, 320),
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type='PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(320, 320),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(320, 320)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(320, 320), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
    dict(type='Pad', size=(320, 320), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    batch_sampler=None,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=128,
    num_workers=8,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/val/'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=128,
    num_workers=8,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='images/test/'),
        pipeline=test_pipeline))

val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val.json',
    classwise=True,
    proposal_nums=(100, 1, 10))
test_evaluator = dict(
    ann_file=data_root + 'annotations/instances_test.json',
    classwise=True,
    proposal_nums=(100, 1, 10),
    outfile_prefix='mmdetection_work_dirs/rtmdet_s/test_predictions')

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
]
optim_wrapper = dict(optimizer=dict(lr=base_lr))
auto_scale_lr = dict(enable=False, base_batch_size=256)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=2))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2),
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        patience=3,
        min_delta=0.001,
        rule='greater'),
]
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'
work_dir = 'mmdetection_work_dirs/rtmdet_s'
randomness = dict(seed=42, deterministic=False)
