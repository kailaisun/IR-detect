_base_ = '../third_party/mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

classes = ('lie', 'sit', 'other', 'off_bed')
metainfo = dict(classes=classes)
data_root = 'mmdetection_data/'
max_epochs = 12

model = dict(roi_head=dict(bbox_head=dict(num_classes=4)))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
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
    batch_size=64,
    num_workers=8,
    batch_sampler=None,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/val/'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='images/test/'),
        pipeline=test_pipeline))

val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val.json', classwise=True)
test_evaluator = dict(
    ann_file=data_root + 'annotations/instances_test.json',
    classwise=True,
    outfile_prefix='mmdetection_work_dirs/faster_rcnn_r50_fpn/test_predictions')

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=2)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1),
]
optim_wrapper = dict(optimizer=dict(lr=0.04))
auto_scale_lr = dict(enable=False, base_batch_size=16)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=2,
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=2))
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        patience=3,
        min_delta=0.001,
        rule='greater')
]
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
work_dir = 'mmdetection_work_dirs/faster_rcnn_r50_fpn'
randomness = dict(seed=42, deterministic=False)
