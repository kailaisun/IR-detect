_base_ = '../../mmdetection_configs/dino_4scale_r50_ir.py'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=4, val_interval=1)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=10))
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        patience=100,
        min_delta=0.001,
        rule='greater')
]
work_dir = 'early_training/retrain_work_dirs/dino_4scale_r50'
