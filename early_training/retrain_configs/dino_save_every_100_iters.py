_base_ = '../../mmdetection_configs/dino_4scale_r50_ir.py'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=100,
        save_last=True,
        max_keep_ckpts=10))
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        patience=100,
        min_delta=0.001,
        rule='greater')
]
work_dir = 'early_training/retrain_work_dirs/dino_iter_snapshots'
