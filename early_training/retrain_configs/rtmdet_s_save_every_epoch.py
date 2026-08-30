_base_ = '../../mmdetection_configs/rtmdet_s_ir.py'

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=25,
    val_interval=1,
    dynamic_intervals=[(25, 1)])
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=30))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=25,
        switch_pipeline={{_base_.train_pipeline_stage2}}),
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        patience=100,
        min_delta=0.001,
        rule='greater'),
]
work_dir = 'early_training/retrain_work_dirs/rtmdet_s'
