_base_ = [
    '../../_base_/models/segformer_mit-b0.py',
    '../../_base_/datasets/crack500.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (384, 384)
data_preprocessor = dict(size=crop_size)
checkpoint = 'path to pretrained weights'
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(
            type='CCMambaHead',
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            dropout_ratio=0.1,
            num_classes=2,
            out_channels=1,
            threshold=0.38,
            relation=[[3], [3], [3]],
            kernel_size=[[3], [3], [3]],
            dilation=[[2], [1], [0]],
            num_heads=16,
            channels=256,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(224, 224)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=3000),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=3000,
        end=80000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=2, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
