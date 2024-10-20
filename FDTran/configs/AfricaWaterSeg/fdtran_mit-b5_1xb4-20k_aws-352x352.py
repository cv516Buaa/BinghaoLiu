_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/aws16k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (352, 352)
data_preprocessor = dict(size=crop_size)
checkpoint = '' # path of pretrained file
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(type='FDTHead',
                     in_channels=[64, 128, 320, 512],
                     num_classes=2,
                     num_heads=16,
                     loss_decode=[
                         dict(
                             type='CrossEntropyLoss',
                             use_sigmoid=False,
                             loss_weight=1.0),
                         dict(
                             type='OhemCrossEntropy',
                             thres=0.7,
                             min_kept=15488,
                             loss_weight=1.0),
                     ]),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(224, 224)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=20000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
