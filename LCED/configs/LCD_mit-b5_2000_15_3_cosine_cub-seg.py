_base_ = [
    '_base_/models/segformer_mit-b0.py', '_base_/datasets/cub_seg.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_40k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (384, 384)
data_preprocessor = dict(size=crop_size)
checkpoint = './pretrained/mit_b5.pth'
# model settings
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint
            ),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(
        type='LCEDHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=201,
        norm_cfg=norm_cfg,
        align_corners=False,
        mem_length=2000, 
        G=15, # 15
        smooth_kernel=3,
        iter_train=3,
        r_func='cosine',
        save_path='experiments/CUB/LCD_mit-b5_2000_15_3_cosine_alpha1/lcd',
        # when testing, change this path to "'experiments/CUB/LCD_mit-b5_2000_15_3_cosine_alpha1/lcd_{id}'"
        # choose id in {1,...,10}
        version='train',
        alpha=1,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    )

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01), # 0.00006 0.0001
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500), # 3000
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=8, num_workers=8)
val_dataloader = dict(batch_size=1, num_workers=2)
test_dataloader = val_dataloader