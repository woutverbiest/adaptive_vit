_base_ = [
    '../_base_/models/seg_vit-b16.py',
    '../_base_/datasets/coco-stuff10k.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
in_channels = 1024
img_size = (1920,512)
# checkpoint = './pretrained/vit_large_p16_384_20220308-d4efb41d.pth'
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_large_p16_384_20220308-d4efb41d.pth'
out_indices = [7, 15, 23]
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        img_size=img_size,
        embed_dims=1024,
        num_layers=24,
        drop_path_rate=0.3,
        num_heads=16,
        out_indices=out_indices),
    decode_head=dict(
        img_size=img_size,
        in_channels=in_channels,
        channels=in_channels,
        embed_dims=in_channels // 2,
        num_heads=16,
        num_classes=171,
        use_stages=len(out_indices),
        loss_decode=dict(
            type='ATMLoss', num_classes=171, dec_layers=len(out_indices), loss_weight=1.0),
    ),
    # test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
    test_cfg=dict(mode='whole'),
)


# jax use different img norm cfg
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(1920, 512), ratio_range=(1.0, 1.0)),  # No scaling
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(1920, 512), pad_val=0, seg_pad_val=255),

    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1920, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        # img_ratios=[0.5],
        flip=False,
        transforms=[
            # dict(type='Resize', keep_ratio=True, min_size=512),
            dict(type='Resize', img_scale=(1920, 512), ratio_range=(1.0, 1.0)),  # No scaling
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline)
)


optimizer = dict(_delete_=True, type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.02,
                 paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.),
                                                 'ln': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.),
                                                 }))
#
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=2.0, min_lr=0.0, by_epoch=False)

