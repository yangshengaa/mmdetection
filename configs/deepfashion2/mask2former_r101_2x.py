"""
config file for DeepFashion2 Using Mask2Former
"""

_base_ = [
    "../mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py",
    "../_base_/schedules/schedule_2x.py",
    "data_pipeline.py",  # custom data pipeline
]

num_things_classes = 13
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=False,
    batch_augments=None
)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        depth=101,
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet101"),
    ),
    panoptic_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes),
    test_cfg=dict(panoptic_on=False))

# ======== runtime settings ===========

# Change the checkpoint saving interval to iter-based
default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1),  # log model only every 5 epochs
)

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        _delete_=True, 
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0),
    clip_grad=dict(max_norm=0.01, norm_type=2))