"""
config file for DeepFashion2 Using MaskRCNN
"""

_base_ = [
    "../_base_/models/mask-rcnn_r50_fpn.py",
    "../_base_/schedules/schedule_2x.py",
    "../_base_/default_runtime.py",
    "data_pipeline.py",  # custom data pipeline
]
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet101"),
    ),
    roi_head=dict(bbox_head=dict(num_classes=13), mask_head=dict(num_classes=13)),
)

# ======== runtime settings ===========

# Change the checkpoint saving interval to iter-based
default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1),  # log model only every 5 epochs
)
