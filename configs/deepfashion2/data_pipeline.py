"""
data loader and preprocessing pipeline for DeepFashion2
"""

# * dataset root
data_root = "/n/home02/shengy/course/6_8300/code/data/subset/"
batch_size = 1

# =================== info ========================
backend_args = None
dataset_type = "DeepFashionDataset"  # use DeepFashionDataset but overwrite class and palette with DeepFashion2 to avoid registration

METAINFO = {
    'classes': (
        "short_sleeved_shirt",
        "long_sleeved_shirt",
        "short_sleeved_outwear",
        "long_sleeved_outwear",
        "vest",
        "sling",
        "shorts",
        "trousers",
        "skirt",
        "short_sleeved_dress",
        "long_sleeved_dress",
        "vest_dress",
        "sling_dress",
    ),
    # palette is a list of color tuples, which is used for visualization. (remove the final one from DeepFashion)
    'palette': [(0, 192, 64), (0, 64, 96), (128, 192, 192), (0, 64, 64),
                (0, 192, 224), (0, 192, 192), (128, 192, 64), (0, 192, 96),
                (128, 32, 192), (0, 0, 224), (0, 0, 64), (0, 160, 192),
                (128, 0, 96)]
}


# =================== augmentation ========================
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Resize", scale=(750, 1101), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(750, 1101), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

# =================== loaders ========================
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type="RepeatDataset",
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            metainfo=METAINFO,
            ann_file="train/annotation_coco.json",
            data_prefix=dict(img="train/image/"),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            backend_args=backend_args,
        ),
    ),
)
val_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=METAINFO,
        ann_file="validation/annotation_coco.json",
        data_prefix=dict(img="validation/image/"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=METAINFO,
        ann_file="test/annotation_coco.json",
        data_prefix=dict(img="test/image/"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

# =================== evaluation ========================
val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "validation/annotation_coco.json",
    metric=["bbox", "segm"],
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "test/annotation_coco.json",
    metric=["bbox", "segm"],
    format_only=False,
    backend_args=backend_args,
)