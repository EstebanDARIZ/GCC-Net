dataset_type = 'CocoDataset'
data_root = '/home/esteban-dreau-darizcuren/doctorat/dataset/dataset_coco_format/dataset_coco_gccnet'
data_test = '/home/esteban-dreau-darizcuren/doctorat/dataset/datat_test'

classes = (
    'Bait_1_Squid',
    'Bait_2_Sardine',
    'Ray',
    'Sunfish',
    'Pilotfish'
)

img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717],
    std=[1.0, 1.0, 1.0],
    to_rgb=False
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations/instances_train2017.json',
        img_prefix=data_root + '/train',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations/instances_val2017.json',
        img_prefix=data_root + '/val',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=None,  # ← test NON labellisé
        img_prefix=data_test,
        classes=classes,
        pipeline=test_pipeline)
)
