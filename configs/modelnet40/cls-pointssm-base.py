_base_ = ["../_base_/default_runtime.py"]
# misc custom setting
batch_size = 64  # bs: total bs in all gpus
num_worker = 16
batch_size_val = 8
empty_cache = False
enable_amp = False

# model settings
model = dict(
    type="DefaultClassifier",
    num_classes=40,
    backbone_embed_dim=512,
    backbone=dict(
        type="PointSSM",
        in_channels=6,
        order=["hilbert-xyz", "hilbert-yzx", "hilbert-zxy"],
        stride=(2, 2, 2, 2),
        enc_depths=(1, 1, 2, 4, 1),
        enc_channels=(64, 64, 128, 256, 512),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(64, 64, 128, 256),
        drop_path=0.1,
        shuffle_orders=True,
        cls_mode=True,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 300
# optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
# scheduler = dict(type="MultiStepLR", milestones=[0.6, 0.8], gamma=0.1)
optimizer = dict(type="AdamW", lr=0.0006, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.0006, 0.00006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.00006)]

# dataset settings
dataset_type = "ModelNetDataset"
data_root = "data/modelnet40_normal_resampled"
cache_data = False
class_names = [
    "airplane",
    "bathtub",
    "bed",
    "bench",
    "bookshelf",
    "bottle",
    "bowl",
    "car",
    "chair",
    "cone",
    "cup",
    "curtain",
    "desk",
    "door",
    "dresser",
    "flower_pot",
    "glass_box",
    "guitar",
    "keyboard",
    "lamp",
    "laptop",
    "mantel",
    "monitor",
    "night_stand",
    "person",
    "piano",
    "plant",
    "radio",
    "range_hood",
    "sink",
    "sofa",
    "stairs",
    "stool",
    "table",
    "tent",
    "toilet",
    "tv_stand",
    "vase",
    "wardrobe",
    "xbox",
]

data = dict(
    num_classes=40,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        class_names=class_names,
        transform=[
            dict(type="NormalizeCoord"),
            # dict(type="CenterShift", apply_z=True),
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1/24, 1/24], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1/24, 1/24], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.7, 1.5], anisotropic=True),
            dict(type="RandomShift", shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
            # dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                keys=("coord", "normal"),
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=10000, mode="random"),
            # dict(type="CenterShift", apply_z=True),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "category"),
                feat_keys=["coord", "normal"],
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        class_names=class_names,
        transform=[
            dict(type="NormalizeCoord"),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                keys=("coord", "normal"),
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "category"),
                feat_keys=["coord", "normal"],
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        class_names=class_names,
        transform=[
            dict(type="NormalizeCoord"),
        ],
        test_mode=True,
        test_cfg=dict(
            post_transform=[
                dict(
                    type="GridSample",
                    grid_size=0.01,
                    hash_type="fnv",
                    mode="train",
                    keys=("coord", "normal"),
                    return_grid_coord=True,
                ),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord"),
                    feat_keys=["coord", "normal"],
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[1, 1], anisotropic=True)],  # 1
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 2
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 3
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 4
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 5
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 5
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 6
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 7
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 8
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 9
            ],
        ),
    ),
)

# hooks
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="ClsEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# tester
test = dict(type="ClsVotingTester", num_repeat=100)
