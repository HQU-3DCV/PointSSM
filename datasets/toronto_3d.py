import os
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence


from utils.cache import shared_dict
from .builder import DATASETS
from .defaults import DefaultDataset
from .transform import Compose, TRANSFORMS

'''
Road (label 1)
Road marking (label 2)
Natural (label 3)
Building (label 4)
Utility line (label 5)
Pole (label 6)
Car (label 7)
Fence (label 8)
unclassified (label 0)
'''
VALID_CLASS_IDS = np.array([
    1,   # Road
    2,   # Road marking
    3,   # Trucks
    4,   # Natural
    5,   # Utility line
    6,   # Pole
    7,    # Fence
    8   # Fence
])


@DATASETS.register_module()
class Toronto3DDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "intensity",
        "segment",
    ]
    class2id = VALID_CLASS_IDS

    def __init__(
        self,
        lr_file=None,
        la_file=None,
        **kwargs,
    ):
        self.lr = np.loadtxt(lr_file, dtype=str) if lr_file is not None else None
        self.la = torch.load(la_file) if la_file is not None else None
        super().__init__(**kwargs)

    def get_data_list(self):
        if self.lr is None:
            data_list = super().get_data_list()
        else:
            data_list = [
                os.path.join(self.data_root, "train", name) for name in self.lr
            ]
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name
        data_dict["coord"] = data_dict["coord"].astype(np.float32)
        data_dict["color"] = data_dict["color"].astype(np.float32)
        data_dict["strength"] = data_dict["intensity"].astype(np.float32)
        del data_dict['intensity']

        if "segment" in data_dict.keys():
            data_dict["segment"] = (
                (data_dict.pop("segment").reshape([-1])).astype(np.int32)
            )
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        # # 映射函数
        # learning_map = {
        #     1: 0,
        #     2: 1,
        #     3: 2,
        #     4: 3,
        #     5: 4,
        #     6: 5,
        #     7: 6,
        #     8: 7
        # }
        #
        # # 创建一个向量化的映射函数
        # vectorized_map = np.vectorize(learning_map.get)
        #
        # # 对 segment 值进行映射
        # mapped_segment = vectorized_map(data_dict["segment"])
        #
        # # 将映射后的值更新到 data_dict 中
        # data_dict["segment"] = mapped_segment


        # if "instance" in data_dict.keys():
        #     data_dict["instance"] = (
        #         data_dict.pop("instance").reshape([-1]).astype(np.int32)
        #     )
        # else:
        #     data_dict["instance"] = (
        #         np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
        #     )
        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            mask = np.ones_like(data_dict["segment"], dtype=bool)
            mask[sampled_index] = False
            data_dict["segment"][mask] = self.ignore_index
            data_dict["sampled_index"] = sampled_index

        return data_dict
    #
    # @staticmethod
    # def get_learning_map(ignore_index):
    #     learning_map = {
    #         1: 0,
    #         2: 1,
    #         3: 2,
    #         4: 3,
    #         5: 4,
    #         6: 5,
    #         7: 6,
    #         8: 7
    #     }
    #     return learning_map
    #
    # @staticmethod
    # def get_learning_map_inv(ignore_index):
    #     learning_map_inv = {
    #         0: 1,
    #         1: 2,
    #         2: 3,
    #         3: 4,
    #         4: 5,
    #         5: 6,
    #         6: 7,
    #         7: 8,
    #     }
    #     return learning_map_inv

#
# import os
# import numpy as np
#
# from .builder import DATASETS
# from .defaults import DefaultDataset
# from pointcept.utils.cache import shared_dict
#
#
# @DATASETS.register_module()
# class DALESDataset(DefaultDataset):
#     def __init__(self, ignore_index=-1, **kwargs):
#         self.ignore_index = ignore_index
#         self.learning_map = self.get_learning_map(ignore_index)
#         self.learning_map_inv = self.get_learning_map_inv(ignore_index)
#         super().__init__(ignore_index=ignore_index, **kwargs)
#
#     def get_data_list(self):
#
#         if self.lr is None:
#             data_list = super().get_data_list()
#         else:
#             data_list = [
#                 os.path.join(self.data_root, "train", name) for name in self.lr
#             ]
#         return data_list
#
#     def get_data(self, idx):
#
#         data_path = self.data_list[idx % len(self.data_list)]
#         name = self.get_data_name(idx)
#         if self.cache:
#             cache_name = f"pointcept-{name}"
#             return shared_dict(cache_name)
#
#         data_dict = {}
#         assets = os.listdir(data_path)
#         for asset in assets:
#             if not asset.endswith(".npy"):
#                 continue
#             if asset[:-4] not in self.VALID_ASSETS:
#                 continue
#             data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
#         data_dict["name"] = name
#         data_dict["coord"] = data_dict["coord"].astype(np.float32)
#         data_dict["strength"] = data_dict["intensity"].astype(np.float32)
#         del data_dict['intensity']
#
#         if "segment" in data_dict.keys():
#             data_dict["segment"] = (
#                 data_dict.pop("segment").reshape([-1]).astype(np.int32)
#             )
#         else:
#             data_dict["segment"] = (
#                 np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
#             )
#
#         if "instance" in data_dict.keys():
#             data_dict["instance"] = (
#                 data_dict.pop("instance").reshape([-1]).astype(np.int32)
#             )
#         else:
#             data_dict["instance"] = (
#                 np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
#             )
#         if self.la:
#             sampled_index = self.la[self.get_data_name(idx)]
#             mask = np.ones_like(data_dict["segment"], dtype=bool)
#             mask[sampled_index] = False
#             data_dict["segment"][mask] = self.ignore_index
#             data_dict["sampled_index"] = sampled_index
#
#         return data_dict
#
#     def get_data_name(self, idx):
#         file_path = self.data_list[idx % len(self.data_list)]
#         dir_path, file_name = os.path.split(file_path)
#         sequence_name = os.path.basename(os.path.dirname(dir_path))
#         frame_name = os.path.splitext(file_name)[0]
#         data_name = f"{sequence_name}_{frame_name}"
#         return data_name
#
#     @staticmethod
#     def get_learning_map(ignore_index):
#         learning_map = {
#             1: 0,
#             2: 1,
#             3: 2,
#             4: 3,
#             5: 4,
#             6: 5,
#             7: 6,
#             8: 7
#         }
#         return learning_map
#
#     @staticmethod
#     def get_learning_map_inv(ignore_index):
#         learning_map_inv = {
#             0: 1,
#             1: 2,
#             2: 3,
#             3: 4,
#             4: 5,
#             5: 6,
#             6: 7,
#             7: 8,
#         }
#         return learning_map_inv
