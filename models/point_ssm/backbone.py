from functools import partial

import numpy as np
from addict import Dict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.layers import trunc_normal_, DropPath

from .mamba_layer import Mamba
from .dsamba import DSamba
from .attn_layer import Block

try:
    import flash_attn
except ImportError:
    flash_attn = None

from models.builder import MODELS
from models.utils.misc import offset2bincount, offset2batch
from models.utils.structure import Point
from models.modules import PointModule, PointSequential



class MambaBlock(PointModule):
    def __init__(
            self, dim, layer_idx, d_state,
            norm_cls=nn.LayerNorm, drop_path=0., ssm_cfg={}, factory_kwargs={},
            order_index=0,
    ):
        super().__init__()
        self.order_index = order_index
        mixer_cls = partial(Mamba, d_state=d_state, **ssm_cfg, bias=False, expand=1,
                            bimamba_type='v3', layer_idx=layer_idx,
                            **factory_kwargs)
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, point: Point):
        order = point.serialized_order[self.order_index]

        # (N,D) -> (1,N,D)
        hidden_states = point.feat[order].unsqueeze(0)
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        hidden_states = self.drop_path(self.mixer(hidden_states)) + residual
        hidden_states = hidden_states.squeeze(0)
        point.feat = hidden_states[point.serialized_inverse[self.order_index]]
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)

        return point


class LocalSparseConv(PointModule):
    def __init__(
            self,
            in_channels,
            out_channels,
            indice_key,
            norm_fn,
            kernel_size=3,
            padding=1,
            bias=False,
            drop_path=0.

    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spconv = PointSequential(
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    indice_key=indice_key,
                    bias=bias,
                ),
                norm_fn(out_channels),
                nn.GELU(),
                spconv.SubMConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    indice_key=indice_key,
                    bias=bias,
                ),
                norm_fn(out_channels)
            )
        )
        self.out_act = nn.GELU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def forward(self, point: Point):
        residual = point.feat
        point = self.spconv(point)
        point.feat = self.drop_path(point.feat)
        point.feat = self.out_act(point.feat + residual)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)

        return point

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class GridUpPooling(PointModule):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            norm_layer=None,
            act_layer=None,
            traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
            self,
            in_channels,
            embed_channels,
            norm_layer=None,
            act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )

        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)

        # np.save(f"/home/cv/gjl/PointSSM/visual/coord.npy", point.coord.cpu().numpy())
        # np.save(f"/home/cv/gjl/PointSSM/visual/order.npy", point.serialized_order.cpu().numpy())
        return point


@MODELS.register_module("PointSSM")
class PointTransformerV3(PointModule):
    def __init__(
            self,
            in_channels=6,
            order=("hilbert-xyz"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(48, 48, 48, 48, 48),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(48, 48, 48, 48),
            drop_path=0.1,
            shuffle_orders=True,
            cls_mode=False,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                             sum(enc_depths[:s]): sum(enc_depths[: s + 1])
                             ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    DSamba(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        d_state=4,
                        d_conv=4,
                        norm_fn=ln_layer,
                        order_index=0,
                    ),
                    name="down",
                )

            for i in range(enc_depths[s]):
                # enc.add(
                #     FusionLayer(
                #         dim=enc_channels[s],
                #         layer_index=i,
                #         order_index=i % len(self.order),
                #         norm_ln=ln_layer,
                #         norm_bn=bn_layer,
                #         drop_path=enc_drop_path_[i],
                #         mode='mix'
                #     ),
                #     name=f"FusionBlock{i}",
                # )
                enc.add(
                    MambaBlock(
                        dim=enc_channels[s],
                        d_state=32,
                        layer_idx=i,
                        norm_cls=ln_layer,
                        drop_path=enc_drop_path_[i],
                        order_index=i % len(self.order),
                    ),
                    name=f"GlobalMamba{i}",
                )
                enc.add(
                    LocalSparseConv(
                        in_channels=enc_channels[s],
                        out_channels=enc_channels[s],
                        kernel_size=3,
                        padding=1,
                        norm_fn=bn_layer,
                        indice_key=f"lc{i}",
                        bias=False,
                        drop_path=enc_drop_path_[i],
                    ),
                    name=f"SparseBlock{i}",
                )

            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                                 sum(dec_depths[:s]): sum(dec_depths[: s + 1])
                                 ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    GridUpPooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )

                for i in range(dec_depths[s]):

                    # dec.add(
                    #     FusionLayer(
                    #         dim=dec_channels[s],
                    #         layer_index=i,
                    #         order_index=i % len(self.order),
                    #         norm_ln=ln_layer,
                    #         norm_bn=bn_layer,
                    #         drop_path=dec_drop_path_[i],
                    #         mode='mix'
                    #     ),
                    #     name=f"FusionBlock{i}",
                    # )
                    dec.add(
                        MambaBlock(
                            dim=dec_channels[s],
                            d_state=32,
                            layer_idx=i,
                            norm_cls=ln_layer,
                            drop_path=dec_drop_path_[i],
                            order_index=i % len(self.order),
                        ),
                        name=f"GlobalMamba{i}",
                    )
                    dec.add(
                        LocalSparseConv(
                            in_channels=dec_channels[s],
                            out_channels=dec_channels[s],
                            kernel_size=3,
                            padding=1,
                            norm_fn=bn_layer,
                            indice_key=f"lc{i}",
                            bias=False,
                            drop_path=dec_drop_path_[i],
                        ),
                        name=f"SparseBlock{i}",
                    )

                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        point = self.embedding(point)
        point = self.enc(point)
        if not self.cls_mode:
            point = self.dec(point)
        return point
