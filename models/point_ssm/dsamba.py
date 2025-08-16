from copy import copy
from functools import partial

import spconv.pytorch as spconv
from addict import Dict
from timm.layers import trunc_normal_
import math
import torch
import torch.nn as nn
import torch_scatter
from .mamba_layer import Mamba
from timm.models.layers import DropPath

try:
    import flash_attn
except ImportError:
    flash_attn = None

from models.utils.structure import Point
from models.modules import PointModule, PointSequential


class MambaBlock(PointModule):
    def __init__(
            self, dim, d_state, d_conv,
            norm_cls=nn.LayerNorm, drop_path=0., ssm_cfg={}, factory_kwargs={},
            order_index=0, pooling_depth=2, out_dim=None
    ):
        super().__init__()
        self.order_index = order_index
        mixer_cls = partial(Mamba, d_state=d_state, out_dim=out_dim, bias=False, d_conv=d_conv,
                            **ssm_cfg, **factory_kwargs)
        self.mixer = mixer_cls(dim)
        # self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.pooling_depth = pooling_depth

    def forward(self, point: Point):
        order = point.serialized_order[self.order_index]
        hidden_states = point.feat[order].unsqueeze(0)
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        hidden_states = self.mixer(hidden_states)
        hidden_states = self.drop_path(hidden_states)
        point.feat = hidden_states.squeeze(0)

        return point


class DSamba(PointModule):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride=2,
            d_state=1,
            d_conv=4,
            norm_fn=nn.LayerNorm,
            order_index=0,
    ):
        super().__init__()
        self.mamba = MambaBlock(
            dim=in_channels,
            d_state=d_state,
            d_conv=d_conv,
            norm_cls=norm_fn,
            drop_path=0.1,
            order_index=order_index,
            out_dim=out_channels
        )
        self.in_channels = in_channels
        self.stride = stride
        self.order_index = order_index
        # self.out_proj = nn.Linear(in_features=in_channels, out_features=out_channels, bias=False)

        # self.norm = PointSequential(norm_fn(out_channels))
        # self.act = PointSequential(nn.GELU())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        idx = idx_ptr[1:] - 1
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        shortcut = copy(point)
        shortcut = self.mamba(shortcut)
        feat = shortcut.feat[idx, :]    # Selected the last point of each sample stride.
        point_dict = Dict(
            feat=feat[inverse[self.order_index], :].squeeze(0),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        point_dict["pooling_inverse"] = cluster
        point_dict["pooling_parent"] = point
        point = Point(point_dict)

        # if self.norm is not None:
        #     point = self.norm(point)
        # if self.act is not None:
        #     point = self.act(point)

        point.sparsify()
        return point
