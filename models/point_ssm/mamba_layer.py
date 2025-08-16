import math

import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from .selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, \
        mamba_inner_fn_no_out_proj
except ImportError:
    selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None, None


class Mamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,  # Fused kernel options
            layer_idx=None,
            device=None,
            dtype=None,
            out_dim=None,
            bimamba_type="none",
            norm_fn=nn.LayerNorm,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.out_dim = out_dim

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        if out_dim is None:
            self.d_inner = int(self.expand * self.d_model)
        else:
            self.d_inner = out_dim  # Use DSamba when out_dim is not None
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type



        if self.out_dim is not None and use_fast_path is False:
            self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        else:
            self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            self.activation = "silu"
            self.act = nn.SiLU()


        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True


        # bimamba from vim
        if self.bimamba_type == 'v2':
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True



        if self.out_dim is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        else:
            self.out_proj = nn.Linear(self.d_inner, self.out_dim, bias=bias, **factory_kwargs)

        # Add an extra LayerNorm here.
        if self.bimamba_type == 'v3':
            self.norm = norm_fn(self.d_inner)
            # self.alpha = nn.Parameter(torch.ones(self.d_model) / 2)
            # self.norm_fwd = norm_fn(self.d_inner)
            # self.norm_bwd = norm_fn(self.d_inner)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        if self.bimamba_type == 'v3':
            hidden_states = torch.cat([hidden_states, hidden_states.flip([-2])], dim=0)

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            if self.bimamba_type == "v2":  # Structure bi-Mamba from vim
                A_b = -torch.exp(self.A_b_log.float())
                out, _ = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b, _ = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                # F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
                out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight,
                               self.out_proj.bias)
            elif self.bimamba_type == 'v3':  # Point-cloud-suited bi-Mamba
                out, params_for_debug = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )

                # |-------------------------------------|
                # |            Mark Attn-Matrix         |
                # |-------------------------------------|
                # if out.shape[1] == 512:
                #     torch.save(params_for_debug,
                #                f"vis_attn/params_layer_D_{out.shape[1]}_L_{out.shape[2]}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.pt")

                out = rearrange(out, "b d l -> b l d")
                out = out[0, :, :] + out[1, :, :].flip([-2])
                # out = ((self.alpha[None, :] * self.norm_fwd(out[0, :, :]))
                #        + ((1 - self.alpha[None, :]) * self.norm_bwd(out[1, :, :]).flip([-2])))
                out = self.norm(out)
                out = F.linear(out, self.out_proj.weight, self.out_proj.bias)
            else:
                out, params_for_debug = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )

                # |-------------------------------------|
                # |            Mark Attn-Matrix         |
                # |-------------------------------------|
                # if out.shape[2] == 512:
                #     torch.save(params_for_debug,
                #                f"vis_attn/dsamba_params_D_{out.shape[2]}_L_{out.shape[1]}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.pt")

        else:
            x = xz
            # Compute short convolution
            # if conv_state is not None:
            #     conv_state.copy_(x[:, :, -self.d_conv:])  # Update state (B D W)

            # x = self.act(self.conv1d(x)[..., :seqlen])

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            # assert self.activation in ["silu", "swish"]
            if self.out_dim is None:
                y = selective_scan_fn(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    self.D.float(),
                    z=None,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=ssm_state is not None,
                )
            else:
                y = selective_scan_fn(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    None,
                    z=None,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=ssm_state is not None,
                )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")

            if self.bimamba_type == 'v3':
                y = y[0, :, :] + y[1, :, :].flip([-2])
                y = self.norm(y)

            out = self.out_proj(y)
            # out = y

        return out

# class MambaBlock(PointModule):
#     def __init__(
#             self, dim, layer_idx, d_state,
#             norm_cls=nn.LayerNorm, drop_path=0., ssm_cfg={}, factory_kwargs={},
#             order_index=0,
#     ):
#         super().__init__()
#         self.order_index = order_index
#         mixer_cls = partial(Mamba, d_state=d_state, **ssm_cfg, bias=False, expand=1,
#                             bimamba_type='v3', layer_idx=layer_idx,
#                             **factory_kwargs)
#         self.mixer = mixer_cls(dim)
#         self.norm = norm_cls(dim)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#     def forward(self, point: Point):
#         order = point.serialized_order[self.order_index]
#
#         # (N,D) -> (1,N,D)
#         hidden_states = point.feat[order].unsqueeze(0)
#         hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
#         residual = hidden_states
#         hidden_states = self.drop_path(self.mixer(hidden_states)) + residual
#         hidden_states = hidden_states.squeeze(0)
#         point.feat = hidden_states[point.serialized_inverse[self.order_index]]
#         point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
#
#         return point
#
#     def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
#         return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
