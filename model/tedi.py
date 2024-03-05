"""
Unet denoiser for diffusion
"""

import logging
from functools import partial

import torch
from einops import rearrange
from torch import nn

from utils.tedi_utils import SinusoidalPosEmb, Downsample, Prenorm, Residual, ResnetBlock, Upsample, Attention, LinearAttention
from model.mdm_any import MdmBase


class Unet(MdmBase):
    def __init__(self, input_features: int, diffusion, dim: int=256, dim_mults: list[int]=[2, 4],
                  resnet_block_groups: int=8, norm: str='group', kernel: int=3,
                    padding: int=1, stride: int=2):
        super().__init__()
        self.input_features = input_features
        self.diffusion = diffusion

        self.init_conv = nn.Conv1d(input_features, dim, 7, padding=3)
        dims = [dim] + [dim * i for i in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 4
        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim

        block = partial(
            ResnetBlock,
            time_dim=time_dim,
            groups=resnet_block_groups,
            norm=norm,
            kernel_size=kernel,
            padding=padding,
        )

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs, self.ups = nn.ModuleList([]), nn.ModuleList([])
        num_resolutions = len(in_out)

        print(f"{num_resolutions=}, {in_out=}")

        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i == (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block(dim_in, dim_in),
                        block(dim_in, dim_in),
                        Residual(Prenorm(dim_in, LinearAttention(dim_in))),
                        Downsample(
                            dim_in,
                            dim_out,
                            kernel_size=kernel,
                            stride=stride,
                            padding=padding,
                        )
                        if not is_last
                        else nn.Conv1d(
                            dim_in, dim_out, kernel, padding=padding
                        ),
                        Downsample(
                            time_dim,
                            time_dim,
                            kernel_size=kernel,
                            stride=stride,
                            padding=padding,
                        )
                        if not is_last
                        else nn.Conv1d(
                            time_dim, time_dim, kernel, padding=padding
                        ),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block(mid_dim, mid_dim)
        self.mid_attn = Residual(Prenorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block(mid_dim, mid_dim)

        for i, (dim_out, dim_in) in enumerate(reversed(in_out)):
            is_last = i == (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block(dim_in + dim_out, dim_in),
                        block(dim_in + dim_out, dim_in),
                        Residual(Prenorm(dim_in, LinearAttention(dim_in))),
                        Upsample(
                            dim_in,
                            dim_out,
                            kernel_size=kernel,
                            padding=padding,
                        )
                        if not is_last
                        else nn.Conv1d(
                            dim_in, dim_out, kernel, padding=padding
                        ),
                        Upsample(
                            time_dim,
                            time_dim,
                            kernel_size=kernel,
                            padding=padding,
                        )
                        if not is_last
                        else nn.Conv1d(
                            time_dim, time_dim, kernel, padding=padding
                        ),
                    ]
                )
            )

        self.final_res_block = block(dim * 2, dim)
        self.final_conv = nn.Conv1d(dim, input_features, 1)

    def forward(self, x, t):
        x = self.init_conv(x)
        x_ = x.clone()

        t = self.time_mlp(t)

        skip_connects = []

        for block1, block2, *attn, down_x, down_t in self.downs:
            x = block1(x, t)
            skip_connects.append(x)
            x = block2(x, t)
            x = attn[0](x)
            skip_connects.append(x)
            x = down_x(x)
            t = rearrange(t, "t c -> c t")
            t = down_t(t)
            t = rearrange(t, "c t-> t c")

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, *attn, up_x, up_t in self.ups:
            x = torch.cat((x, skip_connects.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, skip_connects.pop()), dim=1)
            x = block2(x, t)
            x = attn[0](x)
            x = up_x(x)
            t = rearrange(t, "t c -> 1 c t")
            t = up_t(t)
            t = rearrange(t, "1 c t-> t c")

        print(f"x has shape {x.shape}, x_ has shape {x_.shape}")
        x = torch.cat((x, x_), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)
