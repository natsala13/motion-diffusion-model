"""
Positional embeddings
"""
import math
from inspect import isfunction

import torch
from torch import nn, einsum
from einops import rearrange


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
    
class Residual(nn.Module):
    def __init__(self, fct):
        super().__init__()
        self.fct = fct

    def forward(self, x, *args):
        return self.fct(x, *args) + x


def Upsample(dim, dim_out=None, kernel_size=5, padding=2):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="linear"),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding=padding),
    )


def Downsample(dim, dim_out=None, kernel_size=5, stride=2, padding=1):
    return nn.Conv1d(dim, default(dim_out, dim), kernel_size, stride, padding)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.gamma


class Prenorm(nn.Module):
    def __init__(self, dim, fct):
        super().__init__()
        self.fct = fct
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fct(self.norm(x))


class Block(nn.Module):
    def __init__(self, dim, dim_out, norm="group", groups=8, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim_out, kernel_size, padding=padding)

        if norm == "group":
            self.norm = nn.GroupNorm(groups, dim_out)
        elif norm == "instance":
            self.norm = nn.InstanceNorm1d(dim_out)
        else:
            raise ValueError(f"Normalization {norm} not supported.")

        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.norm(self.conv(x))

        if scale_shift is not None:
            scale, shift = scale_shift
            print.debug(
                f"scale shape: {scale.shape}, shift shape: {shift.shape}, x shape: {x.shape}"
            )
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        time_dim=None,
        groups=8,
        norm="group",
        kernel_size=3,
        padding=1,
    ):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, dim_out * 2))
        self.block1 = Block(dim, dim_out, norm, groups, kernel_size, padding)
        self.block2 = Block(dim_out, dim_out, norm, groups, kernel_size, padding)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        # time_emb = rearrange(self.mlp(time_emb), 'b c -> b c 1')
        print(f"time emb shape: {time_emb.shape}")
        time_emb = rearrange(self.mlp(time_emb), "t c -> 1 c t")
        print(f"time emb shape after reshape: {time_emb.shape}")
        scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)



class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x):
        *_, f = x.shape
        print(f"dimension of input: {x.shape}")
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) f -> b h c f", h=self.heads),
            self.to_qkv(x).chunk(3, dim=1),
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c f -> b (h c) f")

        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        *_, f = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t, "b (h c) f -> b h c f", h=self.heads), qkv)
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h i d -> b (h d) i")

        return self.to_out(out)
