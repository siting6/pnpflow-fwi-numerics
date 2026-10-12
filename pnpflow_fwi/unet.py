"""Time-conditioned 2D U-Net used as the flow-matching velocity network."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, dim, time_dim):
        super().__init__()
        self.time_embed = SinusoidalPositionEmbeddings(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))

    def forward(self, time):
        return self.mlp(self.time_embed(time))


class ResidualBlock(nn.Module):
    """GroupNorm-SiLU-conv twice, with the time embedding added between, plus a skip connection."""

    def __init__(self, in_channels, out_channels, time_dim, num_groups=8, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual_conv = (nn.Conv2d(in_channels, out_channels, kernel_size=1)
                              if in_channels != out_channels else nn.Identity())

    def forward(self, x, time_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(time_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention over all pixels."""

    def __init__(self, channels, num_heads=4, num_groups=8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = (a.reshape(B, self.num_heads, C // self.num_heads, H * W).transpose(-2, -1) for a in (q, k, v))
        h = F.scaled_dot_product_attention(q, k, v, scale=(C // self.num_heads) ** -0.5)
        h = h.transpose(-2, -1).reshape(B, C, H, W)
        return x + self.proj(h)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


class UNet2D(nn.Module):
    """U-Net with `num_res_blocks` residual blocks per level, channel widths base_channels * channel_mults,
    and self-attention at the coarsest level and in the middle."""

    def __init__(self, in_channels=1, out_channels=1, base_channels=64, channel_mults=(1, 2, 2, 4),
                 num_res_blocks=2, dropout=0.0, time_embed_dim=256, num_heads=4, num_groups=8):
        super().__init__()
        self.time_embed = TimeEmbedding(base_channels, time_embed_dim)
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        last = len(channel_mults) - 1

        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        in_ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            block = nn.ModuleList()
            for _ in range(num_res_blocks):
                block.append(ResidualBlock(in_ch, out_ch, time_embed_dim, num_groups, dropout))
                channels.append(out_ch)
                in_ch = out_ch
                if i == last:
                    block.append(AttentionBlock(in_ch, num_heads, num_groups))
            if i < last:
                block.append(Downsample(in_ch))
                channels.append(in_ch)
            self.down_blocks.append(block)

        self.mid = nn.ModuleList([ResidualBlock(in_ch, in_ch, time_embed_dim, num_groups, dropout),
                                  AttentionBlock(in_ch, num_heads, num_groups),
                                  ResidualBlock(in_ch, in_ch, time_embed_dim, num_groups, dropout)])

        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            block = nn.ModuleList()
            for j in range(num_res_blocks + 1):
                block.append(ResidualBlock(in_ch + channels.pop(), out_ch, time_embed_dim, num_groups, dropout))
                in_ch = out_ch
                if i == last and j == 0:
                    block.append(AttentionBlock(in_ch, num_heads, num_groups))
            if i > 0:
                block.append(Upsample(in_ch))
            self.up_blocks.append(block)

        self.norm_out = nn.GroupNorm(num_groups, in_ch)
        self.conv_out = nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x, time):
        time_emb = self.time_embed(time)
        h = self.conv_in(x)
        skips = [h]
        for block in self.down_blocks:
            for module in block:
                if isinstance(module, ResidualBlock):
                    h = module(h, time_emb)
                    skips.append(h)
                elif isinstance(module, Downsample):
                    h = module(h)
                    skips.append(h)
                else:
                    h = module(h)
        for module in self.mid:
            h = module(h, time_emb) if isinstance(module, ResidualBlock) else module(h)
        for block in self.up_blocks:
            for module in block:
                if isinstance(module, ResidualBlock):
                    h = module(torch.cat([h, skips.pop()], dim=1), time_emb)
                else:
                    h = module(h)
        return self.conv_out(F.silu(self.norm_out(h)))
