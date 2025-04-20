from abc import abstractmethod

import math

# TODO: EVERYTHING !!!!!!
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .unet import Upsample, Downsample, TimestepBlock
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    zero_module,
    LayerNorm,
    normalization,
)


# Fusion SAR with Optical Featrue
# We need to handle Optical Feature (Q) and SAR feature (K, V)
# Based on QKVAttention (unet.py by NVIDIA)
# Q from Optical, KV from SAR. need to concat it before send it to SFBlock


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=4, dims=2):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1) if dims == 2 else nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            conv_nd(dims, channels, reduced, 1),
            nn.SiLU(),
            conv_nd(dims, reduced, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class MBConv(nn.Module):
    """
    point-wise → depth-wise → SE(채널 어텐션) → projection
    residual if in_channels == out_channels
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=6,
        reduction=4,
        dropout=0.0,
        dims=2,
    ):
        super().__init__()
        self.use_residual = (in_channels == out_channels)
        hidden = in_channels * expansion

        # 1×1 pointwise expansion
        self.expand = conv_nd(dims, in_channels, hidden, 1)
        # 3×3 depthwise conv
        self.dw = conv_nd(dims, hidden, hidden, 3, padding=1, groups=hidden)
        # channel attention
        self.se = SqueezeExcite(hidden, reduction, dims)
        # 1×1 projection
        self.project = conv_nd(dims, hidden, out_channels, 1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        h = self.act(self.expand(x))
        h = self.act(self.dw(h))
        h = self.se(h)
        h = self.project(h)
        h = self.dropout(h)
        return x + h if self.use_residual else h


'''
Basic Architecture

1. LN (Layer Norm)
2. MBConv
3. + X
4. LN (Layer Norm)
5. FFN
6. + Z
7. Simple Gat

*** PARAMETERS *** 

- we dont use multi-head attention. (n_heads = 1)
- QKV from Optical + SAR 

encoder_kv 
- Upper the if (about encoder_kv), Code is talking about Self-attention (* 3)
- but we only use kv from SAR (it means -> * 2)

after coding, we can rewrite it briefly

- bs: bench_size, k = key, v = value
'''
class SFBlock(nn.Module):
    """
    Cross-attention fusion block for DB-CR backbone.
    Q from Optical branch, K/V from SAR branch.

    Steps:
    1. Normalize & project Q, K, V via 1x1 conv
    2. Flatten spatial dims → sequence length HW
    3. Split into heads and compute scaled dot-product attention
    4. Concat heads, MLP (2 x hidden) + residual
    5. Reshape back to (B, C, H, W), final 1x1 conv
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        mlp_ratio=2,
        dims=2,
    ):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # 1×1 projections with normalization
        self.norm_q  = normalization(channels)
        self.q_proj  = conv_nd(dims, channels, channels, 1)
        self.norm_kv = normalization(channels)
        self.k_proj  = conv_nd(dims, channels, channels, 1)
        self.v_proj  = conv_nd(dims, channels, channels, 1)

        # MLP after attention
        hidden_dim = channels * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )

        # Final 1×1 conv, zero-init for stable fusion
        self.out_proj = zero_module(conv_nd(dims, channels, channels, 1))

    def forward(self, feat_opt, feat_sar):
        B, C, H, W = feat_opt.shape
        HW = H * W

        # 1. Norm + 1×1 conv
        q = self.q_proj(self.norm_q(feat_opt))   # [B, C, H, W]
        k = self.k_proj(self.norm_kv(feat_sar))
        v = self.v_proj(self.norm_kv(feat_sar))

        # 2. Flatten to [B, HW, C]
        q_flat = q.view(B, C, HW).transpose(1, 2)
        k_flat = k.view(B, C, HW).transpose(1, 2)
        v_flat = v.view(B, C, HW).transpose(1, 2)

        # 3. Split heads → [B, num_heads, HW, head_dim]
        def split_heads(x):
            return x.view(B, HW, self.num_heads, self.head_dim).transpose(1, 2)
        qh = split_heads(q_flat)
        kh = split_heads(k_flat)
        vh = split_heads(v_flat)

        # 4. Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = th.softmax(th.matmul(qh, kh.transpose(-2, -1)) * scale, dim=-1)
        attn_out = th.matmul(attn, vh)  # [B, num_heads, HW, head_dim]

        # 5. Concat heads → [B, HW, C]
        out_flat = attn_out.transpose(1, 2).reshape(B, HW, C)

        # 6. MLP + residual
        mlp_out = self.mlp(out_flat)
        fused = out_flat + mlp_out  # [B, HW, C]

        # 7. Reshape back → [B, C, H, W], final conv
        fused = fused.transpose(1, 2).reshape(B, C, H, W)
        return self.out_proj(fused)
'''
NAFBlock

rewriting it (as NVIDIA style)

- SimpleGate
- LayerNorm
- NAFBlock

From Residual Block (NVIDIA)
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.


Basic Architecture

1. LN (Layer Norm)
2. MBConv
3. + X
4. LN (Layer Norm)
5. FFN
6. + Z
7. Simple Gate
'''
class NAFBlock(nn.Module):
    """
    1. LN → 2. MBConv → 3. +X
    4. LN → 5. FFN → 6. +Z
    7. SimpleGate
    """
    def __init__(
        self,
        channels,
        dropout=0.0,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        assert channels % 2 == 0, "channels must be divisible by 2 for SimpleGate"
        self.use_checkpoint = use_checkpoint

        # 1) first normalization + MBConv
        self.norm1 = normalization(channels)
        self.mbconv = MBConv(channels, channels, dims=dims, dropout=dropout)

        # 2) second normalization + FFN
        self.norm2 = normalization(channels)
        self.ffn = nn.Sequential(
            conv_nd(dims, channels, channels * 2, 1),
            SimpleGate(),
            conv_nd(dims, channels, channels, 1),
            nn.Dropout(p=dropout),
        )

        # residual weights
        self.beta = nn.Parameter(th.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(th.zeros(1, channels, 1, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        # Stage 1: LN → MBConv → +X
        h = self.norm1(x)
        h = self.mbconv(h)
        x1 = x + self.beta * h

        # Stage 2: LN → FFN → +Z
        h2 = self.norm2(x1)
        h2 = self.ffn(h2)
        x2 = x1 + self.gamma * h2

        # Stage 3: final gating
        return SimpleGate()(x2)



class NAFUNetModel(nn.Module):
    def __init__(
        self,
        in_channels_opt,
        in_channels_sar,
        out_channels,
        model_channels=22,
        channel_mult=(1, 2, 4, 8),
        num_naf_blocks=1,
        num_heads_per_level=(1, 1, 2, 4),
        dropout=0.0,
        dims=2,
        use_checkpoint=False,
        conv_resample=True,
        use_fp16=False,
    ):
        super().__init__()
        self.num_levels = len(channel_mult)
        self.channel_list = [model_channels * m for m in channel_mult]
        self.dtype = th.float16 if use_fp16 else th.float32

        # 1. Pre-embedding
        self.opt_embed = conv_nd(dims, in_channels_opt, self.channel_list[0], 3, padding=1)
        self.sar_embed = conv_nd(dims, in_channels_sar, self.channel_list[0], 3, padding=1)

        # 2. Encoders + Fusion
        self.encoder_opt = nn.ModuleList()
        self.encoder_sar = nn.ModuleList()
        self.fusion_blocks = nn.ModuleList()
        self.downsamples_opt = nn.ModuleList()
        self.downsamples_sar = nn.ModuleList()

        ch = self.channel_list[0]
        for lvl in range(self.num_levels):
            out_ch = self.channel_list[lvl]

            # each level: stack of NAFBlocks
            self.encoder_opt.append(nn.Sequential(*[
                NAFBlock(ch, dropout, dims, use_checkpoint)
                for _ in range(num_naf_blocks)
            ]))
            self.encoder_sar.append(nn.Sequential(*[
                NAFBlock(ch, dropout, dims, use_checkpoint)
                for _ in range(num_naf_blocks)
            ]))

            # fusion SFBlock
            self.fusion_blocks.append(SFBlock(
                channels=out_ch,
                num_heads=num_heads_per_level[lvl],
                dims=dims
            ))

            # downsample if not last
            if lvl != self.num_levels - 1:
                self.downsamples_opt.append(
                    Downsample(out_ch, use_conv=conv_resample, dims=dims)
                )
                self.downsamples_sar.append(
                    Downsample(out_ch, use_conv=conv_resample, dims=dims)
                )

            ch = out_ch

        # 3. Middle block
        self.middle_block = NAFBlock(ch, dropout, dims, use_checkpoint)

        # 4. Decoder (Opt only)
        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for lvl in reversed(range(self.num_levels)):
            out_ch = self.channel_list[lvl]
            # concatenated channels = current ch + skip ch
            self.decoder.append(NAFBlock(ch + out_ch, dropout, dims, use_checkpoint))
            ch = out_ch
            if lvl != 0:
                self.upsamples.append(
                    Upsample(ch, use_conv=conv_resample, dims=dims)
                )

        # 5. Output
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1))
        )

    def forward(self, opt, sar):
        h_opt = self.opt_embed(opt)
        h_sar = self.sar_embed(sar)

        skips = []
        # Encoder + Fusion
        for i in range(self.num_levels):
            h_opt = self.encoder_opt[i](h_opt)
            h_sar = self.encoder_sar[i](h_sar)
            h_opt = self.fusion_blocks[i](h_opt, h_sar)
            skips.append(h_opt)
            if i != self.num_levels - 1:
                h_opt = self.downsamples_opt[i](h_opt)
                h_sar = self.downsamples_sar[i](h_sar)

        # Middle
        h = self.middle_block(h_opt)

        # Decoder + skip
        for idx, dec_block in enumerate(self.decoder):
            skip = skips.pop()
            h = th.cat([h, skip], dim=1)
            h = dec_block(h)
            if idx < len(self.upsamples):
                h = self.upsamples[idx](h)

        return self.out(h)