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
    normalization,
    linear,
    SiLU,
    timestep_embedding,
    LayerNorm
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
        pooled = self.pool(x)
        return x * self.fc(pooled.to(x.dtype))


class MBConv(nn.Module):

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

        # expansion
        self.expand = conv_nd(dims, in_channels, hidden, 1)
        # depthwise
        self.dw = conv_nd(dims, hidden, hidden, 3, padding=1, groups=hidden)
        # channel‑attention
        self.se = SqueezeExcite(hidden, reduction, dims)
        # projection
        self.project = conv_nd(dims, hidden, out_channels, 1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        dtype = x.dtype
        # keep original for residual
        x_in = x

        # Pointwise expansion
        h = self.expand(x).to(dtype)
        h = self.act(h)

        # Depthwise
        h = self.dw(h).to(dtype)
        h = self.act(h)

        # SE
        h = self.se(h).to(dtype)

        # Projection
        h = self.project(h).to(dtype)
        h = self.dropout(h)

        return x_in + h if self.use_residual else h


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
    Channel-wise cross-attention fusion block.
    Q from optical branch, K/V from SAR branch.
    Computes C×C attention over channels.
    """
    def __init__(self, channels, num_heads=4, mlp_ratio=2, dims=2):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = (channels * (dims == 2 and 1 or 1))  # we'll split features below
        # 1) Norm + 1×1 conv
        self.norm_q  = normalization(channels)
        self.q_proj  = conv_nd(dims, channels, channels, 1)
        self.norm_kv = normalization(channels)
        self.k_proj  = conv_nd(dims, channels, channels, 1)
        self.v_proj  = conv_nd(dims, channels, channels, 1)
        # 2) MLP on channel embeddings
        hidden = channels * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        # 3) Final 1×1 conv
        self.out_proj = zero_module(conv_nd(dims, channels, channels, 1))

    def forward(self, feat_opt, feat_sar):
        B, C, H, W = feat_opt.shape
        HW = H * W

        # 1) Norm + conv1×1
        q = self.norm_q(feat_opt).to(feat_opt.dtype)
        k = self.norm_kv(feat_sar).to(feat_opt.dtype)
        v = self.norm_kv(feat_sar).to(feat_opt.dtype)

        q = self.q_proj(q)    # [B, C, H, W]
        k = self.k_proj(k)    # [B, C, H, W]
        v = self.v_proj(v)    # [B, C, H, W]

        # 2) Flatten spatial dims → feature dim
        #    shape becomes [B, C, HW]
        q = q.reshape(B, C, HW)
        k = k.reshape(B, C, HW)
        v = v.reshape(B, C, HW)

        # 3) Multi-head channel attention
        #    Sequence length = C, embed dim = HW // num_heads
        head_dim = HW // self.num_heads
        assert head_dim * self.num_heads == HW, "HW must be divisible by num_heads"

        # Split heads on feature axis
        # q_h: [B, num_heads, C, head_dim]
        q_h = q.reshape(B, C, self.num_heads, head_dim) \
               .permute(0, 2, 1, 3)
        k_h = k.reshape(B, C, self.num_heads, head_dim) \
               .permute(0, 2, 1, 3)
        v_h = v.reshape(B, C, self.num_heads, head_dim) \
               .permute(0, 2, 1, 3)

        # Scaled dot-product over channels
        scale = head_dim ** -0.5
        # attn: [B, num_heads, C, C]
        attn = (q_h @ k_h.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        # apply to V
        # out_h: [B, num_heads, C, head_dim]
        out_h = attn @ v_h

        # 4) Merge heads → [B, C, HW]
        out = out_h.permute(0, 2, 1, 3)   # [B, C, num_heads, head_dim]
        out = out.reshape(B, C, HW)

        # 5) MLP over channel embeddings
        #    We treat each of the HW positions independently:
        #    out.permute -> [B*HW, C], MLP, then back
        out = out.permute(0, 2, 1).reshape(B*HW, C)  # [B*HW, C]
        out = self.mlp(out)                          # [B*HW, C]
        out = out.reshape(B, HW, C).permute(0, 2, 1)  # [B, C, HW]

        # 6) reshape to (B,C,H,W), residual + final conv
        out = out.reshape(B, C, H, W) + feat_opt
        return self.out_proj(out)
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
    1. LN → MBConv → +X
    2. LN → FFN (with SimpleGate) → +Z
    """
    def __init__(
        self,
        channels,
        dropout=0.0,
        dims=2,
        use_checkpoint=False,
        emb_channels=None,
        use_scale_shift_norm=True,
    ):
        super().__init__()
        assert channels % 2 == 0, "channels must be divisible by 2 for SimpleGate"
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        # 1) first normalization + MBConv
        self.norm1 = normalization(channels)
        self.mbconv = MBConv(channels, channels, dims=dims, dropout=dropout)

        # 2) second normalization + FFN w/ SimpleGate baked in
        self.norm2 = normalization(channels)
        hidden = channels * 2
        # In Paper, Apply SimpleGate after ffn ... ?
        self.ffn = nn.Sequential(
            # expand to 2×channels
            conv_nd(dims, channels, hidden, 3, padding=1),
            conv_nd(dims, hidden, hidden, 3, padding=1),
            SimpleGate(),
        )

        # optional scale‑&‑shift time embedding
        if emb_channels is not None:
            out_dim = 2 * channels if use_scale_shift_norm else channels
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(emb_channels, out_dim), # 22 -> 44, 44 -> 88, 88 -> 176
            )
        else:
            self.emb_layers = None

    def forward(self, x, emb=None):
        # gradient checkpointing if desired
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):

        h = self.norm1(x).to(x.dtype)
        h = self.mbconv(h)
        x1 = x + h

        h2 = self.norm2(x1).to(x.dtype)

        if self.emb_layers is not None and emb is not None:
            emb_out = self.emb_layers(emb.to(x.dtype))
            if self.use_scale_shift_norm:
                scale, shift = emb_out.chunk(2, dim=1)
                h2 = h2 * (1 + scale[..., None, None]) + shift[..., None, None]
            else:
                h2 = h2 + emb_out[..., None, None]

        h2 = self.ffn(h2)
        return x1 + h2


class NAFUNetModel(nn.Module):
    def __init__(
        self,
        in_channels=13,    # 13 -> 3 
        sar_channels=2,
        out_channels=13,
        model_channels=22,
        channel_mult=(1,2,4,8),
        num_naf_blocks=1,
        num_heads_per_level=(1,1,2,4),
        dropout=0.0,
        dims=2,
        use_checkpoint=False,
        conv_resample=True,
        use_fp16=False,
    ):
        super().__init__()
        self.dtype = th.float16 if use_fp16 else th.float32

        # time‑embed dimension, 22
        self.emb_channels = model_channels

        # 3x3 input embeddings
        self.opt_embed = conv_nd(dims, in_channels, model_channels, 3, padding=1)
        self.sar_embed = conv_nd(dims, sar_channels, model_channels, 3, padding=1)

        # compute channels at each level
        self.channel_list = [model_channels * m for m in channel_mult]
        self.num_levels   = len(self.channel_list)

        # 1) encoders and fusion modules
        self.encoder_opt      = nn.ModuleList()
        self.encoder_sar      = nn.ModuleList()
        self.fusion_blocks    = nn.ModuleList()
        self.downsamples_opt  = nn.ModuleList()
        self.downsamples_sar  = nn.ModuleList()

        for lvl, ch in enumerate(self.channel_list):
            # stack of NAFBlocks
            self.encoder_opt.append(nn.Sequential(*[
                NAFBlock(ch, dropout, dims, use_checkpoint, emb_channels=self.emb_channels)
                for _ in range(num_naf_blocks)
            ]))
            self.encoder_sar.append(nn.Sequential(*[
                NAFBlock(ch, dropout, dims, use_checkpoint, emb_channels=self.emb_channels)
                for _ in range(num_naf_blocks)
            ]))

            # fusion SFBlock defined elsewhere
            self.fusion_blocks.append(
                SFBlock(ch, num_heads=num_heads_per_level[lvl], dims=dims)
            )

            # downsample except last
            if lvl < self.num_levels-1:
                self.downsamples_opt.append(
                    Downsample(ch, use_conv=conv_resample, dims=dims)
                )
                self.downsamples_sar.append(
                    Downsample(ch, use_conv=conv_resample, dims=dims)
                )

        # 2) middle
        mid_ch = self.channel_list[-1]
        self.middle_block = NAFBlock(mid_ch, dropout, dims, use_checkpoint, emb_channels=self.emb_channels)

        # 3) decoder (optical only)
        self.decoder   = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for lvl, ch in enumerate(reversed(self.channel_list)):
            self.decoder.append(
                NAFBlock(ch, dropout, dims, use_checkpoint, emb_channels=self.emb_channels)
            )
            if lvl < self.num_levels-1:
                self.upsamples.append(
                    Upsample(ch, use_conv=conv_resample, dims=dims)
                )

        # 4) final output conv
        self.out = zero_module(conv_nd(dims, model_channels, out_channels, 1))

        # convert to fp16 if requested
        if use_fp16:
            self.convert_to_fp16()

    def convert_to_fp16(self):
        for m in self.modules():
            if isinstance(m, LayerNorm):
                m.float()
            else:
                m.half()

    def convert_to_fp32(self):
        for m in self.modules():
            m.float()

    def forward(self, x, t, opt, sar):
        # x: noisy cloudy input (unused here, we start from y directly)
        # t: time steps, opt: optical cloudy, sar: SAR
        dtype = self.dtype

        # Dataset
        # debug_stats("opt (in)", opt)
        # debug_stats("sar (in)", sar)

        # debug_stats("opt_embed.W", self.opt_embed.weight)
        # if self.opt_embed.bias is not None:
        #     debug_stats("opt_embed.b", self.opt_embed.bias)
        # debug_stats("sar_embed.W", self.sar_embed.weight)
        # if self.sar_embed.bias is not None:
        #     debug_stats("sar_embed.b", self.sar_embed.bias)

        # embed time once
        t_emb = timestep_embedding(t.to(dtype), dim=self.emb_channels).to(dtype)
        # debug_stats("t_emb", t_emb)

        # embed inputs
        try:
            h_opt = self.opt_embed(opt.to(dtype))
        except Exception as e:
            # print("[ERROR] opt_embed conv failed:", e)
            raise
        # debug_stats("h_opt after emb", h_opt)

        try:
            h_sar = self.sar_embed(sar.to(dtype))
        except Exception as e:
            # print("[ERROR] sar_embed conv failed:", e)
            raise
        # debug_stats("h_sar after emb", h_sar)

        # encoder + fusion
        for lvl in range(self.num_levels):
            # modality‑specific NAFBlocks (with time emb)
            for blk in self.encoder_opt[lvl]:
                h_opt = blk(h_opt, t_emb)
            for blk in self.encoder_sar[lvl]:
                h_sar = blk(h_sar, t_emb)

            # cross‑modal fusion
            h_opt = self.fusion_blocks[lvl](h_opt, h_sar)
            # debug_stats(f"h_opt after fusion{lvl}", h_opt)

            # downsample if not last
            if lvl < self.num_levels-1:
                h_opt = self.downsamples_opt[lvl](h_opt)
                h_sar = self.downsamples_sar[lvl](h_sar)
        # middle
        h = self.middle_block(h_opt, t_emb)
        # decoder (only optical path)
        for i, dec in enumerate(self.decoder):
            h = dec(h, t_emb)
            if i < len(self.upsamples):
                h = self.upsamples[i](h)
        # debug_stats("pre-out h", h)

        out = self.out(h)
        # debug_stats("final out", out)

        # final projection
        return out



# def debug_stats(name, x):
#     """ Tensor x 의 shape, dtype, min/max, NaN/Inf 여부를 출력 """
#     with th.no_grad():
#         x_cpu = x.detach().float().cpu()
#         mn = x_cpu.min().item()
#         mx = x_cpu.max().item()
#         has_nan = th.isnan(x_cpu).any().item()
#         has_inf = th.isinf(x_cpu).any().item()
#     # tuple(x.shape) → 문자열로 바꿔서 출력
#     shape_str = str(tuple(x.shape))
#     print(f"[DEBUG] {name:12s} shape={shape_str:15s} dtype={str(x.dtype):7s}"
#           f" min/max=({mn:.6g},{mx:.6g}) nan={has_nan} inf={has_inf}")