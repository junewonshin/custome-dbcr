from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    LayerNorm2d,
)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    

class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
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
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            LayerNorm2d(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

    def _get_module_type(self):
        return 'ResBlock'


class MBConv(nn.Module):
    """
    MBConv block with expansion factor
    """
    # TODO: expansion_factor=4
    def __init__(self, in_channels, out_channels, expansion_factor=4, dims=2):
        super().__init__()

        self.expanded_channels = in_channels * expansion_factor
        self.start_norm = LayerNorm2d(in_channels)
        
        # Expansion
        self.expand = nn.Sequential(
            conv_nd(dims, in_channels, self.expanded_channels, 1, bias=False),
            LayerNorm2d(self.expanded_channels),
            nn.SiLU()
        )
        
        # Depthwise
        self.depthwise = nn.Sequential(
            conv_nd(dims, self.expanded_channels, self.expanded_channels, 3, padding=1, groups=self.expanded_channels, bias=False),
            LayerNorm2d(self.expanded_channels),
            nn.SiLU()
        )
        
        # Projection
        self.project = conv_nd(dims, self.expanded_channels, out_channels, 1, bias=False)
        
        self.final_norm = LayerNorm2d(out_channels)

    def forward(self, x):
        x = self.start_norm(x)

        h = self.expand(x)
        h = self.depthwise(h)
        h = self.project(h)        
        return self.final_norm(h + x)


class NAFBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.norm1 = LayerNorm2d(channels, affine=False)
        self.norm2 = LayerNorm2d(channels)

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        self.mbconv = MBConv(channels, self.out_channels, dims=dims)
        self.ffn_norm = LayerNorm2d(self.out_channels)
        
        self.ffn_conv1 = conv_nd(dims, self.out_channels, self.out_channels*2, 1)
        self.ffn_conv2 = conv_nd(dims, self.out_channels, self.out_channels, 1)
        self.gate = SimpleGate()


    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        h = self.norm1(x)
        emb_out = self.emb_layers(emb).type(h.dtype)

        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = h * (1 + scale) + shift
        else:
            h = h + emb_out

        h = self.mbconv(self.norm2(h)) + h
        h_ffn = self.ffn_conv1(self.ffn_norm(h))
        h_ffn = self.gate(h_ffn)
        h = self.ffn_conv2(h_ffn) + h
        return h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = LayerNorm2d(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        self.use_checkpoint = use_checkpoint
        
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels

        self.norm = normalization(channels)
        self.q_proj = conv_nd(1, channels, channels, 1)
        self.k_proj = conv_nd(1, channels, channels, 1)
        self.v_proj = conv_nd(1, channels, channels, 1)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)

    def forward(self, x, sar):
        """
        :param x: OPT (query) [B, channels, *spatial]
        :param sar: SAR (key, value) [B, channels, *spatial]
        :return: attention
        """
        # assert x.shape[1] == self.channels, f"OPT channels {x.shape[1]} != expected {self.channels}"
        # assert sar.shape[1] == self.channels, f"SAR channels {sar.shape[1]} != expected {self.channels}"
        
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        sar = sar.reshape(b, c, -1)
        
        q = self.q_proj(self.norm(x))  # [B, channels, T]
        k = self.k_proj(sar)           # [B, channels, T]
        v = self.v_proj(sar)           # [B, channels, T]

        qkv = torch.cat([q, k, v], dim=1)  # [B, 3*channels, T]
        
        h = self.attention(qkv)
        
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        q, k, v = q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1)
        a = F.scaled_dot_product_attention(q, k, v)
        return a.transpose(-2, -1).reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Fallback from Blocksparse if use_fp16=False
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        q, k, v = (
            q.reshape(bs * self.n_heads, ch, length),
            k.reshape(bs * self.n_heads, ch, length),
            v.reshape(bs * self.n_heads, ch, length),
        )
        q, k, v = q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1)
        a = F.scaled_dot_product_attention(q, k, v)
        return a.transpose(-2, -1).reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class SFFusion(nn.Module):
    """
    SAR-Optical Fusion Block
    SAR: K, V
    Optical: Q
    Cross Attention based fusion with C*C attention matrix per head
    Head outputs are concatenated and projected to input channels
    """
    def __init__(self, channels, num_heads=8, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.head_dim = channels // num_heads
        assert self.head_dim * num_heads == channels, "channels must be divisible by num_heads"
           
        self.opt_norm = LayerNorm2d(channels)
        self.sar_norm = LayerNorm2d(channels)
        
        self.q_proj = conv_nd(2, channels, channels, 1)
        self.k_proj = conv_nd(2, channels, channels, 1)
        self.v_proj = conv_nd(2, channels, channels, 1)
        
        self.mlp = nn.Sequential(
                nn.Linear(self.channels, self.channels * 2),
                nn.GELU(),
                nn.Linear(self.channels * 2, self.channels)
            )
        
        self.output_conv = conv_nd(2, channels, channels, 1)


    def forward(self, x_opt, x_sar):
        if self.use_checkpoint:
            return checkpoint(self._forward, (x_opt, x_sar), self.parameters(), self.use_checkpoint)
        else:
            return self._forward(x_opt, x_sar)
    
    def _forward(self, x_opt, x_sar):
        B, C, H, W = x_opt.shape
        HW = H * W
        HN = self.num_heads
        HD = self.head_dim

        x_opt = self.opt_norm(x_opt) # B, C, H, W
        x_sar = self.sar_norm(x_sar) # B, C, H, W

        q = self.q_proj(x_opt)       # B, C, H, W
        k = self.k_proj(x_sar)       # B, C, H, W
        v = self.v_proj(x_sar)       # B, C, H, W

        # C = HN * HD, HN: Head Num, HW: Head Dim
        q = q.reshape(B, HN, HD, HW) # B, HN, HD, HW
        k = k.reshape(B, HN, HD, HW) # B, HN, HD, HW
        v = v.reshape(B, HN, HD, HW).transpose(-2, -1) # B, HN, HD, HW -> B, HN, HW, HD

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q.float(), k.transpose(-2,-1).float()) * (HD ** -0.5)  # B, HN, HD, HD
        attn = (attn - attn.max(dim=-1, keepdim=True)[0]).to(q.dtype)
        attn = F.softmax(attn, dim=-1)

        z = torch.matmul(v, attn)  # B, HN, HW, HD | B, HN, HD, HD
        z = z.permute(0, 2, 1, 3).reshape(B, HW, C) 
        
        z_opt = x_opt.reshape(B, C, HW).transpose(1, 2)  # (B, HW, C)
        z_sum = z + z_opt  # (B, HW, C)
        z_output = self.mlp(z_sum) + z_sum  # (B, HW, C)
        z_output = z_output.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)
        z_output = self.output_conv(z_output)

        return x_opt + z_output


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 1, 2, 2, 4, 4),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        condition_mode=None,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        self.condition_mode = condition_mode

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList()
        self.sar_input_blocks = nn.ModuleList()
        
        self.input_blocks.append(
            TimestepEmbedSequential(
                conv_nd(dims, in_channels, model_channels, 3, padding=1)
            )
        )

        self.sar_input_blocks.append(
            TimestepEmbedSequential(
                conv_nd(dims, 2, model_channels, 3, padding=1)
            )
        )
        
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ds = 1

        self.attention_blocks = nn.ModuleList()
        self.cross_attention_blocks = nn.ModuleList()

        ch = model_channels
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks):
                current_size = image_size // ds
                layers = [
                    NAFBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=out_ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                
                if current_size in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = out_ch // num_heads
                    else:
                        num_heads = out_ch // num_head_channels
                        dim_head = num_head_channels
                    
                    layers.append(
                        AttentionBlock(
                            out_ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )

                sar_layers = [
                    NAFBlock(
                        ch,
                        None,
                        dropout,
                        out_channels=out_ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.sar_input_blocks.append(TimestepEmbedSequential(*sar_layers))
                self._feature_size += out_ch

            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        NAFBlock(
                            out_ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(out_ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                self.sar_input_blocks.append(
                    TimestepEmbedSequential(
                        NAFBlock(
                            out_ch,
                            None,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(out_ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ds *= 2
                self._feature_size += out_ch
            
            ch = out_ch

        self.middle_block = TimestepEmbedSequential(
            NAFBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            NAFBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    NAFBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        NAFBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, xT=None, y=None, sar=None):
        """
        Apply the model to an input batch.
        """
        if self.condition_mode == "concat":
            x = torch.cat([x, xT], dim=1)

        timesteps = timesteps.to(self.dtype)
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None and y is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h_opt = x.to(self.dtype)
        h_sar = sar.to(self.dtype) if sar is not None else None
        hs = []

        current_size = h_opt.shape[-1]
        for i in range(len(self.input_blocks)):
            if current_size in self.attention_resolutions:
                if h_sar is not None:
                    h_opt = self.input_blocks[i][0](h_opt, emb)  # NAFBlock
                    h_opt = CrossAttentionBlock(
                        h_opt.shape[1],
                        use_checkpoint=self.use_checkpoint,
                        num_heads=self.num_heads,
                        num_head_channels=self.num_head_channels,
                        use_new_attention_order=self.use_new_attention_order,
                    )(h_opt, h_sar)
                else:       
                    h_opt = self.input_blocks[i](h_opt, emb)
            else:
                h_opt = self.input_blocks[i](h_opt, emb)
            
            if h_sar is not None and i < len(self.sar_input_blocks):
                h_sar = self.sar_input_blocks[i](h_sar)

            if isinstance(self.input_blocks[i], TimestepEmbedSequential) and \
               any(isinstance(m, Downsample) for m in self.input_blocks[i]):
                current_size = current_size // 2

            hs.append(h_opt)

        h = self.middle_block(h_opt, emb)
            
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        h = self.out(h)
        return h.to(x.dtype)


class NAFUNetModel(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        num_classes=None,
        use_checkpoint=False,
        num_heads=8,  
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        use_fp16=False,
        resblock_updown=False,
        use_new_attention_order=False,
        condition_mode=None,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.num_classes = None
        self.channel_mult = channel_mult
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_fp16 = use_fp16
        self.resblock_updown = resblock_updown
        self.use_new_attention_order = use_new_attention_order
        self.condition_mode = condition_mode
        self.dtype = torch.float16 if use_fp16 else torch.float32

        #TODO: work13 = 4, work2 = 2
        self.time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, self.time_embed_dim),
            nn.SiLU(),
            linear(self.time_embed_dim, self.time_embed_dim),
        )

        # if self.num_classes is not None:
        #     self.label_emb = nn.Embedding(num_classes, self.time_embed_dim).to(self.dtype)
        
        # TODO: workdir2, work_13
        self.opt_input_conv = TimestepEmbedSequential(conv_nd(2, in_channels, model_channels, kernel_size=3, padding=1, bias=False))  # 13 -> 22
        self.sar_input_conv = TimestepEmbedSequential(conv_nd(2, 2, model_channels, kernel_size=3, padding=1, bias=False))  # 2 -> 22
        # TODO: 
        # self.opt_input_conv = TimestepEmbedSequential(conv_nd(2, in_channels, model_channels, kernel_size=1))  # 13 -> 22
        # self.sar_input_conv = TimestepEmbedSequential(conv_nd(2, 2, model_channels, kernel_size=1))  # 2 -> 22

        self.encoder_res_blocks = [1, 1, 1, 28]
        self.decoder_res_blocks = [1, 1, 1, 1]

        self.opt_input_blocks = nn.ModuleList()
        self.sar_input_blocks = nn.ModuleList()
        
        self.opt_down_blocks = nn.ModuleList()
        self.sar_down_blocks = nn.ModuleList()
        
        self.fusion_blocks = nn.ModuleList()
        self.middle_blocks = nn.ModuleList()

        self.output_blocks = nn.ModuleList()
        self.output_upsample = nn.ModuleList()
        self.fusion_heads = [1, 1, 2, 4]  
        
        ch = model_channels
        for level, num_blocks in enumerate(self.encoder_res_blocks):
            for _ in range(num_blocks):
                self.opt_input_blocks.append(
                    TimestepEmbedSequential(
                        NAFBlock(ch, self.time_embed_dim, dropout, use_checkpoint=use_checkpoint)
                    )
                )
                self.sar_input_blocks.append(
                    TimestepEmbedSequential(
                        NAFBlock(ch, self.time_embed_dim, dropout, use_checkpoint=use_checkpoint)
                    )
                )

            self.fusion_blocks.append(
                SFFusion(ch, num_heads=self.fusion_heads[level], use_checkpoint=use_checkpoint)
            )

            self.opt_down_blocks.append(
                TimestepEmbedSequential(
                    conv_nd(2, ch, 2*ch, kernel_size=2, stride=2)
                )
            )
            if level != len(self.encoder_res_blocks) - 1:   
                self.sar_down_blocks.append(
                    TimestepEmbedSequential(
                        conv_nd(2, ch, 2*ch, kernel_size=2, stride=2)
                    )
                )
            ch = ch * 2

        self.middle_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                NAFBlock(ch, self.time_embed_dim, dropout, use_checkpoint=use_checkpoint)
            )
            for _ in range(4)
        ])

        for level, num_blocks in enumerate(self.decoder_res_blocks):
            
            self.output_upsample.append(
                TimestepEmbedSequential(
                    nn.Sequential(
                        conv_nd(2, ch, ch*2, 1, bias=False),
                        nn.PixelShuffle(2)
                    )
                )   
            )
            ch = ch // 2
            for _ in range(num_blocks):
                self.output_blocks.append(
                    TimestepEmbedSequential(    
                        NAFBlock(ch, self.time_embed_dim, dropout, use_checkpoint=use_checkpoint)
                    )
                )

        self.out = conv_nd(2, ch, out_channels, kernel_size=1)
        

    def forward(self, x, timesteps, xT=None, sar=None, y=None):
        if self.condition_mode == "concat":
            x = torch.cat([x, xT], dim=1)

        timesteps = timesteps.to(self.dtype)
        emb = self.time_embed(timestep_embedding(timesteps.float(), self.model_channels))

        # if self.num_classes is not None and y is not None:
        #     assert y.shape == (x.shape[0],)
        #     emb = emb + self.label_emb(y)
        
        emb = emb.to(self.dtype)
        x = x.to(self.dtype)
        sar = sar.to(self.dtype)

        h_opt = self.opt_input_conv(x, emb)
        h_sar = self.sar_input_conv(sar, emb)

        for level, num_blocks in enumerate(self.encoder_res_blocks):
            block_start_idx = sum(self.encoder_res_blocks[:level])
            for block_idx in range(num_blocks):
                current_idx = block_start_idx + block_idx
                h_opt = self.opt_input_blocks[current_idx](h_opt, emb)
                h_sar = self.sar_input_blocks[current_idx](h_sar, emb)

            h_opt = self.fusion_blocks[level](h_opt, h_sar)

            h_opt = self.opt_down_blocks[level](h_opt, emb)
            if level != len(self.encoder_res_blocks) - 1:
                h_sar = self.sar_down_blocks[level](h_sar, emb)

        h = self.middle_blocks[0](h_opt, emb)
        for module in self.middle_blocks[1:]:
            h = module(h, emb)

        for level in range(len(self.decoder_res_blocks)):
            h = self.output_upsample[level](h, emb)
            h = self.output_blocks[level](h, emb)

        h = self.out(h)
        h = h.to(x.dtype)
        return h


# if __name__ == "__main__":
#     from torchinfo import summary
#     batch_size = 1
#     channels = 13
#     height = 256
#     width = 256
    
#     model = NAFUNetModel(
#         image_size=256,
#         in_channels=channels,
#         model_channels=22, 
#         out_channels=channels,
#         num_res_blocks=1,
#         attention_resolutions=(8, 16, 32),
#         dropout=0.0,
#         channel_mult=(1, 2, 4, 8),
#         num_heads=(1, 1, 2, 4),
#         use_scale_shift_norm=True,
#         resblock_updown=True,
#         use_fp16=False,
#         use_new_attention_order=False,
#         use_checkpoint=False,
#     )
#     x_opt = torch.randn(batch_size, channels, height, width)
#     x_sar = torch.randn(batch_size, 2, height, width)
#     timesteps = torch.randint(0, 1000, (batch_size,))
#     dummy_tensor = torch.zeros(1)
#     summary(model, 
#             input_data=[x_opt, timesteps, dummy_tensor, x_sar, dummy_tensor],
#             device='cpu',
#             depth=4,
#             col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
#             row_settings=["var_names"])
    
#     with torch.no_grad():
#         output = model(x_opt, timesteps, dummy_tensor, x_sar, dummy_tensor)
#         print(output.min(), output.max())
