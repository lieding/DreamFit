# Copyright (c) 2022 CompVis team. 
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. 
# SPDX-License-Identifier: CreativeML Open RAIL M License OR Apache-2.0.
#
# This file has been modified by Bytedance Ltd. and/or its affiliates on 2025/5/6.
#
# Original file was released under CreativeML Open RAIL M License, with the full license text
# available at https://github.com/CompVis/stable-diffusion/blob/main/LICENSE.
#
# This modified file is released under the same license.



import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.stablediffusion.modules.distributions import DiagonalGaussianDistribution
import itertools
import math
import einops


def nonlinearity(x):
    # swish
    return F.silu(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: bool = None,
            conv_shortcut: bool = False,
            dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels,
                                               out_channels,
                                               3, 1, 1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels,
                                              out_channels,
                                              1, 1, 0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)

        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape

        q = einops.rearrange(q, "b c h w -> b 1 (h w) c")
        k = einops.rearrange(k, "b c h w -> b 1 (h w) c")
        v = einops.rearrange(v, "b c h w -> b 1 (h w) c")

        h_ = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        h_ = einops.rearrange(h_, "b 1 (h w) c -> b c h w", h=h)

        h_ = self.proj_out(h_)

        return x+h_


class Encoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            ch: int,
            ch_mult=(1, 2, 4, 8),
            num_res_blocks=2,
            dropout=0.0,
            resamp_with_conv=True,
            double_z=True):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(block_in,
                                         block_out,
                                         dropout=dropout))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)

        # end
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6)
        self.conv_out = nn.Conv2d(block_in, conv_out_channels, 3, 1, 1)

    def forward(self, x):

        # downsampling
        h = self.conv_in(x)

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
            if i_level != self.num_resolutions-1:
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)


        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h


class Decoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            ch: int,
            ch_mult=(1, 2, 4, 8),
            num_res_blocks=2,
            dropout=0.0,
            resamp_with_conv=True,
            tanh_out=False):
        super().__init__()
        self.in_channels = in_channels
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.tanh_out = tanh_out

        block_in = ch*ch_mult[self.num_resolutions-1]
        # z to block_in
        self.conv_in = torch.nn.Conv2d(in_channels, block_in, 3, 1, 1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(block_in, block_out, dropout=dropout))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6)
        self.conv_out = nn.Conv2d(block_in, out_channels, 3, 1, 1)

    def forward(self, z):

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class AutoencoderKL(nn.Module):
    def __init__(
            self,
            in_channels,
            z_channels,
            ch,
            ch_mult,
            num_res_blocks,
            embed_dim,
            dropout: float = 0.0,
            double_z: bool = True,
            scale_factor: float = 1.0) -> None:
        super().__init__()

        self.encoder = Encoder(
            in_channels,
            z_channels,
            ch,
            ch_mult,
            num_res_blocks,
            dropout,
            double_z=double_z
        )
        self.decoder = Decoder(
            z_channels,
            in_channels,
            ch,
            ch_mult,
            num_res_blocks,
            dropout
        )
        self.quant_conv = nn.Conv2d(2*z_channels, 2*embed_dim, 1, 1, 0)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1, 1, 0)
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor
        self.feature_stride = 2 ** (len(ch_mult)-1)

    def freeze_encoder(self, mode=True):
        for param in self.encoder.parameters():
            param.requires_grad = not mode
        for param in self.quant_conv.parameters():
            param.requires_grad = not mode

    def freeze(self, mode=True):
        self.freeze_encoder(mode)
        for param in self.decoder_parameters():
            param.requires_grad = not mode

    def decoder_parameters(self):
        return itertools.chain(
            self.decoder.parameters(), self.post_quant_conv.parameters())

    def encode(self, x):
        h = self.encoder(x) # x:
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        rec = self.decode(z)
        return rec, posterior
