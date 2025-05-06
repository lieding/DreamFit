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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel

from src.utils.util import checkpoint, zero_module
from src.stablediffusion.lora import LoRACompatibleConv, LoRACompatibleLinear, LoRAConv2dLayer, LoRALinearLayer

from omegaconf import ListConfig
from einops import rearrange, repeat
from inspect import isfunction
import os
from typing import Optional, Any
import math

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def get_n_heads_d_head(dim, n_heads, d_head):
    if n_heads <= 0:
        return dim // d_head, d_head
    else:
        return n_heads, dim//n_heads

def timestep_embedding(
        timesteps: torch.Tensor,
        dim: int,
        downscale_freq_shift: float = 0.,
        scale: float = 1.0,
        max_period: int = 10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half_dim = dim // 2

    exponent = - math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )

    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)

    emb = timesteps[:, None].float() * emb[None]
    emb = scale * emb

    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)

    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0), 'constant', 0.)
    return emb

def build_down_blocks(
        in_channels: int = 4,
        ch: int = 320,
        ch_mult=[1, 2, 4, 4],
        down_attn=[True, True, True, False],
        down_self_attn=[True, True, True, True],
        n_heads: int = -1,
        d_head: int = 64,
        transformer_depth: int = 1,
        context_dim: int = 1024,
        embed_dim: int = 1280,
        dropout: float = 0.0,
        num_res_blocks: int = 2,
        conv_resample: bool = True,
        use_checkpoint: bool = False,
        use_FiLM: bool = False):

    input_blocks = nn.ModuleList([
        TimestepAttentionBlock(
            nn.Conv2d(in_channels, ch, 3, 1, 1), block_index=-1)
    ])

    block_in = ch
    block_out = ch_mult[0] * ch

    # UNet down modules
    for level, mult in enumerate(ch_mult):
        use_attn = down_attn[level]
        disable_self_attn = not down_self_attn[level]

        is_final_block = level == len(ch_mult) - 1

        block_in = block_out
        block_out = ch * mult

        for i in range(num_res_blocks):
            layer_in = block_in if i == 0 else block_out
            layers = [
                ResBlock2D(
                    layer_in,
                    embed_dim,
                    dropout,
                    out_channels=block_out,
                    use_checkpoint=use_checkpoint,
                    use_FiLM=use_FiLM,
                )
            ]
            if use_attn:
                _n_heads, _d_head = get_n_heads_d_head(
                    block_out, n_heads, d_head)
                layers.append(
                    SpatialTransformer(
                        block_out,
                        _n_heads,
                        _d_head,
                        depth=transformer_depth,
                        context_dim=context_dim,
                        disable_self_attn=disable_self_attn,
                        use_checkpoint=use_checkpoint
                    )
                )
            input_blocks.append(
                TimestepAttentionBlock(*layers, block_index=level))

        if not is_final_block:
            input_blocks.append(
                TimestepAttentionBlock(
                    Downsample2D(block_out, conv_resample, block_out)
                )
            )

    return input_blocks


def build_mid_block(
        channels: int = 4,
        mid_self_attn: bool = True,
        n_heads: int = -1,
        d_head: int = 64,
        transformer_depth: int = 1,
        context_dim: int = 1024,
        embed_dim: int = 1280,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
        use_FiLM: bool = False,
        block_index: int = 5):

    _n_heads, _d_head = get_n_heads_d_head(channels, n_heads, d_head)
    middle_block = TimestepAttentionBlock(
        ResBlock2D(
            channels,
            embed_dim,
            dropout,
            use_checkpoint=use_checkpoint,
            use_FiLM=use_FiLM,
        ),
        SpatialTransformer(
            channels,
            _n_heads,
            _d_head,
            depth=transformer_depth,
            context_dim=context_dim,
            disable_self_attn=not mid_self_attn,
            use_checkpoint=use_checkpoint),
        ResBlock2D(
            channels,
            embed_dim,
            dropout,
            use_checkpoint=use_checkpoint,
            use_FiLM=use_FiLM
        ),
        block_index=block_index
    )

    return middle_block

class TimestepModule(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
        pass

class CaptionModule(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
        pass

class TimestepAttentionBlock(nn.Sequential):
    """
    The building block of Diffusion Unet, which takes pooled embedding and sequential context as input. 
    see Figure A.28, https://arxiv.org/pdf/2205.11487.pdf
    """

    def __init__(self, *args, block_index=-1, unet_ref_flag=False):
        super().__init__(*args)
        self._block_index = block_index
        self._use_context = False
        self.unet_ref_flag = unet_ref_flag
        self.in_channels = self[0].in_channels
        self.out_channels = self[-1].out_channels
        for layer in self:
            if isinstance(layer, SpatialTransformer):
                self._use_context = True

    @property
    def use_context(self):
        return self._use_context

    @property
    def block_index(self):
        """For a UNet model, the block-index --down--,mid,--up--
        """
        return self._block_index

    def forward(self, x, emb, context=None, ref_kv_pair_list_used_in_target_tsb=None, # ref_kv_pair_list_used_in_target_tsb:[[[kg1,vg1],[kg2,vg2]],[[kb1,vb1],[kb2,vb2]]]
                lora_scale=1.0, ip_scale=1.0, ip_hidden_states=None, ref_scale=1.0, return_attn_feature=False):
        self._ref_kv_pair_list = []
        for layer in self:
            if isinstance(layer, TimestepModule) or \
               (isinstance(layer, FullyShardedDataParallel) and 
                 isinstance(layer._fsdp_wrapped_module, TimestepModule)):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer) or \
               (isinstance(layer, FullyShardedDataParallel) and
                 isinstance(layer._fsdp_wrapped_module, SpatialTransformer)):
                if ref_kv_pair_list_used_in_target_tsb:
                    ref_kv_pair_used_in_target_asa = []
                    for ref_kv_pair_list_used_in_target_spt in ref_kv_pair_list_used_in_target_tsb: # spt->spatialtransformer
                        ref_kv_pair_used_in_target_asa.append(ref_kv_pair_list_used_in_target_spt.pop(0)) # ref_kv_pair_used_in_target_asa: [[kg1,vg1], [kb1,vb1]]
                    x = layer(x, context, ref_kv_pair_used_in_target_asa,\
                              lora_scale=lora_scale, ip_scale=ip_scale, ip_hidden_states=ip_hidden_states, ref_scale=ref_scale)
                else:
                    x = layer(x, context,\
                              lora_scale=lora_scale, ip_scale=ip_scale, ip_hidden_states=ip_hidden_states, ref_scale=ref_scale)
                
                if return_attn_feature:
                    attn_feature = x.clone()

                if self.unet_ref_flag:
                    self._ref_kv_pair_list += layer.ref_kv_pair_list
            else:
                x = layer(x)

        if return_attn_feature:
            return x, attn_feature
        else:
            return x

    @property
    def ref_kv_pair_list(self):
        if self.unet_ref_flag:
            return [self._ref_kv_pair_list]
        else:
            raise Exception('No ref_kv_pair_list since unet_ref_flag is False')


class Upsample2D(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied afterwards
    """

    def __init__(self, in_channels, use_conv=False, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        if use_conv:
            # self.conv = nn.Conv2d(in_channels, self.out_channels, 3, 1, 1)
            self.conv = LoRACompatibleConv(in_channels, self.out_channels, 3, 1, 1)
        else:
            assert self.in_channels == self.out_channels
            self.conv = nn.Identity()

    def forward(self, x):
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        if dtype == torch.bfloat16:
            x = x.to(dtype)
        x = self.conv(x)
        return x


class Downsample2D(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, in_channels, use_conv=False, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        if use_conv:
            self.op = LoRACompatibleConv(
                self.in_channels, self.out_channels, 3, 2, 1)
        else:
            assert self.in_channels == self.out_channels
            self.op = nn.AvgPool2d(2, 2)

    def forward(self, x):
        return self.op(x)


class ResBlock2D(TimestepModule):
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
        in_channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_FiLM=False,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_FiLM = use_FiLM

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, self.in_channels),
            nn.SiLU(),
            LoRACompatibleConv(self.in_channels, self.out_channels, 3, 1, 1)
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample2D(self.in_channels, False)
            self.x_upd = Upsample2D(self.in_channels, False)
        elif down:
            self.h_upd = Downsample2D(self.in_channels, False)
            self.x_upd = Downsample2D(self.in_channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            LoRACompatibleLinear(
                emb_channels,
                2 * self.out_channels if use_FiLM else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(LoRACompatibleConv(self.out_channels,
                        self.out_channels, 3, 1, 1))
        )

        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = LoRACompatibleConv(
                self.in_channels, self.out_channels, 3, 1, 1)
        else:
            self.skip_connection = LoRACompatibleConv(
                self.in_channels, self.out_channels, 1, 1, 0)


    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

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
        if self.use_FiLM:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        h = self.skip_connection(x) + h

        return h

class ResBlockText(CaptionModule):
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
        in_channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_FiLM=False,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_FiLM = use_FiLM

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, self.in_channels),
            nn.SiLU(),
            nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1)
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample2D(self.in_channels, False)
            self.x_upd = Upsample2D(self.in_channels, False)
        elif down:
            self.h_upd = Downsample2D(self.in_channels, False)
            self.x_upd = Downsample2D(self.in_channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_FiLM else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels,
                        self.out_channels, 3, 1, 1))
        )

        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(
                self.in_channels, self.out_channels, 3, 1, 1)
        else:
            self.skip_connection = nn.Conv2d(
                self.in_channels, self.out_channels, 1, 1, 0)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

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
        if self.use_FiLM:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h
    

class SpatialTransformer(nn.Module):
    """
    Transformer block for 2D image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False,
                 use_checkpoint=True, use_linear=True,
                 unet_ref_flag= False,
                 use_lora=False,
                 use_ip_adapter=False,
                 rank=128,use_adapter=True,):
        super().__init__()
        if exists(context_dim):
            if isinstance(context_dim, ListConfig):
                context_dim = list(context_dim)
            if not isinstance(context_dim, list):
                context_dim = [context_dim] * depth
        else:
            context_dim = [None] * depth

        assert len(
            context_dim) == depth, 'when context_dim is provided as a list, its length must match the transformer depth exactly'

        self.in_channels = in_channels
        self.out_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.use_linear = use_linear
        self.unet_ref_flag = unet_ref_flag
        self.use_adapter = use_adapter

        if not self.use_linear:
            self.proj_in = LoRACompatibleConv(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = LoRACompatibleLinear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim[d],
                disable_self_attn=disable_self_attn,
                checkpoint=use_checkpoint,
                unet_ref_flag=self.unet_ref_flag,
                use_lora=use_lora,
                use_ip_adapter=use_ip_adapter,
                rank=rank,
                use_adapter=use_adapter,)
            for d in range(depth)]
        )

        if not self.use_linear:
            self.proj_out = zero_module(LoRACompatibleConv(inner_dim,
                                                            in_channels,
                                                            kernel_size=1,
                                                            stride=1,
                                                            padding=0))
        else:
            self.proj_out = zero_module(LoRACompatibleLinear(in_channels, inner_dim))

    def forward(self, x, context=None, ref_kv_pair_used_in_target=None, lora_scale=1.0, ip_scale=1.0, ip_hidden_states=None, ref_scale=1.0):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)

        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                context_in = context[0]
            else:
                context_in = context[i]

            if ref_kv_pair_used_in_target:
                x = block(x, context=context_in, ref_kv_pair_used_in_target=ref_kv_pair_used_in_target, \
                          lora_scale=lora_scale, ip_scale=ip_scale, ip_hidden_states=ip_hidden_states, ref_scale=ref_scale)
            else:
                x = block(x, context=context_in,\
                          lora_scale=lora_scale, ip_scale=ip_scale, ip_hidden_states=ip_hidden_states, ref_scale=ref_scale)


        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)

        x = x + x_in

        return x

    @property
    def ref_kv_pair_list(self):
        if self.unet_ref_flag:
            ref_kv_pair_list = []
            for block in self.transformer_blocks:
                ref_kv_pair_list.append(block.ref_kv_pair)
            return ref_kv_pair_list
        else:
            raise('No kv_pair_list since it unet_ref_flag is False')

class UNet2DModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        ch: int = 320,
        ch_mult=[1, 2, 4, 4],
        down_attn=[True, True, True, False],
        down_self_attn=[True, True, True, True],
        mid_self_attn=True,
        up_attn=[False, True, True, True],
        up_self_attn=[True, True, True, True],
        num_res_blocks: int = 2,
        dropout=0.0,
        conv_resample=True,
        n_heads=-1,
        d_head=64,
        use_FiLM=False,
        resblock_updown=False,
        transformer_depth=1,
        down_transformer_depth=None,
        mid_transformer_depth=None,
        up_transformer_depth=None,
        context_dim=1024,
        embed_dim=1024,
        use_checkpoint=False,
        use_linear_in_transformer=True, # 2.1 True, 1.5 False
        use_extra_embedding=True,
        unet_ref_flag=False, 
        use_lora=False,
        use_ip_adapter=False,
        rank=128,
        use_adapter=True,
        ref_fuse_stage = ['mid', 'decoder'],
    ):
        super().__init__()

        # save the config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ch = ch
        self.ch_mult = ch_mult
        self.down_attn = down_attn
        self.down_self_attn = down_self_attn
        self.mid_self_attn = mid_self_attn
        self.up_attn = up_attn
        self.up_self_attn = up_self_attn
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.conv_resample = conv_resample
        self.use_FiLM = use_FiLM
        self.resblock_updown = resblock_updown
        self.transformer_depth = transformer_depth
        self.use_checkpoint = use_checkpoint
        self.use_extra_embedding = use_extra_embedding

        self.unet_ref_flag = unet_ref_flag
        self.ref_fuse_stage = ref_fuse_stage

        # cross-attention settings
        self.n_heads = n_heads
        self.d_head = d_head
        self.context_dim = context_dim
        self.embed_dim = embed_dim

        if use_lora: print('using lora')
        if use_ip_adapter: print('using ip-adapter')

        if not ((self.n_heads == -1) ^ (self.d_head == -1)):
            raise ValueError(
                'Please use either num_heads or num_head_channels, but not both')

        def get_n_heads_d_head(dim):
            if self.n_heads == -1:
                return dim // self.d_head, self.d_head
            else:
                return self.n_heads, dim//self.n_heads

        time_embed_dim = ch * 4
        self.time_embed = nn.Sequential(
            nn.Linear(ch, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if use_extra_embedding:
            if embed_dim > 0:
                self.embed_proj = nn.Sequential(
                    nn.Linear(embed_dim, time_embed_dim),
                    nn.SiLU(),
                    nn.Linear(time_embed_dim, time_embed_dim)
                )
        else:
            self.embed_proj = None

        self.input_blocks = nn.ModuleList([
            TimestepAttentionBlock(
                nn.Conv2d(in_channels, ch, 3, 1, 1), block_index=-1)
        ])

        block_in = ch
        block_out = ch_mult[0] * ch

        def get_n_heads_d_head(dim):
            if self.n_heads == -1:
                return dim // self.d_head, self.d_head
            else:
                return self.n_heads, dim//self.n_heads

        # UNet down modules
        for level, mult in enumerate(ch_mult):
            use_attn = down_attn[level]
            if down_transformer_depth is not None:
                transformer_depth = down_transformer_depth[level]

            disable_self_attn = not down_self_attn[level]

            is_final_block = level == len(ch_mult) - 1

            block_in = block_out
            block_out = ch * mult

            for i in range(self.num_res_blocks):
                layer_in = block_in if i == 0 else block_out
                layers = [
                    ResBlock2D(
                        layer_in,
                        time_embed_dim,
                        dropout,
                        out_channels=block_out,
                        use_checkpoint=use_checkpoint,
                        use_FiLM=use_FiLM,
                    )
                ]
                if use_attn:
                    n_heads, d_head = get_n_heads_d_head(block_out)
                    layers.append(
                        SpatialTransformer(
                            block_out,
                            n_heads,
                            d_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            disable_self_attn=disable_self_attn,
                            use_checkpoint=use_checkpoint,
                            use_linear=use_linear_in_transformer,
                            unet_ref_flag = self.unet_ref_flag if 'encoder' in self.ref_fuse_stage else False,
                            use_lora=use_lora,
                            use_ip_adapter=use_ip_adapter,
                            rank=rank,
                        )
                    )
                self.input_blocks.append(
                    TimestepAttentionBlock(*layers, 
                                           block_index=level,
                                           unet_ref_flag = self.unet_ref_flag if 'encoder' in self.ref_fuse_stage else False,)
                    )

            if not is_final_block:
                self.input_blocks.append(
                    TimestepAttentionBlock(
                        Downsample2D(block_out, conv_resample, block_out)
                    )
                )

        # mid block
        block_out = ch * ch_mult[-1]
        if mid_transformer_depth is not None:
            transformer_depth = mid_transformer_depth

        n_heads, d_head = get_n_heads_d_head(block_out)
        self.middle_block = TimestepAttentionBlock(
            ResBlock2D(
                block_out,
                time_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_FiLM=use_FiLM,
            ),
            SpatialTransformer(
                block_out,
                n_heads,
                d_head,
                depth=transformer_depth,
                context_dim=context_dim,
                disable_self_attn=not mid_self_attn,
                use_checkpoint=use_checkpoint,
                use_linear=use_linear_in_transformer,
                unet_ref_flag = self.unet_ref_flag if 'mid' in self.ref_fuse_stage else False,
                use_lora=use_lora,
                use_ip_adapter=use_ip_adapter,
                rank=rank,
                use_adapter=use_adapter
                ),
            ResBlock2D(
                block_out,
                time_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_FiLM=use_FiLM
            ),
            block_index=len(ch_mult),
            unet_ref_flag = self.unet_ref_flag if 'mid' in self.ref_fuse_stage else False,
        )

        # up_blocks
        self.output_blocks = nn.ModuleList([])

        ch_mult_reversed = list(reversed(ch_mult))
        if isinstance(self.context_dim, list):
            context_dim_reversed = list(reversed(self.context_dim))
            num_res_blocks_reversed = list(reversed(self.num_res_blocks))
        block_out = ch_mult_reversed[0] * ch
        for level, mult in enumerate(ch_mult_reversed):

            use_attn = up_attn[level]
            disable_self_attn = not up_self_attn[level]

            if up_transformer_depth is not None:
                transformer_depth = up_transformer_depth[level]

            prev_block_out = block_out
            block_out = ch * mult
            block_in = ch * ch_mult_reversed[min(level+1, len(ch_mult) - 1)]

            is_final_block = level == len(ch_mult) - 1

            for i in range(self.num_res_blocks + 1):
                skip_channels = block_in if (
                    i == self.num_res_blocks) else block_out
                layer_in = prev_block_out if i == 0 else block_out

                layers = [
                    ResBlock2D(
                        layer_in + skip_channels,
                        time_embed_dim,
                        dropout,
                        out_channels=block_out,
                        use_checkpoint=use_checkpoint,
                        use_FiLM=use_FiLM,
                    )
                ]
                if use_attn:
                    n_heads, d_head = get_n_heads_d_head(block_out)
                    layers.append(
                        SpatialTransformer(
                            block_out,
                            n_heads,
                            d_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            use_linear=use_linear_in_transformer,
                            disable_self_attn=disable_self_attn,
                            unet_ref_flag = self.unet_ref_flag if 'decoder' in self.ref_fuse_stage else False,
                            use_lora=use_lora,
                            use_ip_adapter=use_ip_adapter,
                            rank=rank,
                            use_adapter=use_adapter,
                        ))
                if (not is_final_block) and i == self.num_res_blocks:
                    layers.append(
                        Upsample2D(block_out, conv_resample, block_out)
                    )

                self.output_blocks.append(
                    TimestepAttentionBlock(
                        *layers,
                        block_index=len(ch_mult)+1+level,
                        unet_ref_flag = self.unet_ref_flag if 'decoder' in self.ref_fuse_stage else False,)
                        )

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(ch, out_channels, 3, 1, 1)),
        )

    def forward(self, 
                x, 
                timesteps, 
                context=None, 
                vit_embeddings=None,
                tryon_embeddings = None,
                pose_feature=None, 
                embeddings=None, 
                ref_kv_pair_lists_used_in_target=None, 
                controls=None,
                lora_scale=1.0, 
                ip_scale=1.0, 
                ip_hidden_states=None,
                ref_scale=1.0,
                supervise_block_index=None,
                ):
        r"""Apply Conditional Denoising Diffusion UNet prediction

        Args:
            x: an [N x C x ...] Tensor of inputs.
            timesteps: a 1-D batch of timesteps.
            context: conditioning plugged in via cross-attn [N x L x D]
            embeddings: additional 1-D conditional embedding [N x L]
        """
        if supervise_block_index is not None:
            supervise_block_feature_dict = {_block_index: [] for _block_index in supervise_block_index}
            masked_supervise_block_feature_dict = {_block_index: [] for _block_index in supervise_block_index}
        
        if self.unet_ref_flag:
            ref_kv_pair_list = []

        if not isinstance(context, list):
            context = [context]
        
        if len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)
        timesteps = timesteps.expand(x.shape[0])

        t_emb = timestep_embedding(timesteps, self.ch).type(x.dtype)
        emb = self.time_embed(t_emb)

        if vit_embeddings is not None and tryon_embeddings is not None:
            embedding = torch.cat([vit_embeddings,tryon_embeddings],dim=1)
            emb = emb + self.cond_embed_proj(embedding)

        if self.embed_proj is not None:
            assert embeddings is not None

        if embeddings is not None:
            emb = emb + self.embed_proj(embeddings)

        use_control = controls is not None

        hs = []
        h = x
        
        def get_context(module):
            if module.use_context and module.block_index >= 0:
                if len(context) == 1:
                    context_in = context[0]
                else:
                    context_in = context[module.block_index]
            else:
                context_in = None
            return context_in

        # input blocks
        for i, module in enumerate(self.input_blocks):

            context_in = get_context(module)

            if ref_kv_pair_lists_used_in_target and 'encoder' in self.ref_fuse_stage and module.block_index!=-1:
                ref_kv_pair_list_used_in_target_tsb = []
                for ref_kv_pair_list_used_in_target in ref_kv_pair_lists_used_in_target:
                    ref_kv_pair_list_used_in_target_tsb.append(ref_kv_pair_list_used_in_target.pop(0)) # [[[kg1,vg1],[kg2,vg2]],[[kb1,vb1],[kb2,vb2]]]
                if ref_kv_pair_list_used_in_target_tsb[0]:
                    h = module(h, emb, context_in, ref_kv_pair_list_used_in_target_tsb,\
                                        lora_scale=lora_scale, ip_scale=ip_scale, ip_hidden_states=ip_hidden_states, ref_scale=ref_scale)
                else:
                    h = module(h, emb, context_in,\
                                        lora_scale=lora_scale, ip_scale=ip_scale, ip_hidden_states=ip_hidden_states, ref_scale=ref_scale)
            else:
                h = module(h, emb, context_in,\
                                    lora_scale=lora_scale, ip_scale=ip_scale, ip_hidden_states=ip_hidden_states, ref_scale=ref_scale)

            if self.unet_ref_flag and 'encoder' in self.ref_fuse_stage and module.block_index!=-1:
                ref_kv_pair_list += module.ref_kv_pair_list # [[[...]], [], [], [], [[...]], [[...]]]

            hs.append(h)

        # middle block
        context_in = get_context(self.middle_block)
        if ref_kv_pair_lists_used_in_target and 'mid' in self.ref_fuse_stage:
            ref_kv_pair_list_used_in_target_tsb = []
            for ref_kv_pair_list_used_in_target in ref_kv_pair_lists_used_in_target:
                ref_kv_pair_list_used_in_target_tsb.append(ref_kv_pair_list_used_in_target.pop(0)) # [[[kg1,vg1],[kg2,vg2]],[[kb1,vb1],[kb2,vb2]]]
            if ref_kv_pair_list_used_in_target_tsb[0]: 
                h = self.middle_block(h, emb, context_in, ref_kv_pair_list_used_in_target_tsb,\
                                      lora_scale=lora_scale, ip_scale=ip_scale, ip_hidden_states=ip_hidden_states, ref_scale=ref_scale)
            else:
                h = self.middle_block(h, emb, context_in,\
                                      lora_scale=lora_scale, ip_scale=ip_scale, ip_hidden_states=ip_hidden_states, ref_scale=ref_scale)
        else:
            h = self.middle_block(h, emb, context_in,\
                                  lora_scale=lora_scale, ip_scale=ip_scale, ip_hidden_states=ip_hidden_states, ref_scale=ref_scale)

        if use_control:
            h = h + controls.pop()

        if self.unet_ref_flag and 'mid' in self.ref_fuse_stage:
            ref_kv_pair_list += self.middle_block.ref_kv_pair_list
        

        # output blocks
        for i, module in enumerate(self.output_blocks):
            context_in = get_context(module)
            
            if use_control:
                h = torch.cat([h, hs.pop() + controls.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop()], dim=1)

            if ref_kv_pair_lists_used_in_target and 'decoder' in self.ref_fuse_stage: #[ [[[...]], [], [], [], [[...]], [[...]]], [[[...]], [], [], [], [[...]], [[...]]] ]
                ref_kv_pair_list_used_in_target_tsb = []
                for ref_kv_pair_list_used_in_target in ref_kv_pair_lists_used_in_target:
                    ref_kv_pair_list_used_in_target_tsb.append(ref_kv_pair_list_used_in_target.pop(0)) # tsb->timestepblock, ref_kv_pair_list_used_in_target_tsb:[[[kg1,vg1],[kg2,vg2]],[[kb1,vb1],[kb2,vb2]]]
                if ref_kv_pair_list_used_in_target_tsb[0]:
                    h = module(h, emb, context_in, ref_kv_pair_list_used_in_target_tsb,\
                               lora_scale=lora_scale, ip_scale=ip_scale, ip_hidden_states=ip_hidden_states, ref_scale=ref_scale)
                else:
                    h = module(h, emb, context_in,\
                               lora_scale=lora_scale, ip_scale=ip_scale, ip_hidden_states=ip_hidden_states, ref_scale=ref_scale)
            else:
                if (supervise_block_index is not None) and (module.block_index in supervise_block_feature_dict):
                    h, attn_feature= module(h, emb, context_in, return_attn_feature=True)
                    supervise_block_feature_dict[module.block_index].append(attn_feature)
                else:
                    h = module(h, emb, context_in,\
                            lora_scale=lora_scale, ip_scale=ip_scale, ip_hidden_states=ip_hidden_states, ref_scale=ref_scale)

            if self.unet_ref_flag and 'decoder' in self.ref_fuse_stage:
                ref_kv_pair_list += module.ref_kv_pair_list # [[[...]], [], [], [], [[...]], [[...]]]

        if supervise_block_index is not None:
            return supervise_block_feature_dict

        if self.unet_ref_flag:
            return ref_kv_pair_list
        else:
            h = self.out(h)
            return h

class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, network_alpha=None):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}")

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, input):
        input_dtype = input.dtype
        dtype = self.down.weight.dtype

        h = self.down(input.to(dtype))
        h = self.up(h)

        if self.network_alpha is not None:
            h *= self.network_alpha / self.rank

        return h.to(input_dtype)

class MemoryEfficientCrossAttention(nn.Module):
    def __init__(self, query_dim, 
                 context_dim=None, 
                 heads=8, 
                 dim_head=64, 
                 dropout=0.0, 
                 unet_ref_flag=False, 
                 use_lora=False, 
                 use_ip_adapter=False,
                 rank=128,
                 use_lora_layer=True,
                 use_adapter=True,
                 ):
        
        super().__init__()
        self.unet_ref_flag = unet_ref_flag
        self.use_lora = use_lora
        self.use_ip_adapter = use_ip_adapter
        self.is_crossattn = context_dim is not None
        self.use_lora_layer = use_lora_layer
        self.use_adapter = use_adapter

        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        if self.unet_ref_flag and self.use_adapter:
            self.ref_to_k = nn.Linear(context_dim, inner_dim, bias=False)
            self.ref_to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        if self.use_lora:
            self.to_q_lora = LoRALinearLayer(query_dim, inner_dim, rank)
            self.to_k_lora = LoRALinearLayer(context_dim, inner_dim, rank)
            self.to_v_lora = LoRALinearLayer(context_dim, inner_dim, rank)
            self.to_out_lora = LoRALinearLayer(query_dim, inner_dim, rank)

        if self.use_ip_adapter and self.is_crossattn:
            self.to_k_ip = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_ip = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None, ref_kv_pair_used_in_target=None, lora_scale=1.0, ip_scale=1.0, ip_hidden_states=None, ref_scale=1.0):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if ref_kv_pair_used_in_target:
            ref_q = torch.clone(q).to(q.dtype)
            # _ref_q = q

        if self.use_lora and self.use_lora_layer:
            q = q + lora_scale * self.to_q_lora(x)
            k = k + lora_scale * self.to_k_lora(context)
            v = v + lora_scale * self.to_v_lora(context)

        if self.unet_ref_flag and self.use_adapter:
            self.ref_k = self.ref_to_k(x)
            self.ref_v = self.ref_to_v(x)
        else:
            self.ref_k = torch.clone(k).to(q.dtype)
            self.ref_v = torch.clone(v).to(q.dtype)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)
        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )

        if ref_kv_pair_used_in_target:
            for _ref_kv_pair_used_in_target in ref_kv_pair_used_in_target:
                _ref_q = torch.clone(ref_q).to(q.dtype)
                _ref_k, _ref_v = _ref_kv_pair_used_in_target
                _ref_q, _ref_k, _ref_v = map(
                    lambda t: t.unsqueeze(3)
                    .reshape(b, t.shape[1], self.heads, self.dim_head)
                    .permute(0, 2, 1, 3)
                    .reshape(b * self.heads, t.shape[1], self.dim_head)
                    .contiguous(),
                    (_ref_q, _ref_k, _ref_v),
                )
                temp = xformers.ops.memory_efficient_attention(_ref_q, _ref_k.to(_ref_q.dtype), _ref_v.to(_ref_q.dtype), attn_bias=None, op=self.attention_op)
                temp = (
                    temp.unsqueeze(0)
                    .reshape(b, self.heads, out.shape[1], self.dim_head)
                    .permute(0, 2, 1, 3)
                    .reshape(b, out.shape[1], self.heads * self.dim_head)
                )    
                out = out + ref_scale * temp        
                # print('out.shape:', out.shape)

        if self.use_ip_adapter and self.is_crossattn:
            assert ip_hidden_states is not None
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)
            ip_key, ip_value = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * self.heads, t.shape[1], self.dim_head)
                .contiguous(),
                (ip_key, ip_value),
            )
            ip_hidden_states = xformers.ops.memory_efficient_attention(q, ip_key, ip_value, attn_bias=None, op=self.attention_op)
            ip_hidden_states = (
                ip_hidden_states.unsqueeze(0)
                .reshape(b, self.heads, out.shape[1], self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, out.shape[1], self.heads * self.dim_head)
            )    
            out = out + ip_scale * ip_hidden_states

        out = self.to_out[0](out) + lora_scale * self.to_out_lora(out) if self.use_lora and self.use_lora_layer else self.to_out[0](out)

        out = self.to_out[1](out)

        return out
    
    @property
    def ref_kv_pair(self):
        k_temp, v_temp = self.ref_k, self.ref_v
        self.ref_k = self.ref_v = None
        return [k_temp, v_temp]

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        _ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)
        debug = True
        if context is not None and debug is True:
            sim = rearrange(sim, '(b h) i j -> b h i j', h=h)
            #sim = sim.mean(axis=1)
            print(sim.shape)

        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        # self.proj = nn.Linear(dim_in, dim_out * 2)
        self.proj = LoRACompatibleLinear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            # nn.Linear(dim, inner_dim),
            LoRACompatibleLinear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            # nn.Linear(inner_dim, dim_out)
            LoRACompatibleLinear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(
            self, 
            dim, 
            n_heads, 
            d_head, 
            dropout=0., 
            context_dim=None, 
            gated_ff=True, 
            checkpoint=True,
            disable_self_attn=False,
            unet_ref_flag=False,
            use_lora=False,
            use_ip_adapter=False,
            rank=128,
            use_adapter=True,):

        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax" # XFORMERS_IS_AVAILBLE=True
        
        assert attn_mode in self.ATTENTION_MODES

        # Gradient can not backward to ref_unet correctly when checkpoint is True
        # the reason of this phenomenon it not clear now.
        assert checkpoint == False
        
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.unet_ref_flag = unet_ref_flag

        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = attn_cls(
            query_dim=dim, 
            heads=n_heads, 
            dim_head=d_head, 
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
            unet_ref_flag=self.unet_ref_flag,
            use_lora=use_lora,
            use_ip_adapter=use_ip_adapter,
            rank=rank,
            use_adapter=use_adapter,
            )

        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = attn_cls(
            query_dim=dim, 
            context_dim=context_dim,
            heads=n_heads, 
            dim_head=d_head, 
            dropout=dropout,
            use_lora=use_lora,
            use_ip_adapter=use_ip_adapter,
            rank=rank,)

        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

        self.checkpoint = checkpoint

    def forward(self, x, context=None, ref_kv_pair_used_in_target=None, lora_scale=1.0, ip_scale=1.0, ip_hidden_states=None, ref_scale=1.0):
        return checkpoint(self._forward, (x, context, ref_kv_pair_used_in_target, lora_scale, ip_scale, ip_hidden_states, ref_scale), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, ref_kv_pair_used_in_target=None, lora_scale=1.0, ip_scale=1.0, ip_hidden_states=None, ref_scale=1.0):
        if ref_kv_pair_used_in_target:
            x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, ref_kv_pair_used_in_target=ref_kv_pair_used_in_target,\
                            lora_scale=lora_scale, ip_scale=ip_scale, ip_hidden_states=ip_hidden_states, ref_scale=ref_scale) + x
        else:
            x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None,\
                            lora_scale=lora_scale, ip_scale=ip_scale, ip_hidden_states=ip_hidden_states, ref_scale=ref_scale) + x

        x = self.attn2(self.norm2(x), context=context,\
                        lora_scale=lora_scale, ip_scale=ip_scale, ip_hidden_states=ip_hidden_states, ref_scale=ref_scale) + x

        x = self.ff(self.norm3(x)) + x

        return x

    @property
    def ref_kv_pair(self,):
        return self.attn1.ref_kv_pair