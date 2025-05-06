# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import os
import os.path as osp
import shutil
import sys
from pathlib import Path

import numpy as np
from einops import rearrange
from PIL import Image
import importlib
from typing import List,Optional,Tuple,Union

import torch
import torchvision


def seed_everything(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def delete_additional_ckpt(base_path, num_keep):
    dirs = []
    for d in os.listdir(base_path):
        if d.startswith("checkpoint-"):
            dirs.append(d)
    num_tot = len(dirs)
    if num_tot <= num_keep:
        return
    # ensure ckpt is sorted and delete the ealier!
    del_dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))[: num_tot - num_keep]
    for d in del_dirs:
        path_to_dir = osp.join(base_path, d)
        if osp.exists(path_to_dir):
            shutil.rmtree(path_to_dir)

def get_class(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents

def from_config(config, dict_params=True):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate object")
    module = get_class(config["target"])
    if dict_params:
        return module(**config.get("params",dict()))
    else:
        return module(config.get("params",[]))
    
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad(), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

def convert_unet_state_dict_from_open_animate_anyone(src_state_dict):

    dst_state_dict = {}

    mapper = {
      # conv_in
      "conv_in.weight": "input_blocks.0.0.weight",
      "conv_in.bias": "input_blocks.0.0.bias",

      # time embedding
      "time_embedding.linear_1.weight": "time_embed.0.weight",
      "time_embedding.linear_1.bias": "time_embed.0.bias",
      "time_embedding.linear_2.weight": "time_embed.2.weight",
      "time_embedding.linear_2.bias": "time_embed.2.bias",

      # conv_out
      "conv_norm_out.weight": "out.0.weight",
      "conv_norm_out.bias": "out.0.bias",
      "conv_out.weight": "out.2.weight",
      "conv_out.bias": "out.2.bias",
    }

    # input_blocks:
    dst_offset = [1, 4, 7, 10] 
    for k in range(4):
        dst_layer_idx = dst_offset[k]

        mapper.update({
            f"down_blocks.{k}.resnets.0.norm1.weight":f"input_blocks.{dst_layer_idx}.0.in_layers.0.weight",
            f"down_blocks.{k}.resnets.0.norm1.bias":f"input_blocks.{dst_layer_idx}.0.in_layers.0.bias",
            f"down_blocks.{k}.resnets.0.conv1.weight":f"input_blocks.{dst_layer_idx}.0.in_layers.2.weight",
            f"down_blocks.{k}.resnets.0.conv1.bias":f"input_blocks.{dst_layer_idx}.0.in_layers.2.bias",
            f"down_blocks.{k}.resnets.0.time_emb_proj.weight":f"input_blocks.{dst_layer_idx}.0.emb_layers.1.weight",
            f"down_blocks.{k}.resnets.0.time_emb_proj.bias":f"input_blocks.{dst_layer_idx}.0.emb_layers.1.bias",
            f"down_blocks.{k}.resnets.0.norm2.weight":f"input_blocks.{dst_layer_idx}.0.out_layers.0.weight",
            f"down_blocks.{k}.resnets.0.norm2.bias":f"input_blocks.{dst_layer_idx}.0.out_layers.0.bias",
            f"down_blocks.{k}.resnets.0.conv2.weight":f"input_blocks.{dst_layer_idx}.0.out_layers.3.weight",
            f"down_blocks.{k}.resnets.0.conv2.bias":f"input_blocks.{dst_layer_idx}.0.out_layers.3.bias",
            f"down_blocks.{k}.resnets.1.norm1.weight":f"input_blocks.{dst_layer_idx+1}.0.in_layers.0.weight",
            f"down_blocks.{k}.resnets.1.norm1.bias":f"input_blocks.{dst_layer_idx+1}.0.in_layers.0.bias",
            f"down_blocks.{k}.resnets.1.conv1.weight":f"input_blocks.{dst_layer_idx+1}.0.in_layers.2.weight",
            f"down_blocks.{k}.resnets.1.conv1.bias":f"input_blocks.{dst_layer_idx+1}.0.in_layers.2.bias",
            f"down_blocks.{k}.resnets.1.time_emb_proj.weight":f"input_blocks.{dst_layer_idx+1}.0.emb_layers.1.weight",
            f"down_blocks.{k}.resnets.1.time_emb_proj.bias":f"input_blocks.{dst_layer_idx+1}.0.emb_layers.1.bias",
            f"down_blocks.{k}.resnets.1.norm2.weight":f"input_blocks.{dst_layer_idx+1}.0.out_layers.0.weight",
            f"down_blocks.{k}.resnets.1.norm2.bias":f"input_blocks.{dst_layer_idx+1}.0.out_layers.0.bias",
            f"down_blocks.{k}.resnets.1.conv2.weight":f"input_blocks.{dst_layer_idx+1}.0.out_layers.3.weight",
            f"down_blocks.{k}.resnets.1.conv2.bias":f"input_blocks.{dst_layer_idx+1}.0.out_layers.3.bias",
        })

        
        if k != 0 and k != 3:
            mapper.update({
                f"down_blocks.{k}.resnets.0.conv_shortcut.weight":f"input_blocks.{dst_layer_idx}.0.skip_connection.weight",
                f"down_blocks.{k}.resnets.0.conv_shortcut.bias":f"input_blocks.{dst_layer_idx}.0.skip_connection.bias",
            })


        if k == 3:
            continue

        mapper.update({
            f"down_blocks.{k}.attentions.0.norm.weight":f"input_blocks.{dst_layer_idx}.1.norm.weight",
            f"down_blocks.{k}.attentions.0.norm.bias":f"input_blocks.{dst_layer_idx}.1.norm.bias",
            f"down_blocks.{k}.attentions.0.proj_in.weight":f"input_blocks.{dst_layer_idx}.1.proj_in.weight",
            f"down_blocks.{k}.attentions.0.proj_in.bias":f"input_blocks.{dst_layer_idx}.1.proj_in.bias",
            f"down_blocks.{k}.attentions.0.transformer_blocks.0.norm1.weight":f"input_blocks.{dst_layer_idx}.1.transformer_blocks.0.norm1.weight",
            f"down_blocks.{k}.attentions.0.transformer_blocks.0.norm1.bias":f"input_blocks.{dst_layer_idx}.1.transformer_blocks.0.norm1.bias",
            f"down_blocks.{k}.attentions.0.transformer_blocks.0.attn1.to_q.weight":f"input_blocks.{dst_layer_idx}.1.transformer_blocks.0.attn1.to_q.weight",
            f"down_blocks.{k}.attentions.0.transformer_blocks.0.attn1.to_k.weight":f"input_blocks.{dst_layer_idx}.1.transformer_blocks.0.attn1.to_k.weight",
            f"down_blocks.{k}.attentions.0.transformer_blocks.0.attn1.to_v.weight":f"input_blocks.{dst_layer_idx}.1.transformer_blocks.0.attn1.to_v.weight",
            f"down_blocks.{k}.attentions.0.transformer_blocks.0.attn1.to_out.0.weight":f"input_blocks.{dst_layer_idx}.1.transformer_blocks.0.attn1.to_out.0.weight",
            f"down_blocks.{k}.attentions.0.transformer_blocks.0.attn1.to_out.0.bias":f"input_blocks.{dst_layer_idx}.1.transformer_blocks.0.attn1.to_out.0.bias",
            f"down_blocks.{k}.attentions.0.transformer_blocks.0.norm2.weight":f"input_blocks.{dst_layer_idx}.1.transformer_blocks.0.norm2.weight",
            f"down_blocks.{k}.attentions.0.transformer_blocks.0.norm2.bias":f"input_blocks.{dst_layer_idx}.1.transformer_blocks.0.norm2.bias",
            f"down_blocks.{k}.attentions.0.transformer_blocks.0.attn2.to_q.weight":f"input_blocks.{dst_layer_idx}.1.transformer_blocks.0.attn2.to_q.weight",
            f"down_blocks.{k}.attentions.0.transformer_blocks.0.attn2.to_k.weight":f"input_blocks.{dst_layer_idx}.1.transformer_blocks.0.attn2.to_k.weight",
            f"down_blocks.{k}.attentions.0.transformer_blocks.0.attn2.to_v.weight":f"input_blocks.{dst_layer_idx}.1.transformer_blocks.0.attn2.to_v.weight",
            f"down_blocks.{k}.attentions.0.transformer_blocks.0.attn2.to_out.0.weight":f"input_blocks.{dst_layer_idx}.1.transformer_blocks.0.attn2.to_out.0.weight",
            f"down_blocks.{k}.attentions.0.transformer_blocks.0.attn2.to_out.0.bias":f"input_blocks.{dst_layer_idx}.1.transformer_blocks.0.attn2.to_out.0.bias",
            f"down_blocks.{k}.attentions.0.transformer_blocks.0.norm3.weight":f"input_blocks.{dst_layer_idx}.1.transformer_blocks.0.norm3.weight",
            f"down_blocks.{k}.attentions.0.transformer_blocks.0.norm3.bias":f"input_blocks.{dst_layer_idx}.1.transformer_blocks.0.norm3.bias",
            f"down_blocks.{k}.attentions.0.transformer_blocks.0.ff.net.0.proj.weight":f"input_blocks.{dst_layer_idx}.1.transformer_blocks.0.ff.net.0.proj.weight",
            f"down_blocks.{k}.attentions.0.transformer_blocks.0.ff.net.0.proj.bias":f"input_blocks.{dst_layer_idx}.1.transformer_blocks.0.ff.net.0.proj.bias",
            f"down_blocks.{k}.attentions.0.transformer_blocks.0.ff.net.2.weight":f"input_blocks.{dst_layer_idx}.1.transformer_blocks.0.ff.net.2.weight",
            f"down_blocks.{k}.attentions.0.transformer_blocks.0.ff.net.2.bias":f"input_blocks.{dst_layer_idx}.1.transformer_blocks.0.ff.net.2.bias",
            f"down_blocks.{k}.attentions.0.proj_out.weight":f"input_blocks.{dst_layer_idx}.1.proj_out.weight",
            f"down_blocks.{k}.attentions.0.proj_out.bias":f"input_blocks.{dst_layer_idx}.1.proj_out.bias",
            f"down_blocks.{k}.attentions.1.norm.weight":f"input_blocks.{dst_layer_idx+1}.1.norm.weight",
            f"down_blocks.{k}.attentions.1.norm.bias":f"input_blocks.{dst_layer_idx+1}.1.norm.bias",
            f"down_blocks.{k}.attentions.1.proj_in.weight":f"input_blocks.{dst_layer_idx+1}.1.proj_in.weight",
            f"down_blocks.{k}.attentions.1.proj_in.bias":f"input_blocks.{dst_layer_idx+1}.1.proj_in.bias",
            f"down_blocks.{k}.attentions.1.transformer_blocks.0.norm1.weight":f"input_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.norm1.weight",
            f"down_blocks.{k}.attentions.1.transformer_blocks.0.norm1.bias":f"input_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.norm1.bias",
            f"down_blocks.{k}.attentions.1.transformer_blocks.0.attn1.to_q.weight":f"input_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.attn1.to_q.weight",
            f"down_blocks.{k}.attentions.1.transformer_blocks.0.attn1.to_k.weight":f"input_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.attn1.to_k.weight",
            f"down_blocks.{k}.attentions.1.transformer_blocks.0.attn1.to_v.weight":f"input_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.attn1.to_v.weight",
            f"down_blocks.{k}.attentions.1.transformer_blocks.0.attn1.to_out.0.weight":f"input_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.attn1.to_out.0.weight",
            f"down_blocks.{k}.attentions.1.transformer_blocks.0.attn1.to_out.0.bias":f"input_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.attn1.to_out.0.bias",
            f"down_blocks.{k}.attentions.1.transformer_blocks.0.norm2.weight":f"input_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.norm2.weight",
            f"down_blocks.{k}.attentions.1.transformer_blocks.0.norm2.bias":f"input_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.norm2.bias",
            f"down_blocks.{k}.attentions.1.transformer_blocks.0.attn2.to_q.weight":f"input_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.attn2.to_q.weight",
            f"down_blocks.{k}.attentions.1.transformer_blocks.0.attn2.to_k.weight":f"input_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.attn2.to_k.weight",
            f"down_blocks.{k}.attentions.1.transformer_blocks.0.attn2.to_v.weight":f"input_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.attn2.to_v.weight",
            f"down_blocks.{k}.attentions.1.transformer_blocks.0.attn2.to_out.0.weight":f"input_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.attn2.to_out.0.weight",
            f"down_blocks.{k}.attentions.1.transformer_blocks.0.attn2.to_out.0.bias":f"input_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.attn2.to_out.0.bias",
            f"down_blocks.{k}.attentions.1.transformer_blocks.0.norm3.weight":f"input_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.norm3.weight",
            f"down_blocks.{k}.attentions.1.transformer_blocks.0.norm3.bias":f"input_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.norm3.bias",
            f"down_blocks.{k}.attentions.1.transformer_blocks.0.ff.net.0.proj.weight":f"input_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.ff.net.0.proj.weight",
            f"down_blocks.{k}.attentions.1.transformer_blocks.0.ff.net.0.proj.bias":f"input_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.ff.net.0.proj.bias",
            f"down_blocks.{k}.attentions.1.transformer_blocks.0.ff.net.2.weight":f"input_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.ff.net.2.weight",
            f"down_blocks.{k}.attentions.1.transformer_blocks.0.ff.net.2.bias":f"input_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.ff.net.2.bias",
            f"down_blocks.{k}.attentions.1.proj_out.weight":f"input_blocks.{dst_layer_idx+1}.1.proj_out.weight",
            f"down_blocks.{k}.attentions.1.proj_out.bias":f"input_blocks.{dst_layer_idx+1}.1.proj_out.bias",
            f"down_blocks.{k}.downsamplers.0.conv.weight":f"input_blocks.{dst_layer_idx+2}.0.op.weight",
            f"down_blocks.{k}.downsamplers.0.conv.bias":f"input_blocks.{dst_layer_idx+2}.0.op.bias"}
        )

    # mid_blocks
    mapper.update({
        "mid_block.attentions.0.norm.weight":"middle_block.1.norm.weight",
        "mid_block.attentions.0.norm.bias":"middle_block.1.norm.bias",
        "mid_block.attentions.0.proj_in.weight":"middle_block.1.proj_in.weight",
        "mid_block.attentions.0.proj_in.bias":"middle_block.1.proj_in.bias",
        "mid_block.attentions.0.transformer_blocks.0.norm1.weight":"middle_block.1.transformer_blocks.0.norm1.weight",
        "mid_block.attentions.0.transformer_blocks.0.norm1.bias":"middle_block.1.transformer_blocks.0.norm1.bias",
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_q.weight":"middle_block.1.transformer_blocks.0.attn1.to_q.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_k.weight":"middle_block.1.transformer_blocks.0.attn1.to_k.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_v.weight":"middle_block.1.transformer_blocks.0.attn1.to_v.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0.weight":"middle_block.1.transformer_blocks.0.attn1.to_out.0.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0.bias":"middle_block.1.transformer_blocks.0.attn1.to_out.0.bias",
        "mid_block.attentions.0.transformer_blocks.0.norm2.weight":"middle_block.1.transformer_blocks.0.norm2.weight",
        "mid_block.attentions.0.transformer_blocks.0.norm2.bias":"middle_block.1.transformer_blocks.0.norm2.bias",
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_q.weight":"middle_block.1.transformer_blocks.0.attn2.to_q.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_k.weight":"middle_block.1.transformer_blocks.0.attn2.to_k.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_v.weight":"middle_block.1.transformer_blocks.0.attn2.to_v.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0.weight":"middle_block.1.transformer_blocks.0.attn2.to_out.0.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0.bias":"middle_block.1.transformer_blocks.0.attn2.to_out.0.bias",
        "mid_block.attentions.0.transformer_blocks.0.norm3.weight":"middle_block.1.transformer_blocks.0.norm3.weight",
        "mid_block.attentions.0.transformer_blocks.0.norm3.bias":"middle_block.1.transformer_blocks.0.norm3.bias",
        "mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj.weight":"middle_block.1.transformer_blocks.0.ff.net.0.proj.weight",
        "mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj.bias":"middle_block.1.transformer_blocks.0.ff.net.0.proj.bias",
        "mid_block.attentions.0.transformer_blocks.0.ff.net.2.weight":"middle_block.1.transformer_blocks.0.ff.net.2.weight",
        "mid_block.attentions.0.transformer_blocks.0.ff.net.2.bias":"middle_block.1.transformer_blocks.0.ff.net.2.bias",
        "mid_block.attentions.0.proj_out.weight":"middle_block.1.proj_out.weight",
        "mid_block.attentions.0.proj_out.bias":"middle_block.1.proj_out.bias",
        "mid_block.resnets.0.norm1.weight":"middle_block.0.in_layers.0.weight",
        "mid_block.resnets.0.norm1.bias":"middle_block.0.in_layers.0.bias",
        "mid_block.resnets.0.conv1.weight":"middle_block.0.in_layers.2.weight",
        "mid_block.resnets.0.conv1.bias":"middle_block.0.in_layers.2.bias",
        "mid_block.resnets.0.time_emb_proj.weight":"middle_block.0.emb_layers.1.weight",
        "mid_block.resnets.0.time_emb_proj.bias":"middle_block.0.emb_layers.1.bias",
        "mid_block.resnets.0.norm2.weight":"middle_block.0.out_layers.0.weight",
        "mid_block.resnets.0.norm2.bias":"middle_block.0.out_layers.0.bias",
        "mid_block.resnets.0.conv2.weight":"middle_block.0.out_layers.3.weight",
        "mid_block.resnets.0.conv2.bias":"middle_block.0.out_layers.3.bias",
        "mid_block.resnets.1.norm1.weight":"middle_block.2.in_layers.0.weight",
        "mid_block.resnets.1.norm1.bias":"middle_block.2.in_layers.0.bias",
        "mid_block.resnets.1.conv1.weight":"middle_block.2.in_layers.2.weight",
        "mid_block.resnets.1.conv1.bias":"middle_block.2.in_layers.2.bias",
        "mid_block.resnets.1.time_emb_proj.weight":"middle_block.2.emb_layers.1.weight",
        "mid_block.resnets.1.time_emb_proj.bias":"middle_block.2.emb_layers.1.bias",
        "mid_block.resnets.1.norm2.weight":"middle_block.2.out_layers.0.weight",
        "mid_block.resnets.1.norm2.bias":"middle_block.2.out_layers.0.bias",
        "mid_block.resnets.1.conv2.weight":"middle_block.2.out_layers.3.weight",
        "mid_block.resnets.1.conv2.bias":"middle_block.2.out_layers.3.bias",
        }
    )

    # output_blocks
    offset = [0, 3, 6, 9]
    for k in range(4):
        dst_layer_idx = offset[k]
        mapper.update({
            f"up_blocks.{k}.resnets.0.norm1.weight":f"output_blocks.{dst_layer_idx}.0.in_layers.0.weight",
            f"up_blocks.{k}.resnets.0.norm1.bias":f"output_blocks.{dst_layer_idx}.0.in_layers.0.bias",
            f"up_blocks.{k}.resnets.0.conv1.weight":f"output_blocks.{dst_layer_idx}.0.in_layers.2.weight",
            f"up_blocks.{k}.resnets.0.conv1.bias":f"output_blocks.{dst_layer_idx}.0.in_layers.2.bias",
            f"up_blocks.{k}.resnets.0.time_emb_proj.weight":f"output_blocks.{dst_layer_idx}.0.emb_layers.1.weight",
            f"up_blocks.{k}.resnets.0.time_emb_proj.bias":f"output_blocks.{dst_layer_idx}.0.emb_layers.1.bias",
            f"up_blocks.{k}.resnets.0.norm2.weight":f"output_blocks.{dst_layer_idx}.0.out_layers.0.weight",
            f"up_blocks.{k}.resnets.0.norm2.bias":f"output_blocks.{dst_layer_idx}.0.out_layers.0.bias",
            f"up_blocks.{k}.resnets.0.conv2.weight":f"output_blocks.{dst_layer_idx}.0.out_layers.3.weight",
            f"up_blocks.{k}.resnets.0.conv2.bias":f"output_blocks.{dst_layer_idx}.0.out_layers.3.bias",
            f"up_blocks.{k}.resnets.0.conv_shortcut.weight":f"output_blocks.{dst_layer_idx}.0.skip_connection.weight",
            f"up_blocks.{k}.resnets.0.conv_shortcut.bias":f"output_blocks.{dst_layer_idx}.0.skip_connection.bias",
            f"up_blocks.{k}.resnets.1.norm1.weight":f"output_blocks.{dst_layer_idx+1}.0.in_layers.0.weight",
            f"up_blocks.{k}.resnets.1.norm1.bias":f"output_blocks.{dst_layer_idx+1}.0.in_layers.0.bias",
            f"up_blocks.{k}.resnets.1.conv1.weight":f"output_blocks.{dst_layer_idx+1}.0.in_layers.2.weight",
            f"up_blocks.{k}.resnets.1.conv1.bias":f"output_blocks.{dst_layer_idx+1}.0.in_layers.2.bias",
            f"up_blocks.{k}.resnets.1.time_emb_proj.weight":f"output_blocks.{dst_layer_idx+1}.0.emb_layers.1.weight",
            f"up_blocks.{k}.resnets.1.time_emb_proj.bias":f"output_blocks.{dst_layer_idx+1}.0.emb_layers.1.bias",
            f"up_blocks.{k}.resnets.1.norm2.weight":f"output_blocks.{dst_layer_idx+1}.0.out_layers.0.weight",
            f"up_blocks.{k}.resnets.1.norm2.bias":f"output_blocks.{dst_layer_idx+1}.0.out_layers.0.bias",
            f"up_blocks.{k}.resnets.1.conv2.weight":f"output_blocks.{dst_layer_idx+1}.0.out_layers.3.weight",
            f"up_blocks.{k}.resnets.1.conv2.bias":f"output_blocks.{dst_layer_idx+1}.0.out_layers.3.bias",
            f"up_blocks.{k}.resnets.1.conv_shortcut.weight":f"output_blocks.{dst_layer_idx+1}.0.skip_connection.weight",
            f"up_blocks.{k}.resnets.1.conv_shortcut.bias":f"output_blocks.{dst_layer_idx+1}.0.skip_connection.bias",
            f"up_blocks.{k}.resnets.2.norm1.weight":f"output_blocks.{dst_layer_idx+2}.0.in_layers.0.weight",
            f"up_blocks.{k}.resnets.2.norm1.bias":f"output_blocks.{dst_layer_idx+2}.0.in_layers.0.bias",
            f"up_blocks.{k}.resnets.2.conv1.weight":f"output_blocks.{dst_layer_idx+2}.0.in_layers.2.weight",
            f"up_blocks.{k}.resnets.2.conv1.bias":f"output_blocks.{dst_layer_idx+2}.0.in_layers.2.bias",
            f"up_blocks.{k}.resnets.2.time_emb_proj.weight":f"output_blocks.{dst_layer_idx+2}.0.emb_layers.1.weight",
            f"up_blocks.{k}.resnets.2.time_emb_proj.bias":f"output_blocks.{dst_layer_idx+2}.0.emb_layers.1.bias",
            f"up_blocks.{k}.resnets.2.norm2.weight":f"output_blocks.{dst_layer_idx+2}.0.out_layers.0.weight",
            f"up_blocks.{k}.resnets.2.norm2.bias":f"output_blocks.{dst_layer_idx+2}.0.out_layers.0.bias",
            f"up_blocks.{k}.resnets.2.conv2.weight":f"output_blocks.{dst_layer_idx+2}.0.out_layers.3.weight",
            f"up_blocks.{k}.resnets.2.conv2.bias":f"output_blocks.{dst_layer_idx+2}.0.out_layers.3.bias",
            f"up_blocks.{k}.resnets.2.conv_shortcut.weight":f"output_blocks.{dst_layer_idx+2}.0.skip_connection.weight",
            f"up_blocks.{k}.resnets.2.conv_shortcut.bias":f"output_blocks.{dst_layer_idx+2}.0.skip_connection.bias",
        })

        if k != 0:
            mapper.update({
                f"up_blocks.{k}.attentions.0.norm.weight":f"output_blocks.{dst_layer_idx}.1.norm.weight",
                f"up_blocks.{k}.attentions.0.norm.bias":f"output_blocks.{dst_layer_idx}.1.norm.bias",
                f"up_blocks.{k}.attentions.0.proj_in.weight":f"output_blocks.{dst_layer_idx}.1.proj_in.weight",
                f"up_blocks.{k}.attentions.0.proj_in.bias":f"output_blocks.{dst_layer_idx}.1.proj_in.bias",
                f"up_blocks.{k}.attentions.0.transformer_blocks.0.norm1.weight":f"output_blocks.{dst_layer_idx}.1.transformer_blocks.0.norm1.weight",
                f"up_blocks.{k}.attentions.0.transformer_blocks.0.norm1.bias":f"output_blocks.{dst_layer_idx}.1.transformer_blocks.0.norm1.bias",
                f"up_blocks.{k}.attentions.0.transformer_blocks.0.attn1.to_q.weight":f"output_blocks.{dst_layer_idx}.1.transformer_blocks.0.attn1.to_q.weight",
                f"up_blocks.{k}.attentions.0.transformer_blocks.0.attn1.to_k.weight":f"output_blocks.{dst_layer_idx}.1.transformer_blocks.0.attn1.to_k.weight",
                f"up_blocks.{k}.attentions.0.transformer_blocks.0.attn1.to_v.weight":f"output_blocks.{dst_layer_idx}.1.transformer_blocks.0.attn1.to_v.weight",
                f"up_blocks.{k}.attentions.0.transformer_blocks.0.attn1.to_out.0.weight":f"output_blocks.{dst_layer_idx}.1.transformer_blocks.0.attn1.to_out.0.weight",
                f"up_blocks.{k}.attentions.0.transformer_blocks.0.attn1.to_out.0.bias":f"output_blocks.{dst_layer_idx}.1.transformer_blocks.0.attn1.to_out.0.bias",
                f"up_blocks.{k}.attentions.0.transformer_blocks.0.norm2.weight":f"output_blocks.{dst_layer_idx}.1.transformer_blocks.0.norm2.weight",
                f"up_blocks.{k}.attentions.0.transformer_blocks.0.norm2.bias":f"output_blocks.{dst_layer_idx}.1.transformer_blocks.0.norm2.bias",
                f"up_blocks.{k}.attentions.0.transformer_blocks.0.attn2.to_q.weight":f"output_blocks.{dst_layer_idx}.1.transformer_blocks.0.attn2.to_q.weight",
                f"up_blocks.{k}.attentions.0.transformer_blocks.0.attn2.to_k.weight":f"output_blocks.{dst_layer_idx}.1.transformer_blocks.0.attn2.to_k.weight",
                f"up_blocks.{k}.attentions.0.transformer_blocks.0.attn2.to_v.weight":f"output_blocks.{dst_layer_idx}.1.transformer_blocks.0.attn2.to_v.weight",
                f"up_blocks.{k}.attentions.0.transformer_blocks.0.attn2.to_out.0.weight":f"output_blocks.{dst_layer_idx}.1.transformer_blocks.0.attn2.to_out.0.weight",
                f"up_blocks.{k}.attentions.0.transformer_blocks.0.attn2.to_out.0.bias":f"output_blocks.{dst_layer_idx}.1.transformer_blocks.0.attn2.to_out.0.bias",
                f"up_blocks.{k}.attentions.0.transformer_blocks.0.norm3.weight":f"output_blocks.{dst_layer_idx}.1.transformer_blocks.0.norm3.weight",
                f"up_blocks.{k}.attentions.0.transformer_blocks.0.norm3.bias":f"output_blocks.{dst_layer_idx}.1.transformer_blocks.0.norm3.bias",
                f"up_blocks.{k}.attentions.0.transformer_blocks.0.ff.net.0.proj.weight":f"output_blocks.{dst_layer_idx}.1.transformer_blocks.0.ff.net.0.proj.weight",
                f"up_blocks.{k}.attentions.0.transformer_blocks.0.ff.net.0.proj.bias":f"output_blocks.{dst_layer_idx}.1.transformer_blocks.0.ff.net.0.proj.bias",
                f"up_blocks.{k}.attentions.0.transformer_blocks.0.ff.net.2.weight":f"output_blocks.{dst_layer_idx}.1.transformer_blocks.0.ff.net.2.weight",
                f"up_blocks.{k}.attentions.0.transformer_blocks.0.ff.net.2.bias":f"output_blocks.{dst_layer_idx}.1.transformer_blocks.0.ff.net.2.bias",
                f"up_blocks.{k}.attentions.0.proj_out.weight":f"output_blocks.{dst_layer_idx}.1.proj_out.weight",
                f"up_blocks.{k}.attentions.0.proj_out.bias":f"output_blocks.{dst_layer_idx}.1.proj_out.bias",
                f"up_blocks.{k}.attentions.1.norm.weight":f"output_blocks.{dst_layer_idx+1}.1.norm.weight",
                f"up_blocks.{k}.attentions.1.norm.bias":f"output_blocks.{dst_layer_idx+1}.1.norm.bias",
                f"up_blocks.{k}.attentions.1.proj_in.weight":f"output_blocks.{dst_layer_idx+1}.1.proj_in.weight",
                f"up_blocks.{k}.attentions.1.proj_in.bias":f"output_blocks.{dst_layer_idx+1}.1.proj_in.bias",
                f"up_blocks.{k}.attentions.1.transformer_blocks.0.norm1.weight":f"output_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.norm1.weight",
                f"up_blocks.{k}.attentions.1.transformer_blocks.0.norm1.bias":f"output_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.norm1.bias",
                f"up_blocks.{k}.attentions.1.transformer_blocks.0.attn1.to_q.weight":f"output_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.attn1.to_q.weight",
                f"up_blocks.{k}.attentions.1.transformer_blocks.0.attn1.to_k.weight":f"output_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.attn1.to_k.weight",
                f"up_blocks.{k}.attentions.1.transformer_blocks.0.attn1.to_v.weight":f"output_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.attn1.to_v.weight",
                f"up_blocks.{k}.attentions.1.transformer_blocks.0.attn1.to_out.0.weight":f"output_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.attn1.to_out.0.weight",
                f"up_blocks.{k}.attentions.1.transformer_blocks.0.attn1.to_out.0.bias":f"output_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.attn1.to_out.0.bias",
                f"up_blocks.{k}.attentions.1.transformer_blocks.0.norm2.weight":f"output_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.norm2.weight",
                f"up_blocks.{k}.attentions.1.transformer_blocks.0.norm2.bias":f"output_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.norm2.bias",
                f"up_blocks.{k}.attentions.1.transformer_blocks.0.attn2.to_q.weight":f"output_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.attn2.to_q.weight",
                f"up_blocks.{k}.attentions.1.transformer_blocks.0.attn2.to_k.weight":f"output_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.attn2.to_k.weight",
                f"up_blocks.{k}.attentions.1.transformer_blocks.0.attn2.to_v.weight":f"output_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.attn2.to_v.weight",
                f"up_blocks.{k}.attentions.1.transformer_blocks.0.attn2.to_out.0.weight":f"output_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.attn2.to_out.0.weight",
                f"up_blocks.{k}.attentions.1.transformer_blocks.0.attn2.to_out.0.bias":f"output_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.attn2.to_out.0.bias",
                f"up_blocks.{k}.attentions.1.transformer_blocks.0.norm3.weight":f"output_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.norm3.weight",
                f"up_blocks.{k}.attentions.1.transformer_blocks.0.norm3.bias":f"output_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.norm3.bias",
                f"up_blocks.{k}.attentions.1.transformer_blocks.0.ff.net.0.proj.weight":f"output_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.ff.net.0.proj.weight",
                f"up_blocks.{k}.attentions.1.transformer_blocks.0.ff.net.0.proj.bias":f"output_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.ff.net.0.proj.bias",
                f"up_blocks.{k}.attentions.1.transformer_blocks.0.ff.net.2.weight":f"output_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.ff.net.2.weight",
                f"up_blocks.{k}.attentions.1.transformer_blocks.0.ff.net.2.bias":f"output_blocks.{dst_layer_idx+1}.1.transformer_blocks.0.ff.net.2.bias",
                f"up_blocks.{k}.attentions.1.proj_out.weight":f"output_blocks.{dst_layer_idx+1}.1.proj_out.weight",
                f"up_blocks.{k}.attentions.1.proj_out.bias":f"output_blocks.{dst_layer_idx+1}.1.proj_out.bias",
                f"up_blocks.{k}.attentions.2.norm.weight":f"output_blocks.{dst_layer_idx+2}.1.norm.weight",
                f"up_blocks.{k}.attentions.2.norm.bias":f"output_blocks.{dst_layer_idx+2}.1.norm.bias",
                f"up_blocks.{k}.attentions.2.proj_in.weight":f"output_blocks.{dst_layer_idx+2}.1.proj_in.weight",
                f"up_blocks.{k}.attentions.2.proj_in.bias":f"output_blocks.{dst_layer_idx+2}.1.proj_in.bias",
                f"up_blocks.{k}.attentions.2.transformer_blocks.0.norm1.weight":f"output_blocks.{dst_layer_idx+2}.1.transformer_blocks.0.norm1.weight",
                f"up_blocks.{k}.attentions.2.transformer_blocks.0.norm1.bias":f"output_blocks.{dst_layer_idx+2}.1.transformer_blocks.0.norm1.bias",
                f"up_blocks.{k}.attentions.2.transformer_blocks.0.attn1.to_q.weight":f"output_blocks.{dst_layer_idx+2}.1.transformer_blocks.0.attn1.to_q.weight",
                f"up_blocks.{k}.attentions.2.transformer_blocks.0.attn1.to_k.weight":f"output_blocks.{dst_layer_idx+2}.1.transformer_blocks.0.attn1.to_k.weight",
                f"up_blocks.{k}.attentions.2.transformer_blocks.0.attn1.to_v.weight":f"output_blocks.{dst_layer_idx+2}.1.transformer_blocks.0.attn1.to_v.weight",
                f"up_blocks.{k}.attentions.2.transformer_blocks.0.attn1.to_out.0.weight":f"output_blocks.{dst_layer_idx+2}.1.transformer_blocks.0.attn1.to_out.0.weight",
                f"up_blocks.{k}.attentions.2.transformer_blocks.0.attn1.to_out.0.bias":f"output_blocks.{dst_layer_idx+2}.1.transformer_blocks.0.attn1.to_out.0.bias",
                f"up_blocks.{k}.attentions.2.transformer_blocks.0.norm2.weight":f"output_blocks.{dst_layer_idx+2}.1.transformer_blocks.0.norm2.weight",
                f"up_blocks.{k}.attentions.2.transformer_blocks.0.norm2.bias":f"output_blocks.{dst_layer_idx+2}.1.transformer_blocks.0.norm2.bias",
                f"up_blocks.{k}.attentions.2.transformer_blocks.0.attn2.to_q.weight":f"output_blocks.{dst_layer_idx+2}.1.transformer_blocks.0.attn2.to_q.weight",
                f"up_blocks.{k}.attentions.2.transformer_blocks.0.attn2.to_k.weight":f"output_blocks.{dst_layer_idx+2}.1.transformer_blocks.0.attn2.to_k.weight",
                f"up_blocks.{k}.attentions.2.transformer_blocks.0.attn2.to_v.weight":f"output_blocks.{dst_layer_idx+2}.1.transformer_blocks.0.attn2.to_v.weight",
                f"up_blocks.{k}.attentions.2.transformer_blocks.0.attn2.to_out.0.weight":f"output_blocks.{dst_layer_idx+2}.1.transformer_blocks.0.attn2.to_out.0.weight",
                f"up_blocks.{k}.attentions.2.transformer_blocks.0.attn2.to_out.0.bias":f"output_blocks.{dst_layer_idx+2}.1.transformer_blocks.0.attn2.to_out.0.bias",
                f"up_blocks.{k}.attentions.2.transformer_blocks.0.norm3.weight":f"output_blocks.{dst_layer_idx+2}.1.transformer_blocks.0.norm3.weight",
                f"up_blocks.{k}.attentions.2.transformer_blocks.0.norm3.bias":f"output_blocks.{dst_layer_idx+2}.1.transformer_blocks.0.norm3.bias",
                f"up_blocks.{k}.attentions.2.transformer_blocks.0.ff.net.0.proj.weight":f"output_blocks.{dst_layer_idx+2}.1.transformer_blocks.0.ff.net.0.proj.weight",
                f"up_blocks.{k}.attentions.2.transformer_blocks.0.ff.net.0.proj.bias":f"output_blocks.{dst_layer_idx+2}.1.transformer_blocks.0.ff.net.0.proj.bias",
                f"up_blocks.{k}.attentions.2.transformer_blocks.0.ff.net.2.weight":f"output_blocks.{dst_layer_idx+2}.1.transformer_blocks.0.ff.net.2.weight",
                f"up_blocks.{k}.attentions.2.transformer_blocks.0.ff.net.2.bias":f"output_blocks.{dst_layer_idx+2}.1.transformer_blocks.0.ff.net.2.bias",
                f"up_blocks.{k}.attentions.2.proj_out.weight":f"output_blocks.{dst_layer_idx+2}.1.proj_out.weight",
                f"up_blocks.{k}.attentions.2.proj_out.bias":f"output_blocks.{dst_layer_idx+2}.1.proj_out.bias",
            })

        if k == 0:
            mapper.update({
                f"up_blocks.{k}.upsamplers.0.conv.weight":f"output_blocks.{dst_layer_idx+2}.1.conv.weight",
                f"up_blocks.{k}.upsamplers.0.conv.bias":f"output_blocks.{dst_layer_idx+2}.1.conv.bias",
            })
        elif k != 3:
            mapper.update({
                f"up_blocks.{k}.upsamplers.0.conv.weight":f"output_blocks.{dst_layer_idx+2}.2.conv.weight",
                f"up_blocks.{k}.upsamplers.0.conv.bias":f"output_blocks.{dst_layer_idx+2}.2.conv.bias",
            })


    for k, v in src_state_dict.items():
        new_key = mapper[k]
        # print(k, "->", new_key)
        dst_state_dict[new_key] = v

    return dst_state_dict

def merge_diffuser_vae_v2(sd_dict,vae_dict):

    out_dict = sd_dict.copy()

    block_remap  = {
        'encoder.conv_in':'encoder.conv_in',
        'decoder.conv_in':'decoder.conv_in',

        'encoder.down.0':'encoder.down_blocks.0', # sd_vae: diffuser_vae
        'encoder.down.1':'encoder.down_blocks.1',
        'encoder.down.2':'encoder.down_blocks.2',
        'encoder.down.3':'encoder.down_blocks.3',
        'decoder.up.0':'decoder.up_blocks.3',
        'decoder.up.1':'decoder.up_blocks.2',
        'decoder.up.2':'decoder.up_blocks.1',
        'decoder.up.3':'decoder.up_blocks.0',

        'encoder.mid.block_1':'encoder.mid_block.resnets.0',
        'encoder.mid.attn_1':'encoder.mid_block.attentions.0',
        'encoder.mid.block_2':'encoder.mid_block.resnets.1',
        'decoder.mid.block_1':'decoder.mid_block.resnets.0',
        'decoder.mid.attn_1':'decoder.mid_block.attentions.0',
        'decoder.mid.block_2':'decoder.mid_block.resnets.1',

        'encoder.norm_out':'encoder.conv_norm_out',
        'encoder.conv_out':'encoder.conv_out',
        'decoder.norm_out':'decoder.conv_norm_out',
        'decoder.conv_out':'decoder.conv_out',

        'quant':'quant',
        'post_quant':'post_quant',
        }
    
    block_dict_sd = {}
    block_dict_vae = {}

    for sd,vae in block_remap.items():
        block_dict_sd[sd] = {}
        block_dict_vae[vae] = {}

    for key,val in sd_dict.items():
        for block in block_remap.keys():
            if key.startswith(block):
                block_dict_sd[block][key] = val

    for key,val in vae_dict.items():
        for block in block_remap.values():
            if key.startswith(block):
                block_dict_vae[block][key] = val


    for sd_block,vae_block in block_remap.items():
        for dest,src in zip(block_dict_sd[sd_block],block_dict_vae[vae_block]):
            src_weight = vae_dict[src]
            if not sd_dict[dest].shape == src_weight.shape:
                src_weight = src_weight[...,None,None]
            assert sd_dict[dest].shape == src_weight.shape

            # print(f'remapping {dest} from {src}')

            out_dict[dest] = src_weight.to(dtype=sd_dict[dest].dtype)

    return out_dict
