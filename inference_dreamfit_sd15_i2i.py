# Copyright (c) 2024 Black Forest Labs and The XLabs-AI Team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0 
#
# This file has been modified by Bytedance Ltd. and/or its affiliates on 2025/5/6.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/XLabs-AI/x-flux/blob/main/LICENSE.
#
# This modified file is released under the same license.


import argparse
import numpy as np
import os.path
from omegaconf import OmegaConf
import PIL
from PIL import Image
import json
import cv2
import matplotlib
from safetensors.torch import load_file as safe_load_file

import torch
import pytorch_lightning as pl

from src.utils.util import get_class


DEFAULT_LOGDIR = "logs"

def to_single_channel(array: np.ndarray):
    if array.ndim == 3:
        return array[..., :1]
    else:
        return array[..., np.newaxis]

def _load_image(img_path, size=(512, 384), dtype=np.float32, normalize=False, force_rgb=True, use_resize=False):
    im = PIL.Image.open(img_path)
    if use_resize:
        size = (size[1], size[0])
        im = im.resize(size)
    if force_rgb:
        im = im.convert("RGB")
    image = np.array(im)
    im.close()
    if dtype is None:
        return image
    elif np.issubdtype(dtype, np.integer):
        return image.astype(dtype)

    if np.issubdtype(image.dtype, np.integer):
        denom = np.iinfo(image.dtype).max
        image = image.astype(dtype) / denom
    else:
        image = image.astype(dtype)

    if normalize:
        image = image * 2 - 1

    return image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="path to config file",
    )

    parser.add_argument(
        "--vae_ckpt",
        type=str,
        default="",
        help="path to other vae model file",
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="",
        help="path to other base model file, like RealisticVision",
    )

    parser.add_argument(
        "--base_model_load_method",
        type=str,
        choices=["origin", "diffusers"],
        default="origin",
        help="Method to load the base model. Choices: ['origin', 'diffusers']. Default is 'origin'.",
    )

    parser.add_argument(
        "--ref_model",
        type=str,
        default="",
        help="path to reference model file",
    )

    parser.add_argument(
        "--ref_model_load_method",
        type=str,
        choices=["from_exist_model"],
        default="from_exist_model",
        help="Method to load the reference model. Choices: ['from_exist_model']. Default is 'from_exist_model'.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default='.',
        required=False,
        help="save directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=28,
        help="random seed. Default is 28",
    )
    parser.add_argument(
        "--ref_scale",
        type=float,
        default=1.0,
        help="control reference strength",
    )
    parser.add_argument(
        "--cloth_path",
        type=str,
        default=None,
        required=True,
        help="path to cloth",
    )

    parser.add_argument(
        "--image_text",
        type=str,
        default=None,
        required=True,
        help="sentence describing the output image",
    )
    return parser.parse_known_args()


def inference(model, shape, _input, filename, num_inference_timesteps=50, guidance_scale=7.5, eta=0.0):
    with torch.no_grad():
        output = model.inference(shape, _input, num_inference_timesteps=num_inference_timesteps, guidance_scale=guidance_scale, eta=eta)

    output = (output+1)/2
    output = output.clamp(0.0, 1.0)
    output = output.transpose(0, 1).transpose(1, 2).squeeze(-1)
    output = output.detach().cpu().numpy()
    output = (output * 255).astype(np.uint8)
    output = Image.fromarray(output)
    output.save(f'./{filename}')


if __name__ == "__main__":
    args, cli = parse_args()

    # load config
    config = OmegaConf.load(args.config)

    # prepare model
    model_class = get_class(config.model.target)
    model = model_class.from_config(config.model.params)

    # prepare weights
    base_model_path = args.base_model
    if os.path.isfile(base_model_path):
        model.load_target_unet(base_model_path, load_method=args.base_model_load_method)
        print('init base model')

    ref_model_path = args.ref_model
    if os.path.isfile(ref_model_path):
        model.load_ref_unet(ref_model_path, load_method=args.ref_model_load_method)
        print('init reference model')

    vae_ckpt = args.vae_ckpt
    if os.path.isfile(vae_ckpt):
        model.load_vae(vae_ckpt, 'diffusers')
        print('init vae from vae-sft')

    # input
    _input = {}
    shape = (768, 512)
    save_dir = args.save_dir
    torch.manual_seed(args.seed)

    # load image text
    _input['image_text'] = [args.image_text]

    # load cloth text
    _input['cloth_text'] = ['cloth']

    # load cloth
    cloth_path = args.cloth_path
    _input['cloth'] = torch.tensor(_load_image(cloth_path, normalize=True, use_resize=True, size=(shape[0], shape[1]))).cuda().unsqueeze(0).permute(0,3,1,2)
    
    _input['ref_scale'] = args.ref_scale
    
    # inference
    model.cuda()
    model.eval()

    filename = cloth_path.split('/')[-1]
    file_path = f'{save_dir}/{filename}'
    inference(model, shape, _input, filename)
