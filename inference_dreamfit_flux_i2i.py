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
from PIL import Image
from datetime import datetime
import logging
import torch
import PIL

from omegaconf import OmegaConf

from src.flux.xflux_pipeline_dreamfit_optimised import XFluxPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
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
        default=164143088151,
        help="random seed. Default is 164143088151",
    )
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M%S")
    LOG_FORMAT = "%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s"
    logging.basicConfig(filename=f'run_{date_str}_{time_str}.log', level=logging.INFO, \
        datefmt='%a, %d %b %Y %H:%M:%S', format=LOG_FORMAT, filemode='w')

    xflux_pipeline = XFluxPipeline(config.model_type, config.device, config.offload, \
                                   image_encoder_path=config.image_encoder_path,\
                                    lora_path=config.lora_local_path, model_path=config.model_path)

    if config.use_ip:
        print('>>> load ip-adapter:', config.ip_local_path, config.ip_repo_id, config.ip_name)
        xflux_pipeline.set_ip(config.ip_local_path, config.ip_repo_id, config.ip_name)

    if config.use_lora:
        print('>>> load lora:', config.lora_local_path, config.lora_repo_id, config.lora_name)
        xflux_pipeline.set_lora(config.lora_local_path, config.lora_repo_id, \
            config.lora_name, config.lora_weight, config.network_alpha, config.double_blocks, config.single_blocks)

    if config.use_controlnet:
        print('>>> load controlnet:', config.ctlnet_local_path, config.repo_id, config.name)
        xflux_pipeline.set_controlnet(config.control_type, config.local_path, config.repo_id, config.name)
    
    # set params
    weight_dtype = torch.float16
    height = config.inference_params.height
    width = config.inference_params.width
    init_seed = args.seed
    size = (width, height) 

    # input 
    save_dir = args.save_dir
    cloth_path = args.cloth_path
    image_text = args.image_text
    cloth_text = 'cloth' # fixed

    # load cloth image
    cloth = PIL.Image.open(cloth_path).convert("RGB").resize(size)

    # inference
    for ind in range(config.num_images_per_prompt):
        result = xflux_pipeline(
            prompt=image_text,
            controlnet_image=None,
            width=width,
            height=height,
            guidance=config.guidance,
            num_steps=config.num_steps,
            seed=init_seed,
            true_gs=config.true_gs,
            control_weight=config.control_weight,
            neg_prompt=config.neg_prompt,
            timestep_to_start_cfg=config.timestep_to_start_cfg,
            ref_img=cloth,
            ref_prompt=cloth_text,
            neg_image_prompt=None,
            ip_scale=config.ip_scale,
            neg_ip_scale=config.neg_ip_scale,
        )

        filename = cloth_path.split('/')[-1]
        file_path = f'{save_dir}/{ind}_{filename}'

        result.save(file_path)
        print("save to ", file_path)

        init_seed = init_seed + 1


if __name__ == "__main__":
    main()
