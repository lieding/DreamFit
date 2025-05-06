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

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from src.utils.util import from_config , convert_unet_state_dict_from_open_animate_anyone, merge_diffuser_vae_v2

from src.utils.util import randn_tensor
from src.stablediffusion.schedulers import DDIMScheduler
import torchvision.transforms as T
from omegaconf import DictConfig
from src.stablediffusion.lora import LoRACompatibleConv, LoRACompatibleLinear, LoRAConv2dLayer, LoRALinearLayer

from PIL import Image
import numpy as np
import copy

from safetensors import safe_open
from safetensors.torch import save_file
from safetensors.torch import load_file as safe_load_file

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
)


def _strong_match(src, dest):
    return src.shape == dest.shape


def _attention_match(src, dest):
    if len(src.shape) == 4 and src.shape[:2] == dest.shape[:2] and \
            src.shape[2] == 1 and src.shape[3] == 1:
        return True
    return False

def _conv_match(src, dest):
    if len(src.shape) == 2 and src.shape[:2] == dest.shape[:2]:
        return True
    return False

def _match_input_blocks(src, dest):
    add_channel = dest.shape[1] - src.shape[1]
    if add_channel > 0:
        add_weight = torch.zeros(
            src.shape[0], add_channel, *src.shape[2:], dtype=src.dtype)
        nn.init.xavier_uniform_(add_weight, 1e-5)
        new_weight = torch.cat([src, add_weight], dim=1)
    elif add_channel < 0:
        new_weight = src[:, :dest.shape[1]]
    else:
        new_weight = src
    return new_weight


class DreamFit_inference(pl.LightningModule):
    def __init__(
            self,
            target_unet,
            vae,
            ref_unet=None,
            noise_scheduler=None,
            text_encoder=None,
            prediction_type: str = "v",
            ref_names = ["cloth"],
            att_cond_names = ['text'],
            scheduler_type = 'ddim',
            proj_lora_rank=64,
            mapped_network_alphas=8,
            use_network_alpha=True,
            ):

        super().__init__()
        self.target_unet = target_unet
        self.vae = vae
        self.ref_unet = ref_unet
        self.noise_scheduler = noise_scheduler
        self.prediction_type = prediction_type
        self.ref_names = ref_names
        self.att_cond_names = att_cond_names
        self.text_encoder = text_encoder
        self.scheduler_type = scheduler_type

        self._init_lora(unet=self.ref_unet, mapped_network_alphas=mapped_network_alphas, rank=proj_lora_rank, use_network_alpha=use_network_alpha,)

    @staticmethod
    def _parse_config_kwargs(config):
        kwargs = {}
        for key, val in dict(config).items():
            if isinstance(val, DictConfig):
                val = dict(val)
            if isinstance(val, dict) and 'target' in val:
                continue
            kwargs.update({key: val})
        return kwargs

    @classmethod
    def from_config(cls, config):
        vae = from_config(config.vae)
        target_unet = from_config(config.target_unet)

        if "text_encoder" in config:
            print("init w. text_encoder")
            text_encoder = from_config(config.text_encoder)
        else:
            text_encoder = None
            print("init w.o ref_text_encoder_unet")

        noise_scheduler = from_config(config.noise_scheduler)
        kwargs = cls._parse_config_kwargs(config)

        if "ref_unet" in config:
            print("init w. ref_unet")
            ref_unet = from_config(config.ref_unet)
        else:
            ref_unet = None
            print("init w.o ref_unet")

        return cls(target_unet=target_unet,
                    vae=vae,
                    ref_unet=ref_unet,
                    noise_scheduler=noise_scheduler, 
                    text_encoder=text_encoder, 
                    **kwargs)

    def encode_latents(self, input, sample=True):
        if isinstance(self.vae, AutoencoderKL):
            latent = self.vae.encode(input).latent_dist
            scaling_factor = self.vae.config.scaling_factor
        else:
            latent = self.vae.encode(input)
            scaling_factor = self.vae.scale_factor

        if sample:
            return (latent.sample() * scaling_factor).type(input.dtype)        
        else:
            return (latent.mode() * scaling_factor).type(input.dtype)

    def decode_latents(self, input):
        if isinstance(self.vae, AutoencoderKL):
            scaling_factor = self.vae.config.scaling_factor
        else:
            scaling_factor = self.vae.scale_factor
        return self.vae.decode(input / scaling_factor)

    def _init_lora(self, unet, mapped_network_alphas=8, rank=64, use_network_alpha=True):
        attn_processors = []
        non_attn_lora_layers = []
        state_dict = unet.state_dict()

        for key, value_dict in state_dict.items():

            attn_processor = unet
            for sub_key in key.split('.'):
                attn_processor = getattr(attn_processor, sub_key) 
                
                if isinstance(attn_processor, LoRACompatibleConv):
                    in_features = attn_processor.in_channels
                    out_features = attn_processor.out_channels
                    kernel_size = attn_processor.kernel_size

                    lora = LoRAConv2dLayer(
                        in_features=in_features,
                        out_features=out_features,
                        rank=rank,
                        kernel_size=kernel_size,
                        stride=attn_processor.stride,
                        padding=attn_processor.padding,
                        network_alpha=mapped_network_alphas if use_network_alpha else None,
                    )
                    non_attn_lora_layers.append((attn_processor, lora))
                elif isinstance(attn_processor, LoRACompatibleLinear):
                    lora = LoRALinearLayer(
                        attn_processor.in_features,
                        attn_processor.out_features,
                        rank,
                        mapped_network_alphas if use_network_alpha else None,
                    )
                    non_attn_lora_layers.append((attn_processor, lora))

                elif isinstance(attn_processor, LoRALinearLayer):
                    attn_processors.append(attn_processor)

        # set correct dtype & device
        non_attn_lora_layers = [(t, l) for t, l in non_attn_lora_layers]

        # set layers
        for attn_processor in attn_processors:
            attn_processor.network_alpha = mapped_network_alphas if use_network_alpha else None

        # set ff layers
        for target_module, lora_layer in non_attn_lora_layers:
            target_module.set_lora_layer(lora_layer)

    @staticmethod
    def cat_condition(cond, uncond):
        if cond is None:
            return None
        is_list = isinstance(cond, list)
        if not is_list:
            cond = [cond] # [2,77,768]
            uncond = [uncond] # [2, 77, 768]
        ret = [torch.cat([c, uc], dim=0) for c, uc in zip(cond, uncond)] # [4, 77, 768]
        if not is_list:
            ret = ret[0]
        return ret

    def prepare_model_input(self, batch, shape):
        # load data
        dtype = batch['cloth'].dtype

        cloth = batch['cloth']

        cloth_text = batch['cloth_text'] if 'cloth_text' in batch else ['']
        image_text = batch['image_text'] if 'image_text' in batch else ['']


        x_0_ref_list = []
        cond_target_attn_list = []
        cond_ref_attn_list = []
        uncond_target_attn_list = []
        uncond_ref_attn_list = []
        for name in self.ref_names: # ref_names=['cloth']

            latent = self.encode_latents(batch[name], sample=False)

            x_0_ref_list.append(latent)

        for name in self.att_cond_names:
            cond_target_attn = self.text_encoder(image_text)
            cond_ref_attn = self.text_encoder(cloth_text)
            if 'negtive_prompt' in batch and 'image_text' in batch['negtive_prompt']:
                uncond_target_attn = self.text_encoder(batch['negtive_prompt']['image_text'])
            else:
                uncond_target_attn = self.text_encoder([''])
            if 'negtive_prompt' in batch and 'cloth_text' in batch['negtive_prompt']:
                uncond_ref_attn = self.text_encoder(batch['negtive_prompt']['cloth_text'])
            else:
                uncond_ref_attn = self.text_encoder([''])

            cond_target_attn_list.append(cond_target_attn)
            cond_ref_attn_list.append(cond_ref_attn)
            uncond_target_attn_list.append(uncond_target_attn)
            uncond_ref_attn_list.append(uncond_ref_attn)

        # CFG by concat
        for k in range(len(cond_target_attn_list)):
            cond_target_attn_list[k] = self.cat_condition(uncond_target_attn_list[k], cond_target_attn_list[k])
        for k in range(len(cond_ref_attn_list)):
            cond_ref_attn_list[k] = self.cat_condition(uncond_ref_attn_list[k], cond_ref_attn_list[k])


        cond_attn_list = [cond_target_attn_list, cond_ref_attn_list]

        return x_0_ref_list, cond_attn_list

    def inference(
            self,
            shape,
            _input,
            num_inference_timesteps=100, guidance_scale=5.0,  eta=0.0, 
            noise=None,
            x_0 = None):

        if self.scheduler_type == 'ddim':
            scheduler = DDIMScheduler(
                self.noise_scheduler.num_train_timesteps,
                self.noise_scheduler.beta_start,
                self.noise_scheduler.beta_end,
                self.noise_scheduler.beta_schedule,
                steps_offset=self.noise_scheduler.steps_offset,
                prediction_type=self.prediction_type)
        else:
            raise ValueError("scheduler_type should be 'ddim'")
            
        device = self.device

        latent_shape = (1,4,int(shape[0]/8),int(shape[1]/8))
        x_0_ref_list, cond_attn_list = self.prepare_model_input(_input, shape)

        cond_target_attn_list, cond_ref_attn_list = cond_attn_list
        add_target_embeds = add_ref_embeds = None

        scheduler.set_timesteps(num_inference_timesteps, device=self.device)

        if noise is None:
            x_t = randn_tensor(latent_shape, device=device)
        else:
            x_t = noise.to(device)

        timesteps = scheduler.timesteps

        x_0_ref_list_ = []
        for k in range(len(x_0_ref_list)):
            x_0_ref_list_.append(torch.cat([x_0_ref_list[k]] * 2, dim=0)) 

        ref_scheduler = {
          0: list(range(len(self.ref_names))) # ref_names=['text'], key=0, value = 0 or [0, 1],
        }

        idx_list = ref_scheduler[0]
        if self.ref_unet is not None:
            ref_kv_pair_lists = []
            for x_0_ref_, cond_attn in zip(
                [x_0_ref_list_[idx] for idx in idx_list],  # [4, 77, 768],
                [cond_ref_attn_list[idx] for idx in idx_list] # [4, 4, 64, 48]
            ): 
                t_ref = torch.zeros_like(timesteps)[0]
                ref_kv_pair_lists.append(self.ref_unet(x_0_ref_, t_ref, context=cond_attn, embeddings=add_ref_embeds))
        else:
            ref_kv_pair_lists = None

        for i, t in enumerate(timesteps):
            x_t_ = torch.cat([x_t]*2, dim=0)
            model_input = x_t_

            ref_kv_pair_lists_used_in_target = copy.deepcopy(ref_kv_pair_lists) if ref_kv_pair_lists else None

            model_output = self.target_unet(
                        model_input, 
                        t, 
                        context=torch.cat(cond_target_attn_list, dim=1), 
                        embeddings=add_target_embeds,
                        ref_kv_pair_lists_used_in_target=ref_kv_pair_lists_used_in_target,
                        ref_scale = _input['ref_scale'] if 'ref_scale' in _input else 1.0,
                        )

            model_output_uc, model_output = model_output.chunk(2)
            model_output = model_output_uc  + guidance_scale * (model_output - model_output_uc)

            x_t = scheduler.step(model_output, t, x_t, eta=eta)[0]

        output = self.decode_latents(x_t)

        return output[0]

    def load_stable_diffusion(self, 
            model_path_ref_unet=None,
            model_path_target_unet=None,
            model_path_vae=None,
            ):

        model_path_vae = model_path_vae if model_path_vae else model_path_target_unet
        if model_path_vae:
            self.load_vae(model_path_vae)
        if model_path_ref_unet:
            self.load_ref_unet(model_path_ref_unet)
        if model_path_target_unet:
            self.load_target_unet(model_path_target_unet)

    def load_vae(self, model_path, load_method='origin'):
        """
          load vae from full sd model
        """
        if load_method is not None:
            self.vae_load_method = load_method

        if model_path.endswith(".ckpt") or model_path.endswith(".bin"):
            loaded_state_dict = torch.load(model_path, map_location='cpu')
        elif model_path.endswith(".safetensors"):
            loaded_state_dict = safe_load_file(model_path)
        elif model_path.endswith("vae"):
            self.vae = AutoencoderKL.from_pretrained(
                model_path,
            )
            return
        if "state_dict" in loaded_state_dict:
            loaded_state_dict = loaded_state_dict["state_dict"]
        remap = {'first_stage_model.': ''}
        dest_state_dict = self.vae.state_dict()
        state_dict = {}

        if self.vae_load_method == 'origin':
            for key in loaded_state_dict:
                for old_key in remap:
                    if key.startswith(old_key):
                        new_key = key.replace(old_key, remap[old_key])
                        src_weight = loaded_state_dict[key]
                        dest_weight = dest_state_dict[new_key]

                        if _strong_match(src_weight, dest_weight):
                            state_dict[new_key] = src_weight
                        else:
                            raise ValueError(
                                f'Unexpected shape mismatch {new_key}:{dest_weight.shape} <- {key}:{src_weight.shape}')
        elif self.vae_load_method == 'diffusers':
            state_dict = merge_diffuser_vae_v2(dest_state_dict, loaded_state_dict)

        print("load vae {}".format(model_path))
        missing_keys, unexpected_keys = self.vae.load_state_dict(state_dict, strict=True)
        print("missing_keys", missing_keys)
        print("unexcepted_keys", unexpected_keys)
        return

    def load_ref_unet(self, model_path, load_method='origin'):
        if model_path.endswith(".ckpt") or model_path.endswith(".pth"):
            loaded_state_dict = torch.load(model_path, map_location='cpu')
        elif model_path.endswith(".safetensors"):
            loaded_state_dict = safe_load_file(model_path)

        if 'state_dict' in loaded_state_dict:
            loaded_state_dict = loaded_state_dict['state_dict']

        if load_method == 'origin':
            dest_state_dict = self.ref_unet.state_dict()
            state_dict = {}
            remap = {'model.diffusion_model.': ''}
            dest_state_dict = self.ref_unet.state_dict()
            state_dict = {}
            for key in loaded_state_dict:
                for old_key in remap:
                    if key.startswith(old_key):
                        new_key = key.replace(old_key, remap[old_key])
                        src_weight = loaded_state_dict[key]
                        dest_weight = dest_state_dict[new_key]

                        if _strong_match(src_weight, dest_weight):
                            state_dict[new_key] = src_weight
                            if (new_key.startswith('middle_block') or new_key.startswith('output_blocks')) and ('attn1.to_k' in new_key or 'attn1.to_v' in new_key):
                                to_kv = new_key.split('.')[-2] 
                                addtion_key = new_key.split('attn1')[0] + 'attn1' + f'.ref_{to_kv}' + '.weight'
                                state_dict[addtion_key] = src_weight
                        elif _attention_match(src_weight, dest_weight):
                            state_dict[new_key] = src_weight[..., 0, 0]
                            if (new_key.startswith('middle_block') or new_key.startswith('output_blocks')) and ('attn1.to_k' in new_key or 'attn1.to_v' in new_key):
                                to_kv = new_key.split('.')[-2] 
                                addtion_key = new_key.split('attn1')[0] + 'attn1' + f'ref_{to_kv}' + '.weight'
                                state_dict[addtion_key] = src_weight[..., 0, 0]
                        elif new_key == 'input_blocks.0.0.weight':
                            new_weight = _match_input_blocks(
                                src_weight, dest_weight)
                            state_dict[new_key] = new_weight
                        else:
                            raise ValueError(
                                f'Unexpected shape mismatch {new_key}:{dest_weight.shape} <- {key}:{src_weight.shape}')
        elif load_method == 'from_exist_model':
            state_dict = {}
            for key, value in loaded_state_dict.items():
                if 'target_unet' in key:
                    new_key = key.replace('target_unet.', '')
                    state_dict[new_key] = value

        print("load ref_unet {}".format(model_path))
        missing_keys, unexpected_keys = self.ref_unet.load_state_dict(state_dict, strict=False)
        print("missing_keys", missing_keys)
        print("unexcepted_keys", unexpected_keys)

    def load_target_unet(self, model_path, load_method='origin'):
        if model_path.endswith(".ckpt") or model_path.endswith(".pth"):
            loaded_state_dict = torch.load(model_path, map_location='cpu')
        elif model_path.endswith(".safetensors"):
            loaded_state_dict = safe_load_file(model_path)
            if 'state_dict' in loaded_state_dict:
                loaded_state_dict = loaded_state_dict['state_dict']
        elif model_path.endswith("unet"):
            self.target_unet = UNet2DConditionModel.from_pretrained(
                model_path,
            )
            return
        elif model_path.endswith(".bin"):
            loaded_state_dict = torch.load(model_path, map_location='cpu')
            if 'state_dict' in loaded_state_dict:
                loaded_state_dict = loaded_state_dict['state_dict']
        else:
            loaded_state_dict = torch.load(model_path, map_location='cpu')

        if 'state_dict' in loaded_state_dict:
            loaded_state_dict = loaded_state_dict['state_dict']

        if load_method == 'origin':
            remap = {'model.diffusion_model.': ''}
            dest_state_dict = self.target_unet.state_dict()
            state_dict = {}
            for key in loaded_state_dict:
                for old_key in remap:
                    if key.startswith(old_key):
                        new_key = key.replace(old_key, remap[old_key])
                        src_weight = loaded_state_dict[key]
                        dest_weight = dest_state_dict[new_key]

                        if _strong_match(src_weight, dest_weight):
                            state_dict[new_key] = src_weight
                        elif _attention_match(src_weight, dest_weight):
                            state_dict[new_key] = src_weight[..., 0, 0]
                        elif _conv_match(src_weight, dest_weight):
                            state_dict[new_key] = src_weight[..., None, None]
                        elif new_key == 'input_blocks.0.0.weight':
                            new_weight = _match_input_blocks(
                                src_weight, dest_weight)
                            state_dict[new_key] = new_weight
                        else:
                            raise ValueError(
                                f'Unexpected shape mismatch {new_key}:{dest_weight.shape} <- {key}:{src_weight.shape}')
        elif load_method =='diffusers':
            state_dict = convert_unet_state_dict_from_open_animate_anyone(loaded_state_dict)

        print("load target_unet {}".format(model_path))
        missing_keys, unexpected_keys = self.target_unet.load_state_dict(state_dict, strict=False)
        print("missing_keys", missing_keys)
        print("unexcepted_keys", unexpected_keys)