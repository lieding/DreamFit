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


from PIL import Image 
import numpy as np
import torch
from torch import Tensor
import re
from einops import rearrange

from src.flux.modules.layers_dreamfit  import (
    SingleStreamBlockLoraProcessor,
    DoubleStreamBlockLoraProcessor,
    IPDoubleStreamBlockProcessor,
    ImageProjModel,
)
from src.flux.sampling import denoise, denoise_controlnet_dreamfit, get_noise, get_schedule, prepare, unpack
from src.flux.util import (
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    Annotator,
    load_controlnet,
    load_flow_model_quintized,
    get_lora_rank,
    load_checkpoint,
    load_flow_model_by_type
)

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers import FluxControlNetModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from src.flux.util import TORCH_FP8

class XFluxPipeline:
    def __init__(self, model_type, device, offload: bool = False, image_encoder_path="", lora_path=None, model_path=None):
        self.device = torch.device(device)
        self.offload = offload
        self.model_type = model_type
        self.lora_path = lora_path
        self.clip = load_clip(self.device)
        self.t5 = load_t5(self.device, max_length=512)
        self.ae = load_ae(model_type, self.device)

        if "fp8" in model_type:
            self.model = load_flow_model_quintized(model_type, device="cpu" if offload else self.device)
        else:
            if model_path is None:
                self.model = load_flow_model(model_type, self.device, lora_path=lora_path)
            else:
                self.model = load_flow_model_by_type(model_type, device="cpu" if offload else self.device, lora_path=lora_path, model_type=model_path)

        self.hf_lora_collection = "XLabs-AI/flux-lora-collection"
        self.lora_types_to_names = {
            "realism": "lora.safetensors",
        }
        self.controlnet_loaded = False
        self.ip_loaded = False
        self.lora_loaded = False

        self.image_encoder_path = image_encoder_path 

        self.vae_scale_factor = 16
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_resize=True, do_convert_rgb=True, do_normalize=True)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_resize=True,
            do_convert_grayscale=True,
            do_normalize=False,
            do_binarize=True,
        )

    def set_lora(self, local_path: str = None, repo_id: str = None,
                 name: str = None, lora_weight: int = 0.7, network_alpha=None, double_blocks=None, single_blocks=None):
        checkpoint = load_checkpoint(local_path, repo_id, name)

        self.update_model_with_lora(checkpoint, lora_weight, network_alpha, double_blocks, single_blocks)
        self.lora_loaded = True

    def set_lora_from_collection(self, lora_type: str = "realism", lora_weight: int = 0.7):
        checkpoint = load_checkpoint(
            None, self.hf_lora_collection, self.lora_types_to_names[lora_type]
        )
        self.update_model_with_lora(checkpoint, lora_weight)

    def update_model_with_lora(self, checkpoint, lora_weight, network_alpha, double_blocks, single_blocks):
        rank =  get_lora_rank(checkpoint)

        print("rank ", rank)
        lora_attn_procs = {}
            
        if double_blocks is None:
            double_blocks_idx = list(range(19))
        else:
            double_blocks_idx = [int(idx) for idx in double_blocks.split(",")]

        if single_blocks is None:
            single_blocks_idx = list(range(38))
        elif single_blocks is not None:
            if single_blocks == "":
                single_blocks_idx = []
            else:
                single_blocks_idx = [int(idx) for idx in single_blocks.split(",")]

        # load lora ckpt for modulation
        dit_state_dict = self.model.state_dict()
        modulation_lora_state_dict = {}
        for name in dit_state_dict.keys():
            if 'lin_lora' in name:
                modulation_lora_state_dict[name] = checkpoint[name]
        missing, unexpected = self.model.load_state_dict(modulation_lora_state_dict, strict=False)
        print('missing parameters:', missing)
        print('unexpected parameters:', unexpected)

        # load lora ckpt for attn processor
        for name, attn_processor in self.model.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))

            if name.startswith("double_blocks") and layer_index in double_blocks_idx:
                print("setting LoRA Processor for", name)
                lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(
                    dim=3072, rank=rank, network_alpha=network_alpha
                )

                lora_state_dict = {}
                for k in checkpoint.keys():
                    if name in k:
                        lora_state_dict[k[len(name) + 1:]] = checkpoint[k] * lora_weight

                lora_attn_procs[name].load_state_dict(lora_state_dict)
                lora_attn_procs[name].to(self.device, dtype=TORCH_FP8)

            elif name.startswith("single_blocks") and layer_index in single_blocks_idx:
                print("setting LoRA Processor for", name)
                lora_attn_procs[name] = SingleStreamBlockLoraProcessor(
                    dim=3072, rank=rank, network_alpha=network_alpha
                )

                lora_state_dict = {}
                for k in checkpoint.keys():
                    if name in k:
                        lora_state_dict[k[len(name) + 1:]] = checkpoint[k] * lora_weight

                lora_attn_procs[name].load_state_dict(lora_state_dict)
                lora_attn_procs[name].to(self.device, dtype=TORCH_FP8)

            else:
                lora_attn_procs[name] = attn_processor
                
        self.model.set_attn_processor(lora_attn_procs)


    def set_ip(self, local_path: str = None, repo_id = None, name: str = None):
        self.model.to(self.device)

        # unpack checkpoint
        checkpoint = load_checkpoint(local_path, repo_id, name)
        prefix = "double_blocks."
        blocks = {}
        proj = {}
  
        for key, value in checkpoint.items():
            if key.startswith(prefix):
                blocks[key[len(prefix):].replace('.processor.', '.')] = value

            if key.startswith("ip_adapter_proj_model"):
                proj[key[len("ip_adapter_proj_model."):]] = value

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()

        # setup image embedding projection model
        self.improj = ImageProjModel(4096, 768, 4)
        self.improj.load_state_dict(proj)
        self.improj = self.improj.to(self.device, dtype=TORCH_FP8)

        ip_attn_procs = {}

        for name, _ in self.model.attn_processors.items():
            ip_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    ip_state_dict[k.replace(f'{name}.', '')] = checkpoint[k]
            
            if ip_state_dict:
                ip_attn_procs[name] = IPDoubleStreamBlockProcessor(4096, 3072)
                ip_attn_procs[name].load_state_dict(ip_state_dict)
                ip_attn_procs[name].to(self.device, dtype=TORCH_FP8)
            else:
                ip_attn_procs[name] = self.model.attn_processors[name]

        self.model.set_attn_processor(ip_attn_procs)
        self.ip_loaded = True


    def set_controlnet(self, control_type: str, local_path: str = None, repo_id: str = None, name: str = None, load_method='origin', use_annotator=False):
        self.model.to(self.device)
        if load_method == 'origin':
            self.controlnet = load_controlnet(self.model_type, self.device).to(TORCH_FP8)
            checkpoint = load_checkpoint(local_path, repo_id, name)
            missing_keys, unexpected_keys = self.controlnet.load_state_dict(checkpoint, strict=False)
            print("missing_keys", missing_keys)
            print("unexcepted_keys", unexpected_keys)
            self.annotator = Annotator(control_type, self.device)
        elif load_method == 'diffusers':
            self.controlnet = FluxControlNetModel.from_pretrained(local_path, torch_dtype=TORCH_FP8).to(self.device)
        
        if use_annotator:
            self.annotator = Annotator(control_type, self.device)

        self.controlnet_loaded = True
        self.control_type = control_type

    def get_image_proj(
        self,
        image_prompt: Tensor,
    ):
        # encode image-prompt embeds
        image_prompt = self.clip_image_processor(
            images=image_prompt,
            return_tensors="pt"
        ).pixel_values

        image_prompt = image_prompt.to(self.image_encoder.device)
        image_prompt_embeds = self.image_encoder(
            image_prompt
        ).image_embeds.to(
            device=self.device, dtype=TORCH_FP8,
        )        
        
        # encode image
        image_proj = self.improj(image_prompt_embeds)
        return image_proj

    def __call__(self,
                 prompt: str,
                 ref_prompt: str = "a cloth",
                 ref_img = None,
                 controlnet_image = None,
                 controlnet_mask: Image = None,
                 pose_image: Image = None,
                 width: int = 512,
                 height: int = 512,
                 guidance: float = 4,
                 num_steps: int = 50,
                 seed: int = 123456789,
                 true_gs: float = 3,
                 control_weight: float = 0.9,
                 ip_scale: float = 1.0,
                 neg_ip_scale: float = 1.0,
                 neg_prompt: str = '',
                 neg_image_prompt: Image = None,
                 timestep_to_start_cfg: int = 0,
                 use_annotator=False,
                 control_mode=None,
                 ):
        width = 16 * (width // 16)
        height = 16 * (height // 16)
        image_proj = None
        neg_image_proj = None
  
        if self.ip_loaded and (ref_img is not None):
            image_proj = self.get_image_proj(ref_img)
            
            if neg_image_prompt is None:
                neg_image_prompt = np.zeros((width, height, 3), dtype=np.uint8)
            neg_image_proj = self.get_image_proj(neg_image_prompt)

        if self.lora_loaded and (ref_img is not None):
            ref_prompts  = [ref_prompt] * 1
            ref_img_tensor = torch.from_numpy((np.array(ref_img) / 127.5) - 1)
            ref_img = ref_img_tensor.permute(
                2, 0, 1).unsqueeze(0).to(torch.float32).to(self.device)
            ref_img = self.ae.encode(ref_img).to(TORCH_FP8)

            neg_ref_img = torch.zeros_like(ref_img_tensor)
            neg_ref_img = neg_ref_img.permute(
                2, 0, 1).unsqueeze(0).to(torch.float32).to(self.device)
            # Optimized: If neg_ref_img is all zeros, directly create a zero tensor for neg_image_proj to save VRAM.
            if torch.all(neg_ref_img == 0):
                num_resolutions = len(self.ae.encoder.down)
                F = 2 ** (num_resolutions - 1)
                z_channels = self.ae.encoder.conv_out.out_channels // 2
                B = neg_ref_img.shape[0]
                H_out = neg_ref_img.shape[2] // F
                W_out = neg_ref_img.shape[3] // F
                output_shape = (B, z_channels, H_out, W_out)
                neg_image_proj = torch.zeros(output_shape, device=self.device, dtype=TORCH_FP8)
            else:
                neg_image_proj = self.ae.encode(neg_ref_img).to(TORCH_FP8)

        else:
            ref_prompts  = None
            ref_img = None

        if self.controlnet_loaded:
            if use_annotator:
                controlnet_image = self.annotator(controlnet_image, width, height)
                controlnet_image = torch.from_numpy((np.array(controlnet_image) / 127.5) - 1)
                controlnet_image = controlnet_image.permute(
                    2, 0, 1).unsqueeze(0).to(TORCH_FP8).to(self.device)

        return self.forward(
            prompt,
            width,
            height,
            guidance,
            num_steps,
            seed,
            controlnet_image,
            controlnet_mask=controlnet_mask,
            pose_image=pose_image,
            timestep_to_start_cfg=timestep_to_start_cfg,
            true_gs=true_gs,
            image_proj=image_proj,
            neg_image_proj=neg_image_proj,
            control_weight=control_weight,
            neg_prompt=neg_prompt,
            ref_img=ref_img,
            ref_prompts=ref_prompts,
            ip_scale=ip_scale,
            neg_ip_scale=neg_ip_scale,
            control_mode=control_mode,
        )
 
    def forward(
        self,
        prompt,
        width,
        height,
        guidance,
        num_steps,
        seed,
        controlnet_image = None,
        controlnet_mask = None,
        pose_image = None,
        timestep_to_start_cfg = 0,
        true_gs = 3.5,
        control_weight = 0.9,
        neg_prompt="",
        ref_img=None,
        ref_prompts="",
        image_proj=None,
        neg_image_proj=None,        
        ip_scale=1.0,
        neg_ip_scale=1.0,
        control_mode=None,
    ):
    
        torch.manual_seed(seed)

        x = get_noise(
            1, height, width, device=self.device,
            dtype=TORCH_FP8, seed=seed
        )
        ### same as training
        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // 4,  ## (16 * 16)
            shift=True,
        )
        
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            
            inp_person = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)

            neg_inp_cond = prepare(t5=self.t5, clip=self.clip, img=neg_image_proj, prompt=neg_prompt)
            
            if ref_img is not None:
                inp_cloth = prepare(t5=self.t5, clip=self.clip, img=ref_img, prompt=ref_prompts)
            else:
                inp_cloth = None

            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)

            if self.controlnet_loaded:
                if controlnet_image is not None:
                    if controlnet_mask is not None:
                        controlnet_image, height, width = self.prepare_image_with_mask(
                                                image=controlnet_image,
                                                mask=controlnet_mask,
                                                width=width,
                                                height=height,
                                                batch_size=1,
                                                num_images_per_prompt=1,
                                                device=self.device,
                                                dtype=TORCH_FP8,
                                                pose_image=pose_image,
                                        )
                    else:
                        controlnet_image = self.ae.encode(controlnet_image.to(dtype=torch.float)).to(TORCH_FP8)
                        if self.controlnet.input_hint_block is None:
                            # pack
                            height_control_image, width_control_image = controlnet_image.shape[2:]
                            controlnet_image = self._pack_latents(
                                controlnet_image,
                                controlnet_image.shape[0],
                                controlnet_image.shape[1],
                                height_control_image,
                                width_control_image,
                            )
                    if control_mode is not None:
                        if not isinstance(control_mode, int):
                            raise ValueError(" For `FluxControlNet`, `control_mode` should be an `int` or `None`")
                        control_mode = torch.tensor(control_mode).to(self.model.device, dtype=torch.long)
                        control_mode = control_mode.view(-1, 1).expand(controlnet_image.shape[0], 1)

                print(">>> Using controlnet ...")
                x = denoise_controlnet_dreamfit(
                    self.model,
                    **inp_person,
                    controlnet=self.controlnet,
                    timesteps=timesteps,
                    guidance=guidance,
                    controlnet_cond=controlnet_image,
                    timestep_to_start_cfg=timestep_to_start_cfg,
                    neg_txt=neg_inp_cond['txt'],
                    neg_txt_ids=neg_inp_cond['txt_ids'],
                    neg_vec=neg_inp_cond['vec'],
                    true_gs=true_gs,
                    controlnet_gs=control_weight,
                    image_proj=image_proj,
                    neg_image_proj=neg_image_proj,
                    ip_scale=ip_scale,
                    neg_ip_scale=neg_ip_scale,
                    control_mode=control_mode,
                    inp_cloth=inp_cloth
                )
            else:
                print(">>> Using denoise ...")
                x = denoise(
                    self.model,
                    inp_person=inp_person,
                    inp_cloth=inp_cloth,
                    timesteps=timesteps,
                    guidance=guidance,
                    timestep_to_start_cfg=timestep_to_start_cfg,
                    neg_inp_cond=neg_inp_cond,
                    true_gs=true_gs,
                    image_proj=None,
                    neg_image_proj=None,
                    ip_scale=ip_scale,
                    neg_ip_scale=neg_ip_scale,
                    num_steps=num_steps
                )

            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(x.device)
            
            x = unpack(x.float(), height, width)
            x = self.ae.decode(x)
            self.offload_model_to_cpu(self.ae.decoder)

        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
        return output_img

    def prepare_image_with_mask(
        self,
        image,
        mask,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        pose_image=None,
    ):
        # Prepare image
        if isinstance(image, torch.Tensor):
            pass
        else:
            image = self.image_processor.preprocess(image, height=height, width=width)

        image_batch_size = image.shape[0]
        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt
        image = image.repeat_interleave(repeat_by, dim=0)
        image = image.to(device=device, dtype=dtype)

        # Prepare mask
        if isinstance(mask, torch.Tensor):
            pass
        else:
            mask = self.mask_processor.preprocess(mask, height=height, width=width)
        mask = mask.repeat_interleave(repeat_by, dim=0)
        mask = mask.to(device=device, dtype=dtype)

        # Get masked image
        masked_image = image.clone()

        # Encode to latents
        image_latents = self.ae.encode(masked_image.to(dtype=torch.float))
        image_latents = image_latents.to(dtype)

        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor * 2, width // self.vae_scale_factor * 2)
        )
        mask = 1 - mask

        control_image = torch.cat([image_latents, mask], dim=1)

        if pose_image is not None:
            if isinstance(pose_image, torch.Tensor):
                pass
            else:
                pose_image = self.image_processor.preprocess(pose_image, height=height, width=width)
            pose_image = pose_image.repeat_interleave(repeat_by, dim=0)
            pose_image = pose_image.to(device=device, dtype=dtype)
            pose_latents = self.ae.encode(pose_image.to(dtype=torch.float))
            pose_latents = pose_latents.to(dtype)
            control_image = torch.cat([pose_latents, image_latents, mask], dim=1)

        # Pack cond latents( has been written in controlnet)
        packed_control_image = self._pack_latents(
            control_image,
            batch_size * num_images_per_prompt,
            control_image.shape[1],
            control_image.shape[2],
            control_image.shape[3],
        )
        return packed_control_image, height, width

    def _pack_latents(self, latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    def offload_model_to_cpu(self, *models):
        if not self.offload: return
        for model in models:
            model.cpu()
            torch.cuda.empty_cache()
