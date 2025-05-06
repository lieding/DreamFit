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
from torch.utils.checkpoint import checkpoint
import open_clip
import einops
import os
from src.utils.util import get_class

try:
    from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
    HF_TRANSFORMER_AVALIABLE = True
except:
    HF_TRANSFORMER_AVALIABLE = False


#This is a for backward compability for models based on StableDiffusion-1.x
class FrozenCLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self, model_path=None,version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        if not HF_TRANSFORMER_AVALIABLE:
            raise ImportError
        if model_path:
            self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
            self.transformer = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(version)
            self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, return_pooled_embeds=False):
        batch_encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_length=True,return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)

        outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedderWithProjection(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self, model_path=None,version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        if not HF_TRANSFORMER_AVALIABLE:
            raise ImportError
        if model_path:
            self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2")
            self.transformer = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder_2")
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(version)
            self.transformer = CLIPTextModelWithProjection.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, return_pooled_embeds=False):
        batch_encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_length=True,return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)

        outputs = self.transformer(input_ids=tokens, output_hidden_states=True)
        z = outputs.hidden_states[-2]
        if return_pooled_embeds:
            return z, outputs[0]
        else:
            return z

    def encode(self, text):
        return self(text)

#This is a for backward compability for models based on StableDiffusion-2.0
class OpenCLIPTextEncoder(nn.Module):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        # "pooled",
        "last",
        "penultimate"
    ]

    def __init__(
            self,
            model_name="ViT-L-14",
            model_path=None,
            device="cuda",
            max_length=77,
            freeze=True,
            pretrained=None,
            layer="penultimate"):
        super().__init__()
        assert layer in self.LAYERS
        model = open_clip.create_model(model_name, device="cpu", pretrained=pretrained)
        self.positional_embedding = model.positional_embedding
        self.token_embedding = model.token_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.attn_mask = model.attn_mask
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

        if model_path is not None:
            self.load_pretrained(model_path)
            print('load pretrained text openclip')

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False


    def forward(self, text):
        batch_size = len(text)
        x = self.encode(text)

        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, fn in enumerate(self.transformer.resblocks):
            if i == len(self.transformer.resblocks) - self.layer_idx:
                break
            if self.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(fn, x, attn_mask)
            else:
                x = fn(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        tokens = open_clip.tokenize(text)
        x = self.token_embedding(tokens.to(self.device))
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        dtype = x.dtype
        attn_mask = self.attn_mask.to(dtype)
        attn_mask = attn_mask.to(x.device)
        x = self.text_transformer_forward(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        return x

    def load_pretrained(self, path):
        if not os.path.exists(path):
            return

        remap = {
            'visual': 'model', }
        loaded_state_dict = torch.load(path, map_location='cpu')
        state_dict = {}
        for key in loaded_state_dict:
            for old_key in remap:
                if key.startswith(old_key):
                    new_key = key.replace(old_key, remap[old_key])
                    state_dict[new_key] = loaded_state_dict[key]

        # 1. load visual model
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        print("missing_keys", missing_keys)
        print("unexcepted_keys", unexpected_keys)

        return


class ConcatTextEncoder(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.encoders = nn.ModuleList()
        for data in config:
            model_class = get_class(data["target"])
            params = data["params"]
            self.encoders.append(model_class(**params))

    def freeze(self):
        for encoder in self.encoders:
            encoder.freeze()

    def free_parameters(self):
        params = []
        for encoder in self.encoders:
            params = params + list(encoder.free_parameters())
        return params

    def forward(self, data, return_pooled_embeds=False):

        output = []
        pooled_embeds = []
        for encoder in self.encoders:
            if return_pooled_embeds:
                z, pooled_embed = encoder(data, return_pooled_embeds=return_pooled_embeds)
                output.append(z)
                pooled_embeds.append(pooled_embed)
            else:
                output.append(encoder(data))

        # N x L x C
        output = torch.cat(output, dim=2)

        if return_pooled_embeds:
            return output, pooled_embeds
        else:
            return output


if __name__ == '__main__':
    model = FrozenCLIPEmbedder()
