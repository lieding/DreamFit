# Copyright (c) 2024 Black Forest Labs and The XLabs-AI Team. 

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch import Tensor, nn
from transformers import (CLIPTextModel, CLIPTokenizer, T5EncoderModel,
                          T5Tokenizer)
from nunchaku import NunchakuT5EncoderModel


class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            clip_tokenizer_path = "/home/featurize/work/pretrained_models/tokenizer"
            clip_enc_path = "/home/featurize/work/pretrained_models/text_encoder"
            
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(clip_tokenizer_path, max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(clip_enc_path, **hf_kwargs)

        else:
            t5_tokenizer_path = "/home/featurize/work/pretrained_models/tokenizer_2"
            t5_enc_path = "/home/featurize/work/pretrained_models/text_encoder_2"
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(t5_tokenizer_path, max_length=max_length)
            self.hf_module: T5EncoderModel = NunchakuT5EncoderModel.from_pretrained("mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors", **hf_kwargs)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
