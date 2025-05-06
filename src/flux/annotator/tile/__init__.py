# Copyright (c) 2024 Black Forest Labs and The XLabs-AI Team. 
# Original file was released at https://github.com/XLabs-AI/x-flux

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import cv2
from .guided_filter import FastGuidedFilter


class TileDetector:
    # https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0
    def __init__(self):
        pass

    def __call__(self, image):
        blur_strength = random.sample([i / 10. for i in range(10, 201, 2)], k=1)[0]
        radius = random.sample([i for i in range(1, 40, 2)], k=1)[0]
        eps = random.sample([i / 1000. for i in range(1, 101, 2)], k=1)[0]
        scale_factor = random.sample([i / 10. for i in range(10, 181, 5)], k=1)[0]

        ksize = int(blur_strength)
        if ksize % 2 == 0:
            ksize += 1

        if random.random() > 0.5:
            image = cv2.GaussianBlur(image, (ksize, ksize), blur_strength / 2)
        if random.random() > 0.5:
            filter = FastGuidedFilter(image, radius, eps, scale_factor)
            image = filter.filter(image)
        return image
