# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/haotian-liu/LLaVA/

import os

from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .intern_encoder import InternVisionTower, InternVisionTowerS2
# from .ps3_encoder import PS3VisionTower
from .radio_encoder import RADIOVisionTower
from .siglip_encoder import SiglipVisionTower, SiglipVisionTowerDynamicS2, SiglipVisionTowerS2


def build_vision_tower(model_name_or_path: str, config: PretrainedConfig) -> PreTrainedModel:
    ## skip vision tower instantiation
    if model_name_or_path is None:
        return None

    vision_tower_arch = None
    if config.resume_path and "radio" not in model_name_or_path:
        assert os.path.exists(model_name_or_path), f"Resume vision tower path {model_name_or_path} does not exist!"
        vision_tower_cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        vision_tower_arch = vision_tower_cfg.architectures[0].lower()
    vision_tower_name = vision_tower_arch if vision_tower_arch is not None else model_name_or_path

    use_s2 = getattr(config, "s2", False)
    use_dynamic_s2 = getattr(config, "dynamic_s2", False)

    if config.ps3:
        vision_tower = PS3VisionTower(model_name_or_path, config)
    elif "intern" in vision_tower_name.lower():
        drop_path_rate = getattr(config, "drop_path_rate", 0.0)
        if use_s2:
            vision_tower = InternVisionTowerS2(model_name_or_path, config=config, drop_path_rate=drop_path_rate)
        else:
            vision_tower = InternVisionTower(model_name_or_path, config=config, drop_path_rate=drop_path_rate)
    elif "radio" in vision_tower_name:
        vision_tower = RADIOVisionTower(model_name_or_path, config)
    elif "clip" in vision_tower_name:
        if use_s2:
            vision_tower = CLIPVisionTowerS2(model_name_or_path, config)
        else:
            vision_tower = CLIPVisionTower(model_name_or_path, config)
    elif "siglip" in vision_tower_name:
        if use_dynamic_s2:
            vision_tower = SiglipVisionTowerDynamicS2(model_name_or_path, config)
        elif use_s2:
            vision_tower = SiglipVisionTowerS2(model_name_or_path, config)
        else:
            vision_tower = SiglipVisionTower(model_name_or_path, config)
    else:
        raise ValueError(f"Unknown vision tower: {model_name_or_path}")

    config.mm_hidden_size = (
        vision_tower.config.hidden_size if not (use_s2 or use_dynamic_s2) else vision_tower.hidden_size
    )
    if config.ps3:
        config.mm_low_res_token_num = vision_tower.vision_tower.vision_model.low_res_token_num
        config.mm_scale_num = len(vision_tower.vision_tower.vision_model.s3_scales)
    return vision_tower
