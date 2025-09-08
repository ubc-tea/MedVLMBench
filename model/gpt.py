import os
from io import BytesIO
import shutil
import warnings
import torch
import base64
from openai import OpenAI
from torchvision.transforms.functional import to_pil_image

from model.chat import DeerAPIModel


class o3(DeerAPIModel):
    def __init__(self, args):
        super().__init__(args)

        self.name = "o3"
        self.model_type = "medical"
        self.api_model_name = "o3-2025-04-16"
        self.max_try_num = 10
