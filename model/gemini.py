import os
from io import BytesIO
import shutil
import warnings
import torch
import base64
from openai import OpenAI
from torchvision.transforms.functional import to_pil_image

from model.chat import DeerAPIModel


class Gemini25Pro(DeerAPIModel):
    def __init__(self, args):
        super().__init__(args)

        self.name = "gemini-2.5-pro"
        self.model_type = "medical"
        # self.api_model_name = "gemini-2.5-pro-preview-06-05"
        self.api_model_name = "gemini-2.5-flash-preview-05-20"


class Gemini31Pro(DeerAPIModel):
    def __init__(self, args):
        super().__init__(args)

        self.name = "gemini-3.1-pro"
        self.model_type = "medical"
        # gemini-3-pro-preview was deprecated/shut down 2026-03-09 and now aliases here.
        self.api_model_name = "gemini-3.1-pro-preview"
