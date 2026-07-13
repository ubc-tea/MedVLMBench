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


class GPT52(DeerAPIModel):
    def __init__(self, args):
        super().__init__(args)

        self.name = "gpt-5.2"
        self.model_type = "medical"
        # "gpt-5.2" resolves to the Thinking variant on the Responses/Chat Completions
        # API; use "gpt-5.2-chat-latest" for the Instant variant if lower latency is needed.
        self.api_model_name = "gpt-5.2"
        self.max_try_num = 10
