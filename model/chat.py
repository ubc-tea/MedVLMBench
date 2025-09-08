import torch.nn as nn
from model.base import BaseModel
from easydict import EasyDict as edict
import os
from io import BytesIO
import shutil
import warnings
import torch
import base64
from openai import OpenAI
from torchvision.transforms.functional import to_pil_image


class ChatMetaModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        self.constants = edict(
            IGNORE_INDEX=-100,
            IMAGE_TOKEN_INDEX=-200,
            DEFAULT_IMAGE_TOKEN="<image>",
            DEFAULT_IMAGE_PATCH_TOKEN="<im_patch>",
            DEFAULT_IM_START_TOKEN="<im_start>",
            DEFAULT_IM_END_TOKEN="<im_end>",
            IMAGE_PLACEHOLDER="<image-placeholder>",
        )

    def infer_vision_language(self, image, qs, image_size=None):
        # input image should be type of Tensor
        pass

    def infer_language(self, qs):
        pass

    def load_for_training(self):
        pass

    def save(self, output_folder):
        pass


def image_to_base64(image, fmt="PNG"):
    buf = BytesIO()
    fmt = (fmt or image.format or "PNG").upper()

    # JPEG doesn't support alpha; convert if needed
    if fmt in ["JPG", "JPEG"] and (
        image.mode in ["RGBA", "LA"] or (image.mode == "P" and "transparency" in image.info)
    ):
        image = image.convert("RGB")

    image.save(
        buf,
        format=fmt,
    )
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class DeerAPIModel(ChatMetaModel):
    def __init__(self, args):
        super().__init__(args)

        os.environ["HTTP_PROXY"] = ""
        os.environ["HTTPS_PROXY"] = ""

        self.client = OpenAI(
            base_url="https://api.deerapi.com/v1",
            api_key="sk-XwFQI7rxo3l9GQXsVkmQDaWulTDHJm3n47uWa84BNGDUqmp8",
        )
        self.api_model_name = "o3-2025-04-16"

        self.max_try_num = 5

    def load_from_pretrained(self, model_path, **kwargs):
        pass

    def infer_vision_language(self, image, qs, image_size=None):
        image = to_pil_image(image)
        img_base64 = image_to_base64(image)
        if "yes or no" not in qs:
            qs = qs + "\n\nPlease answer the question concisely in no more than 2 sentences."

        messages = [
            {"type": "text", "text": qs},
            {"type": "image_url", "image_url": f"data:image/png;base64,{img_base64}"},
        ]

        for _ in range(self.max_try_num):
            try:
                response = self.client.chat.completions.create(
                    model=self.api_model_name, messages=[{"role": "user", "content": messages}]
                )

                result = response.choices[0].message.content
                break
            except Exception as e:
                result = "invalid response"
        if result == "invalid response":
            if hasattr(e, "message"):
                print(e.message)
            raise Exception
        print(result)

        return result.strip()
