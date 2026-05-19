import warnings

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode, to_pil_image
from transformers import AutoModel, AutoTokenizer

from model.chat import ChatMetaModel


class InternVL3(ChatMetaModel):
    def __init__(self, args):
        super().__init__(args)

        self.name = "InternVL3"
        self.model_type = "general"
        self.tokenizer = None
        self.processor = None
        self.image_size = 448
        self.patch_size = 14

    def load_from_pretrained(self, model_path, **kwargs):
        device = kwargs.pop("device", None)
        if device is not None:
            self.device = device

        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        model_kwargs.update(kwargs)

        self.model = AutoModel.from_pretrained(model_path, **model_kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        self.processor = self.tokenizer
        self.image_size = getattr(self.model.config, "force_image_size", 448) or 448
        self.patch_size = getattr(getattr(self.model.config, "vision_config", None), "patch_size", 14) or 14

    def infer_vision_language(self, image, qs, image_size=None, temperature=None):
        if not isinstance(image, list):
            image = [image]

        if len(image) > 1:
            warnings.warn(
                f"InternVL3 received {len(image)} images; using only the first image in the current wrapper.",
                stacklevel=2,
            )

        pil_image = to_pil_image(image[0]).convert("RGB")
        resize_size = self._resolve_resize_size(image_size)
        pixel_values = self._build_transform(resize_size)(pil_image).unsqueeze(0)
        pixel_values = pixel_values.to(device=self.device, dtype=torch.bfloat16)

        generation_config = {
            "max_new_tokens": 200,
            "do_sample": temperature is not None and temperature > 0,
        }
        if temperature is not None:
            generation_config["temperature"] = temperature

        return self.model.chat(
            self.tokenizer,
            pixel_values,
            qs,
            generation_config=generation_config,
            verbose=False,
        ).strip()

    def _resolve_resize_size(self, image_size):
        """Normalize caller-provided image metadata into a square resize target."""
        fallback_size = self._validated_resize_size(self.image_size) or 448

        if isinstance(image_size, int):
            return self._validated_resize_size(image_size) or fallback_size

        if isinstance(image_size, (tuple, list)):
            # Dataset loaders often pass original (width, height) here. That is
            # metadata, not a resize target for InternVL's square vision encoder.
            return fallback_size

        if isinstance(image_size, dict):
            for key in ("shortest_edge", "height", "width"):
                value = image_size.get(key)
                if isinstance(value, int):
                    return self._validated_resize_size(value) or fallback_size

        return fallback_size

    def _validated_resize_size(self, size):
        if isinstance(size, int):
            if size > 0 and size % self.patch_size == 0:
                return size

        return None

    def _build_transform(self, input_size):
        return T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
