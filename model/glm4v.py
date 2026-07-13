import os
import warnings
import torch
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

from model.chat import ChatMetaModel


class GLM4V(ChatMetaModel):
    """Wrapper for Zhipu's GLM-4.1V-9B-Thinking (dense) checkpoint.

    AutoModelForImageTextToText dispatches to Glm4vForConditionalGeneration for this
    checkpoint. Requires transformers>=4.57.1.
    """

    def __init__(self, args):
        super().__init__(args)

        self.name = "GLM-4.1V-Thinking"
        self.model_type = "general"

    def load_from_pretrained(self, model_path, **kwargs):
        device = kwargs.get("device", getattr(self.args, "device", "cuda"))
        self.set_device(device)

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

    def infer_vision_language(self, image, qs, image_size=None, temperature=None):
        if not type(image) == list:
            image = [image]

        pil_images = [to_pil_image(x).convert("RGB") for x in image]
        image_contents = [{"type": "image", "image": img} for img in pil_images]

        messages = [{"role": "user", "content": [*image_contents, {"type": "text", "text": qs}]}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]
        # Thinking models emit a <think>...</think> reasoning trace before the final
        # answer, so they need a much larger generation budget than plain chat models.
        default_max_new_tokens = 2048

        if temperature is None:
            generated_ids = self.model.generate(**inputs, max_new_tokens=default_max_new_tokens)
        else:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=default_max_new_tokens,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
            )

        output_text = self.processor.decode(generated_ids[0][input_len:], skip_special_tokens=True)

        return self._extract_answer(output_text.strip())

    def _extract_answer(self, text):
        if "<answer>" in text and "</answer>" in text:
            return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
        if "</think>" in text:
            return text.split("</think>", 1)[1].strip()
        return text

    def _load_context_images(self):
        context = getattr(self, "_inference_context", None) or {}
        image_paths = context.get("image_paths")
        self._inference_context = {}

        if not image_paths:
            return []

        if isinstance(image_paths, str):
            image_paths = [p for p in image_paths.split(";") if p]

        loaded_images = []
        for image_path in image_paths:
            candidate_path = image_path
            if not os.path.isabs(candidate_path):
                base_dir = getattr(self.args, "image_path", "") or ""
                candidate_path = os.path.join(base_dir, image_path)
            if not os.path.exists(candidate_path):
                warnings.warn(f"[GLM-4.1V] Image path not found: {candidate_path}")
                continue
            try:
                with Image.open(candidate_path) as img:
                    loaded_images.append(img.convert("RGB"))
            except Exception as exc:
                warnings.warn(f"[GLM-4.1V] Failed to open image {candidate_path}: {exc}")

        return loaded_images


class GLM45V(GLM4V):
    """Wrapper for GLM-4.5V (MoE flagship). Same inference API as GLM-4.1V-Thinking;
    AutoModelForImageTextToText dispatches to Glm4vMoeForConditionalGeneration instead.
    """

    def __init__(self, args):
        super().__init__(args)

        self.name = "GLM-4.5V"
