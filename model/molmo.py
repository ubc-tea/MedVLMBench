import os
import warnings
import torch
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image

from model.chat import ChatMetaModel


class Molmo(ChatMetaModel):
    """Wrapper for AI2's Molmo family (1B/7B/72B), e.g. allenai/Molmo-7B-D-0924.

    Molmo ships as trust_remote_code and exposes a custom `processor.process(...)` /
    `model.generate_from_batch(...)` API rather than the standard chat-template flow
    used by most other wrappers in this repo.
    """

    def __init__(self, args):
        super().__init__(args)

        self.name = "Molmo"
        self.model_type = "general"

    def load_from_pretrained(self, model_path, **kwargs):
        device = kwargs.get("device", getattr(self.args, "device", "cuda"))
        self.set_device(device)

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )

    def infer_vision_language(self, image, qs, image_size=None, temperature=None):
        if not type(image) == list:
            image = [image]

        pil_images = [to_pil_image(x).convert("RGB") for x in image]

        inputs = self.processor.process(images=pil_images, text=qs)
        inputs = {k: v.unsqueeze(0).to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = dict(max_new_tokens=512, stop_strings="<|endoftext|>")
        if temperature is not None and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)

        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(**gen_kwargs),
                tokenizer=self.processor.tokenizer,
            )

        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text.strip()

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
                warnings.warn(f"[Molmo] Image path not found: {candidate_path}")
                continue
            try:
                with Image.open(candidate_path) as img:
                    loaded_images.append(img.convert("RGB"))
            except Exception as exc:
                warnings.warn(f"[Molmo] Failed to open image {candidate_path}: {exc}")

        return loaded_images
