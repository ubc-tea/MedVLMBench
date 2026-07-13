import os
import warnings
import torch
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

from model.chat import ChatMetaModel


class Qwen3_VL(ChatMetaModel):
    """Wrapper for the Qwen3-VL family (dense 2B/4B/8B/32B and MoE 30B-A3B/235B-A22B).

    Uses AutoModelForImageTextToText so both the dense (Qwen3VLForConditionalGeneration)
    and MoE (Qwen3VLMoeForConditionalGeneration) checkpoints resolve automatically from
    config, matching how upstream `transformers` dispatches the two architectures.
    """

    def __init__(self, args):
        super().__init__(args)

        self.name = "Qwen3-VL"
        self.model_type = "general"

    def load_from_pretrained(self, model_path, **kwargs):
        device = kwargs.get("device", getattr(self.args, "device", "cuda"))
        self.set_device(device)

        attn_impl = "flash_attention_2"
        for candidate in ("flash_attention_2", "sdpa", "eager"):
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation=candidate,
                    device_map="auto",
                )
                attn_impl = candidate
                break
            except ImportError as exc:
                warnings.warn(
                    f"[Qwen3-VL] Failed to enable attention implementation '{candidate}': {exc}. "
                    "Trying a fallback option."
                )
        else:
            raise RuntimeError("[Qwen3-VL] Unable to load model with any attention implementation.")

        if attn_impl != "flash_attention_2":
            warnings.warn(
                f"[Qwen3-VL] Loaded without FlashAttention2 (using '{attn_impl}'). "
                "Install flash_attn for better performance."
            )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def infer_vision_language(self, image, qs, image_size=None, temperature=None):
        if not type(image) == list:
            image = [image]

        image_contents = [{"type": "image", "image": to_pil_image(x)} for x in image]

        messages = [{"role": "user", "content": [*image_contents, {"type": "text", "text": qs}]}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.device)
        if temperature is None:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
            )
        else:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
            )

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0].strip()

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
                warnings.warn(f"[Qwen3-VL] Image path not found: {candidate_path}")
                continue
            try:
                with Image.open(candidate_path) as img:
                    loaded_images.append(img.convert("RGB"))
            except Exception as exc:
                warnings.warn(f"[Qwen3-VL] Failed to open image {candidate_path}: {exc}")

        return loaded_images
