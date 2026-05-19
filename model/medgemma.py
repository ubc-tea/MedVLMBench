import os
import shutil
import warnings
import math
import torch
from torchvision.transforms.functional import to_pil_image
import transformers
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from PIL import Image

from model.chat import ChatMetaModel
from peft import LoraConfig


class MedGemma(ChatMetaModel):
    def __init__(self, args):
        super().__init__(args)

        self.name = "MedGemma"
        self.model_type = "medical"
        self.prefers_cpu_image_inputs = True

    def load_from_pretrained(self, model_path, **kwargs):
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    def load_for_training(self, model_path):
        # Check if GPU supports bfloat16
        if torch.cuda.get_device_capability()[0] < 8:
            raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")

        model_kwargs = dict(
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
            bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
        )

        self.model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "right"

        modules_to_save = []
        exclude_modules = ""
        if "M" in self.args.tune_modules:
            modules_to_save.extend(
                [
                    "multi_modal_projector"
                    # "lm_head",
                    # "embed_tokens",
                ]
            )

        # print(self.model)

        if "V" not in self.args.tune_modules:
            exclude_modules = r".*\.vision_tower\..*"
        if "L" not in self.args.tune_modules:
            exclude_modules = r".*\.language_model\.layers\..*"

        self.peft_config = LoraConfig(
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            r=self.args.lora_r,
            bias=self.args.lora_bias,
            target_modules="all-linear",
            exclude_modules=exclude_modules,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save,
        )

    def infer_vision_language(self, image, qs, image_size=None, temperature=None):
        if not type(image) == list:
            image = [image]

        # context_images = self._load_context_images()
        # print(len(context_images))
        # if context_images:
        #     image_contents = [{"type": "image", "image": img} for img in context_images]
        # else:
        #     image_contents = [{"type": "image", "image": to_pil_image(image)}]

        pil_images = [to_pil_image(x).convert("RGB") for x in image]
        if len(pil_images) > 1:
            warnings.warn(
                f"Collapsing {len(pil_images)} MedGemma input images into a single panel to fit context budget.",
                stacklevel=2,
            )
            pil_images = [self._compose_image_panel(pil_images)]

        image_contents = [{"type": "image", "image": img} for img in pil_images]

        default_max_new_tokens = 512
        inputs = self._build_inputs(image_contents, qs)
        sliding_window = self._get_sliding_window()
        input_len = inputs["input_ids"].shape[-1]

        if sliding_window is not None:
            target_input_len = max(1, sliding_window - default_max_new_tokens)
        else:
            target_input_len = None

        if target_input_len is not None and input_len > target_input_len:
            try:
                qs, inputs = self._fit_prompt_within_window(image_contents, qs, target_input_len, sliding_window)
                input_len = inputs["input_ids"].shape[-1]
            except ValueError:
                qs, inputs = self._fit_prompt_within_window(image_contents, qs, sliding_window, sliding_window)
                input_len = inputs["input_ids"].shape[-1]

        inputs = inputs.to(self.model.device, dtype=torch.bfloat16)

        max_new_tokens = self._resolve_max_new_tokens(input_len, default_max_new_tokens=default_max_new_tokens)

        with torch.inference_mode():
            if temperature is None:
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            else:
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                )

            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)

        return decoded.strip()

    def _build_inputs(self, image_contents, qs):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert on understanding medical images."}],
            },
            {
                "role": "user",
                "content": [
                    *image_contents,
                    {"type": "text", "text": qs},
                ],
            },
        ]

        return self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )

    def _compose_image_panel(self, images, panel_size=896, padding=8, background=(0, 0, 0)):
        if len(images) == 1:
            return images[0]

        cols = math.ceil(math.sqrt(len(images)))
        rows = math.ceil(len(images) / cols)

        cell_w = max(1, (panel_size - padding * (cols + 1)) // cols)
        cell_h = max(1, (panel_size - padding * (rows + 1)) // rows)

        canvas = Image.new("RGB", (panel_size, panel_size), background)

        for idx, image in enumerate(images):
            tile = image.copy()
            tile.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)

            row = idx // cols
            col = idx % cols
            x0 = padding + col * (cell_w + padding)
            y0 = padding + row * (cell_h + padding)
            x = x0 + (cell_w - tile.width) // 2
            y = y0 + (cell_h - tile.height) // 2
            canvas.paste(tile, (x, y))

        return canvas

    def _get_sliding_window(self):
        text_config = getattr(self.model.config, "text_config", self.model.config)
        return getattr(text_config, "sliding_window", None)

    def _fit_prompt_within_window(self, image_contents, qs, target_input_len, sliding_window):
        low, high = 1, len(qs)
        best_qs = None
        best_inputs = None

        while low <= high:
            mid = (low + high) // 2
            candidate_qs = self._truncate_text_middle(qs, mid)
            candidate_inputs = self._build_inputs(image_contents, candidate_qs)
            candidate_len = candidate_inputs["input_ids"].shape[-1]

            if candidate_len <= target_input_len:
                best_qs = candidate_qs
                best_inputs = candidate_inputs
                low = mid + 1
            else:
                high = mid - 1

        if best_inputs is None:
            raise ValueError(
                "MedGemma prompt could not be truncated to fit the Gemma sliding-window budget: "
                f"original_chars={len(qs)}, sliding_window={sliding_window}."
            )

        warnings.warn(
            "Truncated MedGemma prompt text to fit Gemma sliding-window budget: "
            f"original_chars={len(qs)}, truncated_chars={len(best_qs)}, "
            f"input_len={best_inputs['input_ids'].shape[-1]}, "
            f"target_input_len={target_input_len}, sliding_window={sliding_window}.",
            stacklevel=2,
        )
        return best_qs, best_inputs

    def _truncate_text_middle(self, text, max_chars):
        if len(text) <= max_chars:
            return text

        marker = "\n[... truncated ...]\n"
        if max_chars <= len(marker) + 8:
            return text[-max_chars:]

        head_chars = (max_chars - len(marker)) // 2
        tail_chars = max_chars - len(marker) - head_chars
        return text[:head_chars] + marker + text[-tail_chars:]

    def _resolve_max_new_tokens(self, input_len, default_max_new_tokens):
        sliding_window = self._get_sliding_window()

        if sliding_window is None:
            return default_max_new_tokens

        available_tokens = sliding_window - input_len
        if available_tokens <= 0:
            raise ValueError(
                "MedGemma prompt exhausted the Gemma sliding-window budget: "
                f"input_len={input_len}, sliding_window={sliding_window}."
            )

        if available_tokens < default_max_new_tokens:
            warnings.warn(
                "Reducing MedGemma max_new_tokens to fit Gemma sliding-window budget: "
                f"input_len={input_len}, sliding_window={sliding_window}, "
                f"requested={default_max_new_tokens}, using={available_tokens}.",
                stacklevel=2,
            )

        return min(default_max_new_tokens, available_tokens)

    def save(self, output_folder, trainer=None):
        trainer.save_model()

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
                warnings.warn(f"[MedGemma] Image path not found: {candidate_path}")
                continue
            try:
                with Image.open(candidate_path) as img:
                    loaded_images.append(img.convert("RGB"))
            except Exception as exc:
                warnings.warn(f"[MedGemma] Failed to open image {candidate_path}: {exc}")

        return loaded_images
