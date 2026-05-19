import os
import warnings
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from peft import PeftModel
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from model.chat import ChatMetaModel


def _to_pil(img: Image.Image | torch.Tensor | str):
    """
    Make sure the vision object is a PIL.Image or a path/URL.
    """
    if isinstance(img, torch.Tensor):
        # If the tensor is float in [0, 1] convert to uint8 first
        if img.dtype.is_floating_point:
            img = (img.clamp(0, 1) * 255).to(torch.uint8)
        img = to_pil_image(img.cpu())
    return img


class Lingshu(ChatMetaModel):

    def __init__(self, args):
        super().__init__(args)
        self.name = "Lingshu"
        self.model_type = "medical"
        self.processor = None
        self.context_len = 2048

    def load_from_pretrained(
        self,
        # model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        model_path: str = "lingshu-medical-mllm/Lingshu-7B",
        device_map: str = "auto",
        torch_dtype: str | torch.dtype = "auto",
        **kwargs,
    ):
        load_path = model_path
        if getattr(self.args, "model_base", None):
            load_path = self.args.model_base

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            load_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )

        if getattr(self.args, "model_base", None):
            self.model = PeftModel.from_pretrained(self.model, model_path)
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            self.processor = AutoProcessor.from_pretrained(model_path)

        self.tokenizer = self.processor.tokenizer


    def load_for_training(
        self,
        model_path: str = "lingshu-medical-mllm/Lingshu-7B",
        modules_to_train: str = "ML",        # any subset of "V", "M", "L"
        device_map: str = "auto",
        torch_dtype: str | torch.dtype = "auto",
    ):
        """
        Load Qwen2.5-VL for training.
        modules_to_train: combination of:
          V = vision-encoder LoRA,
          M = full-tune the merger MLP,
          L = language-decoder LoRA.
        """
        # 1) Base load
        self.load_from_pretrained()

        # 2) Disable use_cache and freeze everything
        self.model.model.config.use_cache = False
        self.model.requires_grad_(False)

        # 3) Optional gradient checkpointing
        if getattr(self.args, "gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()

        # 4) Build LoRA target list
        modules = modules_to_train.upper()
        lora_targets = []

        if "V" in modules:
            # patch_embed & all VisionBlock attn + mlp
            lora_targets += [
                # "model.visual.patch_embed.proj",
                "model.visual.blocks.*.attn.qkv",
            ]

        if "L" in modules:
            # every decoder layer’s self- and cross-attn projections
            lora_targets += [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
            ]

        if lora_targets:
            from peft import LoraConfig, get_peft_model

            lora_cfg = LoraConfig(
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                target_modules=lora_targets,
                lora_dropout=self.args.lora_dropout,
                bias=self.args.lora_bias,
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_cfg)

        # 6) Full-fine-tune merger MLP
        if "M" in modules:
            for name, param in self.model.named_parameters():
                if "visual.merger.mlp" in name:
                    param.requires_grad = True

        # 7) Prepare for k-bit if requested
        bits = getattr(self.args, "bits", None)
        if bits in (4, 8):
            from peft import prepare_model_for_kbit_training

            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=getattr(self.args, "gradient_checkpointing", False),
            )

        # 8) update context length
        self.context_len = getattr(self.model.model.config, "max_position_embeddings", self.context_len)

    def infer_vision_language(self, image, qs, image_size=None, temperature=None):
        context_images = self._load_context_images()
        if context_images:
            image_contents = [{"type": "image", "image": img, "resized_height": 224, "resized_width": 224} for img in context_images]
        else:
            images = image if isinstance(image, list) else [image]
            image_contents = [
                {"type": "image", "image": _to_pil(img), "resized_height": 224, "resized_width": 224}
                for img in images
            ]

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

        max_new_tokens = getattr(self.args, "gen_max_new_tokens", None)
        if max_new_tokens is None:
            max_new_tokens = 192 if getattr(self.args, "usage", None) == "ucagent" else 512

        if temperature is None:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        else:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                use_cache=True,
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

            with Image.open(candidate_path) as img:
                loaded_images.append(img.convert("RGB"))

        return loaded_images


    def save(self, output_folder, trainer=None):
        self.model.save_pretrained(output_folder)
        self.processor.save_pretrained(output_folder)



class LingshuDataset(Dataset):
    def __init__(self, args, base_dataset, processor, ignore_index=-100):
        """
        base_dataset: any list/dataset of dicts with keys
          - 'image': a PIL or tensor image, or None
          - 'prompt': the user text prompt
          - 'response': the target text
        processor: your AutoProcessor (tokenizer + vision preprocess)
        """
        self.args = args
        self.base = base_dataset
        self.proc = processor
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        sample = self.base[i]
        img   = _to_pil(sample["image"])

        if hasattr(sample, "query"):
            text  = sample["query"]
        else:
            text  = sample["prompt_template"].format("")
        resp  = sample["label"]

        # ---------- build chat-style message ----------
        if img is not None:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img,
                    "resized_height": 224, "resized_width": 224},
                    {"type": "text", "text": text},
                ],
            }]
        else:
            messages = [{"role": "user", "content": text}]

        # prompt seen by the model before generation
        chat_txt = self.proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # ---------- FULL prompt = prompt + answer ----------
        full_prompt = (
            chat_txt
            + self.proc.tokenizer.eos_token
            + resp
            + self.proc.tokenizer.eos_token
        )

        # vision pre-processing exactly like inference
        image_inputs, video_inputs = process_vision_info(messages)

        # ---------- single processor call ----------
        inputs = self.proc(
            text   = [full_prompt],          # <-- USE full prompt here
            images = image_inputs,
            videos = video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # input_ids, attention_mask, pixel_values, image_grid_thw are all aligned
        input_ids      = inputs.input_ids.squeeze(0)        # (L,)
        attention_mask = inputs.attention_mask.squeeze(0)

        # ---------- build labels by masking the prompt tokens ----------
        labels = input_ids.clone()

        # tokens that belong to the prompt (chat_txt + <eos>)
        prompt_len = len(
            self.proc.tokenizer(chat_txt + self.proc.tokenizer.eos_token).input_ids
        )
        labels[:prompt_len] = self.ignore_index

        return dict(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            pixel_values   = inputs.pixel_values.squeeze(0),
            image_grid_thw = inputs.image_grid_thw[0],   # shape (3,)
            labels         = labels,
        )


class LingshuCollator:
    def __init__(self, pad_token_id: int, ignore_index: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, features):
        input_ids  = [f["input_ids"]      for f in features]
        attention  = [f["attention_mask"] for f in features]
        labels     = [f["labels"]         for f in features]

        pixel_vals = [f["pixel_values"]   for f in features]
        grid_thw   = [f["image_grid_thw"] for f in features]   # NEW

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention = pad_sequence(attention, batch_first=True, padding_value=0)
        labels    = pad_sequence(labels,    batch_first=True, padding_value=self.ignore_index)

        batch = {
            "input_ids":      input_ids,
            "attention_mask": attention,
            "labels":         labels,
            "pixel_values":   torch.stack(pixel_vals),   # (B,3,224,224)
            "image_grid_thw": torch.stack(grid_thw),                  # list[(T,H,W), …]
        }
        return batch
