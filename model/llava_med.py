import os
import shutil
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

from model.release.llava_med.model import LlavaMistralForCausalLM
from model.release.llava_med.conversation import conv_templates
from model.release.llava_med.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from model.llava import LLaVA, find_lora_modules
from PIL import Image

import warnings


class ImageProcessorCallable:
    def __init__(self, image_processor, model_cfg):
        self.image_processor = image_processor
        self.model_cfg = model_cfg

    def __call__(self, image):
        return process_images([image], self.image_processor, self.model_cfg)[0]


class LLaVAMed(LLaVA):
    def __init__(self, args):
        super().__init__(args)

        self.conv_mode = "mistral_instruct"
        self.name = "LLaVA-Med"
        self.model_type = "medical"

    def load_for_training(self, model_name_or_path):
        compute_dtype = torch.float16 if self.args.fp16 else (torch.bfloat16 if self.args.bf16 else torch.float32)

        self.args.vision_tower = "openai/clip-vit-large-patch14-336"
        attn_implementation = "flash_attention_2"

        bnb_model_from_pretrained_args = {}
        if self.args.bits in [4, 8]:
            from transformers import BitsAndBytesConfig

            bnb_model_from_pretrained_args.update(
                dict(
                    device_map={"": self.args.device},
                    load_in_4bit=self.args.bits == 4,
                    load_in_8bit=self.args.bits == 8,
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=self.args.bits == 4,
                        load_in_8bit=self.args.bits == 8,
                        llm_int8_skip_modules=["mm_projector"],
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_use_double_quant=self.args.double_quant,
                        bnb_4bit_quant_type=self.args.quant_type,  # {'fp4', 'nf4'}
                    ),
                )
            )

        if self.args.vision_tower is not None:
            if "mistral" in self.args.model_path.lower():
                model = LlavaMistralForCausalLM.from_pretrained(
                    self.args.model_path,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if self.args.bf16 else None),
                    **bnb_model_from_pretrained_args,
                )
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

        model.config.use_cache = False
        if self.args.freeze_backbone:
            model.model.requires_grad_(False)

        if self.args.bits in [4, 8]:
            from peft import prepare_model_for_kbit_training

            model.config.torch_dtype = compute_dtype
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=self.args.use_gradient_checkpointing
            )

        if self.args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path,
            cache_dir=self.args.cache_dir,
            model_max_length=self.args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

        if self.args.version == "v0":
            if tokenizer.pad_token is None:
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(pad_token="[PAD]"),
                    tokenizer=tokenizer,
                    model=model,
                )
        elif self.args.version == "v0.5":
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.unk_token

        if self.args.vision_tower is not None:
            model.get_model().initialize_vision_modules(model_args=self.args, fsdp=None)

            vision_tower = model.get_vision_tower()
            vision_tower.to(dtype=torch.bfloat16 if self.args.bf16 else torch.float16, device=self.args.device)

            image_processor = vision_tower.image_processor

            model.config.image_aspect_ratio = self.args.image_aspect_ratio
            model.config.tokenizer_padding_side = tokenizer.padding_side
            model.config.tokenizer_model_max_length = tokenizer.model_max_length

        if self.args.lora_enable:
            from peft import LoraConfig, get_peft_model

            # target_modules, exclude_modules = find_lora_modules(self.args, model)

            lora_config = LoraConfig(
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                target_modules=find_lora_modules(self.args, model),
                lora_dropout=self.args.lora_dropout,
                bias=self.args.lora_bias,
                task_type="CAUSAL_LM",
            )

            if self.args.bits == 16:
                if self.args.bf16:
                    model.to(torch.bfloat16)
                if self.args.fp16:
                    model.to(torch.float16)

            model = get_peft_model(model, lora_config)

        if self.args.vision_tower is not None:
            model.config.tune_mm_mlp_adapter = self.args.tune_mm_mlp_adapter
            if self.args.tune_mm_mlp_adapter:
                if not self.args.lora_enable:
                    model.requires_grad_(False)
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True

            model.config.freeze_mm_mlp_adapter = self.args.freeze_mm_mlp_adapter
            if self.args.freeze_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = False

            if self.args.bits in [4, 8]:
                model.get_model().mm_projector.to(dtype=compute_dtype, device=self.args.device)

            model.config.mm_use_im_start_end = self.args.mm_use_im_start_end
            model.config.mm_projector_lr = self.args.mm_projector_lr
            self.args.use_im_start_end = self.args.mm_use_im_start_end
            model.config.mm_use_im_patch_token = self.args.mm_use_im_patch_token
            model.initialize_vision_tokenizer(self.args, tokenizer=tokenizer)

        if self.args.bits in [4, 8]:
            from peft.tuners.lora import LoraLayer

            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    if self.args.bf16:
                        module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if "lm_head" in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        if self.args.bf16 and module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16)

        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def load_from_pretrained(
        self,
        model_path,
        load_8bit=False,
        load_4bit=False,
        device_map="auto",
        device="cuda",
        use_flash_attn=False,
        **kwargs,
    ):
        model_base = self.args.model_base
        model_name = get_model_name_from_path(model_path)

        kwargs = {}

        if device != "cuda":
            kwargs["device_map"] = {"": device}

        if load_8bit:
            kwargs["load_in_8bit"] = True
        elif load_4bit:
            kwargs["load_in_4bit"] = True
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["torch_dtype"] = torch.float16

        if "llava" in model_name.lower():
            # Load LLaVA model
            if "lora" in model_name.lower() and model_base is None:
                warnings.warn(
                    "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
                )
            if "lora" in model_name.lower() and model_base is not None:
                from .release.llava_med.model.language_model.llava_mistral import LlavaMistralConfig

                # lora_cfg_pretrained = LlavaMistralConfig.from_pretrained(model_path)
                lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)

                tokenizer = AutoTokenizer.from_pretrained(
                    model_base,
                )
                print("Loading LLaVA from base model...")
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_base,
                    low_cpu_mem_usage=False,
                    use_flash_attention_2=False,
                    config=lora_cfg_pretrained,
                    **kwargs,
                )
                token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
                if model.lm_head.weight.shape[0] != token_num:
                    print(f"Warning: {model.lm_head.weight.shape[0]} and token_num {token_num} mismatch!")
                    model.lm_head.weight = torch.nn.Parameter(
                        torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
                    )
                    model.model.embed_tokens.weight = torch.nn.Parameter(
                        torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
                    )

                print("Loading additional LLaVA weights...")
                if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                    non_lora_trainables = torch.load(
                        os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu"
                    )
                else:
                    # this is probably from HF Hub
                    from huggingface_hub import hf_hub_download

                    def load_from_hf(repo_id, filename, subfolder=None):
                        cache_file = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder)
                        return torch.load(cache_file, map_location="cpu")

                    non_lora_trainables = load_from_hf(model_path, "non_lora_trainables.bin")
                non_lora_trainables = {
                    (k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()
                }
                if any(k.startswith("model.model.") for k in non_lora_trainables):
                    non_lora_trainables = {
                        (k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()
                    }
                model.load_state_dict(non_lora_trainables, strict=False)

                from peft import PeftModel

                vision_tower = model.get_vision_tower()
                if not vision_tower.is_loaded:
                    vision_tower.load_model()

                print("Loading LoRA weights...")
                model = PeftModel.from_pretrained(model, model_path)
                print("Merging LoRA weights...")
                model = model.merge_and_unload()
                print("Model is loaded...")
            elif model_base is not None:
                # this may be mm projector only
                print("Loading LLaVA from base model...")
                tokenizer = AutoTokenizer.from_pretrained(model_base)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_base, config=cfg_pretrained, low_cpu_mem_usage=False, use_flash_attention_2=False, **kwargs
                )

                mm_projector_weights = torch.load(os.path.join(model_path, "mm_projector.bin"), map_location="cpu")
                mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
                model.load_state_dict(mm_projector_weights, strict=False)
            else:
                if "mistral" in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = LlavaMistralForCausalLM.from_pretrained(
                        model_path, low_cpu_mem_usage=False, use_flash_attention_2=False, **kwargs
                    )
                else:
                    raise NotImplementedError
        else:
            # Load language model
            if model_base is not None:
                # PEFT model
                from peft import PeftModel

                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
                print(f"Loading LoRA weights from {model_path}")
                model = PeftModel.from_pretrained(model, model_path)
                print(f"Merging weights")
                model = model.merge_and_unload()
                print("Convert to FP16...")
                model.to(torch.float16)
            else:
                use_fast = False
                if "mpt" in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs
                    )
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

        # model.generation_config.pad_token_id = tokenizer.pad_token_id
        image_processor = None

        if "llava" in model_name.lower():  # or 'mistral' in model_name.lower():
            mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
            mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
            if mm_use_im_patch_token:
                tokenizer.add_tokens([self.constants.DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            if mm_use_im_start_end:
                tokenizer.add_tokens(
                    [self.constants.DEFAULT_IM_START_TOKEN, self.constants.DEFAULT_IM_END_TOKEN], special_tokens=True
                )
            model.resize_token_embeddings(len(tokenizer))

            vision_tower = model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower.to(device=device, dtype=torch.float16)
            model.model.mm_projector.to(device=device, dtype=torch.float16)
            model.to(device=device, dtype=torch.float16)
            image_processor = vision_tower.image_processor

        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048

        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_processor_callable = ImageProcessorCallable(image_processor, model.config)
        self.context_len = context_len

        return tokenizer, model, image_processor, context_len

    def infer_vision_language(self, image, qs, temperature=0, image_size=None):
        # Model inference for vision-language tasks
        warnings.filterwarnings("ignore")

        qs = qs.replace(self.constants.DEFAULT_IMAGE_TOKEN, "").strip()
        if self.model.config.mm_use_im_start_end:
            qs = (
                self.constants.DEFAULT_IM_START_TOKEN
                + self.constants.DEFAULT_IMAGE_TOKEN
                + self.constants.DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = self.constants.DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, self.constants.IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.model.device)
        )

        if type(image) is Image.Image:
            image_tensor = process_images([image], self.image_processor, self.model.config)[0]
            image_size = image.size
        else:
            image_tensor = image

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image_size],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=None,
                num_beams=1,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
