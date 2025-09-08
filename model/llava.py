import os
import shutil
import warnings
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, LlamaForCausalLM
from model.release.llava.model import LlavaLlamaForCausalLM, LlavaMptForCausalLM, LlavaMistralForCausalLM
from model.release.llava.conversation import conv_templates, default_conversation
from model.release.llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from model.chat import ChatMetaModel
from utils.utils import maybe_zero_3
from PIL import Image


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ["mm_projector"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f"{current_folder}.bin"))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    projector_keywords = ["mm_projector"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def find_lora_modules(
    args,
    model,
):
    cls = torch.nn.Linear
    lora_module_names = set()
    projector_keywords = ["mm_projector"]
    visiual_keywords = ["vision_tower", "vision_resampler"]

    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in projector_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    target_modules = []
    exclude_parameters = []
    for name, _ in model.named_modules():
        if any(keyword in name for keyword in list(lora_module_names)):
            # vision part
            name_sub = name[6:]
            if any(visual_keyword in name for visual_keyword in visiual_keywords):
                if "V" in args.tune_modules:
                    target_modules.append(name)
                else:
                    exclude_parameters.append(name)

            # LLM part
            if "model.layers" in name:
                if "L" in args.tune_modules:
                    target_modules.append(name)
                else:
                    exclude_parameters.append(name)

    return target_modules


class ImageProcessorCallable:
    def __init__(self, image_processor, model_cfg):
        self.image_processor = image_processor
        self.model_cfg = model_cfg

    def __call__(self, image):
        return process_images([image], self.image_processor, self.model_cfg)[0]


class LLaVA(ChatMetaModel):
    def __init__(self, args):
        super().__init__(args)

        self.conv_mode = "vicuna_v1"
        self.name = "LLaVA-1.5"
        self.model_type = "general"

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
            if "mpt" in self.args.model_path:
                config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
                config.attn_config["attn_impl"] = self.args.mpt_attn_impl
                model = LlavaMptForCausalLM.from_pretrained(
                    model_name_or_path, config=config, cache_dir=self.args.cache_dir, **bnb_model_from_pretrained_args
                )
            else:
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_name_or_path,
                    cache_dir=self.args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if self.args.bf16 else None),
                    **bnb_model_from_pretrained_args,
                )
        else:
            model = LlamaForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir=self.args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if self.args.bf16 else None),
                **bnb_model_from_pretrained_args,
            )
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

        if "mpt" in model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                cache_dir=self.args.cache_dir,
                model_max_length=self.args.model_max_length,
                padding_side="right",
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
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
            # if self.args.version in conversation_lib.conv_templates:
            #     conversation_lib.default_conversation = conversation_lib.conv_templates[self.args.version]
            # else:
            #     conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

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
        # Load models from pretrained weights. For inference only.
        model_base = self.args.model_base
        model_name = get_model_name_from_path(model_path)

        kwargs = {"device_map": device_map, **kwargs}

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

        if use_flash_attn:
            kwargs["attn_implementation"] = "flash_attention_2"

        if "llava" in model_name.lower():
            # Load LLaVA model

            if "lora" in model_name.lower() and model_base is None:
                warnings.warn(
                    "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
                )
            if "lora" in model_name.lower() and model_base is not None:
                from .release.llava.model.language_model.llava_llama import LlavaConfig

                lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                print("Loading LLaVA from base model...")
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
                )
                token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
                if model.lm_head.weight.shape[0] != token_num:
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
                    vision_tower.load_model(device_map=device_map)

                print("Loading LoRA weights...")
                model = PeftModel.from_pretrained(model, model_path)
                print("Merging LoRA weights...")
                model = model.merge_and_unload()

                print("Model is loaded...")
            elif model_base is not None:
                # this may be mm projector only
                print("Loading LLaVA from base model...")
                if "mpt" in model_name.lower():
                    if not os.path.isfile(os.path.join(model_path, "configuration_mpt.py")):
                        shutil.copyfile(
                            os.path.join(model_base, "configuration_mpt.py"),
                            os.path.join(model_path, "configuration_mpt.py"),
                        )
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                    cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                    model = LlavaMptForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                    )
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                    cfg_pretrained = AutoConfig.from_pretrained(model_path)
                    model = LlavaLlamaForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                    )

                mm_projector_weights = torch.load(os.path.join(model_path, "mm_projector.bin"), map_location="cpu")
                mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
                model.load_state_dict(mm_projector_weights, strict=False)
            else:
                if "mpt" in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                    model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                elif "mistral" in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = LlavaMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
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

        image_processor = None

        if "llava" in model_name.lower():
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
                vision_tower.load_model(device_map=device_map)
            if device_map != "auto":
                vision_tower.to(device=device_map, dtype=torch.float16)
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

    def infer_vision_language(self, image, qs, temperature=0, image_size=None):
        # Model inference for vision-language tasks
        # TODO: Make it work for a batch
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

    def infer_language(self, qs, temperature=0):
        # model inference for language only tasks
        conv = default_conversation.copy()
        conv.append_message(conv.roles[0], qs)
        prompt = conv.get_prompt()
        inputs = self.tokenizer([prompt])

        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        output_ids = self.model.generate(
            input_ids,
            do_sample=True,
            use_cache=True,
            temperature=0.7,
            max_new_tokens=1024,
        )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        try:
            index = outputs.index(conv.sep, len(prompt))
        except ValueError:
            outputs += conv.sep
            index = outputs.index(conv.sep, len(prompt))

        outputs = outputs[len(prompt) + len(conv.roles[1]) + 2 : index].strip()

        return outputs

    def save(self, output_folder, trainer=None):
        self.model.config.use_cache = True
        if self.args.lora_enable:
            state_dict = get_peft_state_maybe_zero_3(self.model.named_parameters(), self.args.lora_bias)
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters())
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_folder)
                self.model.save_pretrained(output_folder, state_dict=state_dict)
                torch.save(non_lora_state_dict, os.path.join(output_folder, "non_lora_trainables.bin"))
        else:
            safe_save_model_for_hf_trainer(trainer=trainer, output_dir=output_folder)


if __name__ == "__main__":
    from easydict import EasyDict as edict
    from PIL import Image

    # model download command: git clone https://huggingface.co/liuhaotian/llava-v1.5-7b
    # model_path = "/mnt/hdd/weights/llava-v1.5-7b"
    model_path = "./pretrained_models/llava-v1.5-7b"

    prompt = """
    Please caption the image with findings for medical report.
    """

    # image_file = "/media/yesindeed/DATADRIVE1/mount/remote_cse/datasets/LLaVA-Med/data/images/34630837_F2.jpg"
    image_file = "/bigdata/rjin02/MedVLMBench/data/SLAKE/imgs/xmlab0/source.jpg"
    img = Image.open(image_file).convert("RGB")

    """
    GT report:
    Abdominopelvic CT scan in axial view indicates significant distension of the stomach and intestines with marked luminal dilatation observed in the oesophagus, stomach, small, and large bowel, accompanied by faecal loading. Notably, the distended large bowel is positioned anterior to the liver, causing medial displacement of the liver, which suggests a possible chronic underlying condition. This constellation of findings points to a long-standing obstructive process in the gastrointestinal tract, necessitating further clinical correlation and potential intervention.
    """

    llava_model = LLaVA(args=edict(model_path=model_path, model_base=None))
    llava_model.load_from_pretrained(model_path=model_path)

    output = llava_model.infer_vision_language(img, prompt)
    print(output)
