import os
import math
import shutil
import warnings
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, LlamaForCausalLM
from torchvision import transforms
from model.release.vila.model import LlavaLlamaModel, LlavaTopDownLlamaModel
from model.release.vila.model.utils import is_mm_model
from model.release.vila.model.builder import prepare_config_for_eval
from model.release.vila.conversation import conv_templates, default_conversation
from model.release.vila.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from model.chat import ChatMetaModel
from utils.utils import maybe_zero_3
from PIL import Image
from typing import Dict



def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

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

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        trainer.model.config.save_pretrained(output_dir)

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return
        
class ImageProcessorCallable:
    def __init__(self, image_processor, model_cfg):
        self.image_processor = image_processor
        self.model_cfg = model_cfg
        self.to_pil = transforms.ToPILImage()

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = self.to_pil(image)
        return image
    

class VILA(ChatMetaModel):
    def __init__(self, args):
        super().__init__(args)

        self.name = args.model_path
        self.model_type = "medical"

    def load_for_training(self, model_name_or_path):
        self.load_from_pretrained(model_name_or_path)
        self.model.cpu()
        if self.args.lora_enable:
            from peft import LoraConfig, PeftModel, get_peft_model

            lora_llm = "L" in self.args.tune_modules
            lora_vt = "V" in self.args.tune_modules

            lora_config = LoraConfig(
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                target_modules=self.find_all_linear_names(model=self.model, lora_llm=lora_llm, lora_vt=lora_vt),
                lora_dropout=self.args.lora_dropout,
                bias=self.args.lora_bias,
                task_type="CAUSAL_LM",
            )
            if self.args.bits == 16:
                if self.args.bf16:
                    self.model.to(torch.bfloat16)
                if self.args.fp16:
                    self.model.to(torch.float16)

            self.args.logger.info("Adding LoRA adapters...")
            self.model = get_peft_model(self.model, lora_config)
        if "M" in self.args.tune_modules:
            # for param in self.model.parameters():
            #     param.requires_grad = False
            self.model.get_mm_projector().requires_grad_(True)
            self.args.logger.info("Only training MM Projector...")

        else:
            raise NotImplementedError("Only LoRA is supported for training.")
    
        tokenizer = self.model.tokenizer

        if tokenizer.bos_token is None:
            self.smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(bos_token="[BOS]"),
                tokenizer=tokenizer,
                model=self.model.llm,
            )
        
        tokenizer.pad_token = tokenizer.unk_token
        if tokenizer.pad_token is None:
            self.smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=self.model.llm,
            )

        vision_tower = self.model.get_vision_tower()
        if vision_tower is not None:
            self.args.image_processor = vision_tower.image_processor
            self.args.is_multimodal = True

            # if hasattr(data_args, "num_video_frames") and data_args.num_video_frames != None:
            #     model.config.num_video_frames = data_args.num_video_frames
            # else:
            #     model.config.num_video_frames = 8

            # if hasattr(self.args, "fps"):
            #     self.model.config.fps = self.args.fps
            # else:
            #     self.model.config.fps = 0.0

            self.model.config.image_aspect_ratio = self.args.image_aspect_ratio
            self.model.config.mm_projector_lr = self.args.mm_projector_lr
            self.model.config.vision_tower_lr = self.args.vision_tower_lr
            if self.args.mm_use_im_start_end:
                num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            assert not self.args.mm_use_im_patch_token

            self.model.config.num_time_tokens = self.args.num_time_tokens
            self.model.config.time_token_format = self.args.time_token_format
            if self.args.num_time_tokens > 0:
                time_tokens = [self.model.config.time_token_format.format(t=t) for t in range(self.model.config.num_time_tokens)]
                num_new_tokens = tokenizer.add_tokens(time_tokens)
                assert len(time_tokens) == num_new_tokens or num_new_tokens == 0
                self.model.resize_token_embeddings(len(tokenizer))
                self.model.config.time_token_ids = tokenizer.convert_tokens_to_ids(time_tokens)
            else:
                self.model.config.time_token_ids = []
            self.model.config.soft_ce_std = self.args.soft_ce_std

            num_patches = self.model.get_vision_tower().num_patches
            downsample_rate = self.model.get_mm_projector().downsample_rate
            num_image_tokens = math.ceil(num_patches**0.5 / downsample_rate) ** 2
            self.args.num_image_tokens = num_image_tokens
        else:
            raise RuntimeError("VILA family is designed for multi-modal tasks. Please provide a model with vision tower.")


    def load_from_pretrained(
        self,
        model_path,
        model_base=None,
        load_8bit=False,
        load_4bit=False,
        device_map="auto",
        device="cuda",
        use_flash_attn=False,
        **kwargs,
    ):
        if "NVILA" in model_path:
            if "lora" in model_path.lower():
                model_base = "Efficient-Large-Model/NVILA-8B"
                model_name = "NVILA-8B-LoRA"
            else:
                model_name = "NVILA-8B"
        elif "VILA-M3" in model_path:
            if "lora" in model_path.lower():
                model_base = "Efficient-Large-Model/Llama-3-VILA1.5-8B"
                model_name = "VILA-M3-LoRA"
            else:
                model_name = "VILA-M3"
        elif "VILA1.5" in model_path:
            if "lora" in model_path.lower():
                model_base = "Efficient-Large-Model/Llama-3-VILA1.5-8B"
                model_name = "VILA1.5-8B-LoRA"
            else:
                model_name = "VILA1.5-8B"
        else:
            raise NotImplementedError
        
        kwargs = {"device_map": device_map, **kwargs}
        kwargs = {**kwargs}

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
            # kwargs["torch_dtype"] = torch.bfloat16

        if is_mm_model(model_path):
            # Load LLaVA model
            ## TODO @yunhao: mind fixing lora
            if "lora" in model_name.lower() and model_base is None:
                warnings.warn(
                    "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
                )
            if ("lora" in model_name.lower() or "dora" in model_name.lower()) and model_base is not None:
                lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
                print(lora_cfg_pretrained)
                print("Loading LLaVA from base model...")
                config = AutoConfig.from_pretrained(model_base)
                prepare_config_for_eval(config, kwargs)
                model = LlavaLlamaModel.from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)
                tokenizer = model.tokenizer
                token_num, tokem_dim = model.llm.lm_head.out_features, model.llm.lm_head.in_features
                if model.llm.lm_head.weight.shape[0] != token_num:
                    model.llm.lm_head.weight = torch.nn.Parameter(
                        torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
                    )
                    model.llm.embed_tokens.weight = torch.nn.Parameter(
                        torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
                    )

                print("Loading additional LLaVA weights...")
                if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                    non_lora_trainables = torch.load(
                        os.path.join(model_path, "non_lora_trainables.bin"),
                        map_location="cpu",
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

                print("Loading LoRA weights...")
                model = PeftModel.from_pretrained(model, model_path)
                print("Merging LoRA weights...")
                model = model.merge_and_unload()
                print("Model is loaded...")
            else:
                config = AutoConfig.from_pretrained(model_path)
                config.resume_path = model_path
                prepare_config_for_eval(config, kwargs)
                if "topdown" in config.model_type.lower():
                    model = LlavaTopDownLlamaModel(config=config, low_cpu_mem_usage=True, **kwargs)
                else:
                    model = LlavaLlamaModel(config=config, low_cpu_mem_usage=True, **kwargs)
                tokenizer = model.tokenizer
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
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, legacy=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        model.eval()
        image_processor = None
        if is_mm_model(model_path):
            model.resize_token_embeddings(len(tokenizer))
            vision_tower = model.get_vision_tower()
            if vision_tower is None:
                raise ValueError("Vision tower failed to load!")
            vision_tower.to(device=device, dtype=torch.float16)
            # vision_tower.to(device=device, dtype=torch.bfloat16)
            mm_projector = model.get_mm_projector()
            mm_projector.to(device=device, dtype=torch.float16)
            # mm_projector.to(device=device, dtype=torch.bfloat16)
            image_processor = vision_tower.image_processor

        if hasattr(model.llm.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048
        
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len
    
    def find_all_linear_names(self, model, lora_llm, lora_vt):
        cls = torch.nn.Linear
        lora_module_names = set()
        multimodal_keywords = ["mm_projector", "vision_resampler"]
        assert lora_llm or lora_vt, "Not applying LoRA to any of the modules..."

        if not lora_llm:
            multimodal_keywords += ["llm"]
        if not lora_vt:
            multimodal_keywords += ["vision_tower"]

        for name, module in model.named_modules():
            if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                continue
            if isinstance(module, cls):
                if not "lm_head" in name:
                    lora_module_names.add(name)
                # names = name.split(".")
                # lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        # if "lm_head" in lora_module_names:  # needed for 16-bit
        #     lora_module_names.remove("lm_head")
        return list(lora_module_names)

    def smart_tokenizer_and_embedding_resize(
            self,
            special_tokens_dict: Dict,
            tokenizer: transformers.PreTrainedTokenizer,
            model: transformers.PreTrainedModel,
        ):
        """Resize tokenizer and embedding.

        Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
        """
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg


    def infer_vision_language(self, image, qs, temperature=0, image_size=224):
        # Model inference for vision-language tasks
        # TODO: Make it work for a batch
        image = transforms.ToPILImage()(image)
        # image = image.resize((image_size, image_size), Image.BICUBIC)
        prompt = [image, qs]
        answer_generated = self.model.generate_content(prompt)
        print(answer_generated)
        return answer_generated


    def infer_language(self, qs, temperature=0):
        raise NotImplementedError

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

    # run this command to set absolute import path before testing "export PYTHONPATH=/bigdata/rjin02/MedVLMBench:$PYTHONPATH"

    model_path = "Efficient-Large-Model/NVILA-8B"
    model_name = "NVILA-8B"

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

    llava_model = VILA(args=edict(model_path=model_path, model_name=model_name, model_base=None))
    llava_model.load_from_pretrained(model_path=model_path, model_name=model_name, model_base=None)
    
    output = llava_model.infer_vision_language(img, prompt)
    print(output)