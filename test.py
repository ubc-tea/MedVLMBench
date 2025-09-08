from model.release.llava.model import LlavaLlamaForCausalLM, LlavaMptForCausalLM, LlavaMistralForCausalLM
from model.release.llava.conversation import conv_templates, default_conversation
from model.release.llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from model.release.llava.model.language_model.llava_llama import LlavaConfig
from model.chat import ChatMetaModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, LlamaForCausalLM

import os
import torch

model_path = "/research/d5/gds/yzhong22/experiments/med_vlm_benchmark/vqa/SLAKE/LLaVA-1.5/train_lora_V_seed42_llava"
model_base = "/research/d5/gds/yzhong22/misc/pretrained/llava-v1.5-7b"
device_map = "auto"

lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
print("Loading LLaVA from base model...")
model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained)
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
    non_lora_trainables = torch.load(os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu")
else:
    # this is probably from HF Hub
    from huggingface_hub import hf_hub_download

    def load_from_hf(repo_id, filename, subfolder=None):
        cache_file = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder)
        return torch.load(cache_file, map_location="cpu")

    non_lora_trainables = load_from_hf(model_path, "non_lora_trainables.bin")
non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}
if any(k.startswith("model.model.") for k in non_lora_trainables):
    non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
model.load_state_dict(non_lora_trainables, strict=False)

from peft import PeftModel

vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model(device_map=device_map)

print_count = 10
count = 0
for n, p in model.get_vision_tower().named_parameters():
    print(n)
    print(p)

    count += 1

    if count == print_count:
        break

import copy

before = copy.deepcopy(model.get_vision_tower())

print("Loading LoRA weights...")
model = PeftModel.from_pretrained(model, model_path)

print_count = 10
count = 0
for n, p in model.get_vision_tower().named_parameters():
    print(n)
    print(p)

    count += 1

    if count == print_count:
        break


print(model)
print("Merging LoRA weights...")
model = model.merge_and_unload()


# for n, p in model.get_vision_tower().named_parameters():
#     print(n)
#     print(p)


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print("Mismtach found at", key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print("Models match perfectly! :)")


count = 0
for n, p in model.get_vision_tower().named_parameters():
    print(n)
    print(p)

    count += 1

    if count == print_count:
        break

compare_models(before, model.get_vision_tower())
