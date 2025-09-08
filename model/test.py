from release.llava.model import LlavaLlamaForCausalLM
from release.llava_med.model import LlavaMistralForCausalLM
import torch
from easydict import EasyDict as edict

# model = LlavaLlamaForCausalLM.from_pretrained(
#     "/research/d5/gds/yzhong22/misc/pretrained/llava-v1.5-7b",
#     cache_dir=None,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
# )

model = LlavaMistralForCausalLM.from_pretrained(
    "/research/d5/gds/yzhong22/misc/pretrained/llava-med-v1.5-mistral-7b",
    low_cpu_mem_usage=False,
)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


print(model)
print(find_all_linear_names(model))

for n, m in model.named_modules():
    print(n)

# for n, p in model.named_parameters():
#     if p.requires_grad is True:
#         print(n)

# from peft import LoraConfig, get_peft_model

# lora_config = LoraConfig(
#     r=128,
#     lora_alpha=256,
#     target_modules=find_all_linear_names(model),
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
# )

# model = get_peft_model(model, lora_config)

# model_args = edict(
#     vision_tower="openai/clip-vit-large-patch14-336",
#     mm_vision_select_layer=-2,
#     mm_projector_type="mlp2x_gelu",
#     cache_dir="/research/d5/gds/yzhong22/misc/cache",
#     mm_vision_select_feature=None,
#     pretrain_mm_mlp_adapter=None,
#     mm_patch_merge_type=None,
# )
# model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)

# print(model)

# for n, p in model.named_parameters():
#     if p.requires_grad is True:
#         print(n)
