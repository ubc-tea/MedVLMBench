import os
import math
import torch
from torch import nn
from torch.nn.utils import parametrize
import open_clip
from model.openclip_base import OpenCLIPForDiagnosis, OpenCLIPLPForDiagnosis

# EyeCLIP (Shi et al.) ships as a CLIP ViT-B/32 (QuickGELU) training checkpoint that also carries
# an MAE decoder and optimizer state. Only the CLIP towers are needed here. Official weights:
# https://drive.google.com/file/d/1kWpbDqFCFt4j8RkYqacV4nl-aCKZfqZr (see github.com/Michi-3000/EyeCLIP)
_ARCH = "ViT-B-32-quickgelu"
_DEFAULT_CKPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pretrained_models", "eyeclip", "eyeclip_visual.pt"
)


def _load_eyeclip():
    ckpt_path = os.environ.get("EYECLIP_CKPT", _DEFAULT_CKPT)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"EyeCLIP checkpoint not found at {ckpt_path}. Download `eyeclip_visual.pt` from the official "
            "EyeCLIP repository (https://github.com/Michi-3000/EyeCLIP) and place it there, "
            "or point EYECLIP_CKPT to it."
        )
    model, _, preprocess = open_clip.create_model_and_transforms(_ARCH, pretrained=None)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("visual.decoder")}
    model.load_state_dict(state_dict, strict=True)
    return model, preprocess


class _AttentionOutputLoRA(nn.Module):
    """Add a trainable low-rank update to the weight read by MultiheadAttention."""

    def __init__(self, weight, rank=8, alpha=32):
        super().__init__()
        self.A = nn.Parameter(weight.new_empty(rank, weight.shape[1]))
        self.B = nn.Parameter(weight.new_zeros(weight.shape[0], rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, weight):
        return weight + (self.B @ self.A) * self.scale


def _upgrade_legacy_attention_lora(module, state_dict, prefix, *args):
    """Read checkpoints saved when PEFT's out_proj adapter was bypassed."""
    suffixes = {
        ".attn.out_proj.base_layer.weight": ".attn.out_proj.parametrizations.weight.original",
        ".attn.out_proj.base_layer.bias": ".attn.out_proj.bias",
        ".attn.out_proj.lora_A.default.weight": ".attn.out_proj.parametrizations.weight.0.A",
        ".attn.out_proj.lora_B.default.weight": ".attn.out_proj.parametrizations.weight.0.B",
    }
    for key in list(state_dict):
        if not key.startswith(prefix):
            continue
        for old, new in suffixes.items():
            if key.endswith(old):
                state_dict[key[: -len(old)] + new] = state_dict.pop(key)
                break


class _EyeCLIPMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.usage in {"clip-img-lora", "img-lora-lp"}:
            # nn.MultiheadAttention reads out_proj.weight directly, bypassing
            # the forward method patched by PEFT's Linear LoRA adapter.
            for block in self.model.visual.transformer.resblocks:
                parametrize.register_parametrization(
                    block.attn.out_proj, "weight", _AttentionOutputLoRA(block.attn.out_proj.weight)
                )
            self.args.logger.info(
                "Enabled %d EyeCLIP attention output LoRA parametrizations",
                len(self.model.visual.transformer.resblocks),
            )
            self.model.register_load_state_dict_pre_hook(_upgrade_legacy_attention_lora)

    def find_target_linear_names(self, model, *args, **kwargs):
        verbose = kwargs.pop("verbose", True)
        names = super().find_target_linear_names(model, *args, verbose=False, **kwargs)
        names = [name for name in names if not name.endswith(".attn.out_proj")]
        if verbose:
            self.args.logger.info(f"Found {len(names)} PEFT lora modules: {names}")
        return names

    def build_model(self):
        return _load_eyeclip()

    def tokenize(self, texts):
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = open_clip.get_tokenizer(_ARCH)
        return self._tokenizer(texts)


class EyeCLIPForDiagnosis(_EyeCLIPMixin, OpenCLIPForDiagnosis):
    pass


class EyeCLIPLPForDiagnosis(_EyeCLIPMixin, OpenCLIPLPForDiagnosis):
    pass
