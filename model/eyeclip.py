import os
import torch
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


class _EyeCLIPMixin:
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
