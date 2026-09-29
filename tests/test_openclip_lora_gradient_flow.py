import logging
from types import SimpleNamespace

import pytest
import torch
from open_clip.model import CLIP, CLIPTextCfg, CLIPVisionCfg

from model.openclip_base import OpenCLIPForDiagnosis, OpenCLIPLPForDiagnosis


def _tiny_openclip():
    return CLIP(
        embed_dim=512,
        vision_cfg=CLIPVisionCfg(layers=1, width=64, head_width=64, patch_size=16, image_size=32),
        text_cfg=CLIPTextCfg(context_length=77, vocab_size=49408, width=64, heads=1, layers=1),
    ), lambda image: image


class _TinyZeroShot(OpenCLIPForDiagnosis):
    build_model = staticmethod(_tiny_openclip)

    def tokenize(self, texts):
        return torch.ones(len(texts), 77, dtype=torch.long)


class _TinyLinearProbe(OpenCLIPLPForDiagnosis):
    build_model = staticmethod(_tiny_openclip)

    def tokenize(self, texts):
        return torch.ones(len(texts), 77, dtype=torch.long)


@pytest.mark.parametrize(
    ("usage", "model_class"),
    [("clip-img-lora", _TinyZeroShot), ("img-lora-lp", _TinyLinearProbe)],
)
def test_multihead_attention_lora_updates_actual_image_output(usage, model_class):
    torch.manual_seed(17)
    args = SimpleNamespace(usage=usage, device="cpu", logger=logging.getLogger(__name__))
    model = model_class(args=args, text=["normal", "abnormal"], num_classes=2).eval()
    images = torch.randn(2, 3, 32, 32)
    attention = model.model.visual.transformer.resblocks[0].attn
    adapter = attention.out_proj.parametrizations.weight[0]
    frozen_weight = attention.out_proj.parametrizations.weight.original.detach().clone()

    before = model(images).detach().clone()
    model(images).square().sum().backward()
    assert adapter.B.grad is not None and adapter.B.grad.abs().sum() > 0
    assert not any("attn.out_proj.lora_B" in name for name, _ in model.named_parameters())

    optimizer = torch.optim.SGD([adapter.B], lr=0.01)
    optimizer.step()
    after = model(images).detach()
    assert (after - before).abs().max() > 1e-8
    torch.testing.assert_close(attention.out_proj.parametrizations.weight.original, frozen_weight)
