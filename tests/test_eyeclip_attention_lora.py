import logging
from types import SimpleNamespace

import pytest
import torch
from open_clip.model import CLIP, CLIPTextCfg, CLIPVisionCfg

import model.eyeclip as eyeclip


def _tiny_eyeclip():
    model = CLIP(
        embed_dim=512,
        vision_cfg=CLIPVisionCfg(layers=1, width=64, head_width=64, patch_size=16, image_size=32),
        text_cfg=CLIPTextCfg(context_length=77, vocab_size=49408, width=64, heads=1, layers=1),
        quick_gelu=True,
    )
    return model, lambda image: image


@pytest.mark.parametrize(
    ("usage", "model_class"),
    [
        ("clip-img-lora", eyeclip.EyeCLIPForDiagnosis),
        ("img-lora-lp", eyeclip.EyeCLIPLPForDiagnosis),
    ],
)
def test_eyeclip_attention_lora_receives_gradient_and_roundtrips(monkeypatch, usage, model_class):
    monkeypatch.setattr(eyeclip, "_load_eyeclip", _tiny_eyeclip)
    args = SimpleNamespace(usage=usage, device="cpu", logger=logging.getLogger(__name__))
    model = model_class(args=args, text=["normal retina", "glaucoma fundus"], num_classes=2)

    image = torch.randn(2, 3, 32, 32)
    adapter = model.model.visual.transformer.resblocks[0].attn.out_proj.parametrizations.weight[0]
    model(image).sum().backward()
    assert adapter.B.grad is not None and adapter.B.grad.abs().sum() > 0

    optimizer = torch.optim.SGD((adapter.A, adapter.B), lr=0.01)
    optimizer.step()
    optimizer.zero_grad()
    model(image).sum().backward()
    assert adapter.A.grad is not None and adapter.A.grad.abs().sum() > 0

    model.eval()
    restored = model_class(args=args, text=["normal retina", "glaucoma fundus"], num_classes=2)
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(image), model(image))

    # Historical checkpoints used PEFT Linear keys for attention out_proj.
    legacy_suffixes = {
        ".attn.out_proj.parametrizations.weight.original": ".attn.out_proj.base_layer.weight",
        ".attn.out_proj.bias": ".attn.out_proj.base_layer.bias",
        ".attn.out_proj.parametrizations.weight.0.A": ".attn.out_proj.lora_A.default.weight",
        ".attn.out_proj.parametrizations.weight.0.B": ".attn.out_proj.lora_B.default.weight",
    }
    source = model.model if usage == "clip-img-lora" else model
    target = restored.model if usage == "clip-img-lora" else restored
    legacy_state = {}
    for key, value in source.state_dict().items():
        legacy_key = key
        if ".attn.out_proj." in key and not key.startswith("transformer.resblocks"):
            for new, old in legacy_suffixes.items():
                if key.endswith(new):
                    legacy_key = key[: -len(new)] + old
                    break
        legacy_state[legacy_key] = value
    target.load_state_dict(legacy_state, strict=True)
