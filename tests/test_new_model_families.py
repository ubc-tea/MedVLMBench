import csv
import json
import logging
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch import nn

from model import get_model
import model.vision_foundation as foundation
import model.siglip2 as siglip2
from run_segmentation import evaluate, prompt_from_mask


def test_foundation_probe_updates_head_and_reloads(monkeypatch, tmp_path):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(3, 4)

        def forward(self, images):
            return self.projection(images.mean(dim=(2, 3)))

    monkeypatch.setattr(
        foundation, "_build_backbone",
        lambda name, args: (Encoder(), lambda image: image, 4, "timm"),
    )
    monkeypatch.setattr("model.get_prototype", lambda args: ["healthy", "disease"])
    args = SimpleNamespace(
        task="diagnosis", model="DINOv3", usage="lp", dataset="papila",
        device="cpu", logger=logging.getLogger(__name__),
    )
    torch.manual_seed(1)
    model = get_model(args)
    image = torch.randn(2, 3, 8, 8)
    before = model(image).detach().clone()
    loss = model(image).square().sum()
    loss.backward()
    assert model.head.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.model.parameters())
    torch.optim.SGD(model.head.parameters(), lr=0.1).step()
    assert not torch.allclose(before, model(image))

    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = get_model(args)
    restored.load_from_pretrained(checkpoint, device="cpu")
    torch.testing.assert_close(restored(image), model(image))


def test_segmentation_metrics_and_empty_mask_accounting(tmp_path):
    class Predictor:
        def set_image(self, image):
            self.shape = image.shape[:2]

        def predict(self, **kwargs):
            assert "box" in kwargs
            mask = np.zeros(self.shape, dtype=bool)
            mask[1:3, 1:3] = True
            return np.stack([~mask, mask]), np.array([0.1, 0.9]), None

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    Image.fromarray(image).save(tmp_path / "image.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")
    Image.fromarray(np.zeros_like(mask)).save(tmp_path / "empty.png")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "mask"])
        writer.writerow(["image.png", "mask.png"])
        writer.writerow(["image.png", "empty.png"])
    summary = evaluate(manifest, Predictor(), "box", tmp_path / "results")
    assert summary["samples_evaluated"] == 1
    assert summary["empty_masks_skipped"] == 1
    assert summary["mean_dice"] == summary["mean_iou"] == 1.0
    assert json.loads((tmp_path / "results" / "metrics.json").read_text()) == summary
    assert prompt_from_mask(mask > 0, "point")["point_labels"].tolist() == [1]
    assert prompt_from_mask(mask > 0, "box")["box"].tolist() == [1, 1, 3, 3]


def test_siglip2_zero_shot_and_lp_factory(monkeypatch):
    class Tokens(dict):
        def to(self, device):
            return self

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(3, 4)

        def forward(self, images):
            return self.layer(images.mean(dim=(2, 3)))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = Encoder()
            self.config = SimpleNamespace(
                vision_config=SimpleNamespace(hidden_size=4),
                text_config=SimpleNamespace(hidden_size=4),
            )

        def get_image_features(self, pixel_values):
            return self.vision_model(pixel_values)

        def forward(self, pixel_values, input_ids, attention_mask):
            return SimpleNamespace(logits_per_image=self.get_image_features(pixel_values)[:, :2])

    processor = SimpleNamespace(
        tokenizer=lambda texts, **kwargs: Tokens(
            input_ids=torch.ones(len(texts), 2, dtype=torch.long),
            attention_mask=torch.ones(len(texts), 2, dtype=torch.long),
        ),
        image_processor=lambda image, **kwargs: {"pixel_values": image.unsqueeze(0)},
    )
    monkeypatch.setattr(siglip2.Siglip2Model, "from_pretrained", lambda source: Model())
    monkeypatch.setattr(siglip2.AutoProcessor, "from_pretrained", lambda source: processor)
    monkeypatch.setattr("model.get_prototype", lambda args: ["healthy", "disease"])
    for usage in ("clip-zs", "lp"):
        args = SimpleNamespace(
            task="diagnosis", model="SigLIP2", usage=usage, dataset="papila",
            device="cpu", logger=logging.getLogger(__name__), vision_backbone=None,
        )
        model = get_model(args)
        assert model(torch.randn(2, 3, 8, 8)).shape == (2, 2)
