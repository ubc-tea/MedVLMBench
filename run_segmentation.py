"""Prompted 2D segmentation evaluation for the SAM family.

The CSV manifest must contain ``image`` and ``mask`` paths. A center point or
bounding box is computed from the ground-truth mask for each nonempty sample.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


SEGMENTATION_MODELS = (
    "SAM", "MedSAM", "SAM2", "MedSAM2", "MobileSAM", "TinySAM",
    "SAM-Med2D", "FT-SAM",
)


def load_predictor(args):
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    if args.model in {"SAM2", "MedSAM2"}:
        if not args.sam2_config:
            raise ValueError(f"{args.model} requires --sam2-config")
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise ImportError("Install the official SAM2 package to evaluate SAM2 or MedSAM2") from exc
        model = build_sam2(args.sam2_config, str(args.checkpoint), device=args.device)
        return SAM2ImagePredictor(model)

    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise ImportError("Install the official segment-anything package for SAM-family models") from exc

    if args.model in {"SAM", "MedSAM"}:
        model = sam_model_registry[args.sam_variant](checkpoint=str(args.checkpoint))
    elif args.model in {"MobileSAM", "TinySAM"}:
        from model.segmentation_builders.build_tinysam import sam_model_registry2

        model = sam_model_registry2["vit_t"](checkpoint=str(args.checkpoint))
    else:
        from types import SimpleNamespace
        from model.segmentation_builders.build_sammed2d import sam_model_registry1

        build_args = SimpleNamespace(
            img_size=1024, sam_checkpoint=str(args.checkpoint), encoder_adapter=False
        )
        model = sam_model_registry1["vit_b"](build_args)
    model.to(args.device)
    model.eval()
    return SamPredictor(model)


def prompt_from_mask(mask, prompt):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    if prompt == "box":
        return {"box": np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)}
    center = np.array([xs.mean(), ys.mean()])
    closest = np.argmin((xs - center[0]) ** 2 + (ys - center[1]) ** 2)
    return {
        "point_coords": np.array([[xs[closest], ys[closest]]], dtype=np.float32),
        "point_labels": np.array([1], dtype=np.int32),
    }


def evaluate(manifest, predictor, prompt, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"image", "mask"}.issubset(reader.fieldnames or ()):
            raise ValueError("Manifest must have image and mask columns")
        entries = list(reader)

    rows = []
    empty_masks = 0
    for entry in entries:
        image_path = manifest.parent / entry["image"]
        mask_path = manifest.parent / entry["mask"]
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = mask > 0
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask size mismatch: {image_path}, {mask_path}")
        arguments = prompt_from_mask(mask, prompt)
        if arguments is None:
            empty_masks += 1
            continue
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(multimask_output=True, **arguments)
        best = np.asarray(masks[int(np.argmax(scores))]) > 0
        intersection = np.logical_and(best, mask).sum()
        total = best.sum() + mask.sum()
        union = np.logical_or(best, mask).sum()
        rows.append({
            "image": entry["image"], "mask": entry["mask"],
            "dice": float(2 * intersection / total),
            "iou": float(intersection / union),
        })

    summary = {
        "manifest": str(manifest.resolve()), "prompt": prompt,
        "samples_total": len(entries), "samples_evaluated": len(rows),
        "empty_masks_skipped": empty_masks,
        "mean_dice": float(np.mean([r["dice"] for r in rows])) if rows else None,
        "mean_iou": float(np.mean([r["iou"] for r in rows])) if rows else None,
    }
    with (output_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image", "mask", "dice", "iou"))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=SEGMENTATION_MODELS)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sam-variant", choices=("vit_b", "vit_l", "vit_h"), default="vit_b")
    parser.add_argument("--sam2-config", help="SAM2 YAML model configuration")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--prompt", choices=("box", "point"), default="box")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    predictor = load_predictor(args)
    summary = evaluate(args.manifest, predictor, args.prompt, args.output_dir)
    summary.update({"model": args.model, "checkpoint": str(args.checkpoint.resolve())})
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
