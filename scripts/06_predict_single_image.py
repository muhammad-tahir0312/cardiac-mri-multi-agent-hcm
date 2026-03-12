"""
scripts/10_predict_single_image.py

Run single-image inference for 2D HCM classification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cardiac_image_dataset import build_image_transforms
from src.models.image_backbones import ImageClassifier2D


def _resolve(path_like: Path) -> Path:
    return path_like if path_like.is_absolute() else (PROJECT_ROOT / path_like)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-image HCM prediction with confidence score.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=Path, required=True, help="Path to image file.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained .pt checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("src/config/image2d.yaml"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_json", type=Path, default=None, help="Optional path to save JSON prediction output.")
    return parser.parse_args()


def build_eval_transform(cfg: DictConfig, use_rgb: bool):
    if use_rgb:
        mean = list(cfg.data.normalization.mean_rgb)
        std = list(cfg.data.normalization.std_rgb)
    else:
        mean = list(cfg.data.normalization.mean_grayscale)
        std = list(cfg.data.normalization.std_grayscale)

    return build_image_transforms(
        image_size=int(cfg.data.image_size),
        train=False,
        replicate_to_rgb=use_rgb,
        mean=mean,
        std=std,
    )


def main() -> None:
    args = parse_args()

    cfg: DictConfig = OmegaConf.load(_resolve(args.config))
    image_path = _resolve(args.image)
    checkpoint_path = _resolve(args.checkpoint)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    use_rgb = bool(ckpt.get("use_rgb", bool(cfg.data.replicate_to_rgb_if_pretrained) and bool(cfg.model.pretrained)))
    effective_in_channels = int(ckpt.get("effective_in_channels", 3 if use_rgb else int(cfg.model.in_channels)))

    transform = build_eval_transform(cfg, use_rgb=use_rgb)
    image_tensor = transform(Image.open(image_path).convert("L")).unsqueeze(0).to(device)

    model = ImageClassifier2D(
        backbone=str(cfg.model.backbone),
        pretrained=False,
        in_channels=effective_in_channels,
        num_classes=int(cfg.model.num_classes),
        dropout=float(cfg.model.dropout),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    prob_normal = float(probs[0].item())
    prob_sick = float(probs[1].item())
    pred_label = int(prob_sick >= float(args.threshold))

    idx_to_class: Dict[int, str] = {0: "Normal", 1: "Sick"}
    payload = {
        "image_path": str(image_path),
        "predicted_label": pred_label,
        "predicted_class": idx_to_class[pred_label],
        "prob_normal": prob_normal,
        "prob_sick": prob_sick,
        "confidence": float(max(prob_normal, prob_sick)),
        "threshold": float(args.threshold),
    }

    print(json.dumps(payload, indent=2))

    if args.save_json is not None:
        out_path = _resolve(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
