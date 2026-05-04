from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_lightmedseg import (
    build_model,
    dataset_from_samples,
    format_metrics,
    get_device,
    paired_samples,
    validate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LightMedSeg-2D on PraNet-style test datasets.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--best-metric", choices=["dice", "iou"], default="dice")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--kvasir-split", type=Path, default=Path("data/splits/pranet_light_seed42/test_kvasir.txt"))
    parser.add_argument("--cvc-clinicdb-root", type=Path, default=Path("data/CVC-ClinicDB/test"))
    parser.add_argument("--pranet-test-root", type=Path, default=Path("data/PraNet-TestDataset"))
    return parser.parse_args()


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        return args.checkpoint

    pattern = f"*/lightmedseg_best_{args.best_metric}.pt"
    candidates = [path for path in args.checkpoint_root.glob(pattern) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint matching {pattern} under {args.checkpoint_root}. "
            "Train first or pass --checkpoint explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_split_samples(split_file: Path) -> list[tuple[Path, Path]]:
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    samples: list[tuple[Path, Path]] = []
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_path, mask_path = line.split("\t")
            samples.append((Path(image_path), Path(mask_path)))
    if not samples:
        raise ValueError(f"Split file contains no samples: {split_file}")
    return samples


def build_eval_datasets(args: argparse.Namespace, image_size: int) -> dict[str, object]:
    datasets = {
        "Kvasir": dataset_from_samples(load_split_samples(args.kvasir_split), image_size, augment=False),
        "CVC-ClinicDB": dataset_from_samples(
            paired_samples(args.cvc_clinicdb_root / "images", args.cvc_clinicdb_root / "masks"),
            image_size,
            augment=False,
        ),
        "CVC-300": dataset_from_samples(
            paired_samples(args.pranet_test_root / "CVC-300" / "images", args.pranet_test_root / "CVC-300" / "masks"),
            image_size,
            augment=False,
        ),
        "CVC-ColonDB": dataset_from_samples(
            paired_samples(
                args.pranet_test_root / "CVC-ColonDB" / "images",
                args.pranet_test_root / "CVC-ColonDB" / "masks",
            ),
            image_size,
            augment=False,
        ),
        "ETIS-LaribPolypDB": dataset_from_samples(
            paired_samples(
                args.pranet_test_root / "ETIS-LaribPolypDB" / "images",
                args.pranet_test_root / "ETIS-LaribPolypDB" / "masks",
            ),
            image_size,
            augment=False,
        ),
    }
    return datasets


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    checkpoint_path = resolve_checkpoint(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    variant = checkpoint.get("variant", "tiny")
    image_size = args.image_size or int(checkpoint.get("image_size", 256))
    model = build_model(variant)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Variant: {variant}")
    print(f"Image size: {image_size}")

    datasets = build_eval_datasets(args, image_size)
    all_metrics: dict[str, dict[str, float]] = {}

    for name, dataset in datasets.items():
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        metrics = validate(model, loader, device, desc=name)
        all_metrics[name] = metrics
        print(f"{name} ({len(dataset)} samples) | {format_metrics(name, metrics)}")

    mean_metrics = {
        key: sum(metrics[key] for metrics in all_metrics.values()) / len(all_metrics)
        for key in ["loss", "dice", "iou", "precision", "recall", "mae"]
    }
    print(f"Mean across datasets | {format_metrics('mean', mean_metrics)}")


if __name__ == "__main__":
    main()
