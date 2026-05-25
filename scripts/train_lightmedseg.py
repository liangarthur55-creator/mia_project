from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import LightMedSegOutput, lightmedseg_small, lightmedseg_tiny


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightMedSeg-2D on paired image/mask folders.")
    parser.add_argument("--protocol", choices=["random", "pranet-light"], default="random")
    parser.add_argument("--images-dir", type=Path, default=Path("data/Kvasir-SEG/images"))
    parser.add_argument("--masks-dir", type=Path, default=Path("data/Kvasir-SEG/masks"))
    parser.add_argument("--cvc-root", type=Path, default=Path("data/CVC-ClinicDB"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--best-metric", choices=["dice", "iou"], default="dice")
    parser.add_argument("--variant", choices=["tiny", "small"], default="tiny")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--edge-loss-weight", type=float, default=0.2)
    parser.add_argument("--use-lha", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-egff", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-dfc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multi-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--size-rates", type=str, default="0.75,1,1.25")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional small subset size for quick smoke runs.")
    parser.add_argument("--save-split-dir", type=Path, default=Path("data/splits/kvasir_seed42_70_10_20"))
    parser.add_argument("--kvasir-train-count", type=int, default=900)
    parser.add_argument("--kvasir-test-count", type=int, default=100)
    parser.add_argument("--cvc-train-count", type=int, default=550)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(device_name: str) -> torch.device:
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but PyTorch cannot use MPS in this environment.")
        return torch.device("mps")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot use CUDA in this environment.")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SegmentationFolderDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        image_size: int,
        augment: bool = False,
        samples: list[tuple[Path, Path]] | None = None,
    ) -> None:
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.augment = augment
        self.samples = samples if samples is not None else self._match_samples(images_dir, masks_dir)

    def _match_samples(self, images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

        image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        mask_by_stem = {p.stem: p for p in masks_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
        samples = [(p, mask_by_stem[p.stem]) for p in image_paths if p.stem in mask_by_stem]
        samples.sort(key=lambda pair: pair[0].name)

        if not samples:
            raise ValueError(f"No paired image/mask files found in {images_dir} and {masks_dir}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        if self.augment:
            image, mask = apply_training_augmentations(image, mask)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return image_tensor, mask_tensor


def paired_samples(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    return SegmentationFolderDataset(images_dir, masks_dir, image_size=1)._match_samples(images_dir, masks_dir)


def apply_training_augmentations(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    if random.random() < 0.5:
        angle = random.uniform(-15.0, 15.0)
        image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)
    if random.random() < 0.5:
        brightness = random.uniform(0.9, 1.1)
        contrast = random.uniform(0.9, 1.1)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
    return image, mask


def dataset_from_samples(
    samples: list[tuple[Path, Path]],
    image_size: int,
    augment: bool,
) -> SegmentationFolderDataset:
    return SegmentationFolderDataset(Path("."), Path("."), image_size=image_size, augment=augment, samples=samples)


def split_dataset(
    dataset: SegmentationFolderDataset,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    limit: int | None,
    train_augment: bool,
) -> tuple[SegmentationFolderDataset, SegmentationFolderDataset, SegmentationFolderDataset]:
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("--val-ratio and --test-ratio must be positive and sum to less than 1.")

    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    if limit is not None:
        indices = indices[: min(limit, len(indices))]

    test_size = max(1, int(len(indices) * test_ratio))
    val_size = max(1, int(len(indices) * val_ratio))
    if len(indices) - val_size - test_size <= 0:
        raise ValueError("Dataset split is too small. Increase --limit or reduce validation/test ratios.")

    test_indices = indices[:test_size]
    val_indices = indices[test_size : test_size + val_size]
    train_indices = indices[test_size + val_size :]
    train_samples = [dataset.samples[index] for index in train_indices]
    val_samples = [dataset.samples[index] for index in val_indices]
    test_samples = [dataset.samples[index] for index in test_indices]
    return (
        dataset_from_samples(train_samples, dataset.image_size, augment=train_augment),
        dataset_from_samples(val_samples, dataset.image_size, augment=False),
        dataset_from_samples(test_samples, dataset.image_size, augment=False),
    )


def build_pranet_light_splits(args: argparse.Namespace) -> dict[str, SegmentationFolderDataset]:
    rng = random.Random(args.seed)

    kvasir_samples = paired_samples(args.images_dir, args.masks_dir)
    rng.shuffle(kvasir_samples)
    if len(kvasir_samples) < args.kvasir_train_count + args.kvasir_test_count:
        raise ValueError("Kvasir-SEG does not have enough paired samples for the requested PraNet-light split.")

    kvasir_train_pool = kvasir_samples[: args.kvasir_train_count]
    kvasir_test = kvasir_samples[args.kvasir_train_count : args.kvasir_train_count + args.kvasir_test_count]

    cvc_train_pool = (
        paired_samples(args.cvc_root / "train" / "images", args.cvc_root / "train" / "masks")
        + paired_samples(args.cvc_root / "validation" / "images", args.cvc_root / "validation" / "masks")
    )
    rng.shuffle(cvc_train_pool)
    cvc_train_pool = cvc_train_pool[: min(args.cvc_train_count, len(cvc_train_pool))]
    cvc_test = paired_samples(args.cvc_root / "test" / "images", args.cvc_root / "test" / "masks")

    train_pool = kvasir_train_pool + cvc_train_pool
    rng.shuffle(train_pool)
    if args.limit is not None:
        train_pool = train_pool[: min(args.limit, len(train_pool))]

    val_size = max(1, int(len(train_pool) * args.val_ratio))
    if len(train_pool) - val_size <= 0:
        raise ValueError("PraNet-light training pool is too small. Increase --limit or reduce --val-ratio.")

    val_samples = train_pool[:val_size]
    train_samples = train_pool[val_size:]
    test_samples = kvasir_test + cvc_test

    return {
        "train": dataset_from_samples(train_samples, args.image_size, augment=args.train_augment),
        "val": dataset_from_samples(val_samples, args.image_size, augment=False),
        "test": dataset_from_samples(test_samples, args.image_size, augment=False),
        "test_kvasir": dataset_from_samples(kvasir_test, args.image_size, augment=False),
        "test_cvc_clinicdb": dataset_from_samples(cvc_test, args.image_size, augment=False),
    }


def build_random_splits(args: argparse.Namespace) -> dict[str, SegmentationFolderDataset]:
    dataset = SegmentationFolderDataset(args.images_dir, args.masks_dir, args.image_size, augment=True)
    train_set, val_set, test_set = split_dataset(
        dataset, args.val_ratio, args.test_ratio, args.seed, args.limit, args.train_augment
    )
    return {
        "train": train_set,
        "val": val_set,
        "test": test_set,
    }


def build_splits(args: argparse.Namespace) -> dict[str, SegmentationFolderDataset]:
    if args.protocol == "pranet-light":
        return build_pranet_light_splits(args)
    return build_random_splits(args)


def save_split_files(split_dir: Path, splits: dict[str, SegmentationFolderDataset]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, dataset in splits.items():
        with (split_dir / f"{name}.txt").open("w", encoding="utf-8") as f:
            for image_path, mask_path in dataset.samples:
                f.write(f"{image_path}\t{mask_path}\n")


def dice_loss(logits: Tensor, targets: Tensor, eps: float = 1e-6) -> Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dim=dims)
    cardinality = torch.sum(probs + targets, dim=dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def structure_loss(logits: Tensor, targets: Tensor) -> Tensor:
    boundary_weight = 1 + 5 * torch.abs(F.avg_pool2d(targets, kernel_size=31, stride=1, padding=15) - targets)
    weighted_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weighted_bce = (boundary_weight * weighted_bce).sum(dim=(2, 3)) / boundary_weight.sum(dim=(2, 3))

    probs = torch.sigmoid(logits)
    intersection = ((probs * targets) * boundary_weight).sum(dim=(2, 3))
    union = ((probs + targets) * boundary_weight).sum(dim=(2, 3))
    weighted_iou = 1 - (intersection + 1) / (union - intersection + 1)
    return (weighted_bce + weighted_iou).mean()


def edge_targets_from_masks(masks: Tensor) -> Tensor:
    dilated = nn.functional.max_pool2d(masks, kernel_size=3, stride=1, padding=1)
    eroded = -nn.functional.max_pool2d(-masks, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def parse_size_rates(size_rates: str, multi_scale: bool) -> list[float]:
    if not multi_scale:
        return [1.0]
    rates = [float(rate.strip()) for rate in size_rates.split(",") if rate.strip()]
    if not rates:
        raise ValueError("--size-rates must contain at least one scale.")
    return rates


def scale_batch(images: Tensor, masks: Tensor, rate: float) -> tuple[Tensor, Tensor]:
    if rate == 1:
        return images, masks

    height, width = images.shape[-2:]
    scaled_height = max(32, int(round(height * rate / 32) * 32))
    scaled_width = max(32, int(round(width * rate / 32) * 32))
    images = F.interpolate(images, size=(scaled_height, scaled_width), mode="bilinear", align_corners=False)
    masks = F.interpolate(masks, size=(scaled_height, scaled_width), mode="nearest")
    return images, masks


def segmentation_metrics(logits: Tensor, targets: Tensor, eps: float = 1e-6) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    intersection = torch.sum(preds * targets, dim=(1, 2, 3))
    pred_area = torch.sum(preds, dim=(1, 2, 3))
    target_area = torch.sum(targets, dim=(1, 2, 3))
    union = pred_area + target_area - intersection

    dice = (2.0 * intersection + eps) / (pred_area + target_area + eps)
    iou = (intersection + eps) / (union + eps)
    precision = (intersection + eps) / (pred_area + eps)
    recall = (intersection + eps) / (target_area + eps)
    mae = torch.mean(torch.abs(probs - targets), dim=(1, 2, 3))

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "mae": mae.mean().item(),
    }


def build_model(
    variant: str,
    use_lha: bool = True,
    use_egff: bool = True,
    use_dfc: bool = True,
) -> nn.Module:
    if variant == "small":
        return lightmedseg_small(
            in_channels=3,
            num_classes=1,
            use_lha=use_lha,
            use_egff=use_egff,
            use_dfc=use_dfc,
        )
    return lightmedseg_tiny(
        in_channels=3,
        num_classes=1,
        use_lha=use_lha,
        use_egff=use_egff,
        use_dfc=use_dfc,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    edge_loss_weight: float,
    size_rates: list[float],
    use_edge_loss: bool,
) -> float:
    model.train()
    total_loss = 0.0
    bce = nn.BCEWithLogitsLoss()

    progress = tqdm(loader, desc="train", leave=False)
    for images, masks in progress:
        images = images.to(device)
        masks = masks.to(device)

        batch_loss = 0.0
        for rate in size_rates:
            scaled_images, scaled_masks = scale_batch(images, masks, rate)
            optimizer.zero_grad(set_to_none=True)
            output = model(scaled_images, return_aux=True)
            if not isinstance(output, LightMedSegOutput):
                raise TypeError("LightMedSeg model must return auxiliary outputs during training.")

            mask_loss = structure_loss(output.mask_logits, scaled_masks)
            edge_loss = bce(output.edge_logits, edge_targets_from_masks(scaled_masks)) if use_edge_loss else 0.0
            loss = mask_loss + edge_loss_weight * edge_loss

            loss.backward()
            optimizer.step()

            batch_loss += loss.item()

        average_batch_loss = batch_loss / len(size_rates)
        total_loss += average_batch_loss * images.size(0)
        progress.set_postfix(loss=f"{average_batch_loss:.4f}")

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    desc: str = "val",
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    metric_totals = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "mae": 0.0,
    }
    for images, masks in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        if not isinstance(logits, Tensor):
            logits = logits.mask_logits

        loss = structure_loss(logits, masks)
        metrics = segmentation_metrics(logits, masks)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        for name, value in metrics.items():
            metric_totals[name] += value * batch_size

    size = len(loader.dataset)
    averaged_metrics = {name: value / size for name, value in metric_totals.items()}
    averaged_metrics["loss"] = total_loss / size
    return averaged_metrics


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    return (
        f"{prefix}_loss={metrics['loss']:.4f} | "
        f"{prefix}_dice={metrics['dice']:.4f} | "
        f"{prefix}_iou={metrics['iou']:.4f} | "
        f"{prefix}_precision={metrics['precision']:.4f} | "
        f"{prefix}_recall={metrics['recall']:.4f} | "
        f"{prefix}_mae={metrics['mae']:.4f}"
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    size_rates = parse_size_rates(args.size_rates, args.multi_scale)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = build_splits(args)
    train_set = splits["train"]
    val_set = splits["val"]
    test_set = splits["test"]
    save_split_files(args.save_split_dir, splits)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        args.variant,
        use_lha=args.use_lha,
        use_egff=args.use_egff,
        use_dfc=args.use_dfc,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_scores = {
        "dice": -1.0,
        "iou": -1.0,
    }
    checkpoint_paths = {
        "dice": args.output_dir / "lightmedseg_best_dice.pt",
        "iou": args.output_dir / "lightmedseg_best_iou.pt",
    }
    print(f"Protocol: {args.protocol}")
    print(f"Device: {device}")
    print(f"Dataset: {len(train_set)} train / {len(val_set)} val / {len(test_set)} test")
    print(f"Split files: {args.save_split_dir}")
    print(f"Train augmentation: {'enabled' if args.train_augment else 'disabled'}")
    print(f"ImageNet normalization: enabled")
    print(f"Multi-scale training: {size_rates}")
    print(f"Modules: LHA={args.use_lha} | EGFF={args.use_egff} | DFC={args.use_dfc}")
    print("Saving best checkpoints by: val_dice and val_iou")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.edge_loss_weight,
            size_rates,
            use_edge_loss=args.use_egff,
        )
        val_metrics = validate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | {format_metrics('val', val_metrics)}"
        )

        for metric_name in ["dice", "iou"]:
            current_score = val_metrics[metric_name]
            if current_score > best_scores[metric_name]:
                best_scores[metric_name] = current_score
                checkpoint_path = checkpoint_paths[metric_name]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "variant": args.variant,
                        "image_size": args.image_size,
                        "train_augment": args.train_augment,
                        "normalize": "imagenet",
                        "size_rates": size_rates,
                        "use_lha": args.use_lha,
                        "use_egff": args.use_egff,
                        "use_dfc": args.use_dfc,
                        "protocol": args.protocol,
                        "seed": args.seed,
                        "best_metric": metric_name,
                        "best_score": best_scores[metric_name],
                        "val_metrics": val_metrics,
                    },
                    checkpoint_path,
                )
                print(f"Saved best {metric_name} checkpoint: {checkpoint_path}")

    for metric_name, checkpoint_path in checkpoint_paths.items():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        test_metrics = validate(model, test_loader, device, desc=f"test_best_{metric_name}")
        print(f"Best {metric_name} checkpoint test | {format_metrics('test', test_metrics)}")

        for split_name in ["test_kvasir", "test_cvc_clinicdb"]:
            if split_name not in splits:
                continue
            split_loader = DataLoader(
                splits[split_name],
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            split_metrics = validate(model, split_loader, device, desc=f"{split_name}_best_{metric_name}")
            print(f"{split_name} best_{metric_name} | {format_metrics(split_name, split_metrics)}")


if __name__ == "__main__":
    main()
