"""Train and run a CelebA attribute classifier for GFM controllability evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm import tqdm

from src.data import normalize
from src.attribute_evaluator import binary_auroc
from src.models import create_model_from_config
from src.methods import FlowMatching


HAIR_NAMES = ("black", "blond", "brown", "other")
HAIR_BINARY_NAMES = ("black_hair", "blond_hair", "brown_hair")
FIELD_NAMES = ("smiling", "bangs", "hair_color")
DEFAULT_CONDITIONS = {
    "unconditional": None,
    "smiling_present": [2, 0, 0],
    "smiling_absent": [1, 0, 0],
    "bangs_present": [0, 2, 0],
    "bangs_absent": [0, 1, 0],
    "hair_black": [0, 0, 1],
    "hair_blond": [0, 0, 2],
    "hair_brown": [0, 0, 3],
    "smiling_no_bangs_blond": [2, 1, 2],
}


def build_transform(image_size: int) -> transforms.Compose:
    """Create the shared image preprocessing transform for classifier training."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(normalize),
    ])


def derive_targets(item: Dict) -> Dict[str, torch.Tensor]:
    """
    Convert CelebA binary attributes to classifier targets.

    Args:
        item: Dataset row exposing CelebA attribute fields.

    Returns:
        Dictionary with binary smiling/bangs labels and a hair class. Hair class
        is -100 for conflicting labels so the loss ignores that field.
    """
    smiling = torch.tensor(int(item["Smiling"]), dtype=torch.long)
    bangs = torch.tensor(int(item["Bangs"]), dtype=torch.long)
    hair_binary = [
        int(item["Black_Hair"]),
        int(item["Blond_Hair"]),
        int(item["Brown_Hair"]),
    ]
    active = [
        idx
        for idx, name in enumerate(("Black_Hair", "Blond_Hair", "Brown_Hair"))
        if int(item[name]) == 1
    ]
    if len(active) == 1:
        hair = active[0]
    elif len(active) == 0:
        hair = 3
    else:
        hair = -100
    return {
        "smiling": smiling,
        "bangs": bangs,
        "black_hair": torch.tensor(hair_binary[0], dtype=torch.long),
        "blond_hair": torch.tensor(hair_binary[1], dtype=torch.long),
        "brown_hair": torch.tensor(hair_binary[2], dtype=torch.long),
        "hair": torch.tensor(hair, dtype=torch.long),
    }


class CelebAClassifierDataset(Dataset):
    """
    Dataset wrapper for training the frozen controllability evaluator.

    Args:
        root: HuggingFace saved dataset path or local CelebA split root.
        split: Split name to load.
        image_size: Image size used by the transform.
    """

    def __init__(self, root: str, split: str = "train", image_size: int = 64):
        self.root = root
        self.split = split
        self.transform = build_transform(image_size)
        self.data = self._load_rows()

    def _load_rows(self) -> List[Dict]:
        """Load rows from HuggingFace Arrow cache or local image/attribute files."""
        root_path = Path(self.root)
        if (root_path / "dataset_dict.json").exists() or (root_path / "dataset_info.json").exists():
            from datasets import load_from_disk

            dataset = load_from_disk(self.root)
            hf_split = "validation" if self.split == "valid" else self.split
            return list(dataset[hf_split])

        import csv

        split_dir = "validation" if self.split == "valid" else self.split
        split_root = root_path / split_dir
        attributes_path = split_root / "attributes.csv"
        images_dir = split_root / "images"
        with attributes_path.open("r", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["image"] = str(images_dir / row["image_id"])
        return rows

    def __len__(self) -> int:
        """Return the number of loaded CelebA examples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load one image and its classifier targets.

        Args:
            idx: Dataset index.

        Returns:
            Dictionary with image tensor and attribute targets.
        """
        item = self.data[idx]
        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        targets = derive_targets(item)
        return {
            "image": self.transform(image),
            **targets,
        }


class CelebAAttributeClassifier(nn.Module):
    """
    ResNet-18 evaluator for HW3 controllability fields.

    Args:
        pretrained: Whether to request ImageNet pretrained weights.
        hair_mode: ``binary`` uses CelebA's native hair-attribute heads;
            ``multiclass`` preserves the original 4-way baseline.
    """

    def __init__(self, pretrained: bool = False, hair_mode: str = "binary"):
        super().__init__()
        if hair_mode not in {"binary", "multiclass"}:
            raise ValueError(f"Unsupported hair_mode: {hair_mode}")
        self.hair_mode = hair_mode
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.smiling = nn.Linear(feature_dim, 2)
        self.bangs = nn.Linear(feature_dim, 2)
        if hair_mode == "binary":
            self.black_hair = nn.Linear(feature_dim, 2)
            self.blond_hair = nn.Linear(feature_dim, 2)
            self.brown_hair = nn.Linear(feature_dim, 2)
        else:
            self.hair = nn.Linear(feature_dim, 4)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict logits for the three controllability fields.

        Args:
            images: Normalized image tensor of shape ``(B, 3, H, W)``.

        Returns:
            Dictionary with controllability logits.
        """
        features = self.backbone(images)
        outputs = {
            "smiling": self.smiling(features),
            "bangs": self.bangs(features),
        }
        if self.hair_mode == "binary":
            outputs.update({
                "black_hair": self.black_hair(features),
                "blond_hair": self.blond_hair(features),
                "brown_hair": self.brown_hair(features),
            })
        else:
            outputs["hair"] = self.hair(features)
        return outputs


def deterministic_splits(dataset_size: int, seed: int) -> Tuple[List[int], List[int], List[int]]:
    """
    Create deterministic train/validation/test indices from the available subset.

    Args:
        dataset_size: Number of examples.
        seed: Random seed for permutation.

    Returns:
        Tuple of train, validation, and test index lists.
    """
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    train_end = int(dataset_size * 0.8)
    val_end = int(dataset_size * 0.9)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move tensor batch values to the target device."""
    return {key: value.to(device) for key, value in batch.items()}


def classifier_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    class_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Compute summed multi-task cross-entropy loss."""
    class_weights = class_weights or {}
    loss = (
        F.cross_entropy(outputs["smiling"], batch["smiling"], weight=class_weights.get("smiling"))
        + F.cross_entropy(outputs["bangs"], batch["bangs"], weight=class_weights.get("bangs"))
    )
    if "hair" in outputs:
        return loss + F.cross_entropy(
            outputs["hair"],
            batch["hair"],
            weight=class_weights.get("hair"),
            ignore_index=-100,
        )
    for hair_name in HAIR_BINARY_NAMES:
        loss = loss + F.cross_entropy(
            outputs[hair_name],
            batch[hair_name],
            weight=class_weights.get(hair_name),
        )
    return loss


def compute_class_weights(
    dataset: Dataset,
    train_indices: Sequence[int],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Compute inverse-frequency class weights on the classifier train split.

    Args:
        dataset: Full classifier dataset.
        train_indices: Indices used for classifier training.
        device: Target torch device.

    Returns:
        Weight tensors for each classifier head.
    """
    counts = {
        "smiling": torch.zeros(2, dtype=torch.float32),
        "bangs": torch.zeros(2, dtype=torch.float32),
        "hair": torch.zeros(4, dtype=torch.float32),
        "black_hair": torch.zeros(2, dtype=torch.float32),
        "blond_hair": torch.zeros(2, dtype=torch.float32),
        "brown_hair": torch.zeros(2, dtype=torch.float32),
    }
    for idx in train_indices:
        item = dataset.data[idx]
        targets = derive_targets(item)
        counts["smiling"][targets["smiling"].item()] += 1
        counts["bangs"][targets["bangs"].item()] += 1
        for hair_name in HAIR_BINARY_NAMES:
            counts[hair_name][targets[hair_name].item()] += 1
        hair_target = targets["hair"].item()
        if hair_target >= 0:
            counts["hair"][hair_target] += 1

    weights = {}
    for key, value in counts.items():
        safe_counts = value.clamp_min(1.0)
        # Normalized inverse frequency keeps the loss scale close to unweighted CE.
        weights[key] = (safe_counts.sum() / (safe_counts.numel() * safe_counts)).to(device)
    return weights


def binary_metrics(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    """
    Compute binary classification metrics with zero-safe denominators.

    Args:
        tp: True positives.
        tn: True negatives.
        fp: False positives.
        fn: False negatives.

    Returns:
        Accuracy, balanced accuracy, precision, recall, and F1.
    """
    total = tp + tn + fp + fn
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": (tp + tn) / max(1, total),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def outputs_to_hair_class(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Convert hair logits to an exclusive HW3 hair class.

    Binary hair heads are better evaluator targets, but HW3 conditions still use
    one hair-color slot. If no hair head predicts present, the sample is mapped
    to ``other``; otherwise the highest present probability selects the color.
    """
    if "hair" in outputs:
        return outputs["hair"].argmax(dim=1)

    present_logits = []
    present_flags = []
    for hair_name in HAIR_BINARY_NAMES:
        logits = outputs[hair_name]
        present_logits.append(logits.softmax(dim=1)[:, 1])
        present_flags.append(logits.argmax(dim=1) == 1)
    present_probs = torch.stack(present_logits, dim=1)
    any_present = torch.stack(present_flags, dim=1).any(dim=1)
    selected = present_probs.argmax(dim=1)
    other = torch.full_like(selected, 3)
    return torch.where(any_present, selected, other)


@torch.no_grad()
def evaluate_classifier(
    model: CelebAAttributeClassifier,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate classifier accuracy on a real-image split.

    Args:
        model: Attribute classifier.
        dataloader: Evaluation dataloader.
        device: Torch device.

    Returns:
        Dictionary of per-field and joint accuracies.
    """
    model.eval()
    binary_counts = {
        "smiling": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
        "bangs": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
    }
    if model.hair_mode == "binary":
        for hair_name in HAIR_BINARY_NAMES:
            binary_counts[hair_name] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    correct = {"hair": 0, "joint": 0}
    total = {"hair": 0, "joint": 0}
    score_batches = {name: [] for name in binary_counts}
    label_batches = {name: [] for name in binary_counts}
    for batch in dataloader:
        batch = batch_to_device(batch, device)
        outputs = model(batch["image"])
        smile_pred = outputs["smiling"].argmax(dim=1)
        bangs_pred = outputs["bangs"].argmax(dim=1)
        hair_pred = outputs_to_hair_class(outputs)
        hair_mask = batch["hair"] != -100

        for field_name, pred in (("smiling", smile_pred), ("bangs", bangs_pred)):
            target = batch[field_name]
            score_batches[field_name].append(
                (outputs[field_name][:, 1] - outputs[field_name][:, 0]).detach().cpu()
            )
            label_batches[field_name].append(target.detach().cpu())
            binary_counts[field_name]["tp"] += ((pred == 1) & (target == 1)).sum().item()
            binary_counts[field_name]["tn"] += ((pred == 0) & (target == 0)).sum().item()
            binary_counts[field_name]["fp"] += ((pred == 1) & (target == 0)).sum().item()
            binary_counts[field_name]["fn"] += ((pred == 0) & (target == 1)).sum().item()

        if model.hair_mode == "binary":
            for hair_name in HAIR_BINARY_NAMES:
                pred = outputs[hair_name].argmax(dim=1)
                target = batch[hair_name]
                score_batches[hair_name].append(
                    (outputs[hair_name][:, 1] - outputs[hair_name][:, 0]).detach().cpu()
                )
                label_batches[hair_name].append(target.detach().cpu())
                binary_counts[hair_name]["tp"] += ((pred == 1) & (target == 1)).sum().item()
                binary_counts[hair_name]["tn"] += ((pred == 0) & (target == 0)).sum().item()
                binary_counts[hair_name]["fp"] += ((pred == 1) & (target == 0)).sum().item()
                binary_counts[hair_name]["fn"] += ((pred == 0) & (target == 1)).sum().item()

        correct["hair"] += (hair_pred[hair_mask] == batch["hair"][hair_mask]).sum().item()
        total["hair"] += hair_mask.sum().item()

        joint_mask = hair_mask
        joint_ok = (
            (smile_pred == batch["smiling"])
            & (bangs_pred == batch["bangs"])
            & ((hair_pred == batch["hair"]) | ~hair_mask)
        )
        correct["joint"] += joint_ok[joint_mask].sum().item()
        total["joint"] += joint_mask.sum().item()

    metrics: Dict[str, float] = {}
    for field_name, counts in binary_counts.items():
        field_metrics = binary_metrics(**counts)
        field_metrics["auroc"] = binary_auroc(
            torch.cat(score_batches[field_name]),
            torch.cat(label_batches[field_name]),
        )
        for metric_name, value in field_metrics.items():
            metrics[f"{field_name}_{metric_name}"] = value
    metrics["hair_color_argmax_accuracy"] = correct["hair"] / max(1, total["hair"])
    metrics["joint_accuracy"] = correct["joint"] / max(1, total["joint"])
    # Preserve the legacy hair key used by the first baseline report.
    metrics["hair_accuracy"] = metrics["hair_color_argmax_accuracy"]
    return metrics


def train_classifier(args: argparse.Namespace) -> Dict:
    """
    Train and save the independent CelebA attribute classifier.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Training summary dictionary.
    """
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    split_seed = args.split_seed if args.split_seed is not None else args.seed
    model_seed = args.model_seed if args.model_seed is not None else args.seed
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    dataset = CelebAClassifierDataset(args.data_root, image_size=args.image_size)
    train_idx, val_idx, test_idx = deterministic_splits(len(dataset), split_seed)
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = CelebAAttributeClassifier(pretrained=args.pretrained, hair_mode=args.hair_mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    class_weights = compute_class_weights(dataset, train_idx, device) if args.class_weighted else None

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        batch_count = 0
        pbar = tqdm(train_loader, desc=f"classifier epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = classifier_loss(model(batch["image"]), batch, class_weights=class_weights)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            batch_count += 1
            pbar.set_postfix(loss=f"{loss_sum / batch_count:.4f}")
        val_metrics = evaluate_classifier(model, val_loader, device)
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = loss_sum / max(1, batch_count)
        print(json.dumps(val_metrics, indent=2, sort_keys=True))
        history.append(val_metrics)

    test_metrics = None if args.skip_test else evaluate_classifier(model, test_loader, device)
    checkpoint = {
        "model": model.state_dict(),
        "args": vars(args),
        "field_names": FIELD_NAMES,
        "hair_names": HAIR_NAMES,
        "hair_binary_names": HAIR_BINARY_NAMES,
        "hair_mode": args.hair_mode,
        "val_metrics": history[-1],
        "test_metrics": test_metrics,
        "split_seed": split_seed,
        "model_seed": model_seed,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    summary = {
        "checkpoint": str(output_path),
        "dataset_size": len(dataset),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "val_metrics": history[-1],
        "test_metrics": test_metrics,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def load_generator(checkpoint_path: str, device: torch.device) -> Tuple[FlowMatching, Tuple[int, int, int]]:
    """
    Load a trained Flow Matching generator checkpoint.

    Args:
        checkpoint_path: Path to generator checkpoint.
        device: Torch device.

    Returns:
        Configured method and image shape.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = create_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model"])
    if "ema" in checkpoint:
        from src.utils import EMA

        ema = EMA(model, decay=config["training"]["ema_decay"])
        ema.load_state_dict(checkpoint["ema"])
        ema.apply_shadow()
    method = FlowMatching.from_config(model, config, device)
    data_config = config["data"]
    image_shape = (data_config["channels"], data_config["image_size"], data_config["image_size"])
    return method, image_shape


def parse_condition_specs(specs: Optional[Sequence[str]]) -> Dict[str, Optional[List[int]]]:
    """
    Parse ``name:a,b,c`` condition specs.

    Args:
        specs: Optional CLI condition specifications.

    Returns:
        Mapping from condition name to condition ids or ``None`` for unconditional.
    """
    if not specs:
        return dict(DEFAULT_CONDITIONS)
    parsed: Dict[str, Optional[List[int]]] = {}
    for spec in specs:
        name, value = spec.split(":", 1)
        parsed[name] = None if value == "none" else [int(part) for part in value.split(",")]
    return parsed


def predictions_to_condition(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Convert classifier logits to HW3 condition ids.

    Args:
        outputs: Classifier logits.

    Returns:
        Long tensor with columns ``[smiling, bangs, hair_color]``.
    """
    smiling = outputs["smiling"].argmax(dim=1) + 1
    bangs = outputs["bangs"].argmax(dim=1) + 1
    hair = outputs_to_hair_class(outputs) + 1
    return torch.stack([smiling, bangs, hair], dim=1)


def field_distribution(predictions: torch.Tensor) -> Dict[str, Dict[str, float]]:
    """Summarize predicted condition distributions for generated samples."""
    distributions: Dict[str, Dict[str, float]] = {}
    for field_idx, field_name in enumerate(FIELD_NAMES):
        values = predictions[:, field_idx]
        max_value = 4 if field_name == "hair_color" else 2
        distributions[field_name] = {
            str(value): (values == value).float().mean().item()
            for value in range(1, max_value + 1)
        }
    return distributions


def condition_success(predictions: torch.Tensor, condition: Optional[List[int]]) -> Dict[str, float]:
    """
    Compute target and joint success for one requested condition.

    Args:
        predictions: Predicted HW3 condition ids.
        condition: Requested condition ids, with 0 meaning unspecified.

    Returns:
        Dictionary containing per-target and joint success metrics.
    """
    if condition is None:
        return {"num_specified_fields": 0}
    target = torch.tensor(condition, device=predictions.device)
    specified = target != 0
    metrics = {"num_specified_fields": int(specified.sum().item())}
    if specified.sum() == 0:
        return metrics
    matches = predictions[:, specified] == target[specified]
    for local_idx, field_idx in enumerate(torch.where(specified)[0].tolist()):
        metrics[f"{FIELD_NAMES[field_idx]}_success"] = matches[:, local_idx].float().mean().item()
    metrics["joint_success"] = matches.all(dim=1).float().mean().item()
    return metrics


@torch.no_grad()
def evaluate_controllability(args: argparse.Namespace) -> Dict:
    """
    Generate conditional samples and evaluate classifier-predicted controllability.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Result dictionary written to JSON.
    """
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    classifier_ckpt = torch.load(args.classifier_checkpoint, map_location=device)
    hair_mode = classifier_ckpt.get("hair_mode", classifier_ckpt.get("args", {}).get("hair_mode", "multiclass"))
    classifier = CelebAAttributeClassifier(pretrained=False, hair_mode=hair_mode).to(device)
    classifier.load_state_dict(classifier_ckpt["model"])
    classifier.eval()

    method, image_shape = load_generator(args.generator_checkpoint, device)
    condition_specs = parse_condition_specs(args.conditions)
    guidance_scales = [float(value) for value in args.guidance_scales]

    results = {
        "classifier_checkpoint": args.classifier_checkpoint,
        "generator_checkpoint": args.generator_checkpoint,
        "classifier_test_metrics": classifier_ckpt.get("test_metrics", {}),
        "num_samples_per_condition": args.num_samples,
        "num_steps": args.num_steps,
        "sampler": args.sampler,
        "results": [],
    }

    for guidance_scale in guidance_scales:
        for condition_name, condition in condition_specs.items():
            all_predictions = []
            remaining = args.num_samples
            pbar = tqdm(total=args.num_samples, desc=f"{condition_name} w={guidance_scale}")
            while remaining > 0:
                batch_size = min(args.batch_size, remaining)
                samples = method.sample(
                    batch_size=batch_size,
                    image_shape=image_shape,
                    num_steps=args.num_steps,
                    sampler=args.sampler,
                    condition=condition,
                    guidance_scale=None if condition is None else guidance_scale,
                )
                outputs = classifier(samples.to(device))
                all_predictions.append(predictions_to_condition(outputs).cpu())
                remaining -= batch_size
                pbar.update(batch_size)
            pbar.close()

            predictions = torch.cat(all_predictions, dim=0)
            entry = {
                "condition_name": condition_name,
                "condition": condition,
                "guidance_scale": guidance_scale,
                "field_distribution": field_distribution(predictions),
                **condition_success(predictions, condition),
            }
            results["results"].append(entry)
            print(json.dumps(entry, indent=2, sort_keys=True))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"Wrote controllability results to {output_path}")
    return results


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    train_parser = subparsers.add_parser("train-classifier")
    train_parser.add_argument("--data-root", default="./data/celeba-subset")
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--image-size", type=int, default=64)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--split-seed", type=int, default=None)
    train_parser.add_argument("--model-seed", type=int, default=None)
    train_parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Keep final test labels sealed until the final HW4 evaluation.",
    )
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--pretrained", action="store_true")
    train_parser.add_argument("--class-weighted", action="store_true")
    train_parser.add_argument("--hair-mode", default="binary", choices=["binary", "multiclass"])

    eval_parser = subparsers.add_parser("evaluate-controllability")
    eval_parser.add_argument("--classifier-checkpoint", required=True)
    eval_parser.add_argument("--generator-checkpoint", required=True)
    eval_parser.add_argument("--output", required=True)
    eval_parser.add_argument("--num-samples", type=int, default=256)
    eval_parser.add_argument("--batch-size", type=int, default=64)
    eval_parser.add_argument("--num-steps", type=int, default=50)
    eval_parser.add_argument("--sampler", default="heun", choices=["euler", "heun"])
    eval_parser.add_argument("--guidance-scales", nargs="+", default=["0", "1", "2", "3"])
    eval_parser.add_argument("--conditions", nargs="*", default=None)
    eval_parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    """Run the requested controllability command."""
    args = build_parser().parse_args()
    if args.action == "train-classifier":
        train_classifier(args)
    elif args.action == "evaluate-controllability":
        evaluate_controllability(args)
    else:
        raise ValueError(f"Unknown action: {args.action}")


if __name__ == "__main__":
    main()
