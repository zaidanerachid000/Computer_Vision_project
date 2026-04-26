import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.loader import load_features_and_labels
from src.data.preprocess import reshape_images_chw
from src.models import SimpleCIFAR100CNN


class Cifar100AugmentedDataset(Dataset):
    """CHW float tensor in [0, 1]; optional flip + crop for train-time augmentation."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool) -> None:
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)
        self.augment = augment

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        x = self.images[idx].clone()
        y = self.labels[idx]
        if self.augment:
            if torch.rand(()) > 0.5:
                x = torch.flip(x, dims=(2,))
            pad = 4
            x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
            i = int(torch.randint(0, 2 * pad + 1, (1,)).item())
            j = int(torch.randint(0, 2 * pad + 1, (1,)).item())
            x = x[:, i : i + 32, j : j + 32]
        return x, y


def resolve_dataset_path(dataset_path: Path) -> Path:
    """
    Resolve dataset path robustly across different working directories.
    """
    if dataset_path.is_absolute() and dataset_path.exists():
        return dataset_path

    project_root = Path(__file__).resolve().parents[2]
    repo_root = project_root.parent
    candidate_paths = [
        dataset_path,
        Path.cwd() / dataset_path,
        project_root / dataset_path,
        repo_root / dataset_path,
        repo_root / "CV_dataset" / "train",
        repo_root / "CV_dataset" / "train" / "train",
    ]

    for candidate in candidate_paths:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    checked = "\n".join(f"- {p.resolve()}" for p in candidate_paths)
    raise FileNotFoundError(
        "Could not find CIFAR-100 train file. Checked:\n"
        f"{checked}\n\n"
        "Pass --dataset with the full path to your train file."
    )


def build_dataloaders(
    dataset_path: Path,
    batch_size: int,
    val_size: float,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    X, y = load_features_and_labels(dataset_path)
    X = reshape_images_chw(X).astype(np.float32) / 255.0

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=y
    )

    train_ds = Cifar100AugmentedDataset(X_train, y_train, augment=True)
    val_ds = Cifar100AugmentedDataset(X_val, y_val, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        total_count += xb.size(0)

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        total_count += xb.size(0)

    return total_loss / total_count, total_correct / total_count


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple CNN on CIFAR-100.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("../CV_dataset/train"),
        help="Path to CIFAR-100 train pickle file.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="AdamW weight decay.")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("models/simple_cnn_cifar100.pt"),
        help="Where to save the best model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_path = resolve_dataset_path(args.dataset)
    print(f"Using dataset: {dataset_path}")

    train_loader, val_loader = build_dataloaders(
        dataset_path=dataset_path,
        batch_size=args.batch_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    model = SimpleCIFAR100CNN(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{args.epochs} | lr={lr:.2e} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)

    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Saved best model to: {args.save_path}")


if __name__ == "__main__":
    main()

