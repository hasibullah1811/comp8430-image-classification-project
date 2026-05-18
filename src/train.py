"""
Training script for fine-grained object classification.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam

from dataset import get_dataloaders
from model import create_mobilenet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune MobileNetV3 on Lindt Excellence chocolate bar images."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root folder containing 'train/' and 'val/' subdirectories.",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--num_classes",
        type=int,
        default=20,
        help="Number of output classes (overridden if inferred from dataset).",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["small", "large"],
        help="Use mobilenet_v3_small or mobilenet_v3_large.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="best_mobilenet.pth",
        help="Where to save the best validation checkpoint.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader worker processes.",
    )
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """
    Shared routine for one training or validation pass.

    Returns average loss and classification accuracy for the epoch split.
    """
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        # Disable gradient tracking during validation for speed and memory savings.
        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main() -> None:
    args = parse_args()

    # Check for GPU, otherwise default to CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    train_loader, val_loader, class_names = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    num_classes = len(class_names)
    if args.num_classes != num_classes:
        print(
            f"Note: --num_classes={args.num_classes} but dataset has "
            f"{num_classes} classes; using dataset value."
        )

    model = create_mobilenet(
        num_classes=num_classes,
        pretrained=True,
        freeze_features=True,
        model_size=args.model_size,
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Only parameters with requires_grad=True are updated (the new classifier head).
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_val_acc = 0.0

    print(f"Classes ({num_classes}): {class_names}")
    print("Starting training...\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        elapsed = time.time() - epoch_start
        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Time: {elapsed:.1f}s"
        )

        # Persist the checkpoint with the highest validation accuracy so far.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "num_classes": num_classes,
                "model_size": args.model_size,
            }
            torch.save(checkpoint, args.output_path)
            print(f"  -> New best model saved to {args.output_path} (val acc: {val_acc:.2f}%)")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
