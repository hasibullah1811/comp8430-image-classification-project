"""
Custom dataset and image transforms for fine-grained chocolate bar classification.

Expected directory layout (ImageFolder style):

    data_dir/
        train/
            dark_70_cocoa/
                img001.jpg
            sea_salt/
                ...
        val/
            dark_70_cocoa/
                ...
"""

import os
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ImageNet statistics — required because we start from an ImageNet-pretrained backbone.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool = True) -> transforms.Compose:
    """
    Build preprocessing / augmentation pipelines.

    Training uses light augmentation to improve generalisation without distorting
    fine-grained packaging details too aggressively.
    Validation and inference use deterministic resizing and normalisation only.
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class FineGrainedImageDataset(Dataset):
    """
    PyTorch Dataset that walks a root directory and treats each subfolder as one class.

    This mirrors torchvision.datasets.ImageFolder but is implemented explicitly so the
    assignment clearly demonstrates understanding of the Dataset API.
    """

    def __init__(self, root_dir: str, transform: Optional[Callable] = None) -> None:
        self.root_dir = root_dir
        self.transform = transform

        # Sorted class names keep label indices stable across train / val / inference.
        self.classes: List[str] = sorted(
            name
            for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))
        )
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.samples: List[Tuple[str, int]] = []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            label = self.class_to_idx[class_name]

            for filename in os.listdir(class_dir):
                if os.path.splitext(filename)[1].lower() in image_extensions:
                    image_path = os.path.join(class_dir, filename)
                    self.samples.append((image_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under '{root_dir}'. "
                "Check that each class has its own subfolder with image files."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]

        # Convert to RGB so grayscale or palette-mode images still work with the CNN.
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Create training and validation DataLoaders from a standard train/val split layout.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = FineGrainedImageDataset(train_dir, transform=get_transforms(train=True))
    val_dataset = FineGrainedImageDataset(val_dir, transform=get_transforms(train=False))

    # Sanity check: both splits should expose the same class vocabulary.
    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            "Train and validation class folders do not match. "
            f"Train: {train_dataset.classes}, Val: {val_dataset.classes}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.classes
