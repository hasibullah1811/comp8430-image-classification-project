"""
Single-image inference for deployment (e.g. robot camera pipeline).

Example:
    python inference.py --image_path ../sample.jpg --checkpoint best_mobilenet.pth
"""

import argparse

import torch
from PIL import Image

from dataset import get_transforms
from model import create_mobilenet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on one chocolate bar image."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_mobilenet.pth",
        help="Path to the trained model checkpoint.",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    """Restore architecture and weights saved during training."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]
    model_size = checkpoint.get("model_size", "small")

    model = create_mobilenet(
        num_classes=num_classes,
        pretrained=False,
        freeze_features=False,
        model_size=model_size,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, class_names


def predict(image_path: str, checkpoint_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, class_names = load_model(checkpoint_path, device)
    transform = get_transforms(train=False)

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    predicted_class = class_names[predicted_idx.item()]
    confidence_score = confidence.item()

    # Simple, parseable output for downstream robot integration.
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence_score:.4f}")


def main() -> None:
    args = parse_args()
    predict(args.image_path, args.checkpoint)


if __name__ == "__main__":
    main()
