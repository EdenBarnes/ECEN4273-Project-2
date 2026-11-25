
import argparse
import os
import shutil
from pathlib import Path
from ultralytics import YOLO


def train_model(
    weights: str,
    data_yaml: str,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    freeze: int = 0,
    lr0: float = 0.01,
    augment: bool = False,
    project: str = "runs/train",
    name: str = "exp",
    device: str = "",
    export_weights: str | None = None,
) -> Path:
    """Train a YOLO model with configurable transfer learning and augmentation.

    Args:
        weights: Pretrained weights file or model config (e.g., 'yolo11s.pt').
        data_yaml: Path to the dataset YAML file defining splits and classes.
        epochs: Number of training epochs.
        imgsz: Input image resolution (pixels).
        batch: Batch size (-1 for auto-batch).
        freeze: Number of initial layers to freeze (0 = full fine-tuning,
            10 = freeze backbone for transfer learning).
        lr0: Initial learning rate.
        augment: Whether to enable stronger data augmentation (HSV jitter,
            rotation, mosaic, mixup).
        project: Parent directory for saving training runs.
        name: Subdirectory name for this specific experiment.
        device: Compute device ('0' for GPU 0, 'cpu', or '' for auto-select).
        export_weights: Optional destination path to copy the best weights
            (e.g., 'weights/best.pt').

    Returns:
        Path to the best trained weights file (`best.pt`).
    """
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    model = YOLO(weights)

    # Configure augmentation parameters
    aug_kwargs = (
        {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 10.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.15,
        }
        if augment
        else {
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "fliplr": 0.5,
            "mosaic": 0.0,
            "mixup": 0.0,
        }
    )

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        freeze=freeze,
        lr0=lr0,
        project=project,
        name=name,
        device=device if device else None,
        exist_ok=True,
        **aug_kwargs,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights saved to: {best_weights}")

    if export_weights and best_weights.exists():
        dest = Path(export_weights)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, dest)
        print(f"Copied best weights to: {dest}")

    return best_weights


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train custom YOLO object detection models."
    )
    parser.add_argument(
        "--weights", type=str, default="yolo11s.pt",
        help="Pretrained weights or model architecture (default: yolo11s.pt).",
    )
    parser.add_argument(
        "--data", type=str, default="data.yaml",
        help="Path to data.yaml (default: data.yaml).",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size in pixels (default: 640).",
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size (default: 16).",
    )
    parser.add_argument(
        "--freeze", type=int, default=0,
        help="Layers to freeze: 0 for full fine-tuning, 10 for frozen backbone.",
    )
    parser.add_argument(
        "--lr0", type=float, default=0.01,
        help="Initial learning rate (default: 0.01).",
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Enable data augmentation (HSV jitter, rotation, mosaic, mixup).",
    )
    parser.add_argument(
        "--name", type=str, default="exp",
        help="Experiment name under runs/train/ (default: exp).",
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="Device to train on, e.g. '0' or 'cpu' (default: auto).",
    )
    parser.add_argument(
        "--export-weights", type=str, default=None,
        help="Copy final best.pt to this path (e.g., weights/best.pt).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        weights=args.weights,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        lr0=args.lr0,
        augment=args.augment,
        name=args.name,
        device=args.device,
        export_weights=args.export_weights,
    )