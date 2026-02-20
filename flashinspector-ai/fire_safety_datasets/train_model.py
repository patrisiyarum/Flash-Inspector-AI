#!/usr/bin/env python3
"""
FlashInspector AI - YOLOv8 Training Script
Trains fire safety detection models using Ultralytics YOLOv8.

Usage:
    python fire_safety_datasets/train_model.py
    python fire_safety_datasets/train_model.py --dataset firenet --size medium
    python fire_safety_datasets/train_model.py --dataset fire_extinguisher --size nano --epochs 50
    python fire_safety_datasets/train_model.py --export-only --weights fire_safety_models/fire_extinguisher_nano/weights/best.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "fire_safety_datasets"
MODEL_DIR = BASE_DIR / "fire_safety_models"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(BASE_DIR / "training.log"),
    ],
)
logger = logging.getLogger(__name__)

# Model size presets
MODELS = {
    "nano": "yolov8n.pt",   # ~3.2M params - fast, good for mobile
    "small": "yolov8s.pt",  # ~11.2M params - balanced
    "medium": "yolov8m.pt", # ~25.9M params - accurate
    "large": "yolov8l.pt",  # ~43.7M params - most accurate
}

# Default training hyperparameters
DEFAULT_CONFIG = {
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "patience": 20,
    "workers": 8,
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "mosaic": 1.0,
    "close_mosaic": 10,
}

# Available datasets (must match download_datasets.py names)
AVAILABLE_DATASETS = [
    "fire_extinguisher",
    "emergency_exit",
    "firenet",
]


def find_data_yaml(dataset_name: str) -> Path:
    """Locate the data.yaml for a given dataset."""
    dataset_path = DATASET_DIR / dataset_name
    if not dataset_path.exists():
        logger.error(
            f"Dataset '{dataset_name}' not found at {dataset_path}. "
            "Run download_datasets.py first."
        )
        sys.exit(1)

    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        logger.error(f"data.yaml not found in {dataset_path}")
        sys.exit(1)

    return data_yaml


def train(dataset_name: str, model_size: str, config_overrides: dict | None = None) -> Path:
    """Train a YOLOv8 model on the specified dataset.

    Returns the path to the best weights file.
    """
    data_yaml = find_data_yaml(dataset_name)
    pretrained = MODELS[model_size]
    run_name = f"{dataset_name}_{model_size}"
    output_dir = MODEL_DIR / run_name

    logger.info(f"Training: {run_name}")
    logger.info(f"  Dataset:    {data_yaml}")
    logger.info(f"  Pretrained: {pretrained}")
    logger.info(f"  Output:     {output_dir}")

    # Merge configs
    cfg = {**DEFAULT_CONFIG}
    if config_overrides:
        cfg.update(config_overrides)

    # Load pretrained model
    model = YOLO(pretrained)

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        patience=cfg["patience"],
        workers=cfg["workers"],
        optimizer=cfg["optimizer"],
        lr0=cfg["lr0"],
        lrf=cfg["lrf"],
        mosaic=cfg["mosaic"],
        close_mosaic=cfg["close_mosaic"],
        project=str(MODEL_DIR),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )

    # Print metrics
    best_weights = output_dir / "weights" / "best.pt"
    if best_weights.exists():
        logger.info(f"\nTraining complete! Best weights: {best_weights}")
        _print_metrics(results)
    else:
        logger.warning("Training finished but best.pt not found.")

    return best_weights


def _print_metrics(results):
    """Print key training metrics."""
    try:
        metrics = results.results_dict
        logger.info("\n--- Training Metrics ---")
        for key in ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
            if key in metrics:
                logger.info(f"  {key}: {metrics[key]:.4f}")
    except Exception:
        logger.info("(Metrics not available in results object)")


def export_model(weights_path: Path):
    """Export a trained model to TFLite, CoreML, and ONNX."""
    if not weights_path.exists():
        logger.error(f"Weights file not found: {weights_path}")
        sys.exit(1)

    model = YOLO(str(weights_path))
    export_dir = weights_path.parent

    formats = {
        "onnx": "ONNX (cross-platform)",
        "tflite": "TFLite (Android)",
        "coreml": "CoreML (iOS)",
    }

    for fmt, desc in formats.items():
        logger.info(f"Exporting to {desc}...")
        try:
            model.export(format=fmt)
            logger.info(f"  {desc} export successful")
        except Exception as e:
            logger.warning(f"  {desc} export failed: {e}")

    logger.info(f"Exported models saved alongside weights in {export_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 fire safety detection models")
    parser.add_argument(
        "--dataset",
        choices=AVAILABLE_DATASETS,
        default="fire_extinguisher",
        help="Dataset to train on (default: fire_extinguisher)",
    )
    parser.add_argument(
        "--size",
        choices=list(MODELS.keys()),
        default="nano",
        help="Model size (default: nano)",
    )
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch", type=int, help="Override batch size")
    parser.add_argument("--imgsz", type=int, help="Override image size")
    parser.add_argument("--no-export", action="store_true", help="Skip model export after training")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export an existing model (use with --weights)",
    )
    parser.add_argument("--weights", type=str, help="Path to weights file (for --export-only)")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file with training hyperparameters",
    )
    args = parser.parse_args()

    # Export-only mode
    if args.export_only:
        if not args.weights:
            logger.error("--export-only requires --weights path")
            sys.exit(1)
        export_model(Path(args.weights))
        return

    # Load optional config file
    overrides = {}
    if args.config:
        with open(args.config) as f:
            overrides = yaml.safe_load(f) or {}
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.batch:
        overrides["batch"] = args.batch
    if args.imgsz:
        overrides["imgsz"] = args.imgsz

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Train
    best_weights = train(args.dataset, args.size, overrides)

    # Export
    if not args.no_export and best_weights.exists():
        export_model(best_weights)

    logger.info("\nAll done!")


if __name__ == "__main__":
    main()
