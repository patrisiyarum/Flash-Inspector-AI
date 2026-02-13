#!/usr/bin/env python3
"""
FlashInspector AI - YOLO Training Script
Trains a YOLOv8 model on the merged fire safety dataset.

Usage:
    python train.py
    python train.py --epochs 50 --batch 16 --model yolov8s.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).parent
MERGED_DIR = BASE_DIR / "merged_dataset"


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on fire safety dataset")
    parser.add_argument("--model", default="yolov8s.pt", help="Base model (default: yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", default=None, help="Device: 0 for GPU, cpu for CPU (auto-detect)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    data_yaml = MERGED_DIR / "data.yaml"
    if not data_yaml.exists():
        print("ERROR: merged_dataset/data.yaml not found.")
        print("Run 'python prepare_dataset.py' first to merge datasets.")
        return

    model = YOLO(args.model)

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(BASE_DIR / "runs"),
        name="fire_safety",
        exist_ok=True,
        resume=args.resume,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Training params
        patience=20,
        save=True,
        save_period=10,
        val=True,
        plots=True,
    )

    print(f"\nTraining complete!")
    print(f"Best model: {BASE_DIR / 'runs' / 'fire_safety' / 'weights' / 'best.pt'}")
    print(f"Results: {BASE_DIR / 'runs' / 'fire_safety'}")


if __name__ == "__main__":
    main()
