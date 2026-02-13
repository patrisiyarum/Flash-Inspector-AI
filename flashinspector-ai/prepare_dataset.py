#!/usr/bin/env python3
"""
FlashInspector AI - Dataset Preparation Script
Merges multiple Roboflow YOLOv8 datasets into a single unified dataset
with consistent class IDs for training.

Usage:
    python prepare_dataset.py
"""

import shutil
import logging
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "fire_safety_datasets"
MERGED_DIR = BASE_DIR / "merged_dataset"


def load_dataset_classes(dataset_path: Path) -> list[str]:
    """Load class names from a dataset's data.yaml."""
    yaml_files = list(dataset_path.rglob("data.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No data.yaml found in {dataset_path}")
    with open(yaml_files[0]) as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    return names


def build_class_mapping(datasets: dict[str, list[str]]) -> tuple[list[str], dict[str, dict[int, int]]]:
    """Build unified class list and per-dataset old->new ID mappings."""
    unified_classes = []
    seen = set()
    for classes in datasets.values():
        for c in classes:
            if c not in seen:
                seen.add(c)
                unified_classes.append(c)

    # Map old class IDs to new unified IDs
    mappings = {}
    for name, classes in datasets.items():
        mappings[name] = {}
        for old_id, cls_name in enumerate(classes):
            new_id = unified_classes.index(cls_name)
            mappings[name][old_id] = new_id

    return unified_classes, mappings


def remap_labels(label_file: Path, class_map: dict[int, int]) -> list[str]:
    """Remap class IDs in a YOLO label file."""
    lines = []
    for line in label_file.read_text().strip().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        old_id = int(parts[0])
        if old_id in class_map:
            parts[0] = str(class_map[old_id])
            lines.append(" ".join(parts))
    return lines


def merge_datasets():
    """Merge all downloaded datasets into a unified dataset."""
    combined_config = DATASET_DIR / "combined_config.yaml"
    if not combined_config.exists():
        logger.error("combined_config.yaml not found. Run download_datasets.py first.")
        return

    with open(combined_config) as f:
        config = yaml.safe_load(f)

    # Load classes per dataset
    dataset_classes = {}
    for name, info in config["datasets"].items():
        dataset_path = Path(info["path"])
        if not dataset_path.exists():
            logger.warning(f"Dataset path not found: {dataset_path}, skipping")
            continue
        dataset_classes[name] = info["classes"]

    unified_classes, class_mappings = build_class_mapping(dataset_classes)
    logger.info(f"Unified classes ({len(unified_classes)}): {unified_classes}")

    # Create merged directory structure
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)

    for split in ["train", "valid", "test"]:
        (MERGED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy and remap each dataset
    total_images = 0
    for name, info in config["datasets"].items():
        dataset_path = Path(info["path"])
        if not dataset_path.exists():
            continue

        class_map = class_mappings[name]
        dataset_images = 0

        for split in ["train", "valid", "test"]:
            # Find images and labels directories
            img_dirs = list(dataset_path.rglob(f"{split}/images"))
            lbl_dirs = list(dataset_path.rglob(f"{split}/labels"))

            if not img_dirs:
                continue

            img_dir = img_dirs[0]
            lbl_dir = lbl_dirs[0] if lbl_dirs else None

            for img_file in img_dir.iterdir():
                if not img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue

                # Prefix filename with dataset name to avoid collisions
                new_name = f"{name}_{img_file.name}"
                shutil.copy2(img_file, MERGED_DIR / split / "images" / new_name)

                # Copy and remap label
                if lbl_dir:
                    label_name = img_file.stem + ".txt"
                    label_file = lbl_dir / label_name
                    if label_file.exists():
                        remapped = remap_labels(label_file, class_map)
                        (MERGED_DIR / split / "labels" / f"{name}_{label_name}").write_text(
                            "\n".join(remapped) + "\n" if remapped else ""
                        )

                dataset_images += 1

        logger.info(f"  {name}: {dataset_images} images merged")
        total_images += dataset_images

    # Write unified data.yaml
    data_yaml = {
        "path": str(MERGED_DIR.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(unified_classes),
        "names": unified_classes,
    }
    yaml_path = MERGED_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    logger.info(f"\nMerged dataset created at {MERGED_DIR}")
    logger.info(f"Total images: {total_images}")
    logger.info(f"Config: {yaml_path}")

    # Print split counts
    for split in ["train", "valid", "test"]:
        count = len(list((MERGED_DIR / split / "images").iterdir()))
        logger.info(f"  {split}: {count} images")


if __name__ == "__main__":
    merge_datasets()
