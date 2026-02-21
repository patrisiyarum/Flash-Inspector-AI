#!/usr/bin/env python3
"""
FlashInspector AI - Dataset Preparation Script
Merges multiple YOLOv8 datasets into unified detection and segmentation datasets
with consistent class IDs for training.

- Detection datasets -> merged_dataset/ (for YOLOv8 detect)
- Segmentation datasets (task: segment) -> merged_segmentation_dataset/ (for YOLOv8-seg)

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
MERGED_SEG_DIR = BASE_DIR / "merged_segmentation_dataset"

# Map variant class names to canonical unified names (case-insensitive match)
CLASS_ALIASES = {
    "fire extinguisher": "fire_extinguisher",
    "fire_extinguisher": "fire_extinguisher",
    "Fire_Extinguisher": "fire_extinguisher",
    "fire blanket": "fire_blanket",
    "fire_blanket": "fire_blanket",
    "Fire_Blanket": "fire_blanket",
    "smoke detector": "smoke_detector",
    "smoke_detector": "smoke_detector",
    "White_Domes": "smoke_detector",
    "manual call point": "manual_call_point",
    "manual_call_point": "manual_call_point",
    "Alarm_Activator": "manual_call_point",
    "emergency exit sign": "emergency_exit",
    "emergency_exit": "emergency_exit",
    "Fire_Exit": "emergency_exit",
    "Fire_Suppression_Signage": "fire_suppression_sign",
    "Flashing_Light_Orbs": "flashing_light_orb",
    "flashing_light_orb": "flashing_light_orb",
    "Sounders": "sounder",
}


def _canonical(name: str) -> str:
    """Return canonical class name, or original if no alias."""
    return CLASS_ALIASES.get(name, CLASS_ALIASES.get(name.lower(), name))


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
    """Build unified class list and per-dataset old->new ID mappings.
    Aligns similar classes (e.g. Fire_Extinguisher, fire extinguisher) to one canonical name.
    """
    unified_classes = []
    seen = set()
    for classes in datasets.values():
        for c in classes:
            canon = _canonical(c)
            if canon not in seen:
                seen.add(canon)
                unified_classes.append(canon)

    # Map old class IDs to new unified IDs (via canonical name)
    mappings = {}
    for name, classes in datasets.items():
        mappings[name] = {}
        for old_id, cls_name in enumerate(classes):
            canon = _canonical(cls_name)
            new_id = unified_classes.index(canon)
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


def _merge_task_datasets(
    datasets: dict[str, dict],
    output_dir: Path,
    use_aliases: bool = True,
) -> None:
    """Merge datasets into output_dir. use_aliases=False for segmentation (no class alignment)."""
    dataset_classes = {n: info["classes"] for n, info in datasets.items()}
    if use_aliases:
        unified_classes, class_mappings = build_class_mapping(dataset_classes)
    else:
        unified_classes = []
        seen = set()
        for classes in dataset_classes.values():
            for c in classes:
                if c not in seen:
                    seen.add(c)
                    unified_classes.append(c)
        class_mappings = {}
        for name, classes in dataset_classes.items():
            class_mappings[name] = {i: unified_classes.index(c) for i, c in enumerate(classes)}

    logger.info(f"Unified classes ({len(unified_classes)}): {unified_classes}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ["train", "valid", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    total_images = 0
    for name, info in datasets.items():
        dataset_path = Path(info["path"])
        if not dataset_path.exists():
            continue
        class_map = class_mappings[name]
        dataset_images = 0

        for split in ["train", "valid", "test"]:
            try_splits = [split] if split != "valid" else ["valid", "val"]
            img_dirs, lbl_dirs = [], []
            for s in try_splits:
                img_dirs = list(dataset_path.rglob(f"{s}/images"))
                lbl_dirs = list(dataset_path.rglob(f"{s}/labels"))
                if img_dirs:
                    break
            if not img_dirs:
                continue
            img_dir = img_dirs[0]
            lbl_dir = lbl_dirs[0] if lbl_dirs else None

            for img_file in img_dir.iterdir():
                if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                new_name = f"{name}_{img_file.name}"
                shutil.copy2(img_file, output_dir / split / "images" / new_name)
                if lbl_dir:
                    label_name = img_file.stem + ".txt"
                    label_file = lbl_dir / label_name
                    if label_file.exists():
                        remapped = remap_labels(label_file, class_map)
                        (output_dir / split / "labels" / f"{name}_{label_name}").write_text(
                            "\n".join(remapped) + "\n" if remapped else ""
                        )
                dataset_images += 1

        logger.info(f"  {name}: {dataset_images} images")
        total_images += dataset_images

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(unified_classes),
        "names": unified_classes,
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Merged to {output_dir} ({total_images} images, config: {yaml_path})")
    for split in ["train", "valid", "test"]:
        count = len(list((output_dir / split / "images").iterdir()))
        logger.info(f"  {split}: {count} images")


def merge_datasets():
    """Merge detection and segmentation datasets from combined_config.yaml."""
    combined_config = DATASET_DIR / "combined_config.yaml"
    if not combined_config.exists():
        logger.error("combined_config.yaml not found. Run download_datasets.py first.")
        return

    with open(combined_config) as f:
        config = yaml.safe_load(f)

    detect_datasets = {}
    seg_datasets = {}
    for name, info in config["datasets"].items():
        path_val = info.get("path")
        if not path_val:
            logger.warning(f"Dataset {name} has no 'path', skipping")
            continue
        dataset_path = Path(path_val)
        if not dataset_path.exists():
            logger.warning(f"Dataset path not found: {dataset_path}, skipping")
            continue
        if info.get("task") == "segment":
            seg_datasets[name] = info
        else:
            detect_datasets[name] = info

    # Merge detection datasets
    if detect_datasets:
        logger.info("Merging detection datasets -> merged_dataset")
        _merge_task_datasets(detect_datasets, MERGED_DIR, use_aliases=True)
    else:
        logger.info("No detection datasets to merge.")

    # Merge segmentation datasets (services 3, 4)
    if seg_datasets:
        logger.info("\nMerging segmentation datasets -> merged_segmentation_dataset")
        _merge_task_datasets(seg_datasets, MERGED_SEG_DIR, use_aliases=False)
        logger.info("\nSegmentation dataset ready. Train with: python train.py --task segment")
    else:
        logger.info("\nNo segmentation datasets to merge.")


if __name__ == "__main__":
    merge_datasets()
