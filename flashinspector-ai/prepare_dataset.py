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
MERGED_EQUIP_DIR = BASE_DIR / "merged_equipment_dataset"
MERGED_VIOL_DIR = BASE_DIR / "merged_violation_dataset"

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
    "yellow tag": "yellow_tag",
    "yellow_tag": "yellow_tag",
    "red tag": "red_tag",
    "red_tag": "red_tag",
    "white tag": "white_tag",
    "white_tag": "white_tag",
    "Emergency-Light": "emergency_light",
    "emergency-light": "emergency_light",
    "emergency_light": "emergency_light",
    "Alarm": "alarm",
    "Alarm(Bell)": "alarm_bell",
    "Alarm(Lever)": "alarm_lever",
    "Alarm (Lever)": "alarm_lever",
    "Fire-Extinguisher": "fire_extinguisher",
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


MAX_IMAGES_PER_DATASET = 1500

OVERSAMPLE_DATASETS = {
    "find_empty_mounts": 3,
    "extinguisher_missing": 3,
}


def _collect_split_files(dataset_path: Path, split: str):
    """Collect (img_file, lbl_dir) pairs for a given split."""
    try_splits = [split] if split != "valid" else ["valid", "val"]
    for s in try_splits:
        img_dirs = list(dataset_path.rglob(f"{s}/images"))
        lbl_dirs = list(dataset_path.rglob(f"{s}/labels"))
        if img_dirs:
            img_dir = img_dirs[0]
            lbl_dir = lbl_dirs[0] if lbl_dirs else None
            files = [
                f for f in sorted(img_dir.iterdir())
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ]
            return files, lbl_dir
    return [], None


def _merge_task_datasets(
    datasets: dict[str, dict],
    output_dir: Path,
    use_aliases: bool = True,
) -> None:
    """Merge datasets into output_dir with balancing.

    Caps each dataset at MAX_IMAGES_PER_DATASET and oversamples
    rare datasets listed in OVERSAMPLE_DATASETS.
    """
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
        oversample = OVERSAMPLE_DATASETS.get(name, 1)

        for split in ["train", "valid", "test"]:
            files, lbl_dir = _collect_split_files(dataset_path, split)
            if not files:
                continue

            cap = MAX_IMAGES_PER_DATASET if split == "train" else len(files)
            files_to_use = files[:cap]

            for copy_idx in range(oversample if split == "train" else 1):
                for img_file in files_to_use:
                    suffix = f"_dup{copy_idx}" if copy_idx > 0 else ""
                    new_name = f"{name}{suffix}_{img_file.name}"
                    shutil.copy2(img_file, output_dir / split / "images" / new_name)
                    if lbl_dir:
                        label_name = img_file.stem + ".txt"
                        label_file = lbl_dir / label_name
                        if label_file.exists():
                            remapped = remap_labels(label_file, class_map)
                            (output_dir / split / "labels" / f"{name}{suffix}_{label_name}").write_text(
                                "\n".join(remapped) + "\n" if remapped else ""
                            )
                    dataset_images += 1

        extra = f" (oversampled {oversample}x)" if oversample > 1 else ""
        if len(files) > MAX_IMAGES_PER_DATASET:
            extra += f" (capped from {len(files)} to {MAX_IMAGES_PER_DATASET} train images)"
        logger.info(f"  {name}: {dataset_images} images{extra}")
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


# Classes used for the split-model pipeline
EQUIPMENT_CANONICAL = {
    "fire_extinguisher", "fire_blanket", "smoke_detector", "manual_call_point",
    "emergency_exit", "fire_suppression_sign", "flashing_light_orb", "sounder",
    "notification_appliance", "pull_station", "fire_alarm_panel", "emergency_light",
    "alarm", "alarm_bell", "alarm_lever", "fire", "smoke",
}

VIOLATION_CANONICAL = {
    "empty_mount", "extinguisher_cabinet_empty", "non_compliant_tag",
    "yellow_tag", "red_tag", "white_tag", "exit_sign_dark",
    "smoke_detector_missing", "blocked_exit",
}


def _filter_labels_by_classes(label_file: Path, allowed_ids: set[int]) -> list[str]:
    """Keep only label lines whose class ID is in allowed_ids."""
    lines = []
    for line in label_file.read_text().strip().splitlines():
        parts = line.strip().split()
        if parts and int(parts[0]) in allowed_ids:
            lines.append(line.strip())
    return lines


def build_split_datasets():
    """Build separate equipment-only and violation-only datasets from merged_dataset.

    Reads the already-merged data.yaml to get the unified class list,
    then filters labels to produce two focused datasets.
    """
    merged_yaml = MERGED_DIR / "data.yaml"
    if not merged_yaml.exists():
        logger.info("merged_dataset not found, skipping split-model datasets.")
        return

    with open(merged_yaml) as f:
        merged_cfg = yaml.safe_load(f)

    all_names = merged_cfg.get("names", [])
    if isinstance(all_names, dict):
        all_names = [all_names[k] for k in sorted(all_names.keys())]

    equip_ids = {i for i, n in enumerate(all_names) if n in EQUIPMENT_CANONICAL}
    viol_ids = {i for i, n in enumerate(all_names) if n in VIOLATION_CANONICAL}

    if not equip_ids and not viol_ids:
        logger.info("No equipment or violation classes found in merged_dataset — skipping split.")
        return

    for target_dir, keep_ids, label in [
        (MERGED_EQUIP_DIR, equip_ids, "equipment"),
        (MERGED_VIOL_DIR, viol_ids, "violation"),
    ]:
        if not keep_ids:
            continue

        if target_dir.exists():
            shutil.rmtree(target_dir)

        # Remap kept IDs to contiguous 0..N
        kept_names = [all_names[i] for i in sorted(keep_ids)]
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted(keep_ids))}

        img_count = 0
        for split in ["train", "valid", "test"]:
            src_img = MERGED_DIR / split / "images"
            src_lbl = MERGED_DIR / split / "labels"
            dst_img = target_dir / split / "images"
            dst_lbl = target_dir / split / "labels"
            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lbl.mkdir(parents=True, exist_ok=True)

            if not src_img.exists():
                continue

            for img_file in sorted(src_img.iterdir()):
                lbl_file = src_lbl / (img_file.stem + ".txt")
                if not lbl_file.exists():
                    continue

                # Only include images that have at least one label in the target set
                kept_lines = _filter_labels_by_classes(lbl_file, keep_ids)
                if not kept_lines:
                    continue

                # Remap class IDs
                remapped = []
                for line in kept_lines:
                    parts = line.split()
                    parts[0] = str(old_to_new[int(parts[0])])
                    remapped.append(" ".join(parts))

                shutil.copy2(img_file, dst_img / img_file.name)
                (dst_lbl / lbl_file.name).write_text("\n".join(remapped) + "\n")
                img_count += 1

        data_yaml = {
            "path": str(target_dir.resolve()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(kept_names),
            "names": kept_names,
        }
        with open(target_dir / "data.yaml", "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

        logger.info(f"\n  Split-model {label} dataset: {target_dir}")
        logger.info(f"    Classes ({len(kept_names)}): {kept_names}")
        logger.info(f"    Images: {img_count}")


if __name__ == "__main__":
    merge_datasets()
    build_split_datasets()
