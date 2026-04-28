#!/usr/bin/env python3
"""
Convert a COCO-format dataset export to YOLOv8 format for training.

Reads _annotations.coco.json and images from a COCO export directory,
produces a YOLOv8-compatible dataset with train/valid/test splits.

Usage:
    python convert_coco_to_yolo.py path/to/coco_export/
    python convert_coco_to_yolo.py path/to/coco_export/ --output fire_safety_datasets/coco_custom
    python convert_coco_to_yolo.py path/to/coco_export/ --split-ratio 0.8 0.1 0.1
"""

import argparse
import json
import logging
import random
import shutil
from pathlib import Path
from collections import defaultdict

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent


def load_coco_annotations(coco_dir: Path) -> dict:
    candidates = list(coco_dir.glob("*annotations*.coco.json")) + list(coco_dir.glob("*annotations*.json"))
    if not candidates:
        candidates = list(coco_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No COCO annotation JSON found in {coco_dir}")
    ann_file = candidates[0]
    logger.info(f"Loading annotations from {ann_file}")
    with open(ann_file) as f:
        return json.load(f)


def coco_to_yolo_bbox(ann: dict, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x, y, w, h = ann["bbox"]
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, nw)),
        max(0.0, min(1.0, nh)),
    )


def convert(coco_dir: Path, output_dir: Path, split_ratio: tuple[float, float, float]):
    coco = load_coco_annotations(coco_dir)

    cat_id_to_idx = {}
    class_names = []
    for cat in sorted(coco["categories"], key=lambda c: c["id"]):
        cat_id_to_idx[cat["id"]] = len(class_names)
        class_names.append(cat["name"])
    logger.info(f"Classes ({len(class_names)}): {class_names}")

    img_lookup = {img["id"]: img for img in coco["images"]}
    ann_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    image_ids = sorted(img_lookup.keys())
    random.seed(42)
    random.shuffle(image_ids)

    n = len(image_ids)
    n_train = int(n * split_ratio[0])
    n_valid = int(n * split_ratio[1])
    splits = {
        "train": image_ids[:n_train],
        "valid": image_ids[n_train:n_train + n_valid],
        "test": image_ids[n_train + n_valid:],
    }

    if output_dir.exists():
        shutil.rmtree(output_dir)

    total = 0
    empty = 0
    class_counts = defaultdict(int)

    for split_name, ids in splits.items():
        img_out = output_dir / split_name / "images"
        lbl_out = output_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_id in ids:
            img_info = img_lookup[img_id]
            img_file = img_info["file_name"]
            src = coco_dir / img_file
            if not src.exists():
                src_candidates = list(coco_dir.glob(f"**/{img_file}"))
                if not src_candidates:
                    continue
                src = src_candidates[0]

            shutil.copy2(src, img_out / img_file)

            anns = ann_by_image.get(img_id, [])
            lines = []
            for ann in anns:
                if ann.get("iscrowd", 0):
                    continue
                bbox = ann.get("bbox")
                if not bbox or bbox[2] <= 0 or bbox[3] <= 0:
                    continue
                cls_idx = cat_id_to_idx[ann["category_id"]]
                cx, cy, nw, nh = coco_to_yolo_bbox(ann, img_info["width"], img_info["height"])
                lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                class_counts[class_names[cls_idx]] += 1

            label_path = lbl_out / (Path(img_file).stem + ".txt")
            label_path.write_text("\n".join(lines) + "\n" if lines else "")
            total += 1
            if not lines:
                empty += 1

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    with open(output_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    logger.info(f"\nConverted {total} images ({empty} with no annotations)")
    for split_name, ids in splits.items():
        actual = len(list((output_dir / split_name / "images").iterdir()))
        logger.info(f"  {split_name}: {actual} images")
    logger.info(f"\nClass distribution:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {cls}: {count} annotations")
    logger.info(f"\nDataset saved to {output_dir}")
    logger.info(f"Config: {output_dir / 'data.yaml'}")


def main():
    parser = argparse.ArgumentParser(description="Convert COCO dataset to YOLOv8 format")
    parser.add_argument("coco_dir", type=Path, help="Path to COCO export directory")
    parser.add_argument("--output", type=Path, default=None,
        help="Output directory (default: fire_safety_datasets/coco_custom)")
    parser.add_argument("--split-ratio", nargs=3, type=float, default=[0.8, 0.1, 0.1],
        help="Train/valid/test split ratio (default: 0.8 0.1 0.1)")
    args = parser.parse_args()

    output = args.output or (BASE_DIR / "fire_safety_datasets" / "coco_custom")
    convert(args.coco_dir, output, tuple(args.split_ratio))


if __name__ == "__main__":
    main()
