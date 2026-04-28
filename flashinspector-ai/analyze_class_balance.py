#!/usr/bin/env python3
"""
Analyze class distribution across training datasets and recommend improvements.

Reads merged_dataset labels (or a class_balance.csv) and reports:
- Per-class annotation counts and percentages
- Imbalance ratio (most common vs. least common)
- Recommendations for oversampling, augmentation, and data collection

Usage:
    python analyze_class_balance.py                          # Analyze merged_dataset
    python analyze_class_balance.py --csv class_balance.csv  # Analyze from CSV
    python analyze_class_balance.py --dataset fire_safety_datasets/coco_custom
"""

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent


def count_from_yolo_labels(dataset_dir: Path) -> tuple[list[str], dict[str, int]]:
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No data.yaml in {dataset_dir}")

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]

    counts = defaultdict(int)
    for split in ["train", "valid", "test"]:
        lbl_dir = dataset_dir / split / "labels"
        if not lbl_dir.exists():
            continue
        for lbl_file in lbl_dir.glob("*.txt"):
            for line in lbl_file.read_text().strip().splitlines():
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    if cls_id < len(names):
                        counts[names[cls_id]] += 1

    return names, dict(counts)


def count_from_csv(csv_path: Path) -> tuple[list[str], dict[str, int]]:
    counts = {}
    names = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls_name = row.get("class") or row.get("Class") or row.get("name") or row.get("Name")
            count_val = row.get("count") or row.get("Count") or row.get("annotations") or row.get("Annotations")
            if cls_name and count_val:
                cls_name = cls_name.strip()
                names.append(cls_name)
                counts[cls_name] = int(count_val.strip())
    return names, counts


def analyze(names: list[str], counts: dict[str, int]):
    total = sum(counts.values())
    if total == 0:
        logger.warning("No annotations found.")
        return

    sorted_classes = sorted(counts.items(), key=lambda x: -x[1])
    max_count = sorted_classes[0][1] if sorted_classes else 1
    nonzero = [(n, c) for n, c in sorted_classes if c > 0]
    min_count = nonzero[-1][1] if nonzero else 0
    zero_classes = [n for n in names if counts.get(n, 0) == 0]

    print(f"\n{'='*70}")
    print(f"CLASS BALANCE ANALYSIS")
    print(f"{'='*70}")
    print(f"Total annotations: {total:,}")
    print(f"Total classes: {len(names)}")
    print(f"Classes with data: {len(nonzero)}")
    if zero_classes:
        print(f"Classes with ZERO data: {len(zero_classes)} -> {zero_classes}")
    print(f"Imbalance ratio: {max_count}:{min_count} ({max_count/max(min_count,1):.1f}x)")

    print(f"\n{'Class':<35} {'Count':>8} {'%':>7} {'Bar'}")
    print(f"{'-'*35} {'-'*8} {'-'*7} {'-'*30}")
    for cls_name, count in sorted_classes:
        pct = count / total * 100
        bar_len = int(count / max_count * 30)
        bar = "█" * bar_len
        print(f"{cls_name:<35} {count:>8,} {pct:>6.1f}% {bar}")

    for cls_name in zero_classes:
        if cls_name not in dict(sorted_classes):
            print(f"{cls_name:<35} {'0':>8} {'0.0':>6}% ")

    print(f"\n{'='*70}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*70}")

    median_count = sorted([c for _, c in nonzero])[len(nonzero)//2] if nonzero else 0

    severely_under = [(n, c) for n, c in nonzero if c < median_count * 0.1]
    moderately_under = [(n, c) for n, c in nonzero if median_count * 0.1 <= c < median_count * 0.3]
    over_represented = [(n, c) for n, c in nonzero if c > median_count * 5]

    if severely_under:
        print(f"\n\U0001f534 SEVERELY UNDERREPRESENTED (< 10% of median):")
        for n, c in severely_under:
            target = max(median_count, 200)
            print(f"   {n}: {c} annotations -> need ~{target - c} more")
            print(f"     → Collect more training images with {n}")
            print(f"     → Oversample 5-10x in prepare_dataset.py")

    if moderately_under:
        print(f"\n\U0001f7e1 MODERATELY UNDERREPRESENTED (10-30% of median):")
        for n, c in moderately_under:
            target = int(median_count * 0.7)
            print(f"   {n}: {c} annotations -> need ~{max(0, target - c)} more")
            print(f"     → Oversample 2-3x in prepare_dataset.py")

    if over_represented:
        print(f"\n\U0001f7e2 OVER-REPRESENTED (> 5x median) — consider capping:")
        for n, c in over_represented:
            print(f"   {n}: {c} annotations (median={median_count})")
            print(f"     → Cap at {median_count * 3} in prepare_dataset.py to prevent bias")

    if zero_classes:
        print(f"\n⚠️  ZERO-DATA CLASSES — need training images:")
        for n in zero_classes:
            print(f"   {n}: no annotations at all")
            print(f"     → Collect or find a dataset with {n} examples")
            print(f"     → Consider removing from class list if not needed")

    print(f"\n\U0001f4cb TRAINING TIPS:")
    if max_count / max(min_count, 1) > 20:
        print(f"   • High imbalance ({max_count/max(min_count,1):.0f}x) — use class-weighted loss")
        print(f"     Add to train.py: cls=3.0 (increase classification loss weight)")
    if len(nonzero) > 20:
        print(f"   • Many classes ({len(nonzero)}) — consider training separate equipment/violation models")
    print(f"   • Add your COCO dataset: python convert_coco_to_yolo.py 'My First Project.coco/'")
    print(f"   • Then re-run: python prepare_dataset.py && python train.py")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze class balance in training data")
    parser.add_argument("--csv", type=Path, default=None, help="Path to class_balance.csv")
    parser.add_argument("--dataset", type=Path, default=None,
        help="Path to YOLO dataset directory (default: merged_dataset)")
    args = parser.parse_args()

    if args.csv:
        names, counts = count_from_csv(args.csv)
    else:
        dataset_dir = args.dataset or (BASE_DIR / "merged_dataset")
        names, counts = count_from_yolo_labels(Path(dataset_dir))

    analyze(names, counts)


if __name__ == "__main__":
    main()
