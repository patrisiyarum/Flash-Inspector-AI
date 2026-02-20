#!/usr/bin/env python3
"""
FlashInspector AI - Dataset Download Script
Downloads fire safety datasets from Roboflow for YOLO training.

Usage:
    python download_datasets.py
    python download_datasets.py --dataset fire_extinguisher
    python download_datasets.py --list
"""

import os
import sys
import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("download_datasets.log"),
    ],
)
logger = logging.getLogger(__name__)

# Dataset configurations
DATASETS = {
    "fire_extinguisher": {
        "workspace": "fire-extinguisher",
        "project": "fireextinguisher-z5atr",
        "version": 2,
        "description": "Fire extinguisher detection",
    },
    "fire_smoke": {
        "workspace": "middle-east-tech-university",
        "project": "fire-and-smoke-detection-hiwia",
        "version": 2,
        "description": "Fire and smoke detection",
    },
    "emergency_exit": {
        "workspace": "emergency-exit-signs",
        "project": "emergency-exit-signs",
        "version": 1,
        "description": "Emergency exit sign detection",
    },
    "construction_safety": {
        "workspace": "roboflow-universe-projects",
        "project": "construction-site-safety",
        "version": 27,
        "description": "Construction site safety detection",
    },
    "fire_smoke_extra": {
        "workspace": "fire-and-smoke-detection-yolo",
        "project": "fire-and-smoke-detection-o4uhv",
        "version": 1,
        "description": "Additional fire and smoke detection (9k+ images)",
    },
    "firenet": {
        "workspace": "3d-imaging-ucl",
        "project": "firenet-tahn8",
        "version": 1,
        "description": "Fire safety equipment: extinguisher, strobes, sounders, white domes (smoke detectors), etc.",
    },
    "smoke100": {
        "workspace": "smoke-detection",
        "project": "smoke100-uwe4t",
        "version": 5,
        "description": "Smoke detection (100+ images)",
    },
    "wildfire_smoke": {
        "workspace": "brad-dwyer",
        "project": "wildfire-smoke",
        "version": 1,
        "description": "Wildfire smoke detection (737 images)",
    },
}

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "fire_safety_datasets"


def get_api_key() -> str:
    """Retrieve the Roboflow API key securely."""
    load_dotenv(BASE_DIR / ".env")

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if api_key and api_key != "your_roboflow_api_key_here":
        return api_key

    # Prompt user if key not found
    print("\nROBOFLOW_API_KEY not found in environment or .env file.")
    print("Get your free API key at: https://app.roboflow.com/settings/api")
    api_key = input("Enter your Roboflow API key: ").strip()
    if not api_key:
        logger.error("No API key provided. Exiting.")
        sys.exit(1)
    return api_key


def download_dataset(rf: Roboflow, name: str, config: dict, output_dir: Path) -> dict | None:
    """Download a single dataset from Roboflow.

    Returns dataset info dict on success, None on failure.
    """
    logger.info(f"Downloading: {config['description']} ({name})")
    try:
        project = rf.workspace(config["workspace"]).project(config["project"])
        version = project.version(config["version"])

        dataset_path = output_dir / name
        # Remove existing directory to force fresh download (SDK skips if dir exists)
        if dataset_path.exists():
            import shutil
            shutil.rmtree(dataset_path)

        dataset = version.download("yolov8", location=str(dataset_path))

        # Gather dataset stats - Roboflow may download into a subdirectory
        stats = {"name": name, "description": config["description"], "path": str(dataset_path)}
        for split in ["train", "valid", "test"]:
            count = 0
            for images_dir in dataset_path.rglob(f"{split}/images"):
                count += len(list(images_dir.iterdir()))
            stats[f"{split}_images"] = count

        logger.info(
            f"  Downloaded {name}: "
            f"train={stats.get('train_images', 0)}, "
            f"valid={stats.get('valid_images', 0)}, "
            f"test={stats.get('test_images', 0)} images"
        )
        return stats

    except Exception as e:
        logger.error(f"  Failed to download {name}: {e}")
        return None


def build_combined_config(all_stats: list[dict], output_dir: Path):
    """Create a combined YAML config listing all dataset classes."""
    combined = {
        "datasets": {},
        "all_classes": [],
    }

    seen_classes = set()
    for stats in all_stats:
        name = stats["name"]
        yaml_candidates = list((output_dir / name).rglob("data.yaml"))
        dataset_yaml = yaml_candidates[0] if yaml_candidates else output_dir / name / "data.yaml"
        if dataset_yaml.exists():
            with open(dataset_yaml) as f:
                data = yaml.safe_load(f)
            classes = data.get("names", [])
            if isinstance(classes, dict):
                classes = list(classes.values())
            combined["datasets"][name] = {
                "description": stats["description"],
                "path": stats["path"],
                "classes": classes,
                "train_images": stats.get("train_images", 0),
                "valid_images": stats.get("valid_images", 0),
                "test_images": stats.get("test_images", 0),
            }
            for c in classes:
                if c not in seen_classes:
                    seen_classes.add(c)
                    combined["all_classes"].append(c)

    config_path = output_dir / "combined_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(combined, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Combined config saved to {config_path}")
    logger.info(f"Total unique classes: {len(combined['all_classes'])}")


def main():
    parser = argparse.ArgumentParser(description="Download fire safety datasets from Roboflow")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        help="Download a specific dataset only",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets:")
        for name, config in DATASETS.items():
            print(f"  {name:25s} - {config['description']}")
        return

    api_key = get_api_key()
    rf = Roboflow(api_key=api_key)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    datasets_to_download = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS

    print(f"\nDownloading {len(datasets_to_download)} dataset(s) to {DATASET_DIR}/\n")

    all_stats = []
    for name, config in datasets_to_download.items():
        stats = download_dataset(rf, name, config, DATASET_DIR)
        if stats:
            all_stats.append(stats)

    if all_stats:
        build_combined_config(all_stats, DATASET_DIR)

    print(f"\nDone! {len(all_stats)}/{len(datasets_to_download)} datasets downloaded successfully.")
    if len(all_stats) < len(datasets_to_download):
        print("Check download_datasets.log for error details.")


if __name__ == "__main__":
    main()
