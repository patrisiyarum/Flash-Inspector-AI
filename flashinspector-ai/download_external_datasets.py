#!/usr/bin/env python3
"""
FlashInspector AI - Download FireSafetyNet from Zenodo.

Downloads FireSafetyNet sub-datasets from:
https://zenodo.org/records/13358169

Services:
  1  FSE Detection - fire blankets, extinguishers, manual call points, smoke detectors (bbox)
  2  FSE Marking Detection - FSE marking signs (bbox)
  3  Condition Check - Modal - fire extinguisher condition, blocked/non-compliant (segmentation)
  4  Condition Check - Amodal - amodal detection when partially obscured (segmentation)
  5  Details Extraction - Inspection Tags - maintenance dates on tags (bbox)
  6  Details Extraction - Fire Classes Symbols - fire class symbols on extinguishers (bbox)

Services 3 & 4 use segmentation (merged into separate segmentation dataset).

Run after download_datasets.py. Then run prepare_dataset.py to merge.

Usage:
    python download_external_datasets.py              # Service 1 only (default)
    python download_external_datasets.py --services 1,2,5,6   # Detection services
    python download_external_datasets.py --all       # All services
"""

import argparse
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "fire_safety_datasets"
ZENODO_RECORD = "13358169"
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"


def _url(file_name: str) -> str:
    return f"{BASE_URL}/{file_name.replace(' ', '%20')}?download=1"


# Service definitions: (service_id, dataset_key, description, zip files, detection=True/False)
SERVICES = {
    1: (
        "firesafetynet",
        "FSE Detection - extinguishers, blankets, call points, smoke detectors",
        ["1_FSE Detection.zip"],
        True,
    ),
    2: (
        "firesafetynet_s2",
        "FSE Marking Detection - marking signs",
        ["2_FSE Marking Detection.zip"],
        True,
    ),
    3: (
        "firesafetynet_s3_modal",
        "Condition Check - Modal (segmentation, blocked extinguishers)",
        [
            "3_1_FSE Condition Check_modal_train_data.zip",
            "3_1_FSE Condition Check_modal_val_data_and_weights.zip",
        ],
        False,  # segmentation
    ),
    4: (
        "firesafetynet_s4_amodal",
        "Condition Check - Amodal (segmentation, partially obscured)",
        [
            "4_1_FSE Condition Check_amodal_train.zip",
            "4_1_FSE Condition Check_amodal_val_data_and_weights.zip",
        ],
        False,  # segmentation
    ),
    5: (
        "firesafetynet_s5",
        "Details Extraction - Inspection Tags",
        ["5_FSE Details Extraction - inspection tags.zip"],
        True,
    ),
    6: (
        "firesafetynet_s6",
        "Details Extraction - Fire Classes Symbols",
        ["6_FSE Details Extraction - fire classes symbols.zip"],
        True,
    ),
}


def _extract_and_flatten(target_dir: Path, zip_path: Path) -> None:
    """Extract zip and flatten nested root folder if present (for services with train+val zips)."""
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target_dir)
    zip_path.unlink(missing_ok=True)
    known = {"train", "valid", "val", "images", "labels", "weights"}
    dirs = [x for x in target_dir.iterdir() if x.is_dir() and x.name not in known]
    if len(dirs) == 1:
        nested = dirs[0]
        for item in nested.iterdir():
            dst = target_dir / item.name
            if dst.exists() and dst.is_dir():
                shutil.rmtree(dst)
            shutil.move(str(item), str(dst))
        nested.rmdir()


def download_service(service_id: int) -> bool:
    """Download and extract one FireSafetyNet service."""
    if service_id not in SERVICES:
        logger.error(f"Unknown service: {service_id}. Use 1-6.")
        return False
    key, desc, files, _ = SERVICES[service_id]
    target_dir = DATASET_DIR / key
    target_dir.mkdir(parents=True, exist_ok=True)
    if (target_dir / "data.yaml").exists() or (target_dir / "train").exists():
        logger.info(f"Service {service_id} ({key}) already extracted. Remove {target_dir} to re-download.")
        return True
    for fname in files:
        url = _url(fname)
        zip_path = target_dir / fname
        logger.info(f"Downloading FireSafetyNet S{service_id}: {fname}...")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
        logger.info("Extracting...")
        _extract_and_flatten(target_dir, zip_path)
    logger.info(f"Extracted Service {service_id} to {target_dir}")
    return True


def add_to_combined_config(service_ids: list[int]):
    """Add services to combined_config.yaml. Detection (1,2,5,6) and segmentation (3,4)."""
    config_path = DATASET_DIR / "combined_config.yaml"
    if not config_path.exists():
        logger.warning("combined_config.yaml not found. Run download_datasets.py first.")
        return
    with open(config_path) as f:
        config = yaml.safe_load(f)
    updated = False
    for sid in service_ids:
        if sid not in SERVICES:
            continue
        key, desc, _, is_detection = SERVICES[sid]
        target_dir = DATASET_DIR / key
        if key in config.get("datasets", {}):
            continue
        if not target_dir.exists():
            continue
        data_yaml_path = next(target_dir.rglob("data.yaml"), None)
        if not data_yaml_path:
            logger.warning(f"No data.yaml in {key}. Skipping config update.")
            continue
        with open(data_yaml_path) as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys())]
        entry = {
            "description": f"FireSafetyNet S{sid}: {desc}",
            "path": str(target_dir.resolve()),
            "classes": names,
        }
        if not is_detection:
            entry["task"] = "segment"
        config.setdefault("datasets", {})[key] = entry
        if is_detection:
            for c in names:
                if c not in config.get("all_classes", []):
                    config.setdefault("all_classes", []).append(c)
        logger.info(f"Added {key} to combined_config.yaml" + (" (segmentation)" if not is_detection else ""))
        updated = True
    if updated:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Download FireSafetyNet from Zenodo")
    parser.add_argument(
        "--services",
        type=str,
        default="1",
        help="Comma-separated service IDs (1-6). Default: 1",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all services 1-6",
    )
    args = parser.parse_args()
    if args.all:
        ids = list(SERVICES.keys())
    else:
        try:
            ids = [int(x.strip()) for x in args.services.split(",")]
        except ValueError:
            logger.error("--services must be comma-separated integers (e.g. 1,2,5)")
            return
    for sid in ids:
        if sid not in SERVICES:
            logger.error(f"Invalid service {sid}. Use 1-6.")
            return
    for sid in ids:
        download_service(sid)
    add_to_combined_config(ids)
    print("\nRun prepare_dataset.py to merge detection and segmentation datasets.")


if __name__ == "__main__":
    main()
