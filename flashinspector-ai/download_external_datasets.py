#!/usr/bin/env python3
"""
FlashInspector AI - Download external (non-Roboflow) fire safety datasets.

Downloads FireSafetyNet from Zenodo (smoke detectors, extinguishers, etc.).
Fire-ART and others may be added when download URLs are available.

Usage:
    python download_external_datasets.py
"""

import json
import logging
import shutil
import zipfile
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "fire_safety_datasets"
EXTERNAL_DIR = DATASET_DIR / "external"

# External dataset configs: {name: {url or zenodo_id, ...}}
EXTERNAL_DATASETS = {
    "firesafetynet": {
        "zenodo_id": "13358169",
        "description": "Fire safety equipment: extinguishers, smoke detectors, manual call points, etc. (Zenodo)",
    },
}


def download_zenodo_record(record_id: str, output_dir: Path) -> bool:
    """Download all files from a Zenodo record."""
    api_url = f"https://zenodo.org/api/records/{record_id}"
    r = requests.get(api_url, timeout=30)
    if not r.ok:
        logger.error(f"Zenodo API failed: {r.status_code}")
        return False
    data = r.json()
    files = data.get("files", [])
    if not files:
        logger.warning("No files in record")
        return False
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        key = f.get("key", f.get("filename", "file"))
        dl_url = f.get("links", {}).get("self") or f"https://zenodo.org/records/{record_id}/files/{key}"
        logger.info(f"  Downloading {key}...")
        try:
            resp = requests.get(dl_url, stream=True, timeout=60)
            resp.raise_for_status()
            out_path = output_dir / key
            with open(out_path, "wb") as fp:
                for chunk in resp.iter_content(chunk_size=65536):
                    fp.write(chunk)
            if key.endswith(".zip"):
                extract_dir = output_dir / key.replace(".zip", "")
                extract_dir.mkdir(exist_ok=True)
                with zipfile.ZipFile(out_path) as z:
                    z.extractall(extract_dir)
                logger.info(f"  Extracted to {extract_dir}")
        except Exception as e:
            logger.error(f"  Failed to download {key}: {e}")
            return False
    return True


def main():
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in EXTERNAL_DATASETS.items():
        logger.info(f"Downloading {name}: {cfg['description']}")
        out = EXTERNAL_DIR / name
        if out.exists():
            logger.info(f"  {name} already exists at {out}. Remove to re-download.")
            continue
        record_id = cfg.get("zenodo_id")
        if record_id:
            ok = download_zenodo_record(record_id, out)
            if ok:
                logger.info(f"  Done: {out}")
            else:
                logger.error(f"  Failed: {name}")
        else:
            logger.warning(f"  No download URL for {name}")

    print("\nExternal datasets saved to:", EXTERNAL_DIR)
    print("Note: FireSafetyNet may use a different folder structure.")
    print("To merge with Roboflow data, you may need to manually add paths to prepare_dataset or combined_config.")


if __name__ == "__main__":
    main()
