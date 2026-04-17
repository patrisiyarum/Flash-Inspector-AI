#!/usr/bin/env python3
"""
Rebuild poc_report HTML from a PoC stdout log plus annotated frames on disk.

Use when a long run crashed before writing JSON/HTML (e.g. Roboflow HTTP 500).
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "poc"))

from poc_model_v3_demo import (  # noqa: E402
    DEFAULT_VERSION,
    FRAMES_DIR,
    RESULTS_DIR,
    generate_html_report,
)

DET_RE = re.compile(
    r"\[\s*([0-9.]+)s\]\s+(.+?):\s+(\d+)%\s+at\s+\(\s*(\d+)\s*,\s*(\d+)\s*\)"
)
PROC_RE = re.compile(r"Processing video:\s+(.+)")
DUR_RE = re.compile(r"Duration:\s+([0-9.]+)s\s+\|\s+Resolution:\s+([0-9]+x[0-9]+)")
TOTAL_RE = re.compile(r"Total detections:\s+(\d+)\s+across\s+(\d+)\s+frames")

VIOLATION_CLASSES = {
    "empty_mount",
    "extinguisher_cabinet_empty",
    "bracket_empty",
    "missing fire extinguisher",
    "non_compliant_tag",
    "yellow_tag",
    "red_tag",
    "fire extinguisher tagged as noncompliant",
}


def parse_log(text: str) -> list[dict]:
    results: list[dict] = []
    current: dict | None = None
    current_completed = False
    dets_by_ts: dict[float, list[dict]] = {}

    def flush() -> None:
        nonlocal current, dets_by_ts, current_completed
        if current is None:
            return
        if not current_completed:
            current = None
            dets_by_ts = {}
            current_completed = False
            return
        stem = Path(current["source"]).stem
        frame_dir = FRAMES_DIR / stem
        frames: list[dict] = []
        for ts in sorted(dets_by_ts.keys()):
            preds = dets_by_ts[ts]
            jpg = frame_dir / f"frame_{ts:.1f}s.jpg"
            b64 = ""
            if jpg.is_file():
                b64 = base64.b64encode(jpg.read_bytes()).decode()
            frames.append(
                {
                    "timestamp_sec": ts,
                    "detections": preds,
                    "annotated_b64": b64,
                }
            )
        violation_frames = sum(
            1
            for f in frames
            if any(d["class"] in VIOLATION_CLASSES for d in f["detections"])
        )
        current["frames"] = frames
        current["violation_frames"] = violation_frames
        current["frames_with_detections"] = len(frames)
        results.append(current)
        current = None
        dets_by_ts = {}
        current_completed = False

    for line in text.splitlines():
        if m := PROC_RE.search(line):
            flush()
            current = {
                "source": m.group(1).strip(),
                "source_type": "video",
                "duration_sec": 0.0,
                "resolution": "unknown",
                "total_detections": 0,
            }
            dets_by_ts = {}
            current_completed = False
            continue
        if current is None:
            continue
        if m := DUR_RE.search(line):
            current["duration_sec"] = float(m.group(1))
            current["resolution"] = m.group(2)
            continue
        if m := DET_RE.search(line):
            ts = float(m.group(1))
            pred = {
                "class": m.group(2).strip(),
                "confidence": int(m.group(3)) / 100.0,
                "x": int(m.group(4)),
                "y": int(m.group(5)),
                "width": 100,
                "height": 100,
            }
            dets_by_ts.setdefault(ts, []).append(pred)
            continue
        if m := TOTAL_RE.search(line):
            current["total_detections"] = int(m.group(1))
            current_completed = True
            continue
    flush()
    return results


def write_checkpoint_json(all_results: list[dict]) -> None:
    """Write poc_v{N}_results.json + full (for merging the next PoC run)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_results: list[dict] = []
    for r in all_results:
        jr = {k: v for k, v in r.items() if k not in ("annotated_b64", "frames")}
        if "frames" in r:
            jr["detection_frames"] = [
                {"timestamp_sec": f["timestamp_sec"], "detections": f["detections"]}
                for f in r["frames"]
            ]
        json_results.append(jr)
    stem = f"poc_v{DEFAULT_VERSION}_results"
    compact_path = RESULTS_DIR / f"{stem}.json"
    full_path = RESULTS_DIR / f"{stem}_full.json"
    compact_path.write_text(json.dumps(json_results, indent=2))
    full_path.write_text(json.dumps(all_results, indent=2))
    print(f"Checkpoint JSON: {compact_path} and {full_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild HTML report from PoC log + frames/ annotated JPEGs.",
    )
    parser.add_argument(
        "log_file",
        type=Path,
        help="Path to terminal log or captured PoC stdout (contains [INFO] lines).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=BASE_DIR / "poc_results" / "poc_report_rebuilt_from_log.html",
        help="Output HTML path",
    )
    parser.add_argument(
        "--seed-json",
        action="store_true",
        help=f"Also write poc_v{DEFAULT_VERSION}_results*.json for merging a follow-up PoC run.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML (use with --seed-json to only write JSON checkpoints).",
    )
    args = parser.parse_args()

    text = args.log_file.read_text(errors="replace")
    all_results = parse_log(text)
    if not all_results:
        print("No video blocks parsed; check log format.", file=sys.stderr)
        sys.exit(1)

    if args.seed_json:
        write_checkpoint_json(all_results)

    if args.no_html:
        print(f"Parsed {len(all_results)} sources (HTML skipped)")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_html_report(all_results, args.output)

    note = (
        f"<!-- Rebuilt from log: {all_results[0]['source']} … "
        f"{all_results[-1]['source']} ({len(all_results)} sources), model v{DEFAULT_VERSION} -->"
    )
    banner = (
        f"<p class=\"meta\" style=\"color:#b45309;background:#fffbeb;padding:10px 12px;"
        f"border-radius:4px;border:1px solid #fcd34d;margin-bottom:20px\">"
        f"<strong>Partial run</strong> — Rebuilt from captured log + saved frames "
        f"({len(all_results)} videos, model v{DEFAULT_VERSION}). "
        f"Full 44-video batch was interrupted by a server error on the next clip.</p>"
    )
    html = args.output.read_text()
    if "</head>" in html:
        html = html.replace("</head>", f"  {note}\n</head>", 1)
    if '<p class="meta">' in html:
        html = html.replace('<p class="meta">', banner + "\n  <p class=\"meta\">", 1)
    args.output.write_text(html)
    print(f"Wrote {args.output} ({len(all_results)} sources)")


if __name__ == "__main__":
    main()
