#!/usr/bin/env python3
"""
Transcribe all videos in the videos/ folder using OpenAI Whisper.
Outputs one .txt per video in transcripts/ and a combined transcripts_all.txt.
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = SCRIPT_DIR / "videos"
TRANSCRIPTS_DIR = SCRIPT_DIR / "transcripts"


def main():
    import whisper

    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    video_files = sorted(VIDEOS_DIR.glob("*.mp4"))
    if not video_files:
        print("No .mp4 files found in", VIDEOS_DIR)
        sys.exit(1)

    print("Loading Whisper model 'base'...")
    model = whisper.load_model("base")

    all_parts = []
    for i, video_path in enumerate(video_files, 1):
        name = video_path.stem
        out_path = TRANSCRIPTS_DIR / f"{name}.txt"
        print(f"[{i}/{len(video_files)}] {video_path.name} ...")
        result = model.transcribe(str(video_path), fp16=False)
        text = (result.get("text") or "").strip()
        out_path.write_text(text, encoding="utf-8")
        all_parts.append(f"# {name}\n\n{text}\n\n---\n\n")

    combined_path = TRANSCRIPTS_DIR / "transcripts_all.txt"
    combined_path.write_text("".join(all_parts), encoding="utf-8")

    print(f"Done. Transcripts in {TRANSCRIPTS_DIR}")
    print(f"Combined file: {combined_path}")


if __name__ == "__main__":
    main()
