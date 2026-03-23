#!/usr/bin/env python3
"""
download_hotpotqa.py — Download HotpotQA distractor dataset from HuggingFace.

Dataset: hotpotqa/hotpot_qa (distractor config)

Usage:
    python download_hotpotqa.py --output-dir ./data/hotpotqa
    python download_hotpotqa.py --output-dir ./data/hotpotqa --split train
    python download_hotpotqa.py --output-dir ./data/hotpotqa --split validation --format jsonl
"""

import argparse
import json
import os
import sys


DATASET_ID = "hotpotqa/hotpot_qa"
CONFIG = "distractor"
DEFAULT_SPLIT = "validation"


def download_dataset(output_dir: str, split: str, output_format: str):
    """Download HotpotQA distractor dataset using HuggingFace datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ 'datasets' library not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"\n📥 Loading: {DATASET_ID} (config={CONFIG}, split={split})")

    try:
        ds = load_dataset(DATASET_ID, CONFIG, split=split)
    except Exception as e:
        print(f"\n❌ Failed to load dataset: {e}")
        sys.exit(1)

    print(f"   Loaded {len(ds)} examples")

    os.makedirs(output_dir, exist_ok=True)

    out_file = os.path.join(output_dir, f"hotpotqa_distractor_{split}.{output_format}")

    print(f"   Saving to: {out_file}")

    if output_format == "jsonl":
        with open(out_file, "w") as f:
            for row in ds:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif output_format == "json":
        with open(out_file, "w") as f:
            json.dump([row for row in ds], f, ensure_ascii=False, indent=2)
    elif output_format == "parquet":
        ds.to_parquet(out_file)
    else:
        print(f"❌ Unknown format: {output_format}")
        sys.exit(1)

    size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"\n✅ Saved {len(ds)} examples to {out_file} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description=f"Download {DATASET_ID} ({CONFIG}) from HuggingFace.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save the dataset",
    )
    parser.add_argument(
        "--split", type=str, default=DEFAULT_SPLIT,
        choices=["train", "validation"],
        help=f"Dataset split to download (default: {DEFAULT_SPLIT})",
    )
    parser.add_argument(
        "--format", type=str, default="jsonl", dest="output_format",
        choices=["jsonl", "json", "parquet"],
        help="Output file format (default: jsonl)",
    )

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("📦 HotpotQA Distractor Dataset Downloader")
    print("=" * 60)
    print(f"  Dataset:    {DATASET_ID}")
    print(f"  Config:     {CONFIG}")
    print(f"  Split:      {args.split}")
    print(f"  Format:     {args.output_format}")
    print(f"  Output dir: {args.output_dir}")
    print("=" * 60)

    download_dataset(args.output_dir, args.split, args.output_format)


if __name__ == "__main__":
    main()