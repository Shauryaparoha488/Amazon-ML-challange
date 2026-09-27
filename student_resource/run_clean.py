"""
Phase 2 runner: applies the vectorized cleaner (clean_text_fast.py) to all 6
source files, in memory-safe chunks, and writes both the cleaned TSVs and a
before/after summary report.

Usage: edit BASE_TRAIN / BASE_TEST below to your actual paths, then run:
    python run_clean.py
"""

import time
import pandas as pd
from pathlib import Path
from clean_text_fast import process_dataframe

# ---- EDIT THESE TWO PATHS to match your project ----
BASE_TRAIN = Path(r"C:\Users\shaur\OneDrive\project ML challange\Amazon ML challange\student_resource\dataset\train")
BASE_TEST = Path(r"C:\Users\shaur\OneDrive\project ML challange\Amazon ML challange\student_resource\dataset\test")
# ------------------------------------------------------

CHUNK_SIZE = 500_000  # rows per chunk — keeps memory bounded regardless of file size

FILES = [
    (BASE_TRAIN / "train_source1.tsv", BASE_TRAIN / "train_source1_clean.tsv"),
    (BASE_TRAIN / "train_source2.tsv", BASE_TRAIN / "train_source2_clean.tsv"),
    (BASE_TRAIN / "train_source3.tsv", BASE_TRAIN / "train_source3_clean.tsv"),
    (BASE_TEST / "test_source1.tsv", BASE_TEST / "test_source1_clean.tsv"),
    (BASE_TEST / "test_source2.tsv", BASE_TEST / "test_source2_clean.tsv"),
    (BASE_TEST / "test_source3.tsv", BASE_TEST / "test_source3_clean.tsv"),
]

report_lines = []


def log(line: str):
    print(line)
    report_lines.append(line)


def clean_file(src: Path, dst: Path):
    if not src.exists():
        log(f"SKIP (not found): {src}")
        return

    t0 = time.time()
    total_rows = 0
    null_addr_before = 0
    null_addr_after = 0
    sample_rows = None
    first_chunk = True

    reader = pd.read_csv(src, sep="\t", chunksize=CHUNK_SIZE, dtype=str)
    for i, chunk in enumerate(reader):
        null_addr_before += chunk["business_address"].isna().sum() if "business_address" in chunk.columns else 0

        cleaned = process_dataframe(chunk)
        # tokens are frozensets — convert to a readable string for TSV storage
        cleaned["name_tokens"] = cleaned["name_tokens"].apply(lambda s: "|".join(sorted(s)))
        cleaned["addr_tokens"] = cleaned["addr_tokens"].apply(lambda s: "|".join(sorted(s)))

        null_addr_after += (cleaned["clean_address"] == "").sum()
        total_rows += len(cleaned)

        if sample_rows is None:
            sample_rows = cleaned.sample(min(5, len(cleaned)), random_state=42)

        cleaned.to_csv(dst, sep="\t", index=False, mode="w" if first_chunk else "a", header=first_chunk)
        first_chunk = False

    elapsed = time.time() - t0
    log(f"\n=== {src.name} ===")
    log(f"Rows processed: {total_rows:,}")
    log(f"Time: {elapsed:.1f}s ({elapsed/60:.2f} min)")
    log(f"Null address before: {null_addr_before:,} ({100*null_addr_before/max(total_rows,1):.2f}%)")
    log(f"Empty clean_address after: {null_addr_after:,} ({100*null_addr_after/max(total_rows,1):.2f}%)")
    log("Sample before/after:")
    for _, row in sample_rows.iterrows():
        log(f"  NAME: {row.get('business_name','')!r} -> {row['clean_name']!r}")
        log(f"  ADDR: {row.get('business_address','')!r} -> {row['clean_address']!r}")
    log(f"Saved: {dst}")


if __name__ == "__main__":
    overall_start = time.time()
    for src, dst in FILES:
        clean_file(src, dst)
    overall_elapsed = time.time() - overall_start
    log(f"\n=== TOTAL TIME: {overall_elapsed:.1f}s ({overall_elapsed/60:.2f} min) ===")

    out_path = BASE_TRAIN.parent.parent / "clean_output.txt"
    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nFull report saved to: {out_path}")
