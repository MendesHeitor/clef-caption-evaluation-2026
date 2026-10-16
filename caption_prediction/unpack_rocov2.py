#!/usr/bin/env python3
"""
Download the ROCOv2 validation split from Hugging Face and unpack it into
CLEF_ROCOV2_DIR as:

    <CLEF_ROCOV2_DIR>/validation/<ID>.<ext>               (raw image bytes, real extension)
    <CLEF_ROCOV2_DIR>/validation/validation_metadata.csv  (image_id,caption,cui)

Only the validation parquet files are downloaded (~2.6 GB, into the HF cache
under CLEF_ARTIFACTS_DIR). The metadata CSV is written last, so an interrupted
unpack is never counted as complete. Run with the main venv:

    .venvs/main/bin/python caption_prediction/unpack_rocov2.py [--force]
"""
import argparse
import csv
import os
import sys

import config

config.export_cache_env()  # HF downloads go to the artifacts dir

REPO_ID = "eltorio/ROCOv2-radiology"
REVISION = "80ffeef4eb8d34d27cb5c2815305f1d8aee8a83c"  # main as of 2024-11-13

# Leading bytes -> extension; images are written byte for byte, never re-encoded
MAGIC = [
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF8", ".gif"),
    (b"BM", ".bmp"),
    (b"II*\x00", ".tif"),
    (b"MM\x00*", ".tif"),
]


def image_extension(data, image_id):
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    for magic, ext in MAGIC:
        if data.startswith(magic):
            return ext
    raise ValueError(f"Unrecognised image format for {image_id} (first bytes: {data[:8]!r})")


def status():
    """(ready, image count, metadata rows): ready when both counts match."""
    split_dir = os.path.join(config.rocov2_dir(), config.ROCOV2_SPLIT)
    metadata = config.rocov2_metadata()
    if not os.path.exists(metadata):
        n_images = len(config.image_index(split_dir)) if os.path.isdir(split_dir) else 0
        return False, n_images, 0
    with open(metadata, encoding="utf-8", newline="") as f:
        n_rows = sum(1 for _ in csv.DictReader(f))
    n_images = len(config.image_index(split_dir))
    return n_rows > 0 and n_images == n_rows, n_images, n_rows


def clear_split_dir(split_dir):
    """Remove earlier images and metadata so stale files (or the same ID under
    another extension) can't mix with the new unpack."""
    removed = 0
    for entry in os.scandir(split_dir):
        ext = os.path.splitext(entry.name)[1].lower()
        if entry.is_file() and (ext in config.IMAGE_EXTENSIONS or entry.name.startswith(config.ROCOV2_METADATA)):
            os.remove(entry.path)
            removed += 1
    if removed:
        print(f"Removed {removed} files from an earlier unpack in {split_dir}")


def download():
    from huggingface_hub import snapshot_download

    print(f"Downloading {REPO_ID}@{REVISION[:7]} validation parquet files (cached in {os.environ['HF_HOME']})...")
    local = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=[f"data/{config.ROCOV2_SPLIT}-*.parquet"],
    )
    data_dir = os.path.join(local, "data")
    files = sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.startswith(f"{config.ROCOV2_SPLIT}-") and name.endswith(".parquet")
    )
    if not files:
        raise RuntimeError(f"No {config.ROCOV2_SPLIT} parquet files found in {data_dir}")
    return files


def unpack(parquet_files, split_dir):
    import pyarrow.parquet as pq
    from tqdm import tqdm

    os.makedirs(split_dir, exist_ok=True)
    clear_split_dir(split_dir)
    metadata = config.rocov2_metadata()
    partial = metadata + ".partial"
    total = sum(pq.ParquetFile(p).metadata.num_rows for p in parquet_files)
    with open(partial, "w", encoding="utf-8", newline="") as f, tqdm(total=total, desc="Unpacking") as bar:
        writer = csv.writer(f)
        writer.writerow(["image_id", "caption", "cui"])
        for path in parquet_files:
            columns = ["image", "image_id", "caption", "cui"]
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=columns):
                for row in batch.to_pylist():
                    image_id = row["image_id"]
                    data = row["image"]["bytes"]
                    with open(os.path.join(split_dir, image_id + image_extension(data, image_id)), "wb") as img:
                        img.write(data)
                    writer.writerow([image_id, row["caption"], str(row["cui"])])
                    bar.update(1)
    os.replace(partial, metadata)  # last step: marks the unpack as complete


def main():
    parser = argparse.ArgumentParser(description="Download and unpack ROCOv2 (validation) into CLEF_ROCOV2_DIR.")
    parser.add_argument("--force", action="store_true", help="Unpack again even if the folder looks complete")
    args = parser.parse_args()

    try:
        ready, n_images, n_rows = status()
    except config.ConfigError as e:
        print(f"Config error: {e}")
        sys.exit(1)
    if ready and not args.force:
        print(f"ROCOv2 already unpacked: {n_images} images, {n_rows} metadata rows")
        return

    split_dir = os.path.join(config.rocov2_dir(), config.ROCOV2_SPLIT)
    print(f"ROCOv2 not complete in {split_dir} ({n_images} images, {n_rows} metadata rows); unpacking")
    unpack(download(), split_dir)

    ready, n_images, n_rows = status()
    if not ready:
        print(f"Unpack finished but counts differ: {n_images} images vs {n_rows} metadata rows")
        sys.exit(1)
    print(f"Unpacked {n_images} images and {n_rows} metadata rows into {split_dir}")


if __name__ == "__main__":
    main()
