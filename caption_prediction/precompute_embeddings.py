#!/usr/bin/env python3
import os
import base64
import numpy as np
import torch
import sys
from typing import List
from tqdm import tqdm
import argparse

import config

# Ensure MedImageInsight is importable
med_image_insights_dir = os.path.join(config.artifacts_dir(), "MedImageInsights")
if not os.path.exists(med_image_insights_dir):
    raise FileNotFoundError(
        f"MedImageInsights directory not found at {med_image_insights_dir}"
    )

sys.path.insert(0, med_image_insights_dir)

from medimageinsightmodel import MedImageInsight


def encode_batch(image_paths: List[str], scorer: MedImageInsight):
    with open(image_paths[0], "rb") as f:
        pass  # quick existence check to surface early errors

    images = []
    for p in image_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image file not found: {p}")
        with open(p, "rb") as f:
            images.append(base64.b64encode(f.read()).decode("utf-8"))

    with torch.inference_mode():
        outputs = scorer.encode(images=images)
        image_vecs = outputs["image_embeddings"]
        if hasattr(image_vecs, "detach"):
            image_vecs = image_vecs.detach().cpu().numpy()
    return image_vecs


def encode_images(
    image_ids: List[str], image_dir: str, scorer: MedImageInsight, batch_size: int = 32
):
    embeddings = {}
    for start in tqdm(range(0, len(image_ids), batch_size), desc="Encoding images"):
        batch_ids = image_ids[start : start + batch_size]
        batch_paths = [
            os.path.join(image_dir, image_id + ".jpg") for image_id in batch_ids
        ]
        image_vecs = encode_batch(batch_paths, scorer)
        for image_id, vec in zip(batch_ids, image_vecs):
            embeddings[image_id] = vec
    return embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Precompute image embeddings for the ground-truth IDs of a dataset. "
        "Incremental: only IDs missing from the cache are encoded."
    )
    parser.add_argument("--dataset", choices=sorted(config.DATASETS), required=True)
    parser.add_argument(
        "--gt", help="Ground-truth CSV overriding the dataset's valid rows."
    )
    args = parser.parse_args()

    cache_path = config.embeddings_cache(args.dataset)
    embeddings = {}
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        embeddings = {k: data[k] for k in data.files}
    missing = [i for i in config.ground_truth(args.dataset, args.gt) if i not in embeddings]
    if not missing:
        print(f"All embeddings already cached in {cache_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "GPU is required for MedImageInsight model."
    scorer = MedImageInsight(
        model_dir=os.path.join(med_image_insights_dir, "2024.09.27"),
        vision_model_name="medimageinsigt-v1.0.0.pt",
        language_model_name="language_model.pth",
    )
    scorer.load_model()
    if hasattr(scorer, "device"):
        scorer.device = device
    if hasattr(scorer, "to"):
        try:
            scorer.to(device)
        except Exception:
            pass
    print(f"MedImageInsight device: {device}")

    print(f"Encoding {len(missing)} images for {args.dataset}...")
    image_dir = os.path.join(config.dataset_dir(args.dataset), "images")
    embeddings.update(encode_images(missing, image_dir, scorer))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, **embeddings)
    print(f"Saved {len(embeddings)} embeddings to {cache_path}")


if __name__ == "__main__":
    main()
