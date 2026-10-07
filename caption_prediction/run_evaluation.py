#!/usr/bin/env python3
"""
Local (non-Docker) caption evaluation. Run with the main venv:

    .venvs/main/bin/python caption_prediction/run_evaluation.py \
        --dataset standard --submission my_captions.csv [--gt cropped_gt.csv] [--output scores.json]

The ground truth is the dataset's official valid rows unless --gt is given.
Paths come from <repo>/.env (see config.py).
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import traceback

import config
from submission_check import check_submission, SubmissionFormatError

config.export_cache_env()  # before transformers/HF imports in the evaluator
sys.stdout.reconfigure(line_buffering=True)  # progress visible when logging to a file (Docker sets PYTHONUNBUFFERED)

current_dir = os.path.dirname(os.path.abspath(__file__))


def load_evaluator_module():
    # the multi-venv evaluator; its file name is not importable with a plain import
    path = os.path.join(current_dir, "evaluator.ai4mediabench.py")
    spec = importlib.util.spec_from_file_location("evaluator_ai4mediabench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Evaluate a caption submission CSV.")
    parser.add_argument("--dataset", choices=sorted(config.DATASETS), required=True)
    parser.add_argument("--submission", required=True, help="CSV with columns ID,Caption")
    parser.add_argument("--gt", help="Ground-truth CSV overriding the dataset's valid rows")
    parser.add_argument("--output", help="Also write the scores as JSON to this path")
    args = parser.parse_args()

    try:
        gt = config.ground_truth(args.dataset, args.gt)
    except config.ConfigError as e:
        print(f"Config error: {e}")
        sys.exit(1)
    gt_source = args.gt or f"{config.dataset_dir(args.dataset)}/captions.csv (valid rows)"
    print(f"Dataset: {args.dataset}")
    print(f"Ground truth: {gt_source} ({len(gt)} IDs)")
    print(f"Submission: {args.submission}")

    # Run format checks before loading any model
    try:
        check_submission(
            submission_path=args.submission,
            ground_truth_path=args.gt,
            dataset_type=args.dataset,
            ground_truth_ids=list(gt),
        )
        print("Submission format check passed.")
    except SubmissionFormatError as e:
        print("Submission format error detected:\n" + str(e))
        sys.exit(1)
    except Exception:
        print("Unexpected error during submission check. Full traceback:")
        traceback.print_exc()
        sys.exit(1)

    # Image embeddings are cached per dataset; only missing IDs get encoded
    precompute = [
        config.venv_python("medimageinsights"),
        os.path.join(current_dir, "precompute_embeddings.py"),
        "--dataset",
        args.dataset,
    ] + (["--gt", args.gt] if args.gt else [])
    subprocess.run(precompute, check=True)

    artifacts = config.artifacts_dir()
    evaluator_module = load_evaluator_module()
    caption_evaluator = evaluator_module.CaptionEvaluator(
        ground_truth_path=gt_source,
        dataset_type=args.dataset,
        gt=gt,
        embeddings_path=config.embeddings_cache(args.dataset),
        medcat_zip=config.medcat_zip(),
        alignscore_python=config.venv_python("alignscore"),
        alignscore_ckpt=os.path.join(artifacts, "AlignScore", "AlignScore-base.ckpt"),
        medimageinsights_python=config.venv_python("medimageinsights"),
        medimageinsights_model_dir=os.path.join(artifacts, "MedImageInsights", "2024.09.27"),
    )
    result = caption_evaluator._evaluate({"submission_file_path": args.submission}, {})
    result = {k: float(v) for k, v in result.items()}
    print(f"\nEvaluation complete for {args.dataset}!")
    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Scores written to {args.output}")


if __name__ == "__main__":
    main()
