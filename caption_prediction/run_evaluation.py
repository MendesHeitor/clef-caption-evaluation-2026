#!/usr/bin/env python3
"""
Local (non-Docker) caption evaluation. Run with the main venv:

    .venvs/main/bin/python caption_prediction/run_evaluation.py \
        --dataset standard --submission my_captions.csv [--gt cropped_gt.csv] [--output scores.json]
    .venvs/main/bin/python caption_prediction/run_evaluation.py \
        --dataset rocov2 --submission responses.jsonl [--verify-images] [--check-only]

The ground truth is the dataset's official valid rows (all of ROCOv2
validation) unless --gt is given. A .jsonl submission is converted to an
ID,Caption CSV first. For rocov2 the dataset is downloaded and unpacked into
CLEF_ROCOV2_DIR when it isn't complete there. Each stage prints the script it
runs. Paths come from <repo>/.env (see config.py).
"""
import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import traceback

import config
import unpack_rocov2
from convert_responses import default_output
from submission_check import (
    TRIPLE_CHECKS,
    SubmissionFormatError,
    check_submission,
    check_triples,
    format_triple_summary,
)

config.export_cache_env()  # before transformers/HF imports in the evaluator
sys.stdout.reconfigure(line_buffering=True)  # progress visible when logging to a file (Docker sets PYTHONUNBUFFERED)

current_dir = os.path.dirname(os.path.abspath(__file__))


def stage(script, what):
    print(f"\n==> {script}: {what}")


def run_script(script, args, python=sys.executable):
    """Run a pipeline script as its own process; on failure, name it and stop."""
    result = subprocess.run([python, os.path.join(current_dir, script)] + args)
    if result.returncode != 0:
        print(f"\nFAILED in {script} (exit code {result.returncode}). See its output above.")
        sys.exit(result.returncode)


def ensure_rocov2():
    """Unpack ROCOv2 only when the image count and metadata rows don't match."""
    try:
        ready, n_images, n_rows = unpack_rocov2.status()
    except config.ConfigError as e:
        print(f"Config error: {e}")
        sys.exit(1)
    if ready:
        print(f"ROCOv2 ready in {config.rocov2_dir()}: {n_images} images, {n_rows} metadata rows")
        return
    stage("unpack_rocov2.py", f"ROCOv2 incomplete ({n_images} images, {n_rows} metadata rows); downloading and unpacking")
    run_script("unpack_rocov2.py", ["--force"])


def write_scored_submission(submission_path, excluded):
    """Copy of the submission without the excluded IDs: the evaluator requires
    the submission IDs to match the (reduced) ground truth exactly."""
    scored_path = os.path.splitext(submission_path)[0] + ".scored.csv"
    with open(submission_path, encoding="utf-8", newline="") as src, open(
        scored_path, "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(next(reader))
        writer.writerows(row for row in reader if row[0] not in excluded)
    return scored_path


def load_evaluator_module():
    # the multi-venv evaluator; its file name is not importable with a plain import
    path = os.path.join(current_dir, "evaluator.ai4mediabench.py")
    spec = importlib.util.spec_from_file_location("evaluator_ai4mediabench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Evaluate a caption submission (CSV or response JSONL).")
    parser.add_argument("--dataset", choices=config.DATASET_CHOICES, required=True)
    parser.add_argument(
        "--submission", required=True,
        help="CSV with columns ID,Caption, or a response JSONL (image_id, generated_caption)",
    )
    parser.add_argument("--gt", help="Ground-truth CSV overriding the dataset's valid rows")
    parser.add_argument("--output", help="Also write the scores as JSON to this path")
    parser.add_argument(
        "--verify-images", action="store_true",
        help="rocov2: fully decode every image in the sanity check (slower)",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Stop after the format and sanity checks, before any model loads",
    )
    args = parser.parse_args()

    if args.dataset == config.ROCOV2:
        ensure_rocov2()

    try:
        gt = config.ground_truth(args.dataset, args.gt)
    except config.ConfigError as e:
        print(f"Config error: {e}")
        sys.exit(1)
    if args.gt:
        gt_source = args.gt
    elif args.dataset == config.ROCOV2:
        gt_source = config.rocov2_metadata()
    else:
        gt_source = f"{config.dataset_dir(args.dataset)}/captions.csv (valid rows)"
    print(f"Dataset: {args.dataset}")
    print(f"Ground truth: {gt_source} ({len(gt)} IDs)")
    print(f"Submission: {args.submission}")

    if args.submission.endswith(".jsonl"):
        csv_path = default_output(args.submission)
        stage("convert_responses.py", f"converting the JSONL to {csv_path}")
        run_script(
            "convert_responses.py",
            ["--dataset", args.dataset, "--responses", args.submission, "--output", csv_path]
            + (["--gt", args.gt] if args.gt else []),
        )
        args.submission = csv_path

    # Run format checks before loading any model
    stage("submission_check.py", "format check")
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

    excluded = {}
    if args.dataset == config.ROCOV2:
        decode = " with full image decode" if args.verify_images else " (existence only; --verify-images decodes)"
        stage("submission_check.py", f"sanity check of every (image, GT caption, predicted caption){decode}")
        found = check_triples(gt, args.submission, config.image_index(config.image_dir(args.dataset)), args.verify_images)
        if not any(found.values()):
            print(f"Sanity check passed for {len(gt)} IDs.")
        else:
            print(format_triple_summary(found, len(gt)))
            if any(found[key] for key, _, stops in TRIPLE_CHECKS if stops):
                print("\nStopping: fix the problems marked STOPS above and run again.")
                sys.exit(1)
            # Only exclusions left: drop those IDs from the GT and the submission
            excluded = {i: "empty ground-truth caption" for i in found["empty_gt"]}
            gt = {i: c for i, c in gt.items() if i not in excluded}
            args.submission = write_scored_submission(args.submission, excluded)
            print(
                f"\nScoring {len(gt)} IDs; {len(excluded)} excluded because their ground-truth "
                f"caption is empty in the source dataset. Scored submission: {args.submission}"
            )

    if args.check_only:
        print("\n--check-only: stopping before the models load.")
        return

    # Image embeddings are cached per dataset; only missing IDs get encoded
    stage("precompute_embeddings.py", "image embeddings (medimageinsights venv)")
    run_script(
        "precompute_embeddings.py",
        ["--dataset", args.dataset] + (["--gt", args.gt] if args.gt else []),
        python=config.venv_python("medimageinsights"),
    )

    stage("evaluator.ai4mediabench.py", "scoring")
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
        report = dict(result)
        if args.dataset == config.ROCOV2:
            report["n_scored"] = len(gt)
            report["excluded"] = excluded
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Scores written to {args.output}")


if __name__ == "__main__":
    main()
