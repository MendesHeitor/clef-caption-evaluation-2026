#!/usr/bin/env python3
"""
Convert a RAG response JSONL (one {"image_id", "generated_caption", ...} per
line) into the submission CSV the evaluator expects: ID,Caption, every
ground-truth ID once, in ground-truth order.

    .venvs/main/bin/python caption_prediction/convert_responses.py \
        --dataset rocov2 --responses responses.jsonl [--gt cropped_gt.csv] [--output out.csv]

Default output: next to the JSONL, same name with .csv (overwritten).
"""
import argparse
import csv
import json
import os
import re
import sys

import config


class ResponsesError(Exception):
    pass


def default_output(responses_path):
    return os.path.splitext(responses_path)[0] + ".csv"


def load_responses(path):
    captions = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ResponsesError(f"Line {line_no}: not valid JSON ({e})")
            image_id = record.get("image_id")
            if not image_id:
                raise ResponsesError(f"Line {line_no}: missing 'image_id'")
            if image_id in captions:
                raise ResponsesError(f"Line {line_no}: duplicate image_id {image_id}")
            caption = record.get("generated_caption") or ""
            # A raw newline would split the CSV row and trip the blank-line check
            captions[image_id] = re.sub(r"[\r\n]+", " ", caption)
    return captions


def convert(responses_path, gt_ids, output_path):
    captions = load_responses(responses_path)
    gt_set = set(gt_ids)
    unknown = [i for i in captions if i not in gt_set]
    missing = [i for i in gt_ids if i not in captions]
    if unknown or missing:
        raise ResponsesError(
            f"Responses don't match the ground truth: {len(missing)} GT IDs missing "
            f"(e.g. {missing[:3]}), {len(unknown)} IDs not in the GT (e.g. {unknown[:3]})"
        )
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["ID", "Caption"])
        writer.writerows((image_id, captions[image_id]) for image_id in gt_ids)
    return len(gt_ids)


def main():
    parser = argparse.ArgumentParser(description="Convert a response JSONL into an ID,Caption submission CSV.")
    parser.add_argument("--dataset", choices=config.DATASET_CHOICES, required=True)
    parser.add_argument("--responses", required=True, help="Response JSONL")
    parser.add_argument("--gt", help="Ground-truth CSV overriding the dataset's default")
    parser.add_argument("--output", help="Submission CSV to write (default: <responses>.csv)")
    args = parser.parse_args()

    output = args.output or default_output(args.responses)
    try:
        gt_ids = list(config.ground_truth(args.dataset, args.gt))
        n = convert(args.responses, gt_ids, output)
    except (config.ConfigError, ResponsesError) as e:
        print(f"Conversion error: {e}")
        sys.exit(1)
    print(f"Wrote {n} captions to {output}")


if __name__ == "__main__":
    main()
