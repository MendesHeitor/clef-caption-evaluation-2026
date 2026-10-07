import os
import csv
import argparse
import traceback


class SubmissionFormatError(Exception):
    pass


def _read_utf8(path: str) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read()
        # Validate UTF-8 by decoding
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise SubmissionFormatError(
            "File encoding error: submission.csv must be UTF-8.\nDetails: " + str(e)
        )


def _load_ground_truth_ids(ground_truth_path: str) -> list:
    ids = []
    with open(ground_truth_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or len(header) < 1:
            raise SubmissionFormatError(
                "Ground truth file header is missing or invalid."
            )
        id_idx = 0  # ground truth files have ID as first column
        for row in reader:
            if not row:
                # ignore blank lines in ground truth
                continue
            ids.append(row[id_idx].strip())
    return ids


def check_submission(
    submission_path: str,
    ground_truth_path: str,
    dataset_type: str,
    ground_truth_ids: list = None,
) -> None:
    """ground_truth_ids (ordered) replaces reading IDs from ground_truth_path."""
    if not os.path.exists(submission_path):
        raise SubmissionFormatError(f"Submission file not found at {submission_path}.")

    raw_text = _read_utf8(submission_path)

    # Disallow extra blank lines anywhere
    lines = raw_text.splitlines()
    if any(len(line.strip()) == 0 for line in lines if line != lines[0]):
        raise SubmissionFormatError(
            "Empty line detected: The submission must not contain extra blank lines."
        )

    # Parse via csv to validate structure
    try:
        with open(submission_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                raise SubmissionFormatError(
                    "Missing header row. Expected columns: ID,Caption."
                )
            if len(header) != 2 or header[0] != "ID" or header[1] != "Caption":
                raise SubmissionFormatError(
                    f"Invalid header. Expected exactly two columns: ID,Caption. Found: {header}"
                )

            seen_ids = set()
            gt_ids_list = (
                list(ground_truth_ids)
                if ground_truth_ids is not None
                else _load_ground_truth_ids(ground_truth_path)
            )
            gt_ids = set(gt_ids_list)
            missing_in_submission = set(gt_ids)

            # Inspect raw lines for quoting rule when caption contains comma
            for i, (line, row) in enumerate(
                zip(lines[1:], reader), start=2
            ):  # start=2 for human 1-based
                if not row or len(row) != 2:
                    raise SubmissionFormatError(
                        f"Row {i}: Expected exactly 2 columns. Found: {row}"
                    )
                image_id, _caption = row[0], row[1]

                # Leading/trailing whitespace check
                if image_id != image_id.strip():
                    raise SubmissionFormatError(
                        f"Row {i}: ID has leading/trailing whitespace. Found: '{image_id}'."
                    )
                # Captions may include leading/trailing whitespace (allowed)

                # Order + duplicate + membership checks
                if image_id in seen_ids:
                    raise SubmissionFormatError(
                        f"Row {i}: Duplicate ID detected: {image_id}."
                    )
                position = len(seen_ids)
                if position >= len(gt_ids_list):
                    raise SubmissionFormatError(
                        f"Row {i}: Extra ID beyond ground truth length: {image_id}."
                    )
                expected_id = gt_ids_list[position]
                if image_id != expected_id:
                    raise SubmissionFormatError(
                        f"Row {i}: ID order mismatch. Expected '{expected_id}' at position {position + 1}, found '{image_id}'."
                    )
                if image_id not in gt_ids:
                    raise SubmissionFormatError(
                        f"Row {i}: ID '{image_id}' not found in {dataset_type} ground truth set."
                    )
                seen_ids.add(image_id)
                if image_id in missing_in_submission:
                    missing_in_submission.remove(image_id)

                # Quoting rule when caption contains comma
                # Check original raw line: after first comma, ensure caption starts and ends with quotes when it contains comma
                raw_after_first_comma = line.split(",", 1)[1] if "," in line else None
                if raw_after_first_comma and "," in raw_after_first_comma:
                    if not (
                        raw_after_first_comma.startswith('"')
                        and raw_after_first_comma.endswith('"')
                    ):
                        raise SubmissionFormatError(
                            f"Row {i}: Caption contains a comma but is not enclosed in double quotes."
                        )

            # Missing IDs warning/error: require full coverage for strict site checks
            if missing_in_submission:
                missing_count = len(missing_in_submission)
                sample = sorted(list(missing_in_submission))[:5]
                raise SubmissionFormatError(
                    f"Submission incomplete: {missing_count} official IDs are missing. Example: {sample}"
                )

    except csv.Error as e:
        # Show full csv parsing error for edge cases
        raise SubmissionFormatError(f"CSV parsing error: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate caption submission.csv format without Docker."
    )
    parser.add_argument("--submission", required=True, help="Path to submission.csv")
    parser.add_argument(
        "--ground-truth", dest="ground_truth", help="Path to ground truth captions.csv"
    )
    parser.add_argument(
        "--dataset",
        choices=["valid", "test"],
        default="valid",
        help="Dataset split to validate against (used to infer default ground truth path).",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    gt_path = args.ground_truth or os.path.join(
        base_dir, "data", args.dataset, "captions.csv"
    )

    if not os.path.exists(gt_path):
        print(f"Ground truth file not found at {gt_path}")
        raise SystemExit(1)

    try:
        check_submission(args.submission, gt_path, args.dataset)
        print("Submission format check passed.")
    except SubmissionFormatError as e:
        print("Submission format error detected:\n" + str(e))
        raise SystemExit(1)
    except Exception:
        print("Unexpected error during submission check. Full traceback:")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
