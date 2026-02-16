import os
import csv
import argparse


def create_ids_csv(captions_path: str, output_path: str) -> None:
    """
    Extract only IDs from captions.csv and create ids.csv with ID and empty Caption columns.

    Args:
        captions_path: Path to the input captions.csv file
        output_path: Path to the output ids.csv file
    """
    if not os.path.exists(captions_path):
        raise FileNotFoundError(f"Input file not found: {captions_path}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ids = []
    with open(captions_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        if not header or len(header) < 1:
            raise ValueError("Invalid captions.csv: missing header")

        for row in reader:
            if row:  # Skip empty lines
                ids.append(row[0].strip())

    # Write ids.csv with ID and empty Caption columns
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Caption"])  # Header
        for image_id in ids:
            writer.writerow([image_id, ""])  # ID with empty caption

    print(f"Created {output_path} with {len(ids)} IDs")


def main():
    parser = argparse.ArgumentParser(
        description="Create ids.csv files from captions.csv files"
    )
    parser.add_argument(
        "--input",
        help="Path to input captions.csv (if not specified, processes both valid and test)",
    )
    parser.add_argument(
        "--output", help="Path to output ids.csv (required if --input is specified)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process both data/valid/captions.csv and data/test/captions.csv",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory name relative to this script (default: data)",
    )

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, args.data_dir)

    if args.input:
        if not args.output:
            parser.error("--output is required when --input is specified")
        create_ids_csv(args.input, args.output)
    elif args.all:
        # Process both valid and test datasets
        datasets = ["valid", "test"]
        for dataset in datasets:
            captions_path = os.path.join(data_dir, dataset, "captions.csv")
            ids_path = os.path.join(data_dir, dataset, "ids.csv")

            if os.path.exists(captions_path):
                create_ids_csv(captions_path, ids_path)
            else:
                print(f"Warning: {captions_path} not found, skipping...")
    else:
        # Default: process both if they exist
        datasets = ["valid", "test"]
        for dataset in datasets:
            captions_path = os.path.join(data_dir, dataset, "captions.csv")
            ids_path = os.path.join(data_dir, dataset, "ids.csv")

            if os.path.exists(captions_path):
                create_ids_csv(captions_path, ids_path)
            else:
                print(f"Warning: {captions_path} not found, skipping...")


if __name__ == "__main__":
    main()
