import os
import csv
import argparse


def create_ids_csv(concepts_path: str, output_path: str) -> None:
    """
    Extract only IDs from concepts.csv and create ids.csv with ID and empty CUIs columns.

    Args:
        concepts_path: Path to the concepts.csv file
        output_path: Path to the output ids.csv file
    """
    if not os.path.exists(concepts_path):
        raise FileNotFoundError(f"Input file not found: {concepts_path}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ids = []
    with open(concepts_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        if not header or len(header) < 1:
            raise ValueError(f"Invalid file: missing header in {concepts_path}")

        for row in reader:
            if row:  # Skip empty lines
                ids.append(row[0].strip())

    # Write ids.csv with ID and empty CUIs columns
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "CUIs"])  # Header
        for image_id in ids:
            writer.writerow([image_id, ""])  # ID with empty CUIs

    print(f"Created {output_path} with {len(ids)} IDs")


def main():
    parser = argparse.ArgumentParser(
        description="Create ids.csv files from concepts.csv files"
    )
    parser.add_argument(
        "--input",
        help="Path to concepts.csv (if not specified, processes both valid and test)",
    )
    parser.add_argument(
        "--output", help="Path to output ids.csv (required if --input is specified)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process both data/valid and data/test datasets",
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
            concepts_path = os.path.join(data_dir, dataset, "concepts.csv")
            ids_path = os.path.join(data_dir, dataset, "ids.csv")

            if os.path.exists(concepts_path):
                create_ids_csv(concepts_path, ids_path)
            else:
                print(f"Warning: {concepts_path} not found, skipping...")
    else:
        # Default: process both if they exist
        datasets = ["valid", "test"]
        for dataset in datasets:
            concepts_path = os.path.join(data_dir, dataset, "concepts.csv")
            ids_path = os.path.join(data_dir, dataset, "ids.csv")

            if os.path.exists(concepts_path):
                create_ids_csv(concepts_path, ids_path)
            else:
                print(f"Warning: {concepts_path} not found, skipping...")


if __name__ == "__main__":
    main()
