
# ImageCLEFmedical Caption Prediction & Concept Detection 2026

Official evaluation and submission checking scripts for the [ImageCLEFmedical 2026 Caption Prediction and Concept Detection Challenge](https://www.imageclef.org/2026/medical/caption).

## Introduction

This repository provides tools to validate and evaluate your submissions for the two ImageCLEFmedical 2026 tasks:

- **Caption Prediction:** Generate descriptive captions for medical images.
- **Concept Detection:** Identify relevant UMLS concepts (CUIs) for each image.

You can use these scripts to check your submission format and run local evaluations before submitting to the AI4MediaBench platform. Passing the pre-check is required for a valid submission.

## Setup

You need docker to run the evaluations with GPU support for caption prediction evaluation. Please refer to the last section of this README to see submission format instructions and run submission checks locally (see below).

# Caption Prediction Evaluation

1. Copy `captions.csv` and `images` dir into `caption_prediction/data/valid`.
   
2. Request a licence for UMLS and then download the UMLS full model (zip file) from https://medcat.sites.er.kcl.ac.uk/auth-callback-api into `caption_prediction/models/MedCAT`.
   
3. Choose device (GPU) or use `--gpus all`, precompute the image embeddings once, and build the `caption_prediction_evaluator` docker image. This may take a while.

    ```sh
    cd caption_prediction
    docker build --no-cache -f Dockerfile.dev -t caption_prediction_evaluator .
    docker run --rm --gpus '"device=3"' \
      -v "$(pwd)/precomputed:/app/precomputed" \
      caption_prediction_evaluator \
      python3 precompute_embeddings.py --dataset valid
    docker build --no-cache -f Dockerfile.valid -t caption_prediction_evaluator .
    ```
4. Go to dir with your `submission.csv`, choose device (GPU) or use `--gpus all` and run the evaluation. The container will first run a submission format pre-check and print errors if any issues are found.
    ```sh
    docker run \
      --gpus '"device=4"' \
      --rm \
      -v $(pwd)/submission.csv:/app/submission.csv \
      caption_prediction_evaluator
    ```
   Submission format: `submission.csv` with the two columns **ID** and **Caption**.

   (`submission.csv` is the file you submit to AI4MediaBench in a .zip archive)

   Example:
    ```plain
    ID,Caption
    ImageCLEFmedical_Caption_2025_valid_0,"Illustration of the original image with an ROI.T: Tumor."
    ```

    Common pre-check errors and fixes:
    - Encoding: Ensure the file is saved as UTF-8 (no BOM).
    - Header: Must be exactly two columns: ID,Caption.
    - Blank lines: Remove any empty trailing or intermediate lines.
    - Duplicates: Each ID must appear only once.
    - Order: IDs must follow the exact order of the ground truth file.
    - ID set: Use only IDs from the official set; include all official IDs.
    - Quoting: Captions containing commas must be enclosed in double quotes.
    - Edge cases: Full error trace is printed to help diagnose parsing issues.

# Concept Detection Evaluation

1. Copy `concepts.csv` and `concepts_manual.csv` into `concept_detection/data/valid`.

2. Build the `concept_detection_evaluator` docker image. 

    ```sh
    cd concept_detection
    docker build --no-cache -f Dockerfile.valid -t concept_detection_evaluator .
    ```

3. Go to dir with your `submission.csv` and run evaluation. The container will first run a submission format pre-check and print errors if any issues are found.

    ```sh
    docker run \
      --rm \
      -v $(pwd)/submission.csv:/app/submission.csv \
      concept_detection_evaluator \
      valid
    ```
    Submission format: `submission.csv` with the two columns **ID** and **CUIs** with semicolon seperated CUIs. 

   (`submission.csv` is the file you submit to AI4MediaBench in a .zip archive)

   Example:
   ```plain
   ID,CUIs
   ImageCLEFmedical_Caption_2024_valid_000001,C0040405;C0856747
   ```

  Common pre-check errors and fixes:
  - Encoding: Ensure the file is saved as UTF-8 (no BOM).
  - Header: Must be exactly two columns: ID,CUIs.
  - Blank lines: Remove any empty trailing or intermediate lines.
  - Whitespace: Trim leading/trailing spaces in IDs and CUIs.
  - Duplicates: Each ID must appear only once; no duplicate CUIs per ID.
  - Order: IDs must follow the exact order of the ground truth file.
  - ID set: Use only IDs from the official set; include all official IDs.
  - Separator: CUIs must be ';' separated with no empty entries when provided (empty CUI lists are allowed).
  - Format: Each CUI must match C followed by digits (e.g., C0040405) when provided.
  - Edge cases: Full error trace is printed to help diagnose parsing issues.


# Run Submission Checks Locally (No Docker)

Use the built-in checkers if you just want to validate formatting:

```sh
python caption_prediction/submission_check.py \
  --submission /path/to/submission.csv \
  --dataset valid \
  --ground-truth caption_prediction/data/valid/captions.csv
```

```sh
python concept_detection/submission_check.py \
  --submission /path/to/submission.csv \
  --dataset valid \
  --primary-gt concept_detection/data/valid/concepts.csv
```

Arguments:
- `--submission` (required): path to your submission.csv
- `--dataset` (optional): valid|test (default: valid) to auto-pick default ground truth paths
- `--ground-truth`, `--primary-gt`: override ground truth locations if needed


## Notes & Troubleshooting

- **Docker expects your `submission.csv` in the current directory when running the evaluation container.**
- The evaluation scripts do not modify your submission file.
- If you encounter GPU or file mounting issues, check your Docker version and permissions.
- If you see format errors, use the local checker scripts to debug before submitting.

## File Structure

This is how the file structure would look like with UMLS model and submission.csv files:

```plain
.
├── README.md
├── caption_prediction
│   ├── Dockerfile
│   ├── Dockerfile.dev
│   ├── Dockerfile.test
│   ├── Dockerfile.test-check
│   ├── Dockerfile.valid
│   ├── Dockerfile.valid-check
│   ├── create_ids_csv.py
│   ├── data
│   │   ├── test
│   │   │   ├── captions.csv
│   │   │   ├── ids.csv
│   │   │   └── images
│   │   ├── train
│   │   │   └── captions.csv
│   │   └── valid
│   │       ├── captions.csv
│   │       ├── ids.csv
│   │       └── images
│   ├── evaluator.py
│   ├── medcat_scorer.py
│   ├── models
│   │   └── MedCAT
│   │       └── umls_self_train_model_pt2ch_3760d588371755d0.zip
│   ├── precompute_embeddings.py
│   ├── precomputed
│   │   ├── image_embeddings_test.npz
│   │   └── image_embeddings_valid.npz
│   ├── requirements.txt
│   ├── run_evaluation.py
│   └── submission_check.py
└── concept_detection
    ├── Dockerfile
    ├── Dockerfile.test
    ├── Dockerfile.test-check
    ├── Dockerfile.valid
    ├── Dockerfile.valid-check
    ├── create_ids_csv.py
    ├── data
    │   ├── test
    │   │   ├── concepts.csv
    │   │   ├── concepts_manual.csv
    │   │   └── ids.csv
    │   ├── train
    │   │   ├── concepts.csv
    │   │   └── concepts_manual.csv
    │   └── valid
    │       ├── concepts.csv
    │       ├── concepts_manual.csv
    │       └── ids.csv
    ├── evaluator.py
    ├── requirements.txt
    ├── run_evaluation.py
    └── submission_check.py
```

## Submission Format Instructions

Please note the following when using the scripts or submitting to the ai4mediabench website.

* ai4mediabench requires a zip file (name it as you like) with only your submission in csv format (named "submission.csv").
* Your submission can only contain two columns ID,Caption or ID,CUIs.
* The submission check on the site is very strict and does not allow extra (even empty) lines in the file.
* Image IDs can only appear once in the file.
* Only include IDs from the official test set.
* Ensure standard utf-8 encoding.

* For caption prediction:
  * Use quotation marks for all captions, or at least for those that contain a comma (,).

* For concept detection:
  * Remove duplicate CUIs on a line. The same concept can not be specified more than once for the same image ID.
  * Use ; to separate CUIs.