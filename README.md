
# ImageCLEFmedical Caption Prediction & Concept Detection 2026

Official evaluation and submission checking scripts for the [ImageCLEFmedical 2026 Caption Prediction and Concept Detection Challenge](https://www.imageclef.org/2026/medical/caption).

## Introduction

This repository provides tools to validate and evaluate your submissions for the two ImageCLEFmedical 2026 tasks:

- **Caption Prediction:** Generate descriptive captions for medical images.
- **Concept Detection:** Identify relevant UMLS concepts (CUIs) for each image.

You can use these scripts to check your submission format and run local evaluations before submitting to the AI4MediaBench platform. Passing the pre-check is required for a valid submission.

## Native setup (no Docker)

Caption evaluation runs natively with [uv](https://docs.astral.sh/uv/), without Docker or root access. It mirrors `Dockerfile.ai4mediabench`: three venvs (`main`, `alignscore`, `medimageinsights`), because their pinned dependencies conflict. Exact versions live in `caption_prediction/locks/`, so every machine gets the same environment that was tested.

### Requirements

- Linux with an NVIDIA GPU (tested: RTX 3060 6 GB). torch 2.1.2 / CUDA 12.1 and torch 1.13.1 / CUDA 11.7 cover GPUs up to the A100 and RTX 30xx generation, but **not Blackwell** (RTX 50xx, B200).
- `uv`, `git`, `python3` (any version, only used to read the config; uv downloads Python 3.10 itself), and internet access **during setup only**.
- Disk: ~10 GB for the venvs (keep them on a local Linux filesystem) and ~25 GB for artifacts (models + caches).
- Memory: **~16 GB RAM + swap**. The UMLS MedCAT pack takes ~14 GB, loaded last after the GPU scorers are freed. On a cluster, request at least 24 GB for the job. On a 16 GB machine, add ~16 GB of swap.
- The datasets: a folder holding `dev_caption/` (`standard`) and `dev_caption_synth/` (`synth`), each with `captions.csv` and `images/`.
- The UMLS MedCAT model pack `umls_self_train_model_pt2ch_3760d588371755d0.zip`. It's licensed and can't be downloaded by the setup; copy your own (see step 2 of the Docker instructions below).

### 1. Install

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh    # if uv is missing
git clone -b setup/native-uv git@github.com:MendesHeitor/clef-caption-evaluation-2026.git
cd clef-caption-evaluation-2026
```

### 2. Configure

```sh
cp .env.example .env
```

Edit `.env`:

| Variable | Meaning |
|---|---|
| `CLEF_DATASETS_DIR` | folder holding `dev_caption/` and `dev_caption_synth/` |
| `CLEF_ARTIFACTS_DIR` | where models, caches and embeddings go (created if missing; needs ~25 GB) |
| `CLEF_ROCOV2_DIR` | required for `--dataset rocov2`: the unpacked ROCOv2 folder. Set it **before the first run, even if the folder is empty or doesn't exist yet** (its parent must exist); the data is downloaded and unpacked into it. |
| `CLEF_VENVS_DIR` | optional, default `<repo>/.venvs` |

`~` and `*` are expanded, and each pattern must match exactly one path. A mount point whose name changes between boots can be written as `/media/<user>/*/...`. The values in `.env.example` are examples from one machine; replace them.

Then copy the MedCAT pack:

```sh
mkdir -p <CLEF_ARTIFACTS_DIR>/MedCAT && cp umls_self_train_model_pt2ch_3760d588371755d0.zip <CLEF_ARTIFACTS_DIR>/MedCAT/
```

### 3. Set up the environment

```sh
caption_prediction/setup.sh
```

This creates the three venvs from `caption_prediction/locks/` and downloads the model weights at pinned revisions (MedImageInsights, AlignScore). It also prefetches every model the scorers would otherwise download at runtime (DeBERTa-xlarge-mnli, BLEURT-20-D12, roberta, the ROUGE metric, NLTK punkt). Afterwards, evaluation works **offline**: on compute nodes without internet, set `HF_HUB_OFFLINE=1`. It's safe to re-run.

### 4. Evaluate

Prepare a CSV with columns `ID,Caption` (see the format rules under *Caption Prediction Evaluation* below). It must contain every valid ID of the dataset in the order of its `captions.csv`. Then run:

```sh
.venvs/main/bin/python caption_prediction/run_evaluation.py --dataset standard --submission my_captions.csv --output my_scores.json
.venvs/main/bin/python caption_prediction/run_evaluation.py --dataset synth    --submission my_synth_captions.csv
```

- The ground truth is the dataset's official valid rows (IDs containing `_valid_`), read from its `captions.csv` in memory: 19,240 IDs for `standard`, 19,239 for `synth`.
- `--gt file.csv` replaces the ground truth with any `ID,Caption` file, for example a few rows for a quick pipeline test.
- The format check runs first and stops on errors before any model loads.
- Image embeddings are cached per dataset in `<CLEF_ARTIFACTS_DIR>/precomputed/image_embeddings_<dataset>.npz`. Only missing IDs get encoded, so the first full run spends extra time on that step.
- `score` is relevance (mean of BERTScore, ROUGE, similarity, BLEURT). `score_secondary` is factuality (mean of MedCAT, AlignScore).

**On every new machine, run this quick test before a full evaluation.** It exercises every stage, MedCAT included, in minutes. MedCAT is loaded last, after the GPU scorers have been freed, to keep peak memory down; at start-up only its pack is checked, so a MedCAT loading problem would otherwise appear only at the end of a long run.

Quick pipeline test with a 5-row ground truth (`submissions/` is gitignored):

```sh
mkdir -p submissions
.venvs/main/bin/python - <<'PY'
import csv, sys; sys.path.insert(0, "caption_prediction"); import config
gt = list(config.ground_truth("standard").items())[:5]
for name, rows in [("gt_5", gt), ("fake_5", [(i, "Chest X-ray, no abnormality.") for i, _ in gt])]:
    with open(f"submissions/{name}.csv", "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n"); w.writerow(["ID", "Caption"]); w.writerows(rows)
PY
.venvs/main/bin/python caption_prediction/run_evaluation.py --dataset standard --gt submissions/gt_5.csv --submission submissions/fake_5.csv
```

Expect 10–20 minutes on a 16 GB laptop, most of it loading MedCAT (less with enough RAM to avoid swap). Passing the ground truth itself as `--submission` should give ~1.0 for every metric.

#### ROCOv2 responses

`--dataset rocov2` evaluates against the ROCOv2 validation split (9,904 IDs, [eltorio/ROCOv2-radiology](https://huggingface.co/datasets/eltorio/ROCOv2-radiology) at a pinned revision). `--submission` can be the response JSONL directly (one `{"image_id", "generated_caption", ...}` per line):

```sh
.venvs/main/bin/python caption_prediction/run_evaluation.py --dataset rocov2 --submission responses/responses.jsonl --output scores.json
```

One command runs every stage, and each prints the script it runs (`==> <script>: ...`), so a failure names the script that failed:

1. `unpack_rocov2.py`, only if `CLEF_ROCOV2_DIR` is incomplete (the image count in `validation/` differs from the rows of `validation/validation_metadata.csv`). It downloads the validation parquet files (~2.6 GB, into the HF cache under `CLEF_ARTIFACTS_DIR`) and writes `validation/<ID>.<ext>` (raw bytes, PNG for this dataset) plus `validation_metadata.csv` (`image_id,caption,cui`). An existing unpack in this layout with JPEGs works too, since images are found by ID, whatever their extension.
2. `convert_responses.py`: JSONL to `<jsonl name>.csv` next to it (`ID,Caption`, ground-truth order, newlines in captions collapsed). It fails if IDs are missing, duplicated or unknown.
3. `submission_check.py`: the format check, then a sanity check that every ID has an image file, a ground-truth caption and a predicted caption. `--verify-images` also fully decodes every image (~1.5 min more on CPU for the 9,904 images). Any finding prints a summary by category:
   - missing image, undecodable image or empty predicted caption: the run **stops**;
   - empty ground-truth caption (ROCOv2 has one, `ROCOv2_2023_valid_003902`, empty in the source dataset): that ID is **excluded** from scoring and the run continues. The evaluator gets `<submission>.scored.csv` without it, and the `--output` JSON records `n_scored` and `excluded` (ID and reason).
4. Embeddings and scoring, as for the other datasets.

`--check-only` stops after step 3, so the data and the submission can be checked on a machine without a GPU.

### Troubleshooting

- `Killed` / exit 137: not enough RAM (MedCAT needs ~14 GB when it loads, at the end of the run). Add swap or request more memory.
- `Nothing found at '...'. Is the partition mounted?`: a path in `.env` doesn't resolve.
- `no kernel image is available for execution on the device`: the GPU is newer than the pinned torch builds support (see Requirements).
- Upgrading packages: see the comment at the top of `caption_prediction/setup.sh` for how the locks were produced and the known version traps.

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