#!/usr/bin/env python3
"""
Local (non-Docker) path configuration.

Values come from environment variables, falling back to <repo>/.env.
'~' and glob patterns ('*') are expanded, so a mount point whose name
changes between boots (e.g. /media/hmp/<volume-id>/...) can be written as
/media/hmp/*/... . A pattern must match exactly one path.

Stdlib only: also used by setup.sh through the system python3.
Usage from shell: python3 config.py artifacts|venvs|datasets
"""
import csv
import glob
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
ENV_FILE = os.path.join(REPO_ROOT, ".env")

# Dataset name -> directory under CLEF_DATASETS_DIR (holds captions.csv + images/)
DATASETS = {
    "standard": "dev_caption",
    "synth": "dev_caption_synth",
}


class ConfigError(Exception):
    pass


def _load_env_file():
    values = {}
    if not os.path.exists(ENV_FILE):
        return values
    with open(ENV_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


_ENV_FILE_VALUES = _load_env_file()


def get(key, default=None):
    return os.environ.get(key, _ENV_FILE_VALUES.get(key, default))


def resolve(pattern, create=False):
    """Expand '~' and globs; the result must be exactly one path.

    With create=True only the parent has to exist; the leaf is created.
    """
    pattern = os.path.expanduser(pattern)
    matches = glob.glob(pattern)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ConfigError(f"Pattern '{pattern}' matches several paths: {matches}")
    if create:
        parent = resolve(os.path.dirname(pattern))
        path = os.path.join(parent, os.path.basename(pattern))
        os.makedirs(path, exist_ok=True)
        return path
    raise ConfigError(
        f"Nothing found at '{pattern}'. Is the partition mounted? "
        f"Check {ENV_FILE} (see .env.example)."
    )


def _required(key):
    value = get(key)
    if not value:
        raise ConfigError(f"{key} is not set. Copy .env.example to .env and fill it in.")
    return value


def datasets_dir():
    return resolve(_required("CLEF_DATASETS_DIR"))


def artifacts_dir():
    return resolve(_required("CLEF_ARTIFACTS_DIR"), create=True)


def venvs_dir():
    return os.path.expanduser(get("CLEF_VENVS_DIR", os.path.join(REPO_ROOT, ".venvs")))


def dataset_dir(name):
    if name not in DATASETS:
        raise ConfigError(f"Unknown dataset '{name}'. Choose from: {', '.join(DATASETS)}")
    return os.path.join(datasets_dir(), DATASETS[name])


def ground_truth(dataset, gt_path=None):
    """Ordered {ID: caption} to evaluate against.

    Default: the official valid rows ('_valid_' in the ID) of the dataset's
    captions.csv, which mixes train and valid. gt_path overrides it with any
    captions.csv-shaped file (e.g. a cropped GT for quick runs).
    """
    path = gt_path or os.path.join(dataset_dir(dataset), "captions.csv")
    gt = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if row and (gt_path or "_valid_" in row[0]):
                gt[row[0]] = row[1]
    if not gt:
        raise ConfigError(f"No ground-truth rows found in {path}")
    return gt


def embeddings_cache(dataset):
    return os.path.join(artifacts_dir(), "precomputed", f"image_embeddings_{dataset}.npz")


def venv_python(name):
    return os.path.join(venvs_dir(), name, "bin", "python")


def medcat_zip():
    zips = sorted(glob.glob(os.path.join(artifacts_dir(), "MedCAT", "*.zip")))
    if not zips:
        raise ConfigError(f"No MedCAT model pack (.zip) in {artifacts_dir()}/MedCAT")
    return zips[0]


def export_cache_env():
    """Keep HF / NLTK downloads in the artifacts dir; inherited by subprocesses."""
    artifacts = artifacts_dir()
    os.environ.setdefault("HF_HOME", os.path.join(artifacts, "hf"))
    os.environ.setdefault("NLTK_DATA", os.path.join(artifacts, "nltk_data"))


if __name__ == "__main__":
    getters = {"artifacts": artifacts_dir, "venvs": venvs_dir, "datasets": datasets_dir}
    if len(sys.argv) != 2 or sys.argv[1] not in getters:
        print(f"Usage: python3 config.py {'|'.join(getters)}", file=sys.stderr)
        sys.exit(2)
    try:
        print(getters[sys.argv[1]]())
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)
