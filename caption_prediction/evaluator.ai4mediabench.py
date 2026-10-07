import os
import sys
import csv
import json
import string
import subprocess
import numpy as np
import re
import zipfile
import evaluate
from tqdm import tqdm
from bert_score import BERTScorer
from medcat_scorer import MedCatScorer
import torch
from bleurt_pytorch import (
    BleurtConfig,
    BleurtForSequenceClassification,
    BleurtTokenizer,
)

current_dir = os.path.dirname(os.path.abspath(__file__))

# Venv paths for isolated scorers
ALIGNSCORE_PYTHON = "/venv/alignscore/bin/python"
MEDIMAGEINSIGHTS_PYTHON = "/venv/medimageinsights/bin/python"

# AlignScore checkpoint: check local first, then Docker location
ALIGNSCORE_CKPT = os.path.join(current_dir, "models/AlignScore/AlignScore-base.ckpt")
if not os.path.exists(ALIGNSCORE_CKPT):
    ALIGNSCORE_CKPT = "/app/models/AlignScore/AlignScore-base.ckpt"

# MedImageInsights model dir: check local first, then Docker location
MEDIMAGEINSIGHTS_MODEL_DIR = os.path.join(current_dir, "MedImageInsights/2024.09.27")
if not os.path.exists(MEDIMAGEINSIGHTS_MODEL_DIR):
    MEDIMAGEINSIGHTS_MODEL_DIR = "/app/MedImageInsights/2024.09.27"

# Path to the subprocess runner scripts (named *.ai4mediabench.py in the repo,
# renamed when packed into scoring_program.zip)
def _runner(name):
    path = os.path.join(current_dir, f"{name}.py")
    return path if os.path.exists(path) else os.path.join(current_dir, f"{name}.ai4mediabench.py")


ALIGNSCORE_RUNNER = _runner("run_alignscore")
MEDIMAGEINSIGHTS_RUNNER = _runner("run_medimageinsights")


class CaptionEvaluator:
    case_sensitive = False

    def __init__(
        self,
        ground_truth_path="/app/data/valid/captions.csv",
        dataset_type="valid",
        gt=None,
        embeddings_path=None,
        medcat_zip=None,
        alignscore_python=ALIGNSCORE_PYTHON,
        alignscore_ckpt=ALIGNSCORE_CKPT,
        medimageinsights_python=MEDIMAGEINSIGHTS_PYTHON,
        medimageinsights_model_dir=MEDIMAGEINSIGHTS_MODEL_DIR,
    ):
        """gt ({ID: caption}) and the path arguments let a local run (see config.py)
        replace the Docker layout; the defaults keep the AI4MediaBench behaviour."""
        print("Initializing evaluator...")
        self.ground_truth_path = ground_truth_path
        self.dataset_type = dataset_type
        self.gt = gt if gt is not None else self.load_gt()
        self.embeddings_path = embeddings_path or os.path.join(
            current_dir, "precomputed", f"image_embeddings_{dataset_type}.npz"
        )
        self.alignscore_python = alignscore_python
        self.alignscore_ckpt = alignscore_ckpt
        self.medimageinsights_python = medimageinsights_python
        self.medimageinsights_model_dir = medimageinsights_model_dir
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.text_batch_size = int(os.environ.get("TEXT_BATCH_SIZE", "8"))
        print("Loading ROUGE from HuggingFace")
        self.scorers = {
            "rouge": (evaluate.load("rouge"),),
        }
        self.medcat_model_path = medcat_zip or os.path.join(
            current_dir,
            [
                os.path.join("models/MedCAT", filename)
                for filename in os.listdir("models/MedCAT")
                if filename.endswith(".zip")
            ][0],
        )
        self._check_medcat_pack()
        # Loaded in compute_medcats, after the GPU scorers are freed: the UMLS
        # pack takes ~14 GB RAM and would otherwise be held through every metric.
        self.medcat_scorer = None
        self.bert_scorer = None
        self.bleurt_model = None
        self.bleurt_tokenizer = None
        self.bleurt_config = None
        self.image_similarity_scorer = None
        self._image_embeddings = None

    def _evaluate(self, client_payload, _context={}):
        print("Evaluating...")
        submission_file_path = client_payload["submission_file_path"]
        predictions = self.load_predictions(submission_file_path)

        print("Compute BERTScore")
        bertscore = self.compute_bertscore(predictions)
        print("BERTScore:", bertscore)
        print("Compute AlignScore")
        alignscore = self.compute_alignscore(predictions)
        print("AlignScore:", alignscore)
        print("Compute ROUGE")
        rouge = self.compute_rouge(predictions)
        print("ROUGE:", rouge)
        print("Compute Image-Caption Similarity")
        sim = self.compute_similarity(predictions)
        print("Similarity:", sim)
        print("Compute BLEURT")
        bleurt = self.compute_bleurt(predictions)
        print("BLEURT:", bleurt)
        print("Compute MedCAT")
        medcats = self.compute_medcats(predictions)
        print("Medcats:", medcats)

        relevance = np.mean([bertscore, rouge, sim, bleurt])
        factuality = np.mean([medcats, alignscore])

        _result_object = {
            "score": relevance,
            "score_secondary": factuality,
            "bert": bertscore,
            "rouge": rouge,
            "similarity": sim,
            "bleurt": bleurt,
            "medcat": medcats,
            "align": alignscore,
        }
        print(
            "Similarity,BERTScore,ROUGE,BLEURT,Relevance,Medcats,AlignScore,Factuality\n"
            + "{},{},{},{},{},{},{},{}\n".format(
                sim,
                bertscore,
                rouge,
                bleurt,
                relevance,
                medcats,
                alignscore,
                factuality,
            )
        )

        assert "score" in _result_object
        assert "score_secondary" in _result_object

        return _result_object

    def load_gt(self):
        print("Loading ground truth...")
        pairs = {}
        with open(self.ground_truth_path) as csvfile:
            reader = csv.reader(csvfile)
            first_line = next(reader)
            # Check if it's a header
            if first_line and first_line[0].lower() != "id":
                # Process the first line if it's not a header
                pairs[first_line[0]] = first_line[1]
            for row in tqdm(reader):
                pairs[row[0]] = row[1]
        return pairs

    def load_predictions(self, submission_file_path):
        print("Loading predictions...")
        pairs = {}
        image_ids_gt = set(self.gt.keys())
        occured_images = set()
        with open(submission_file_path) as csvfile:
            reader = csv.reader(csvfile)
            first_line = next(reader)
            # Check if it's a header
            if first_line and first_line[0].lower() != "id":
                # Process the first line if it's not a header
                pairs[first_line[0]] = first_line[1]
            for lineCnt, row in enumerate(tqdm(reader)):
                if len(row) < 2:
                    self.raise_exception(
                        "Wrong format: Each line must consist of an image ID followed by a ',' (comma) and a caption ({}).",
                        lineCnt,
                        "<imageID><comma><caption>",
                    )
                image_id = row[0]
                if image_id not in image_ids_gt:
                    self.raise_exception(
                        "Image ID '{}' in submission file does not exist in testset.",
                        lineCnt,
                        image_id,
                    )
                if image_id in occured_images:
                    self.raise_exception(
                        "Image ID '{}' was specified more than once in submission file.",
                        lineCnt,
                        image_id,
                    )
                pairs[image_id] = row[1]
                occured_images.add(image_id)
        if len(occured_images) != len(image_ids_gt):
            self.raise_exception(
                f"Number of image IDs in submission file not equal to number of image IDs in testset.\nNumber in testset: {len(image_ids_gt)}\nNumber in submission: {len(occured_images)}",
                lineCnt,
            )
        print("Number of image IDs in submission file: {}".format(len(occured_images)))
        return pairs

    def raise_exception(self, message, record_count, *args):
        raise Exception(
            message.format(*args)
            + " Error occurred at record line {}.".format(record_count)
        )

    def preprocess_caption(self, caption):
        translator = str.maketrans("", "", string.punctuation)
        number_regex = re.compile(r"\d+")
        if not type(self).case_sensitive:
            caption = caption.lower()
        caption = number_regex.sub("number", caption)
        caption = caption.translate(translator)
        return caption

    def compute_bertscore(self, candidate_pairs):
        print("Computing BERTScore")
        if self.bert_scorer is None:
            idf_sentences = [
                self.preprocess_caption(caption) for caption in self.gt.values()
            ]
            self.bert_scorer = BERTScorer(
                model_type="microsoft/deberta-xlarge-mnli",
                idf=True,
                idf_sents=idf_sentences,
                device=self.device,
            )
        bert_scores = [
            (
                self.bert_scorer.score(
                    cands=[self.preprocess_caption(candidate_pairs[image_key])],
                    refs=[self.preprocess_caption(self.gt[image_key])],
                )[2].item()
                if len(self.gt[image_key]) != 0 or len(candidate_pairs[image_key]) != 0
                else 1
            )
            for image_key in candidate_pairs
        ]
        if self.device == "cuda":
            self.bert_scorer = None
            self._free_cuda()
        return np.mean(bert_scores)

    def compute_rouge(self, candidate_pairs):
        print("Computing ROUGE")
        rouge_scores = [
            (
                self.scorers["rouge"][0].compute(
                    predictions=[self.preprocess_caption(candidate_pairs[image_key])],
                    references=[self.preprocess_caption(self.gt[image_key])],
                    use_aggregator=False,
                    use_stemmer=False,
                )["rouge1"]
                if len(self.gt[image_key]) != 0 or len(candidate_pairs[image_key]) != 0
                else 1
            )
            for image_key in candidate_pairs
        ]
        return np.mean(rouge_scores)

    def compute_alignscore(self, candidate_pairs):
        print("Computing Alignscore")
        contexts = []
        claims = []
        empty_indices = []
        for idx, image_key in enumerate(candidate_pairs):
            if len(self.gt[image_key]) == 0 and len(candidate_pairs[image_key]) == 0:
                empty_indices.append(idx)
                contexts.append("")
                claims.append("")
            else:
                contexts.append(self.gt[image_key])
                claims.append(candidate_pairs[image_key])

        payload = json.dumps({
            "contexts": contexts,
            "claims": claims,
            "ckpt_path": self.alignscore_ckpt,
            "device": self.device,
        })

        result = subprocess.run(
            [self.alignscore_python, ALIGNSCORE_RUNNER],
            input=payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(f"AlignScore subprocess failed:\n{result.stderr}")

        scores = json.loads(result.stdout)["scores"]
        for idx in empty_indices:
            scores[idx] = 1.0

        return np.mean(scores)

    def _check_medcat_pack(self):
        # Cheap early failure (missing/corrupt pack) without loading the model
        path = self.medcat_model_path
        if not zipfile.is_zipfile(path):
            raise Exception(f"MedCAT model pack missing or not a valid zip: {path}")
        names = set(zipfile.ZipFile(path).namelist())
        missing = {"cdb.dat", "vocab.dat"} - names
        if missing:
            raise Exception(f"MedCAT model pack {path} lacks {sorted(missing)}")

    def compute_medcats(self, candidate_pairs):
        print("Computing MEDCATS")
        if self.medcat_scorer is None:
            print("Loading MedCatScorer")
            self.medcat_scorer = MedCatScorer(model_path=self.medcat_model_path)
        medcat_scores = []
        for image_key in candidate_pairs:
            if len(self.gt[image_key]) != 0 or len(candidate_pairs[image_key]) != 0:
                score = self.medcat_scorer.score(
                    self.gt[image_key], candidate_pairs[image_key]
                )
            else:
                score = 1
            medcat_scores.append(score)
        return np.mean(medcat_scores)

    def _ensure_image_embeddings(self):
        if self._image_embeddings is not None:
            return
        cache_path = self.embeddings_path
        if not os.path.exists(cache_path):
            raise Exception(
                f"Precomputed image embeddings not found at {cache_path}. Rebuild the image to generate them."
            )
        data = np.load(cache_path)
        self._image_embeddings = {k: data[k] for k in data.files}

    def _encode_texts(self, texts):
        print("Encoding captions via MedImageInsights subprocess")
        payload = json.dumps({
            "texts": texts,
            "model_dir": self.medimageinsights_model_dir,
            "batch_size": max(1, self.text_batch_size),
        })

        result = subprocess.run(
            [self.medimageinsights_python, MEDIMAGEINSIGHTS_RUNNER],
            input=payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(f"MedImageInsights subprocess failed:\n{result.stderr}")

        parsed = self._parse_subprocess_json(
            result.stdout,
            result.stderr,
            process_name="MedImageInsights",
        )
        if "embeddings" not in parsed:
            raise RuntimeError(
                "MedImageInsights subprocess response did not contain 'embeddings'. "
                f"Response keys: {list(parsed.keys())}"
            )
        embeddings = parsed["embeddings"]
        return np.array(embeddings)

    def _parse_subprocess_json(self, stdout, stderr, process_name):
        raw = (stdout or "").strip()
        if not raw:
            stderr_preview = (stderr or "").strip()[:1000]
            raise RuntimeError(
                f"{process_name} subprocess produced empty stdout while JSON was expected. "
                f"stderr preview: {stderr_preview}"
            )

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Some libraries may print logs before the JSON payload.
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            if lines:
                try:
                    return json.loads(lines[-1])
                except json.JSONDecodeError:
                    pass

            stdout_preview = raw[:1000]
            stderr_preview = (stderr or "").strip()[:1000]
            raise RuntimeError(
                f"{process_name} subprocess returned non-JSON stdout. "
                f"stdout preview: {stdout_preview} | stderr preview: {stderr_preview}"
            )

    def compute_similarity(self, candidate_pairs):
        print("Computing MedImageInsights Similarity")
        self._load_image_similarity_scorer()
        self._ensure_image_embeddings()

        missing = [
            image_key
            for image_key in candidate_pairs
            if image_key not in self._image_embeddings
        ]
        if missing:
            raise Exception(
                f"Missing precomputed embeddings for image IDs: {', '.join(missing)}"
            )

        captions = list(candidate_pairs.values())
        text_embeddings = self._encode_texts(captions)

        sim_scores = []
        for idx, image_key in enumerate(candidate_pairs):
            candidate_caption = candidate_pairs[image_key]
            try:
                if len(candidate_caption) == 0:
                    print("Candidate caption is empty")
                    score = 0
                else:
                    with torch.no_grad():
                        v = self._image_embeddings[image_key]
                        c = text_embeddings[idx]
                        w = 2.5
                        cos = np.dot(c, v) / (np.linalg.norm(c) * np.linalg.norm(v))
                        score = w * np.max([cos, 0])
            except Exception as e:
                print(e)
                score = 1
            sim_scores.append(score)
        if self.device == "cuda":
            self.image_similarity_scorer = None
            self._free_cuda()
        return np.mean(sim_scores)

    def _load_image_similarity_scorer(self):
        # MedImageInsights runs as a subprocess in its own venv
        # No scorer object needed in main process
        pass

    def compute_bleurt(self, candidate_pairs):
        print("Computing BLEURT")
        if self.bleurt_model is None or self.bleurt_tokenizer is None:
            self.bleurt_config = BleurtConfig.from_pretrained(
                "lucadiliello/BLEURT-20-D12"
            )
            self.bleurt_model = BleurtForSequenceClassification.from_pretrained(
                "lucadiliello/BLEURT-20-D12"
            )
            self.bleurt_tokenizer = BleurtTokenizer.from_pretrained(
                "lucadiliello/BLEURT-20-D12"
            )
            self.bleurt_model.to(self.device)
        references = [
            self.preprocess_caption(self.gt[image_key]) for image_key in candidate_pairs
        ]
        candidates = [
            self.preprocess_caption(candidate_pairs[image_key])
            for image_key in candidate_pairs
        ]
        self.bleurt_model.eval()
        scores = []
        bs = max(1, self.text_batch_size)
        for start in tqdm(
            range(0, len(references), bs), desc="BLEURT batches", unit="batch"
        ):
            refs = references[start : start + bs]
            cands = candidates[start : start + bs]
            with torch.inference_mode():
                inputs = self.bleurt_tokenizer(
                    refs,
                    cands,
                    padding="longest",
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                batch_scores = (
                    self.bleurt_model(**inputs).logits.flatten().cpu().tolist()
                )
            scores.extend(batch_scores)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if torch.cuda.is_available() and self.device == "cuda":
            self.bleurt_model = None
            self.bleurt_tokenizer = None
            self.bleurt_config = None
            self._free_cuda()
        return np.mean(scores)

    def _free_cuda(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys

    # Default to 'valid' if no argument is provided
    dataset_type = "valid"
    if len(sys.argv) > 1:
        dataset_type = sys.argv[1].lower()
        if dataset_type not in ["valid", "test"]:
            print("Error: Argument must be either 'valid' or 'test'")
            sys.exit(1)

    print(f"Running evaluator for {dataset_type} dataset...")
    ground_truth_path = os.path.join(current_dir, f"data/{dataset_type}/captions.csv")
    submission_file_path = f"/app/data/{dataset_type}/submission.csv"

    if not os.path.exists(ground_truth_path):
        print(f"Error: Ground truth file not found at {ground_truth_path}")
        sys.exit(1)

    _client_payload = {"submission_file_path": submission_file_path}
    _context = {}
    caption_evaluator = CaptionEvaluator(ground_truth_path)
    result = caption_evaluator._evaluate(_client_payload, _context)
    print(result)
