#!/usr/bin/env bash
# Native (non-Docker) setup mirroring Dockerfile.ai4mediabench:
# three isolated venvs because their pins conflict
#   main             BERTScore, ROUGE, BLEURT, MedCAT   (torch 2.1.2, transformers 4.49)
#   alignscore       AlignScore                         (torch 1.13.1, transformers 4.20.1)
#   medimageinsights MedImageInsights text/image encoder (torch 2.1.2, transformers 4.16.2)
# Exact package versions come from locks/<venv>.txt; model downloads are pinned to
# revisions and prefetched, so evaluation needs no internet afterwards.
# Paths come from <repo>/.env (see .env.example). Safe to re-run.
#
# Regenerating the locks (only if you deliberately upgrade): install from the loose
# specs, test, then freeze. Known traps when doing so:
#   main:  add "pyarrow<21" (medcat 1.x -> datasets 2.x uses pa.PyExtensionType)
#   all:   add "setuptools<81" (medcat / pytorch-lightning 1.x import pkg_resources)
#   uv pip install --python .venvs/main -r requirements.txt "pyarrow<21" "setuptools<81"
#   uv pip install --python .venvs/alignscore "alignscore @ git+https://github.com/yuh-zha/AlignScore.git" \
#       "transformers==4.20.1" "setuptools<81" && .venvs/alignscore/bin/python -m spacy download en_core_web_sm
#   uv pip install --python .venvs/medimageinsights "numpy<2" torch==2.1.2 torchvision==0.16.2 \
#       -r <artifacts>/MedImageInsights/requirements.txt "setuptools<81"
#   uv pip freeze --python .venvs/<venv> > locks/<venv>.txt
set -euo pipefail

cd "$(dirname "$0")"
ARTIFACTS=$(python3 config.py artifacts)
VENVS=$(python3 config.py venvs)
export HF_HOME="$ARTIFACTS/hf"
export NLTK_DATA="$ARTIFACTS/nltk_data"
PY=3.10  # Python of the official image (ubuntu 22.04); uv downloads it if missing
MEDIMAGEINSIGHTS_REV=8b332dbcc542ca6b84b955269e547533960ab197
ALIGNSCORE_REV=8509e78d25bb914939fc585c626500c9b2944249

for venv in main alignscore medimageinsights; do
    echo "== $venv venv"
    [ -x "$VENVS/$venv/bin/python" ] || uv venv --python "$PY" "$VENVS/$venv"
    uv pip sync --python "$VENVS/$venv" "locks/$venv.txt"
done

echo "== model weights -> $ARTIFACTS"
"$VENVS/main/bin/python" -c "
from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download('lion-ai/MedImageInsights', revision='$MEDIMAGEINSIGHTS_REV',
                  local_dir='$ARTIFACTS/MedImageInsights')
hf_hub_download('yzha/AlignScore', 'AlignScore-base.ckpt', revision='$ALIGNSCORE_REV',
                local_dir='$ARTIFACTS/AlignScore')"

echo "== prefetch models the scorers download at runtime (loads each once on CPU)"
"$VENVS/main/bin/python" -c "
import nltk; nltk.download('punkt_tab'); nltk.download('punkt')
import evaluate; evaluate.load('rouge')
from bert_score import BERTScorer; BERTScorer(model_type='microsoft/deberta-xlarge-mnli', device='cpu')
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20-D12')
BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20-D12')
print('main models OK')"
"$VENVS/alignscore/bin/python" -c "
from alignscore import AlignScore
AlignScore(model='roberta-large', batch_size=32, device='cpu', evaluation_mode='nli_sp', verbose=False,
           ckpt_path='$ARTIFACTS/AlignScore/AlignScore-base.ckpt')
print('alignscore OK')"
"$VENVS/medimageinsights/bin/python" -c "
import sys; sys.path.insert(0, '$ARTIFACTS/MedImageInsights')
from medimageinsightmodel import MedImageInsight; print('medimageinsights OK')"

ls "$ARTIFACTS"/MedCAT/*.zip >/dev/null 2>&1 \
    || echo "WARNING: put the UMLS MedCAT .zip in $ARTIFACTS/MedCAT/ (licensed, not downloadable here)"
echo "Setup done."
