#!/usr/bin/env python3
"""
Standalone AlignScore runner.
Runs inside /venv/alignscore.
Called as a subprocess by evaluator.py.

Input  (stdin): JSON {"contexts": [...], "claims": [...], "ckpt_path": "..."}
Output (stdout): JSON {"scores": [...]}
"""
import sys
import json

def main():
    payload = json.loads(sys.stdin.read())
    contexts = payload["contexts"]
    claims = payload["claims"]
    ckpt_path = payload["ckpt_path"]
    device = payload.get("device", "cpu")

    from alignscore import AlignScore

    scorer = AlignScore(
        model="roberta-large",
        batch_size=32,
        device=device,
        ckpt_path=ckpt_path,
        evaluation_mode="nli_sp",
        verbose=False,
    )

    scores = scorer.score(contexts=contexts, claims=claims)
    print(json.dumps({"scores": scores}))

if __name__ == "__main__":
    main()
