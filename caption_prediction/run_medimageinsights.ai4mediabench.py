#!/usr/bin/env python3
"""
Standalone MedImageInsights text encoder.
Runs inside /venv/medimageinsights.
Called as a subprocess by evaluator.py.

Input  (stdin): JSON {"texts": [...], "model_dir": "..."}
Output (stdout): JSON {"embeddings": [[...], ...]}
"""
import sys
import json
import os
import contextlib

def main():
    payload = json.loads(sys.stdin.read())
    texts = payload["texts"]
    model_dir = payload["model_dir"]

    with contextlib.redirect_stdout(sys.stderr):
        # model_dir is <MedImageInsights repo>/2024.09.27
        sys.path.insert(0, os.path.dirname(os.path.normpath(model_dir)))
        from medimageinsightmodel import MedImageInsight

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with contextlib.redirect_stdout(sys.stderr):
        scorer = MedImageInsight(
            model_dir=model_dir,
            vision_model_name="medimageinsigt-v1.0.0.pt",
            language_model_name="language_model.pth",
        )
        scorer.load_model()
        if hasattr(scorer, "device"):
            scorer.device = device
        if hasattr(scorer, "to"):
            try:
                scorer.to(device)
            except Exception:
                pass

        batch_size = payload.get("batch_size", 8)
        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            with torch.inference_mode():
                outputs = scorer.encode(texts=batch)
            if isinstance(outputs, dict) and "text_embeddings" in outputs:
                embeddings = outputs["text_embeddings"]
            else:
                embeddings = outputs
            if hasattr(embeddings, "detach"):
                embeddings = embeddings.detach().cpu().numpy()
            all_embeddings.extend(embeddings.tolist())

    print(json.dumps({"embeddings": all_embeddings}), flush=True)

if __name__ == "__main__":
    main()
