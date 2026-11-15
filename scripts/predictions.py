# Project: Labels vs. Scale: Breaking Even on Banking77 with Encoders and Decoders

"""What every model script calls to produce its prediction file.

  write_predictions(...)  writes ONE CSV in the shared format (results/PREDICTION_FORMAT.md)
                          and re-checks it with evaluation.load_predictions.
  match_intent(text)      turns a decoder's generated text into an intent id (-1 if no match),
                          so every decoder (Qwen zero-shot, Qwen QLoRA, frontier) is matched by
                          the SAME rule. evaluation.py never calls it: it only sees the finished CSVs.

This file depends on evaluation.py (for the format rules); evaluation.py does not depend on this file.

Typical use at the end of a run:
    import sys; sys.path.insert(0, "scripts")       # once, if scripts/ is not already importable
    from predictions import write_predictions, match_intent

    write_predictions("modernbert", "k5", 0, "test", predicted_labels)
    predicted = [match_intent(t) for t in generated_texts]
    write_predictions("qwen_qlora", "k5", 0, "test", predicted, raw_outputs=generated_texts)
"""
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import evaluation as ev


# ---------------------------------------------------------------------------
# Matching decoder text to an intent
# ---------------------------------------------------------------------------
def _normalize(text):
    """Lowercase; drop surrounding whitespace, quotes, backticks and trailing . or ?"""
    return re.sub(r"[.?]+$", "", text.strip().strip("\"'`").strip()).lower()


@lru_cache(maxsize=None)
def _name_to_id():
    mapping = {_normalize(name): i for i, name in enumerate(ev.label_names())}
    assert len(mapping) == len(ev.label_names()), "Two intent names collide after normalizing"
    return mapping


def match_intent(text):
    """Intent id for a decoder's generated text, or -1 if it is not exactly an intent name.

    Matching is strict on purpose: only case, quotes and a trailing '.' or '?' are ignored.
    Any other text (extra words, explanations) counts as unmatched, i.e. wrong.
    """
    return _name_to_id().get(_normalize(str(text)), ev.UNMATCHED)


# ---------------------------------------------------------------------------
# Writing a prediction file
# ---------------------------------------------------------------------------
def write_predictions(model, budget, seed, split, predicted_labels, raw_outputs=None, out_dir=None):
    """Write one prediction file in the shared format and check it before returning.

    You pass only what the model produced, in the order of the split file:
      predicted_labels : integers 0-76, or -1 for an unmatched decoder answer
      raw_outputs      : the generated text (decoders only; leave None for other models)
    example_id, true_label and type are filled in here, so they cannot be misaligned.
    Returns the path of the file.
    """
    if model not in ev.MODEL_TYPE:
        raise ValueError(f"Unknown model '{model}'. Allowed: {sorted(ev.MODEL_TYPE)}")
    if split not in ev.SPLIT_NAMES:
        raise ValueError(f"Unknown split '{split}'. Allowed: {ev.SPLIT_NAMES}")

    mtype = ev.MODEL_TYPE[model]
    true = ev.split_labels(split)
    predicted = np.asarray(predicted_labels)
    if not np.issubdtype(predicted.dtype, np.integer):
        raise ValueError("predicted_labels must be integers (use match_intent for decoder text)")
    if len(predicted) != len(true):
        raise ValueError(f"Expected {len(true)} predictions for split '{split}', got {len(predicted)}")

    if mtype == "decoder":
        if raw_outputs is None:
            raise ValueError("Decoder runs must pass raw_outputs (the generated text)")
        raw = [str(t) for t in raw_outputs]
        if len(raw) != len(true):
            raise ValueError(f"Expected {len(true)} raw_outputs, got {len(raw)}")
    else:
        if raw_outputs is not None:
            raise ValueError(f"'{model}' is not a decoder: leave raw_outputs as None")
        raw = [""] * len(true)

    df = pd.DataFrame({
        "model": model, "type": mtype, "budget": budget, "seed": seed, "split": split,
        "example_id": np.arange(len(true)), "true_label": true,
        "predicted_label": predicted, "raw_output": raw,
    })

    out_dir = Path(out_dir) if out_dir else ev.RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model}_{budget}_seed{seed}_{split}.csv"
    df.to_csv(path, index=False)

    # Re-read the file with the full checks; remove it if it breaks the format
    try:
        ev.load_predictions(path)
    except ValueError:
        path.unlink()
        raise
    return path
