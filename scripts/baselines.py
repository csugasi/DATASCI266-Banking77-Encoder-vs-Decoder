# Project: Labels vs. Scale: Breaking Even on Banking77 with Encoders and Decoders

"""Baselines that need no pretrained model.

  majority : for every data-size point and seed, predict the most frequent intent of that
             training subset for every example.
  tfidf_lr : TF-IDF features + logistic regression, trained on each training subset.
             scikit-learn defaults for both parts (no hyperparameter tuning), except
             max_iter=1000 so the solver can finish. A check of C may be added later.

Files are written in the shared prediction format (results/PREDICTION_FORMAT.md)
through predictions.write_predictions.

Usage (from the project root):
    python scripts/baselines.py                              # both baselines, val and test
    python scripts/baselines.py --models tfidf_lr            # one baseline
    python scripts/baselines.py --models majority --splits test
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline

import evaluation as ev
from predictions import write_predictions
DIAGNOSTICS = ev.RESULTS / "diagnostics"   # Extra outputs that are not prediction files

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def budget_files(manifest):
    """(budget name, file name template) for every data-size point, from the split manifest."""
    budgets = [(f"k{k}", f"train_k{k}_seed{{seed}}.parquet") for k in manifest["k_values"]]
    tag = round(manifest["prop_fraction"] * 100)
    budgets.append((f"p{tag}", f"train_p{tag}_seed{{seed}}.parquet"))
    budgets.append(("full", "full_train.parquet"))     # same file for every seed
    return budgets


def split_data(split):
    """The val or test examples (text and label), in the same order as the prediction files."""
    path = {"val": ev.SPLITS / "val.parquet", "test": ev.RAW / "test.parquet"}[split]
    df = pd.read_parquet(path)
    # Predictions must line up with example_id, so the labels must equal the ones evaluation.py uses
    assert (df["label"].to_numpy() == ev.split_labels(split)).all(), f"Order mismatch in split '{split}'"
    return df


# ---------------------------------------------------------------------------
# Majority class
# ---------------------------------------------------------------------------
def majority_label(train_df):
    """Most frequent intent in a training subset. Ties go to the smallest label id (deterministic)."""
    counts = train_df["label"].value_counts().sort_index()   # sort_index makes idxmax pick the smallest id on ties
    return int(counts.idxmax())


def run_majority(splits):
    manifest = json.load(open(ev.SPLITS / "manifest.json"))
    written = 0
    for budget, template in budget_files(manifest):
        for seed in manifest["seeds"]:
            train_df = pd.read_parquet(ev.SPLITS / template.format(seed=seed))
            label = majority_label(train_df)
            for split in splits:
                # The baseline predicts the same intent for every example
                predictions = np.full(len(ev.split_labels(split)), label)
                write_predictions("majority", budget, seed, split, predictions)
                written += 1
    print("Majority: prediction files written:", written, "| Folder:", ev.RESULTS)


# ---------------------------------------------------------------------------
# TF-IDF + logistic regression
# ---------------------------------------------------------------------------
def make_tfidf_lr():
    """TF-IDF features into a logistic regression. scikit-learn defaults; only max_iter is raised."""
    return make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))


def run_tfidf_lr(splits):
    manifest = json.load(open(ev.SPLITS / "manifest.json"))
    data = {split: split_data(split) for split in splits}   # Read the val/test texts once
    val_df = data["val"] if "val" in data else split_data("val")   # Validation is always needed for the loss
    fitted = {}       # File name -> (fitted model, number of training examples, train loss, validation loss)
    loss_rows = []    # One row per budget and seed, saved at the end
    written = 0
    for budget, template in budget_files(manifest):
        for seed in manifest["seeds"]:
            filename = template.format(seed=seed)

            # The full training set is the same file for every seed, so fit it only once
            if filename not in fitted:
                train_df = pd.read_parquet(ev.SPLITS / filename)
                model = make_tfidf_lr().fit(train_df["text"], train_df["label"])

                # Log-loss = average of -ln(probability given to the true intent); lower is better
                train_loss = log_loss(train_df["label"], model.predict_proba(train_df["text"]), labels=model.classes_)
                val_loss = log_loss(val_df["label"], model.predict_proba(val_df["text"]), labels=model.classes_)
                fitted[filename] = (model, len(train_df), train_loss, val_loss)
            model, n_train, train_loss, val_loss = fitted[filename]

            loss_rows.append({"model": "tfidf_lr", "budget": budget, "seed": seed,
                              "n_train": n_train, "train_loss": train_loss, "val_loss": val_loss})

            for split in splits:
                predictions = model.predict(data[split]["text"])
                write_predictions("tfidf_lr", budget, seed, split, predictions)
                written += 1
        print("Finished budget:", budget)

    # The losses are not predictions, so they go in a subfolder that score_all() does not scan
    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    loss_path = DIAGNOSTICS / "tfidf_lr_loss_by_budget.csv"
    pd.DataFrame(loss_rows).to_csv(loss_path, index=False)
    print("TF-IDF + LR: prediction files written:", written, "| Folder:", ev.RESULTS)
    print("Loss file written:", loss_path)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
RUNNERS = {"majority": run_majority, "tfidf_lr": run_tfidf_lr}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(RUNNERS), choices=list(RUNNERS))
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=list(ev.SPLIT_NAMES))
    args = parser.parse_args()
    for name in args.models:
        RUNNERS[name](args.splits)
