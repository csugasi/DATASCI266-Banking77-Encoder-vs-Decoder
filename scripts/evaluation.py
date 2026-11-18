# Project: Labels vs. Scale: Breaking Even on Banking77 with Encoders and Decoders

"""Evaluation of model predictions.

Reads the prediction CSVs in results/ (format: results/PREDICTION_FORMAT.md), checks that every
file follows the format, scores every run the same way, and builds the per-intent view, the seed
summary and the learning curves.

This module only EVALUATES files that already exist. It never writes predictions and never
interprets model output; predictions.py (in this folder) does the writing.

If the prediction format changes, change the constants below AND results/PREDICTION_FORMAT.md.

Typical use in a notebook:
    import sys; sys.path.insert(0, "scripts")       # once, if scripts/ is not already importable
    import evaluation as ev

    scores = ev.score_all("test")                    # check and score every run
    summary = ev.summarize(scores)
    ev.plot_learning_curve(summary, "accuracy")
"""
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# ---------------------------------------------------------------------------
# Paths (this file lives in <project root>/src/)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"            # prediction CSVs live here
SPLITS = ROOT / "data" / "splits"     # val.parquet, subsets, manifest.json
RAW = ROOT / "data" / "raw"           # test.parquet, label_names.json

# ---------------------------------------------------------------------------
# Format rules (results/PREDICTION_FORMAT.md written as code)
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "model", "type", "budget", "seed", "split",
    "example_id", "true_label", "predicted_label", "raw_output",
]

MODEL_TYPE = {
    "modernbert": "encoder",
    "qwen_zeroshot": "decoder",
    "qwen_qlora": "decoder",
    "qwen_sft": "decoder",
    "frontier": "decoder",
    "tfidf_lr": "baseline",
    "majority": "baseline",
}

BUDGETS = ["zeroshot", "k1", "k3", "k5", "k10", "k20", "k30", "p50", "full"]
SEEDS = [0, 1, 2]
SPLIT_NAMES = ("val", "test")
N_LABELS = 77
UNMATCHED = -1                                   # decoder output that matched no intent
ZERO_SHOT_MODELS = ("qwen_zeroshot", "frontier")  # the only models allowed the 'zeroshot' budget


# ---------------------------------------------------------------------------
# Data the checks and metrics depend on (loaded once, on first use)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def split_labels(split):
    """True labels of a split, in file order. example_id is the position in this array."""
    path = {"val": SPLITS / "val.parquet", "test": RAW / "test.parquet"}[split]
    return pd.read_parquet(path)["label"].to_numpy()


@lru_cache(maxsize=None)
def label_names():
    """Intent names, indexed by label id."""
    return json.load(open(RAW / "label_names.json"))


def _names():
    """Label id -> name, with -1 shown as 'unmatched'."""
    names = dict(enumerate(label_names()))
    names[UNMATCHED] = "unmatched"
    return names


def budget_sizes():
    """Number of training examples behind each budget, read from data/splits/manifest.json."""
    manifest = json.load(open(SPLITS / "manifest.json"))
    sizes = {f"k{k}": k * manifest["n_intents"] for k in manifest["k_values"]}
    sizes[f"p{round(manifest['prop_fraction'] * 100)}"] = manifest["n_prop_subset"]
    sizes["full"] = manifest["n_full_train"]
    return sizes


# ---------------------------------------------------------------------------
# Checking prediction files
# ---------------------------------------------------------------------------
def load_predictions(path):
    """Read one prediction CSV and check it follows the format.

    Returns the DataFrame (sorted by example_id) if every check passes.
    Otherwise raises a ValueError that lists every problem found.
    """
    path = Path(path)
    df = pd.read_csv(path, keep_default_na=False)   # keep an empty raw_output as "", not NaN

    # 1. All required columns are present (stop here if not: the other checks need them)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    # Number columns must hold whole numbers (stop here if not: the other checks need them)
    for col in ["seed", "example_id", "true_label", "predicted_label"]:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any() or (values % 1 != 0).any():
            raise ValueError(f"{path.name}: column '{col}' must hold whole numbers only")
        df[col] = values.astype(int)

    # 2. One file = one run: a single model, type, budget, seed and split
    problems = []
    for col in ["model", "type", "budget", "seed", "split"]:
        if df[col].nunique() != 1:
            problems.append(f"'{col}' should have one value per file, found {sorted(df[col].unique())}")
    if problems:
        raise ValueError(f"{path.name}:\n  - " + "\n  - ".join(problems))

    model, mtype, budget = df["model"].iloc[0], df["type"].iloc[0], df["budget"].iloc[0]
    seed, split = int(df["seed"].iloc[0]), df["split"].iloc[0]

    # 3. Each value is one of the allowed ones
    if model not in MODEL_TYPE:
        problems.append(f"unknown model '{model}'")
    elif mtype != MODEL_TYPE[model]:
        problems.append(f"type '{mtype}' does not match model '{model}' (expected '{MODEL_TYPE[model]}')")
    if budget not in BUDGETS:
        problems.append(f"unknown budget '{budget}'")
    if seed not in SEEDS:
        problems.append(f"seed {seed} not in {SEEDS}")
    if split not in SPLIT_NAMES:
        problems.append(f"unknown split '{split}'")
    # Zero-shot means no training: only for zero-shot models, and seed 0 only
    if (budget == "zeroshot") != (model in ZERO_SHOT_MODELS):
        problems.append(f"budget 'zeroshot' is only for {list(ZERO_SHOT_MODELS)} (got model '{model}', budget '{budget}')")
    if budget == "zeroshot" and seed != 0:
        problems.append("zero-shot runs must use seed 0")

    # 4. The file name matches the columns inside it
    expected_name = f"{model}_{budget}_seed{seed}_{split}.csv"
    if path.name != expected_name:
        problems.append(f"file name should be '{expected_name}'")

    # 5. Labels are in range (-1 is allowed only as a prediction)
    if not df["true_label"].between(0, N_LABELS - 1).all():
        problems.append("true_label has values outside 0-76")
    pred_ok = df["predicted_label"].between(0, N_LABELS - 1) | (df["predicted_label"] == UNMATCHED)
    if not pred_ok.all():
        problems.append("predicted_label has values outside 0-76 (or -1)")

    # 6. example_id is exactly 0..n-1 (each once), and true_label equals the real label at that position
    if split in SPLIT_NAMES:
        expected = split_labels(split)
        if sorted(df["example_id"]) != list(range(len(expected))):
            problems.append(f"example_id must be exactly 0..{len(expected) - 1}, each once (file has {len(df)} rows)")
        elif not (df.sort_values("example_id")["true_label"].to_numpy() == expected).all():
            problems.append("true_label does not match the real labels of the split file")

    if problems:
        raise ValueError(f"{path.name}:\n  - " + "\n  - ".join(problems))
    return df.sort_values("example_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def score(df):
    """Accuracy (primary) and macro-F1 (secondary) for one prediction file.

    An unmatched answer (-1) never equals a true label, so it counts as wrong.
    Note: it lowers recall but creates no false positive, so macro-F1 can end up higher
    than accuracy for a decoder with many unmatched answers. Report both.
    """
    y_true, y_pred = df["true_label"], df["predicted_label"]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=list(range(N_LABELS)), average="macro", zero_division=0),
    }


def per_intent(df):
    """Precision, recall, F1 and support for each of the 77 intents."""
    p, r, f, n = precision_recall_fscore_support(
        df["true_label"], df["predicted_label"], labels=list(range(N_LABELS)), zero_division=0)
    return pd.DataFrame({"intent": label_names(), "precision": p, "recall": r, "f1": f, "support": n})


def confusion(df):
    """Full confusion matrix: rows = true intent, columns = predicted intent (incl. 'unmatched')."""
    names = _names()
    return pd.crosstab(df["true_label"].map(names), df["predicted_label"].map(names),
                       rownames=["true"], colnames=["predicted"])


def top_confusions(df, n=10):
    """The n most frequent mistakes as (true, predicted, count)."""
    names = _names()
    wrong = df[df["true_label"] != df["predicted_label"]]
    return (wrong.groupby(["true_label", "predicted_label"]).size()
                 .sort_values(ascending=False).head(n)
                 .rename("count").reset_index()
                 .assign(true=lambda d: d["true_label"].map(names),
                         predicted=lambda d: d["predicted_label"].map(names))
                 [["true", "predicted", "count"]])


# ---------------------------------------------------------------------------
# Every run at once: seed summary and learning curves
# ---------------------------------------------------------------------------
def score_all(split="test", results_dir=None):
    """Load, check and score every prediction file of one split. One row per run."""
    results_dir = Path(results_dir) if results_dir else RESULTS
    rows = []
    for path in sorted(results_dir.glob(f"*_{split}.csv")):
        df = load_predictions(path)
        info = df.iloc[0][["model", "type", "budget", "seed", "split"]].to_dict()
        rows.append({**info, **score(df)})
    return pd.DataFrame(rows)


def summarize(scores):
    """Mean and standard deviation across seeds for each model and budget."""
    return (scores.groupby(["model", "type", "budget"])
                  .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                       macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                       n_seeds=("seed", "nunique"))
                  .reset_index())


def plot_learning_curve(summary, metric="accuracy", split="test", save_to=None):
    """Score against number of training examples, one colored line per model."""
    import matplotlib.pyplot as plt

    sizes = budget_sizes()
    # One fixed color per model, so zero-shot lines and trained lines never share a color
    colors = {m: f"C{i}" for i, m in enumerate(sorted(summary["model"].unique()))}

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, g in summary.groupby("model"):
        color = colors[model]
        if (g["budget"] == "zeroshot").all():        # no training data: a flat reference line
            ax.axhline(g[f"{metric}_mean"].iloc[0], linestyle="--", color=color, label=f"{model} (zero-shot)")
            continue
        g = g.assign(n=g["budget"].map(sizes)).dropna(subset=["n"]).sort_values("n")
        mean, std = g[f"{metric}_mean"], g[f"{metric}_std"].fillna(0)
        ax.plot(g["n"], mean, marker="o", color=color, label=model)
        ax.fill_between(g["n"], mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(list(sizes.values()), list(sizes.keys()))
    ax.set_xlabel("Training budget (log scale)")
    ax.set_ylabel(metric.replace("_", "-"))
    ax.set_title(f"Learning curves: {metric.replace('_', '-')} on {split}")
    ax.legend()
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150)
    plt.close(fig)       # avoid a duplicate display in notebooks; the returned figure still shows
    return fig
