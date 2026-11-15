# Project: Labels vs. Scale: Breaking Even on Banking77 with Encoders and Decoders

"""Diagnostic plots that are not part of scoring predictions.

evaluation.py scores the prediction files. This module looks at how the models fit the data:

  summarize_tfidf_loss()        mean and standard deviation over seeds of the TF-IDF train and
                                validation log-loss at each budget (reads the file that
                                baselines.py writes to results/diagnostics/).
  plot_tfidf_loss_vs_budget()   the same as a figure: loss against number of training examples.

Training-loss curves of the neural models (loss against step) will be added here once the
training scripts save their logs.

Typical use in a notebook:
    import sys; sys.path.insert(0, "scripts")
    import diagnostics as dg
    dg.summarize_tfidf_loss().round(3)
    dg.plot_tfidf_loss_vs_budget(save_to="reports/figures/tfidf_loss_vs_budget.png")
"""
from pathlib import Path

import numpy as np
import pandas as pd

import evaluation as ev

DIAGNOSTICS = ev.RESULTS / "diagnostics"
TFIDF_LOSS_FILE = DIAGNOSTICS / "tfidf_lr_loss_by_budget.csv"


def load_tfidf_loss(path=None):
    """Read the per-run TF-IDF losses written by `python scripts/baselines.py --models tfidf_lr`."""
    path = Path(path) if path else TFIDF_LOSS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/baselines.py --models tfidf_lr")
    return pd.read_csv(path)


def summarize_tfidf_loss(path=None):
    """Mean and standard deviation over seeds for each budget, ordered by training size."""
    df = load_tfidf_loss(path)
    out = (df.groupby("budget")
             .agg(n_train=("n_train", "mean"),
                  train_loss_mean=("train_loss", "mean"), train_loss_std=("train_loss", "std"),
                  val_loss_mean=("val_loss", "mean"), val_loss_std=("val_loss", "std"))
             .reset_index().sort_values("n_train").reset_index(drop=True))
    out["gap"] = out["val_loss_mean"] - out["train_loss_mean"]   # Validation minus train loss
    return out


def plot_tfidf_loss_vs_budget(path=None, save_to=None):
    """Train and validation log-loss against the number of training examples (log scale)."""
    import matplotlib.pyplot as plt

    summary = summarize_tfidf_loss(path)
    sizes = ev.budget_sizes()
    chance = np.log(ev.N_LABELS)       # Loss of guessing all 77 intents with equal probability

    fig, ax = plt.subplots(figsize=(8, 5))
    for column, label, color in [("train_loss", "Train loss", "C0"), ("val_loss", "Validation loss", "C1")]:
        mean, std = summary[f"{column}_mean"], summary[f"{column}_std"].fillna(0)
        ax.plot(summary["n_train"], mean, marker="o", color=color, label=label)
        ax.fill_between(summary["n_train"], mean - std, mean + std, color=color, alpha=0.2)
    ax.axhline(chance, linestyle=":", color="gray", label=f"Chance (ln 77 = {chance:.2f})")

    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(list(sizes.values()), list(sizes.keys()))
    ax.set_xlabel("Training budget (log scale)")
    ax.set_ylabel("Log-loss (lower is better)")
    ax.set_title("TF-IDF + logistic regression: loss vs. training size")
    ax.legend()
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150)
    plt.close(fig)       
    return fig
