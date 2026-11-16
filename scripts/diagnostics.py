# Project: Labels vs. Scale: Breaking Even on Banking77 with Encoders and Decoders

"""Diagnostic plots that are not part of scoring predictions.

evaluation.py scores the prediction files. This module looks at how the models fit the data:

  summarize_tfidf_loss()        mean and standard deviation over seeds of the TF-IDF train and
                                validation log-loss at each budget (reads the file that
                                baselines.py writes to results/diagnostics/).
  plot_tfidf_loss_vs_budget()   the same as a figure: loss against number of training examples.

summarize_pilot()             table of the ModernBERT learning-rate pilot (mean validation scores).
  plot_training_curves()        loss and validation accuracy against training step for one budget,
                                one line per seed (reads the logs train_modernbert.py saves).

The results folder is results/ by default; in Colab pass the Drive folder as results_dir.

Typical use in a notebook:
    import sys; sys.path.insert(0, "scripts")
    import diagnostics as dg
    dg.summarize_tfidf_loss().round(3)
    dg.plot_tfidf_loss_vs_budget(save_to="reports/figures/tfidf_loss_vs_budget.png")
    dg.summarize_pilot(results_dir, rule="fixed3")
    dg.plot_training_curves("modernbert", "k10", results_dir=results_dir)   # or "modernbert_min300"
"""
import json
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

# ---------------------------------------------------------------------------
# Neural model training logs (written by train_modernbert.py)
# ---------------------------------------------------------------------------
def summarize_pilot(results_dir=None, rule="fixed3"):
    """Mean validation accuracy and loss of each learning rate tried in the pilot, plus the chosen one.
    rule is the epoch rule of the pilot: fixed3 or min300."""
    folder = Path(results_dir) if results_dir else ev.RESULTS
    best_file = folder / "pilot" / f"best_lr_{rule}.json"
    if not best_file.exists():
        raise FileNotFoundError(f"{best_file} not found. Run: python scripts/train_modernbert.py pilot --epoch-rule {rule}")
    best = json.load(open(best_file))
    table = pd.DataFrame(best["table"])
    table["chosen"] = table["lr"] == best["lr"]
    return table


def load_training_log(model, budget, seed, results_dir=None):
    """The logged validation curve of one run: step, epoch, train_loss, val_loss, val_acc, seconds."""
    folder = Path(results_dir) if results_dir else ev.RESULTS
    path = folder / "training_logs" / f"{model}_{budget}_seed{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Has this run finished?")
    return pd.read_csv(path)


def plot_training_curves(model, budget, seeds=(0, 1, 2), results_dir=None, save_to=None):
    """Left: train and validation loss against step. Right: validation accuracy against step.
    One line per seed. train_loss is the mean over the steps since the previous evaluation, so it is
    measured in training mode (with dropout) and is not the same quantity as validation loss."""
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, seed in enumerate(seeds):
        log = load_training_log(model, budget, seed, results_dir)
        color = f"C{i}"
        ax_loss.plot(log["step"], log["train_loss"], color=color, linestyle="--", alpha=0.8)
        ax_loss.plot(log["step"], log["val_loss"], color=color, label=f"Seed {seed}")
        ax_acc.plot(log["step"], log["val_acc"], color=color, label=f"Seed {seed}")
    ax_loss.axhline(np.log(ev.N_LABELS), linestyle=":", color="gray")   # Loss of guessing uniformly
    ax_loss.set_xlabel("Training step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss (dashed = train, solid = validation, dotted = chance)")
    ax_loss.legend()
    ax_acc.set_xlabel("Training step")
    ax_acc.set_ylabel("Validation accuracy")
    ax_acc.set_title("Validation accuracy")
    ax_acc.legend()
    fig.suptitle(f"{model}, budget {budget}")
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150)
    plt.close(fig)
    return fig
