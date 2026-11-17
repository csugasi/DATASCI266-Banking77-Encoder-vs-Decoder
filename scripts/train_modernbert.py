"""## Project: Labels vs. Scale: Breaking Even on Banking77 with Encoders and Decoders

ModernBERT-base fine-tuned with a classification head (full SFT, no LoRA).

Two training protocols (the "epoch rule", chosen with --epoch-rule). Everything else is identical:

  fixed3  : every budget trains exactly 3 epochs. Main protocol, the same rule the Qwen runs use.
            Files are saved under the model name "modernbert".
  min300  : at least 300 optimizer steps per run, between 3 and 50 epochs. Secondary variant that
            shows whether the small-budget results depend on training length.
            Files are saved under the model name "modernbert_min300".

  min600  : extra check, meant for the smallest budgets only (k1 and k3): at least 600 optimizer steps,
            between 3 and 200 epochs (k1 needs 120 epochs to reach 600 steps, so the cap is raised).
            Tests whether k1 and k3 are still limited by training length under min300.
            Files are saved under the model name "modernbert_min600". It has no pilot of its own: run it
            with --lr set to the learning rate that min300 chose, and say so when reporting it.

Each protocol has its own pilot (the best learning rate depends on the number of steps) and its own files,
so the two never overwrite each other.

Two commands:

  pilot : tune the learning rate ONCE, at budget k10, on the validation set only.
          Tries each learning rate in PILOT_LRS on the pilot seeds and saves the winner to
          <results-dir>/pilot/best_lr_{epoch-rule}.json. No prediction files are written by the pilot.
  run   : train every budget x seed with that frozen learning rate. The model after the FINAL
          epoch is used (no best-checkpoint picking). For each run it writes
            - prediction files (val and test) in the shared format, through predictions.write_predictions
            - a training log   <results-dir>/training_logs/{model}_{budget}_seed{seed}.csv
            - a config file    <results-dir>/training_logs/{model}_{budget}_seed{seed}_config.json
          Runs that are already finished are skipped, so after a Colab disconnect you just start the
          same command again and it continues with the first unfinished run.

Everything the experiment must keep identical across budgets is in FROZEN below. Only the learning
rate is chosen by the pilot. Nothing is saved to disk besides logs and predictions (no checkpoints).

The data-size points and the val/test loaders come from baselines.py, so all models use the same
files in the same order.

Usage (from the project root):
    python scripts/train_modernbert.py pilot --epoch-rule fixed3 --results-dir /content/drive/MyDrive/banking77/results
    python scripts/train_modernbert.py run   --epoch-rule fixed3 --results-dir /content/drive/MyDrive/banking77/results
    python scripts/train_modernbert.py run   --epoch-rule min300 --results-dir ... --budgets k1 --seeds 0   # Timing / smoke test
"""
import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import evaluation as ev
from baselines import budget_files, split_data       # Shared data-size points and val/test loaders
from predictions import write_predictions

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
# Frozen for every run. The learning rate is the only thing the pilot chooses.
FROZEN = {
    "model_name": "answerdotai/ModernBERT-base",
    "max_length": 128,        # Longer texts are truncated (check the EDA token lengths)
    "batch_size": 16,
    "weight_decay": 0.01,     # Not applied to biases and norm weights
    "warmup_ratio": 0.10,     # Share of steps spent warming the learning rate up from 0
    "max_grad_norm": 1.0,     # Gradient clipping
    "fixed_epochs": 3,        # Epoch rule fixed3: this many epochs for every budget
    "min_epochs": 3,          # Step-based rules (min300, min600): never fewer epochs than this
    "evals_per_run": 20,      # How many times validation is scored during training (for the loss curves)
    "eval_batch_size": 64,
}
PILOT_BUDGET = "k10"
PILOT_LRS = [2e-5, 5e-5, 1e-4, 2e-4, 4e-4]   # Candidates for the one-time tuning (2e-4 and 4e-4 added after
                                                # the first pilot: accuracy was still rising at 1e-4)
PILOT_SEEDS = [0, 1, 2]          # Validation accuracy is averaged over these seeds
# Epoch rule -> name used in prediction files, logs and the results table
EPOCH_RULES = {"fixed3": "modernbert", "min300": "modernbert_min300", "min600": "modernbert_min600"}

# Step-based rules: at least min_steps optimizer steps, but no more than max_epochs epochs
STEP_RULES = {"min300": {"min_steps": 300, "max_epochs": 50},
              "min600": {"min_steps": 600, "max_epochs": 200}}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def steps_per_epoch(n_train):
    return math.ceil(n_train / FROZEN["batch_size"])


def epochs_for(n_train, rule):
    """Number of epochs for a training set of n_train examples under one epoch rule."""
    if rule == "fixed3":
        return FROZEN["fixed_epochs"]
    # Step-based rules: at least min_steps optimizer steps, but between min_epochs and max_epochs epochs
    rule_settings = STEP_RULES[rule]
    needed = math.ceil(rule_settings["min_steps"] / steps_per_epoch(n_train))
    return max(FROZEN["min_epochs"], min(rule_settings["max_epochs"], needed))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_tag(lr):
    return f"{lr:.0e}".replace("-0", "-")   # 5e-05 -> 5e-5, used in file names


def load_model_and_tokenizer():
    """Pretrained ModernBERT with a fresh 77-way classification head. The head's start is set by set_seed."""
    names = ev.label_names()
    tokenizer = AutoTokenizer.from_pretrained(FROZEN["model_name"])
    # Set the options on the config object first, then pass it in (extra keyword arguments given directly
    # to from_pretrained go to the model constructor in newer transformers, which rejects unknown ones)
    config = AutoConfig.from_pretrained(FROZEN["model_name"])
    config.id2label = dict(enumerate(names))
    config.label2id = {n: i for i, n in enumerate(names)}
    config.num_labels = ev.N_LABELS
    config.reference_compile = False   # Avoids torch.compile slowdowns with changing batch shapes
    model = AutoModelForSequenceClassification.from_pretrained(FROZEN["model_name"], config=config)
    return model, tokenizer


def encode(tokenizer, texts):
    """Tokenize once (no padding yet). Padding is done per batch, to the longest text in the batch."""
    enc = tokenizer(list(texts), truncation=True, max_length=FROZEN["max_length"])
    return enc["input_ids"], enc["attention_mask"]


def make_batch(tokenizer, ids, masks, labels, index):
    """Pad the chosen examples into one batch of tensors on the device."""
    batch = tokenizer.pad({"input_ids": [ids[i] for i in index],
                           "attention_mask": [masks[i] for i in index]}, return_tensors="pt")
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    return batch, torch.as_tensor(labels[index], dtype=torch.long, device=DEVICE)


@torch.no_grad()
def predict(model, tokenizer, data, labels=None):
    """Predicted label ids for already-encoded texts. With labels, also returns mean loss and accuracy."""
    ids, masks = data
    model.eval()
    preds, losses = [], []
    n = len(ids)
    dummy = np.zeros(n, dtype=int) if labels is None else np.asarray(labels)
    for start in range(0, n, FROZEN["eval_batch_size"]):
        index = np.arange(start, min(start + FROZEN["eval_batch_size"], n))
        batch, y = make_batch(tokenizer, ids, masks, dummy, index)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            logits = model(**batch).logits.float()
        preds.append(logits.argmax(-1).cpu().numpy())
        if labels is not None:
            losses.append(torch.nn.functional.cross_entropy(logits, y, reduction="sum").item())
    preds = np.concatenate(preds)
    if labels is None:
        return preds
    return preds, sum(losses) / n, float((preds == np.asarray(labels)).mean())


# ---------------------------------------------------------------------------
# One training run
# ---------------------------------------------------------------------------
def train_one_run(rule, budget, seed, lr, template, log_dir, results_dir=None):
    """Train on one subset and log the validation curve. With results_dir, also writes the val and test
    prediction files (the pilot passes None and writes none). Returns the config dict that was saved."""
    started = time.time()
    model_label = EPOCH_RULES[rule]     # Name used in file names and in the model column
    set_seed(seed)
    train_df = pd.read_parquet(ev.SPLITS / template.format(seed=seed))
    val_df = split_data("val")
    n_train, spe = len(train_df), steps_per_epoch(len(train_df))
    epochs = epochs_for(n_train, rule)
    total_steps = epochs * spe
    eval_every = max(1, total_steps // FROZEN["evals_per_run"])

    model, tokenizer = load_model_and_tokenizer()
    model.to(DEVICE)
    train_data = encode(tokenizer, train_df["text"])
    train_y = train_df["label"].to_numpy()
    val_data = encode(tokenizer, val_df["text"])
    val_y = val_df["label"].to_numpy()

    # AdamW; weight decay only on matrices (not biases or norm weights)
    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": FROZEN["weight_decay"]},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(FROZEN["warmup_ratio"] * total_steps), total_steps)
    shuffler = torch.Generator().manual_seed(seed)   # Fixed data order for a given seed

    log, step, running = [], 0, []

    def evaluate():
        preds, val_loss, val_acc = predict(model, tokenizer, val_data, val_y)
        model.train()
        row = {"step": step, "epoch": round(step / spe, 3),
               "train_loss": float(np.mean(running)) if running else float("nan"),   # Mean since the last evaluation
               "val_loss": val_loss, "val_acc": val_acc, "seconds": round(time.time() - started, 1)}
        log.append(row)
        running.clear()
        return preds, val_loss, val_acc

    model.train()
    evaluate()                                       # Step 0: the untrained head, loss should be near ln(77)
    for epoch in range(epochs):
        order = torch.randperm(n_train, generator=shuffler).numpy()
        for start in range(0, n_train, FROZEN["batch_size"]):
            index = order[start:start + FROZEN["batch_size"]]   # The last batch of an epoch may be smaller
            batch, y = make_batch(tokenizer, train_data[0], train_data[1], train_y, index)
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                logits = model(**batch).logits.float()
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), FROZEN["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            running.append(loss.item())
            if step % eval_every == 0 and step != total_steps:
                evaluate()
    val_preds, val_loss, val_acc = evaluate()        # Final evaluation = the model after the last epoch

    # Predictions come from the final-epoch model. The val file reuses the final evaluation.
    if results_dir is not None:
        test_df = split_data("test")
        test_preds = predict(model, tokenizer, encode(tokenizer, test_df["text"]))
        write_predictions(model_label, budget, seed, "val", val_preds, out_dir=results_dir)
        write_predictions(model_label, budget, seed, "test", test_preds, out_dir=results_dir)   # Written last: marks the run as done

    config = {"model": model_label, "epoch_rule": rule, "model_name": FROZEN["model_name"], "budget": budget, "seed": seed, "lr": lr,
              "n_train": n_train, "epochs": epochs, "steps": total_steps, "eval_every": eval_every,
              **{k: v for k, v in FROZEN.items() if k != "model_name"}, **STEP_RULES.get(rule, {}),
              "final_val_acc": val_acc, "final_val_loss": val_loss,
              "seconds": round(time.time() - started, 1), "device": DEVICE,
              "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
              "torch": torch.__version__}
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_label}_{budget}_seed{seed}" + ("" if results_dir is not None else f"_lr{lr_tag(lr)}")
    pd.DataFrame(log).to_csv(log_dir / f"{stem}.csv", index=False)
    json.dump(config, open(log_dir / f"{stem}_config.json", "w"), indent=2)   # Config last: marks the run as done

    # Free the GPU before the next run
    del model, optimizer, scheduler
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return config


# ---------------------------------------------------------------------------
# Pilot: choose the learning rate once
# ---------------------------------------------------------------------------
def run_pilot(results_dir, rule):
    pilot_dir = Path(results_dir) / "pilot"
    model_label = EPOCH_RULES[rule]
    manifest = json.load(open(ev.SPLITS / "manifest.json"))
    template = dict(budget_files(manifest))[PILOT_BUDGET]
    for lr in PILOT_LRS:
        for seed in PILOT_SEEDS:
            config_file = pilot_dir / f"{model_label}_{PILOT_BUDGET}_seed{seed}_lr{lr_tag(lr)}_config.json"
            if config_file.exists():
                print("Pilot run already done, skipping:", lr, "seed", seed)
                continue
            print("Pilot run: lr", lr, "| seed", seed)
            cfg = train_one_run(rule, PILOT_BUDGET, seed, lr, template, pilot_dir)
            print("  Val accuracy:", round(cfg["final_val_acc"], 4), "| Val loss:", round(cfg["final_val_loss"], 4),
                  "| Seconds:", cfg["seconds"])

    # Summary of all pilot runs, rebuilt from the saved config files (so it also works after a restart)
    rows = [json.load(open(f)) for f in sorted(pilot_dir.glob(f"{model_label}_{PILOT_BUDGET}_seed*_config.json"))]
    table = (pd.DataFrame(rows).groupby("lr")
             .agg(val_acc=("final_val_acc", "mean"), val_loss=("final_val_loss", "mean"), runs=("seed", "count"))
             .reset_index())
    table = table[table["runs"] == len(PILOT_SEEDS)]   # Only learning rates with all pilot seeds finished
    if table.empty:
        raise SystemExit("No learning rate has all pilot seeds finished yet")
    # Best = highest mean validation accuracy; ties go to the lower validation loss
    best = table.sort_values(["val_acc", "val_loss"], ascending=[False, True]).iloc[0]
    print("\nPilot results for epoch rule", rule, "(mean over seeds, validation set):")
    print(table.to_string(index=False))
    if best["lr"] in (min(PILOT_LRS), max(PILOT_LRS)):
        print("Warning: the best learning rate is at the edge of the grid, so a value beyond it might be better")
    best_file = pilot_dir / f"best_lr_{rule}.json"
    json.dump({"lr": float(best["lr"]), "epoch_rule": rule, "budget": PILOT_BUDGET, "seeds": PILOT_SEEDS,
               "table": table.to_dict("records")}, open(best_file, "w"), indent=2)
    print("Chosen learning rate:", best["lr"], "| Saved to:", best_file)


# ---------------------------------------------------------------------------
# Main runs: every budget x seed
# ---------------------------------------------------------------------------
def run_is_done(results_dir, rule, budget, seed, lr):
    """A run is done if both prediction files are valid, its config exists and it used the same learning rate."""
    results_dir = Path(results_dir)
    model_label = EPOCH_RULES[rule]
    config_file = results_dir / "training_logs" / f"{model_label}_{budget}_seed{seed}_config.json"
    pred_files = [results_dir / f"{model_label}_{budget}_seed{seed}_{s}.csv" for s in ev.SPLIT_NAMES]
    if not config_file.exists() or not all(p.exists() for p in pred_files):
        return False
    try:
        for p in pred_files:
            ev.load_predictions(p)             # A half-written file fails here and the run is repeated
        return json.load(open(config_file))["lr"] == lr
    except Exception:
        return False


def run_all(results_dir, rule, lr=None, budgets=None, seeds=None):
    results_dir = Path(results_dir)
    if lr is None:
        best_file = results_dir / "pilot" / f"best_lr_{rule}.json"
        if not best_file.exists():
            raise SystemExit(f"No learning rate yet for epoch rule {rule}: run the pilot first, or pass --lr")
        lr = json.load(open(best_file))["lr"]
    manifest = json.load(open(ev.SPLITS / "manifest.json"))
    plan = [(b, t, s) for b, t in budget_files(manifest) if budgets is None or b in budgets
            for s in (seeds if seeds is not None else manifest["seeds"])]
    print("Epoch rule:", rule, "| Learning rate:", lr, "| Runs planned:", len(plan), "| Results folder:", results_dir)

    for i, (budget, template, seed) in enumerate(plan, 1):
        if run_is_done(results_dir, rule, budget, seed, lr):
            print(f"[{i}/{len(plan)}] Already done, skipping:", budget, "seed", seed)
            continue
        print(f"[{i}/{len(plan)}] Training:", budget, "seed", seed)
        cfg = train_one_run(rule, budget, seed, lr, template, results_dir / "training_logs", results_dir=results_dir)
        print("  Val accuracy:", round(cfg["final_val_acc"], 4), "| Steps:", cfg["steps"],
              "| Seconds:", cfg["seconds"])
    print("Finished. Runs planned:", len(plan))


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("pilot", "run"):
        p = sub.add_parser(name)
        p.add_argument("--epoch-rule", required=True, choices=list(EPOCH_RULES), help="fixed3 (main), min300 (variant) or min600 (extra check for k1 and k3)")
        p.add_argument("--results-dir", default=str(ev.RESULTS), help="Where predictions and logs are written (Drive in Colab)")
    sub.choices["run"].add_argument("--lr", type=float, default=None, help="Override the pilot's learning rate")
    sub.choices["run"].add_argument("--budgets", nargs="+", default=None)
    sub.choices["run"].add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()
    if args.command == "pilot":
        run_pilot(args.results_dir, args.epoch_rule)
    else:
        run_all(args.results_dir, args.epoch_rule, args.lr, args.budgets, args.seeds)
