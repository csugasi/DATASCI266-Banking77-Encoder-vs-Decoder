## Labels vs. Scale: Breaking Even on Banking77 with Encoders and Decoders

How much labeled data does a small fine-tuned encoder need before it matches or beats a decoder language model on fine-grained intent classification?

We compare ModernBERT-base against Qwen2.5-1.5B-Instruct on the Banking77 dataset. The Qwen model is tested both zero-shot and fine-tuned with SFT/QLoRA, and both models are trained on the same examples at several data sizes.

#### Why this project?

Help desks need to work out what each customer message is asking before they can answer it. Sending every message to a large language model is easy, but it costs money per query, sends customer data outside the company, and ties you to a model you don't control. A small fine-tuned model avoids most of that, but only if you know how many labeled examples it needs.

Banking77 makes this hard in an interesting way. Its 77 intents include many close neighbours, such as card_arrival vs. card_delivery_estimate and get_physical_card vs. order_physical_card.

#### Research question

How does increasing labeled supervision affect the relative performance of an encoder and a decoder model on Banking77 intent classification?

Earlier work on this dataset compared fine-tuned encoders only against prompted LLMs. The closest study that also fine-tunes the decoder covered intent detection only on SNIPS, which has 7 intents. We fine-tune both models on the exact same examples at each data size, test the decoder zero-shot as well, and repeat each run three times.

#### Dataset

Banking77 (Casanueva et al., 2020)

13,083 customer banking queries

77 intents

Official split: 10,003 train / 3,080 test

The training set is imbalanced; the test set has exactly 40 examples per intent

#### Models and baselines

| Role     | Model                        | How it's used                                                             |
| -------- | ---------------------------- | ------------------------------------------------------------------------- |
| Encoder  | ModernBERT-base              |                                                                           |
| Decoder  | Qwen/Qwen2.5-1.5B-Instruct   |                                                                           |
| Baseline | Majority class               |                                                                           |
| Baseline | TF-IDF + logistic regression |                                                                           |
| Control  | Frontier Model               |                                                                           |

#### Experimental setup

Training sizes: 1, 3, 5, 10, 20, 30 examples per intent and full training dataset

Seeds: 3 (0, 1, 2) with a different random sample of examples for each seed

Matched data: both models train on the exact same subset files

Nested subsets: the examples at 1 per intent are included in 3, which are included in 5, and so on

Validation: a stratified slice of the training data, used for all tuning

Test set: held out and used only for final numbers

#### Evaluation

Main metric: accuracy

Secondary: macro-F1

Results are reported as average and spread across the three seeds

Success means finding the smallest number of examples per intent at which ModernBERT reliably outperforms Qwen, both zero-shot and fine-tuned. If no such point appears, that is also a result.

#### Repository structure

```
├── data/               # dataset copies, split manifests, subset ID files (large files on Drive)
│   ├── preview/ 
|   │   ├── test_preview.csv        
│   |   ├── train_preview.csv              
│   ├── raw/     # Downloaded Banking77 files (train, test, label names)
|   │   ├── label_names.json        
│   |   ├── test.parquet
|   │   ├── train.parquet            
│   ├── splits/          # Validation set, training subsets and manifest
├── scripts/
│   ├── download_data.py           # Downloads the dataset
│   ├── evaluation.py              # Checks and scores prediction files
│   ├── baselines.py               # Majority-class and TF-IDF + logistic regression
│   └── diagnostics.py             # Loss plots
|   └── predictions.py    # write_predictions + match_intent (used by every model)
│   ├── train_encoder.py      # ModernBERT fine-tuning
│   ├── train_decoder.py      # Qwen SFT fine-tuning
│   ├── zero_shot.py          # Qwen zero-shot prompting
├── notebooks/          # Colab notebooks and EDA
│   ├── eda.ipynb               # Exploratory data analysis
│   ├── make_splits.ipynb       # Creates the validation set and training subsets
│   ├── baselines.ipynb         # Runs and shows the baselines
│   └── results.ipynb           # Shows the results of every model
├── results/            # prediction CSVs, summary tables, plots
│   ├── PREDICTION_FORMAT.md       # The format every prediction file must follow
│   ├── *_val.csv, *_test.csv      # One prediction file per run and split
│   └── diagnostics/               # Loss files that are not predictions
├── reports/figures/               # Saved plots
├── requirements.txt
└── .gitignore
└── README.md
```

## What each file does

### Scripts (`scripts/`)

- **`download_data.py`** downloads Banking77 from parquet files pinned to a specific Hugging Face commit, so we get identical data.
  - It writes `train.parquet`, `test.parquet` and `label_names.json` to `data/raw/`.
  - It does nothing if the files exist. Use `--force` to download again.
  - `--preview` also writes readable CSVs to `data/preview/`.
- **`evaluation.py`** evaluates prediction files that already exist. It never writes predictions.
  - `load_predictions` checks a file against `PREDICTION_FORMAT.md` and lists every problem found.
  - `score` computes accuracy (primary metric) and macro-F1 (secondary). An unmatched answer (`-1`) counts as wrong.
  - `per_intent`, `confusion` and `top_confusions` show where a run fails.
  - `score_all`, `summarize` and `plot_learning_curve` score every run, average over seeds and plot score against training size.
- **`predictions.py`** is what every model calls to produce its prediction file.
  - `write_predictions` writes one CSV in the shared format and re-checks it.
  - `match_intent` turns a decoder's generated text into an intent id, or `-1` if it is not exactly an intent name.
- **`baselines.py`** trains the two baselines for every budget and seed.
  - Majority class predicts the most frequent intent of the training subset.
  - TF-IDF + logistic regression uses scikit-learn defaults, with no tuning (only `max_iter=1000`).
  - Both write prediction files for val and test. TF-IDF also saves its train and validation log-loss.
- **`diagnostics.py`** summarizes and plots the train and validation loss against training size. 

### Notebooks (`notebooks/`)

- **`eda.ipynb`** explores the data.
  - Data quality: nulls, empty texts and duplicates.
  - Class balance: 35 to 187 examples per intent in train, 40 per intent in test.
  - Query length in words and tokens.
  - Train/test overlap and label conflicts.
  - Word frequencies and the share of unseen words.
  - Which intents share vocabulary.
- **`make_splits.ipynb`** builds every training set the experiments use.
  - A validation set of 5 examples per intent (385 in total).
  - A training pool of the remaining 9,618 examples.
  - Balanced subsets with 1, 3, 5, 10, 20 and 30 examples per intent, for seeds 0, 1 and 2. They are nested within each seed.
  - A proportional subset of about 50% of the pool (4,807 examples), which contains the 30-per-intent subset.
  - Sanity checks that reload the saved files and verify counts, overlap and nesting.
- **`evaluation.ipynb`** is the original evaluation notebook. Its logic now lives in `scripts/evaluation.py`.
- **`baselines.ipynb`** is a wrapper around `baselines.py`. It runs both baselines and shows validation and test scores, the learning curve and the loss-vs-size plot.
- **`results.ipynb`** only displays results. It scores every prediction file in `results/` and shows the summary table, a check of which runs exist, learning curves, error analysis of one run, and validation scores.

### Data and results

- **`data/raw/`** holds the downloaded files. Nothing here is edited.
- **`data/splits/`** holds the files made by `make_splits.ipynb`:
  - `val.parquet` and `full_train.parquet` (the training pool)
  - `train_k{k}_seed{s}.parquet` (balanced subsets)
  - `train_p50_seed{s}.parquet` (proportional subset)
  - `manifest.json` (settings and sizes)
- **`data/README.md`** describes the dataset: source, license, columns and notes.
- **`results/PREDICTION_FORMAT.md`** defines the prediction files: file name, columns, allowed values and rules.
- **`results/*_val.csv` and `*_test.csv`** are the prediction files, one per model, budget, seed and split, named `{model}_{budget}_seed{seed}_{split}.csv`.
- **`results/diagnostics/`** holds files that are not predictions, such as `tfidf_lr_loss_by_budget.csv`. It is kept separate so the scorer does not read them.
- **`reports/figures/`** holds the saved plots.

### Other

- **`requirements.txt`** lists the Python packages needed.
- **`.gitignore`** lists files that should not be committed.

## How to run

```bash
# Install packages, then download the data
pip install -r requirements.txt
python scripts/download_data.py

# Run notebooks/eda.ipynb (optional), then notebooks/make_splits.ipynb

# Run the baselines (or run notebooks/baselines.ipynb, which does the same)
python scripts/baselines.py

# Look at the results
# Open notebooks/results.ipynb
```

Still to come: the ModernBERT, Qwen and frontier-model runs. They will be trained from Colab and will write prediction files through `scripts/predictions.py`.