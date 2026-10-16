Labels vs. Scale: Breaking Even on Banking77 with Encoders and Decoders

How much labeled data does a small fine-tuned encoder need before it matches or beats a decoder language model on fine-grained intent classification?

We compare ModernBERT-base against Qwen2.5-1.5B-Instruct on the Banking77 dataset. The Qwen model is tested both zero-shot and fine-tuned with SFT/QLoRA, and both models are trained on the same examples at several data sizes.

Why this project?

Help desks need to work out what each customer message is asking before they can answer it. Sending every message to a large language model is easy, but it costs money per query, sends customer data outside the company, and ties you to a model you don't control. A small fine-tuned model avoids most of that, but only if you know how many labeled examples it needs.

Banking77 makes this hard in an interesting way. Its 77 intents include many close neighbours, such as card_arrival vs. card_delivery_estimate and get_physical_card vs. order_physical_card.

Research question

How does increasing labeled supervision affect the relative performance of an encoder and a decoder model on Banking77 intent classification?

Earlier work on this dataset compared fine-tuned encoders only against prompted LLMs. The closest study that also fine-tunes the decoder covered intent detection only on SNIPS, which has 7 intents. We fine-tune both models on the exact same examples at each data size, test the decoder zero-shot as well, and repeat each run three times.

Dataset

Banking77 (Casanueva et al., 2020)

13,083 customer banking queries

77 intents

Official split: 10,003 train / 3,080 test

The training set is imbalanced; the test set has exactly 40 examples per intent

Models and baselines

| Role     | Model                        | How it's used                                                             |
| -------- | ---------------------------- | ------------------------------------------------------------------------- |
| Encoder  | ModernBERT-base              |                                                                           |
| Decoder  | Qwen/Qwen2.5-1.5B-Instruct   |                                                                           |
| Baseline | Majority class               |                                                                           |
| Baseline | TF-IDF + logistic regression |                                                                           |
| Control  | Frontier Model               |                                                                           |

Experimental setup

Training sizes: 1, 3, 5, 10, 20, 35 examples per intent and full training dataset

Seeds: 3 (0, 1, 2) with a different random sample of examples for each seed

Matched data: both models train on the exact same subset files

Nested subsets: the examples at 1 per intent are included in 3, which are included in 5, and so on

Validation: a stratified slice of the training data, used for all tuning

Test set: held out and used only for final numbers

Evaluation

Main metric: accuracy

Secondary: macro-F1

Results are reported as average and spread across the three seeds

Success means finding the smallest number of examples per intent at which ModernBERT reliably outperforms Qwen, both zero-shot and fine-tuned. If no such point appears, that is also a result.

Repository structure

├── data/               # dataset copies, split manifests, subset ID files (large files on Drive)
├── src/
│   ├── banking77_splits.py   # validation split and nested k-shot subsets
│   ├── baselines.py          # majority class, TF-IDF + logistic regression
│   ├── train_encoder.py      # ModernBERT fine-tuning
│   ├── train_decoder.py      # Qwen SFT fine-tuning
│   ├── zero_shot.py          # Qwen zero-shot prompting
│   └── evaluate.py           # accuracy, macro-F1, confusion matrices, learning curves
├── notebooks/          # Colab notebooks and EDA
├── results/            # prediction CSVs, summary tables, plots
└── README.md