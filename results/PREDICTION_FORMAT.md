# Prediction file format

Every model run writes **one CSV per run and split**. The shared evaluation script reads these files, so every model must follow this format exactly.

## File name

```
{model}_{budget}_seed{seed}_{split}.csv
```

Example: `qwen_qlora_k5_seed0_test.csv`

## Columns

| Column | Example | Allowed values |
|---|---|---|
| `model` | `qwen_qlora` | `modernbert`, `qwen_zeroshot`, `qwen_qlora`, `frontier`, `tfidf_lr`, `majority` |
| `type` | `decoder` | `encoder`, `decoder`, `baseline` (lowercase) |
| `budget` | `k5` | `zeroshot`, `k1`, `k3`, `k5`, `k10`, `k20`, `k30`, `p50`, `full` |
| `seed` | `0` | `0`, `1`, `2` (zero-shot runs: `0` only) |
| `split` | `test` | `val`, `test` |
| `example_id` | `17` | row position in that split's file (`data/splits/val.parquet` or `data/raw/test.parquet`) |
| `true_label` | `11` | integer `0`–`76` |
| `predicted_label` | `11` | integer `0`–`76`, or `-1` if the model's output could not be matched to an intent |
| `raw_output` | `card_arrival` | the model's generated text (decoders only); leave blank for other models |

## Model → type mapping

| `model` | `type` |
|---|---|
| `modernbert` | `encoder` |
| `qwen_zeroshot`, `qwen_qlora`, `frontier` | `decoder` |
| `tfidf_lr`, `majority` | `baseline` |

The evaluation script should reject any file whose `type` does not match this mapping.

## Rules

- **`-1` counts as wrong.** Unmatched decoder outputs are kept and scored as incorrect, never dropped. Dropping them would inflate the decoder's accuracy.
- **`example_id` must mean the same thing for every model**, so per-example comparisons (e.g. "do both models miss the same messages?") line up.
- **Tune on `val`, report on `test`.** Test predictions are used only for final numbers.

## Open questions

- Should `frontier` get its own `type` (e.g. `control`) so filtering on `decoder` returns only Qwen?
