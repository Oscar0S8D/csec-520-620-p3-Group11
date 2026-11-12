# Project 5 (bonus) — `results/metrics.json` format

Set `project: p5` at the top of `config.yaml`. Then `python grading/grade.py`
runs `make reproduce` and checks `results/metrics.json` against this format.

While you are iterating, `python grading/grade.py --skip-reproduce` checks the
results you already have without rerunning the grid search. Run it once without
the flag before you submit.

## Rules

- `make reproduce` must regenerate `results/metrics.json` and every figure.
- All metrics are fractions in [0, 1], computed on the **official testing set**
  (`UNSW_NB15_testing-set.csv`) unless the key says `train` or `cv`.
- `fpr` = false positives / all normal rows. `recall` = detection rate = true
  positives / all attack rows.
- Use exactly the settings in the handout: `C` ∈ {0.01, 0.1, 1, 10, 100} for the
  linear SVM; `C` ∈ {0.1, 1, 10, 100} × `gamma` ∈ {0.001, 0.01, 0.1, 1} for the RBF
  grid; `nu` ∈ {0.01, 0.05, 0.1} for the one-class SVM.
- The top-level metrics in each model block are for the **chosen** setting.
- `per_category` lists the detection rate for each of the nine UNSW-NB15 attack
  categories (`attack_cat`), not `Normal`.

## Example

```json
{
  "project": "p5",
  "data": {
    "train_rows": 175341,
    "test_rows": 82332,
    "subsample_rows": 20000,
    "seed": 42,
    "n_features": 56
  },
  "linear": {
    "C": 1,
    "precision": 0.81, "recall": 0.97, "f1": 0.88, "fpr": 0.25,
    "train_seconds": 4.2,
    "sweep": [
      {"C": 0.01, "train": {"precision": 0.90, "recall": 0.95, "f1": 0.92, "fpr": 0.12},
                  "test":  {"precision": 0.80, "recall": 0.96, "f1": 0.87, "fpr": 0.26}},
      {"C": 0.1,  "train": {"...": "..."}, "test": {"...": "..."}},
      {"C": 1,    "train": {"...": "..."}, "test": {"precision": 0.81, "recall": 0.97, "f1": 0.88, "fpr": 0.25}},
      {"C": 10,   "train": {"...": "..."}, "test": {"...": "..."}},
      {"C": 100,  "train": {"...": "..."}, "test": {"...": "..."}}
    ],
    "top_features": [
      {"feature": "sttl", "weight": 2.31},
      {"feature": "ct_state_ttl", "weight": -1.87},
      "... ten entries in all, sorted by |weight|, largest first ..."
    ]
  },
  "rbf": {
    "C": 10, "gamma": 0.1,
    "precision": 0.83, "recall": 0.98, "f1": 0.90, "fpr": 0.22,
    "n_support_vectors": 4210,
    "train_seconds": 41.0,
    "grid": [
      {"C": 0.1, "gamma": 0.001, "train_f1": 0.88, "cv_f1": 0.88},
      "... all 16 (C, gamma) settings ..."
    ]
  },
  "one_class": {
    "nu": 0.05,
    "precision": 0.70, "recall": 0.62, "f1": 0.66, "fpr": 0.31,
    "n_support_vectors": 1180,
    "runs": [
      {"nu": 0.01, "detection_rate": 0.48, "fpr": 0.20,
       "per_category": {"Analysis": 0.31, "Backdoor": 0.35, "DoS": 0.52, "Exploits": 0.66,
                        "Fuzzers": 0.21, "Generic": 0.40, "Reconnaissance": 0.70,
                        "Shellcode": 0.88, "Worms": 0.95}},
      {"nu": 0.05, "detection_rate": 0.62, "fpr": 0.31, "per_category": {"...": "..."}},
      {"nu": 0.1,  "detection_rate": 0.71, "fpr": 0.38, "per_category": {"...": "..."}}
    ]
  }
}
```

The numbers above only illustrate the format; they are not expected results.

## What the grader checks

| Check | Fails when | Flags for review when |
|---|---|---|
| `p5_metrics_fresh` | `metrics.json` is missing | `make reproduce` did not rewrite it |
| `p5_data` | a field is missing or not an integer, or `source` is `synthetic` | row counts differ from 175,341 / 82,332 / 20,000 |
| `p5_linear` | a C value, metric, `top_features` (10) or `train_seconds` is missing | the summary differs from the sweep at the chosen C; features not sorted by \|weight\| |
| `p5_rbf` | a grid cell, metric, `n_support_vectors` or `train_seconds` is missing | the chosen setting is not the best `cv_f1` |
| `p5_one_class` | a nu run, rate or `per_category` is missing | an attack category is missing, or `Normal` is included |
| `p5_figures` | — | fewer than three figures in `results/` |

A `review` flag is not a deduction by itself; the grader reads your report and code
to decide.
