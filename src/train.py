"""Single entry point: `python -m src.train --config config.yaml`.

Reproducibility standard: this command must recreate your reported results.
It writes results/metrics.json and figures to the output directory.
"""
from __future__ import annotations
import argparse, json, os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from .data import load_data
from .model import build_model
from .evaluate import compute_metrics, save_confusion_matrix, save_roc_curve, save_combined_roc_curve


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    out = cfg["output"]["dir"]; os.makedirs(out, exist_ok=True)

    workload_rate = cfg.get("eval", {}).get("workload_rate", 0.05)

    # Cache datasets by feature_set ("static" vs "all")
    data_cache = {}
    all_results = {}
    summary_rows = []
    roc_entries = []

    for exp in cfg["experiments"]:
        name = exp["name"]
        m_type = exp["type"]
        f_set = exp.get("feature_set", "all")
        weighted = exp.get("use_class_weights", True)

        exp_out = os.path.join(out, name)
        os.makedirs(exp_out, exist_ok=True)

        if f_set not in data_cache:
            data_cache[f_set] = load_data(cfg, feature_set=f_set)
        splits = data_cache[f_set]

        model = build_model(cfg, model_type=m_type, use_class_weights=weighted)
        model.fit(splits["X_tr"], splits["y_tr"], df_tr=splits["df_tr"])

        val_prob = model.predict_proba(splits["X_va"], df_te=splits["df_va"])
        val_metrics = compute_metrics(splits["y_va"], val_prob, workload_rate=workload_rate)

        test_prob = model.predict_proba(splits["X_te"], df_te=splits["df_te"])
        test_metrics = compute_metrics(splits["y_te"], test_prob, workload_rate=workload_rate)

        exp_result = {
            "experiment": name,
            "model": m_type,
            "feature_set": f_set,
            "use_class_weights": weighted,
            "validation": val_metrics,
            "test": test_metrics,
        }
        all_results[name] = exp_result

        with open(os.path.join(exp_out, "metrics.json"), "w") as f:
            json.dump(exp_result, f, indent=2)

        save_confusion_matrix(
            splits["y_te"], test_prob, os.path.join(exp_out, "confusion_matrix.png"), workload_rate=workload_rate
        )
        save_roc_curve(
            splits["y_te"], test_prob, os.path.join(exp_out, "roc_curve.png"), model_name=name
        )
        roc_entries.append({
            "label": name,
            "y_true": splits["y_te"],
            "y_prob": test_prob,
        })
        model.save(exp_out)

        summary_rows.append({
            "experiment": name,
            "model": m_type,
            "feature_set": f_set,
            "weighted": weighted,
            "test_recall@5%": round(test_metrics["recall_at_workload"], 4),
            "test_fpr@5%": round(test_metrics["fpr_at_workload"], 4),
            "test_precision@5%": round(test_metrics["precision_at_workload"], 4),
            "test_f1@5%": round(test_metrics["f1_at_workload"], 4),
            "test_roc_auc": round(test_metrics["roc_auc"], 4),
            "test_pr_auc": round(test_metrics["pr_auc"], 4),
            "meets_criteria": test_metrics["meets_success_criteria"],
        })

    
    save_combined_roc_curve(
        roc_entries,
        os.path.join(out, "combined_roc_curves.png"),
    )

    # Write combined metrics and comparison table
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out, "comparison_summary.csv"), index=False)

    print(f"\nAll baseline and primary experiments written to '{out}/':\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)