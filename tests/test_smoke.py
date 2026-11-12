"""Smoke tests verifying config loading, data splitting, and model interfaces."""
from __future__ import annotations
import yaml
from src.data import load_data
from src.model import build_model
from src.evaluate import compute_metrics


def test_pipeline_smoke():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    splits = load_data(cfg, feature_set="all")
    assert len(splits["X_tr"]) > 0
    assert len(splits["X_va"]) > 0
    assert len(splits["X_te"]) > 0

    for m_type in ("rule", "logreg", "gbdt"):
        model = build_model(cfg, model_type=m_type, use_class_weights=True)
        model.fit(splits["X_tr"], splits["y_tr"], df_tr=splits["df_tr"])
        probs = model.predict_proba(splits["X_te"], df_te=splits["df_te"])
        assert len(probs) == len(splits["y_te"])

        metrics = compute_metrics(splits["y_te"], probs, workload_rate=0.05)
        assert 0.0 <= metrics["recall_at_workload"] <= 1.0
        assert 0.0 <= metrics["fpr_at_workload"] <= 1.0