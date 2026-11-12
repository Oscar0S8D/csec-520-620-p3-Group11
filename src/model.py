"""Model definition — a small PyTorch MLP (swap in your own architecture)."""
from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression




class RuleBasedWrapper:
    # Baseline 1: Predefined domain threshold rules
    def __init__(self, cfg: dict):
        self.thresholds = cfg["model"]["rule_thresholds"]

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray, df_tr: pd.DataFrame | None = None) -> None:
        pass

    def predict_proba(self, X_te: np.ndarray, df_te: pd.DataFrame | None = None) -> np.ndarray:
        if df_te is None:
            raise ValueError("RuleBasedWrapper requires unscaled df_te DataFrame.")

        score = np.zeros(len(df_te), dtype=np.float64)
        active_rules = 0

        if "amount" in df_te.columns and "amount" in self.thresholds:
            score += (df_te["amount"].values >= self.thresholds["amount"]).astype(float)
            active_rules += 1
        if "foreign_transaction" in df_te.columns:
            score += (df_te["foreign_transaction"].values == 1).astype(float)
            active_rules += 1
        if "location_mismatch" in df_te.columns:
            score += (df_te["location_mismatch"].values == 1).astype(float)
            active_rules += 1
        if "velocity_last_24h" in df_te.columns and "velocity_last_24h" in self.thresholds:
            score += (df_te["velocity_last_24h"].values >= self.thresholds["velocity_last_24h"]).astype(float)
            active_rules += 1
        if "device_trust_score" in df_te.columns and "device_trust_score" in self.thresholds:
            score += (df_te["device_trust_score"].values <= self.thresholds["device_trust_score"]).astype(float)
            active_rules += 1

        return score / max(1, active_rules)

    def save(self, out_dir: str) -> None:
        joblib.dump(self.thresholds, f"{out_dir}/rule_thresholds.joblib")
class LogisticRegressionWrapper:
    # Baseline 2: Fast, interpretable linear fraud benchmark.
    def __init__(self, cfg: dict, use_class_weights: bool = True):
        d = cfg["model"]
        cw = "balanced" if use_class_weights else None
        self.model = LogisticRegression(
            C=d.get("logreg_c", 1.0),
            max_iter=d.get("logreg_max_iter", 1000),
            class_weight=cw,
            random_state=cfg["seed"],
        )

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray, df_tr=None) -> None:
        self.model.fit(X_tr, y_tr)

    def predict_proba(self, X_te: np.ndarray, df_te=None) -> np.ndarray:
        return self.model.predict_proba(X_te)[:, 1]

    def save(self, out_dir: str) -> None:
        joblib.dump(self.model, f"{out_dir}/logreg_model.joblib")

class GBDT:
    # Primary Model: Scikit-Learn Gradient Boosted Decision Trees
    def __init__(self, cfg: dict, use_class_weights: bool = True):
        d = cfg["model"]
        cw = "balanced" if use_class_weights else None
        self.model = HistGradientBoostingClassifier(
            max_iter=d.get("gbdt_max_iter", 200),
            learning_rate=d.get("gbdt_learning_rate", 0.05),
            max_depth=d.get("gbdt_max_depth", 6),
            class_weight=cw,
            random_state=cfg["seed"],
        )

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray, df_tr=None) -> None:
        self.model.fit(X_tr, y_tr)

    def predict_proba(self, X_te: np.ndarray, df_te=None) -> np.ndarray:
        return self.model.predict_proba(X_te)[:, 1]

    def save(self, out_dir: str) -> None:
        joblib.dump(self.model, os.path.join(out_dir, "gbdt_model.joblib"))


def build_model(cfg: dict, model_type: str, use_class_weights: bool = True):
    m_type = model_type.lower()
    if m_type == "rule":
        return RuleBasedWrapper(cfg)
    elif m_type == "logreg":
        return LogisticRegressionWrapper(cfg, use_class_weights=use_class_weights)
    elif m_type == "gbdt":
        return GBDT(cfg, use_class_weights=use_class_weights)
    raise ValueError(f"Unknown model type: {model_type}")
