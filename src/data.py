"""Chronological 70/15/15 data loading and controlled feature-set selection."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

STATIC_COLS = [
    "amount",
    "transaction_hour",
    "merchant_category",
    "foreign_transaction",
    "location_mismatch",
    "cardholder_age",
]

HISTORY_COLS = [
    "device_trust_score",
    "velocity_last_24h",
]


def load_data(cfg: dict, feature_set: str | None = None) -> dict:
    data_cfg = cfg["data"]
    df = pd.read_csv(data_cfg["csv_path"])
    target_col = data_cfg["target"]

    if "transaction_id" in df.columns:
        df = df.sort_values("transaction_id").reset_index(drop=True)

    y = df[target_col].values.astype(np.int64)

    f_set = feature_set or data_cfg.get("feature_set", "all")
    if f_set == "static":
        selected_cols = [c for c in STATIC_COLS if c in df.columns]
    elif f_set == "all":
        selected_cols = [c for c in (STATIC_COLS + HISTORY_COLS) if c in df.columns]
    else:
        raise ValueError(f"Unknown feature_set: {f_set}")

    X_df = df[selected_cols].copy()
    X_df = pd.get_dummies(X_df, drop_first=True, dtype=float)
    X = X_df.values

    # Chronological 70% Train / 15% Val / 15% Test split (no shuffling)
    n = len(df)
    train_end = int(n * data_cfg.get("train_ratio", 0.70))
    val_end = int(n * (data_cfg.get("train_ratio", 0.70) + data_cfg.get("val_ratio", 0.15)))

    X_tr, y_tr = X[:train_end], y[:train_end]
    X_va, y_va = X[train_end:val_end], y[train_end:val_end]
    X_te, y_te = X[val_end:], y[val_end:]

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr).astype(np.float32)
    X_va_scaled = scaler.transform(X_va).astype(np.float32)
    X_te_scaled = scaler.transform(X_te).astype(np.float32)

    return {
        "X_tr": X_tr_scaled,
        "y_tr": y_tr,
        "X_va": X_va_scaled,
        "y_va": y_va,
        "X_te": X_te_scaled,
        "y_te": y_te,
        "df_tr": X_df.iloc[:train_end],
        "df_va": X_df.iloc[train_end:val_end],
        "df_te": X_df.iloc[val_end:],
    }