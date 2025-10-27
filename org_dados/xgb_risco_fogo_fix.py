#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost for RiscoFogoMedia (Tabular, Regression) - FIXED
- Removes '_dt_raw' from numeric feature list before ColumnTransformer
- Prunes categorical/ numeric lists to columns that actually exist after split
"""
import warnings
warnings.filterwarnings("ignore")

import os
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
from xgboost import XGBRegressor

DATA_PATH: str = "proc_data/saida_merged_2016.csv"
TARGET: str = "RiscoFogoMedia"
ID_COLS: List[str] = []
TIME_COL: Optional[str] = "datahora"
USE_TIME_BASED_SPLIT: bool = True
TRAIN_FRAC: float = 0.70
VALID_FRAC: float = 0.15
RANDOM_STATE = 42

CATEGORICAL_OVERRIDE: Optional[List[str]] = ["uf_nome", "municipio_nome"]
NUMERIC_OVERRIDE: Optional[List[str]] = None
TREE_METHOD = "hist"
MAKE_PLOTS = True

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.where(y_true == 0, 1e-8, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom))

def infer_feature_types(df: pd.DataFrame, target: str, id_cols: List[str],
                        cat_override: Optional[List[str]], num_override: Optional[List[str]]):
    cols = [c for c in df.columns if c not in id_cols + [target]]
    if cat_override is not None or num_override is not None:
        categorical_cols = cat_override if cat_override is not None else []
        numeric_cols = [c for c in cols if c not in categorical_cols] if num_override is None else num_override
        return categorical_cols, numeric_cols
    cat = [c for c in cols if str(df[c].dtype) in ("object", "category", "bool", "boolean")]
    num = [c for c in cols if c not in cat]
    return cat, num

def build_preprocess(categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
    num_transformer = SimpleImputer(strategy="median")
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])
    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_cols),
            ("cat", cat_transformer, categorical_cols),
        ],
        remainder="drop"
    )
    return preprocess

def add_datetime_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if time_col not in df.columns:
        return df
    dt = pd.to_datetime(df[time_col], errors="coerce")
    df["dt_hour"] = dt.dt.hour
    df["dt_dayofweek"] = dt.dt.dayofweek
    df["dt_month"] = dt.dt.month
    df["dt_dayofyear"] = dt.dt.dayofyear
    df["dt_is_weekend"] = (df["dt_dayofweek"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["dt_hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["dt_hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dt_dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dt_dayofweek"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * df["dt_month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["dt_month"] / 12.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["dt_dayofyear"] / 366.0)
    df["doy_cos"] = np.cos(2 * np.pi * df["dt_dayofyear"] / 366.0)

    to_drop = [time_col, "dt_hour", "dt_dayofweek", "dt_month", "dt_dayofyear"]
    return df.drop(columns=[c for c in to_drop if c in df.columns])

def time_based_split(df: pd.DataFrame, time_col: str, target: str,
                     train_frac: float, valid_frac: float):
    assert 0 < train_frac < 1 and 0 < valid_frac < 1 and train_frac + valid_frac < 1
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    n_valid = int(n * (train_frac + valid_frac))
    train = df.iloc[:n_train]
    valid = df.iloc[n_train:n_valid]
    test  = df.iloc[n_valid:]
    X_train, y_train = train.drop(columns=[target]), train[target]
    X_valid, y_valid = valid.drop(columns=[target]), valid[target]
    X_test,  y_test  = test.drop(columns=[target]),  test[target]
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def random_split(X, y, test_size=0.20, valid_size=0.20, seed=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    valid_ratio_of_temp = valid_size / (1.0 - test_size)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - valid_ratio_of_temp, random_state=seed
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"DATA_PATH não existe: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    assert TARGET in df.columns, f"O target '{TARGET}' não está nas colunas"
    if ID_COLS:
        df = df.drop(columns=[c for c in ID_COLS if c in df.columns]).copy()

    if TIME_COL and TIME_COL in df.columns:
        df["_dt_raw"] = pd.to_datetime(df[TIME_COL], errors="coerce")
    else:
        df["_dt_raw"] = pd.NaT

    if TIME_COL:
        df = add_datetime_features(df, TIME_COL)

    categorical_cols, numeric_cols = infer_feature_types(
        df, TARGET, id_cols=[],
        cat_override=CATEGORICAL_OVERRIDE,
        num_override=NUMERIC_OVERRIDE
    )

    # Remove '_dt_raw' from numeric list if present
    if "_dt_raw" in numeric_cols:
        numeric_cols = [c for c in numeric_cols if c != "_dt_raw"]

    y = df[TARGET]
    X = df.drop(columns=[TARGET]).copy()

    if USE_TIME_BASED_SPLIT and "_dt_raw" in X.columns and X["_dt_raw"].notna().any():
        tmp = X.copy(); tmp[TARGET] = y
        X_train, X_valid, X_test, y_train, y_valid, y_test = time_based_split(
            tmp, "_dt_raw", TARGET, TRAIN_FRAC, VALID_FRAC
        )
        for part in (X_train, X_valid, X_test):
            if "_dt_raw" in part.columns:
                part.drop(columns=["_dt_raw"], inplace=True)
    else:
        X_train, X_valid, X_test, y_train, y_valid, y_test = random_split(
            X, y, test_size=0.20, valid_size=0.20, seed=RANDOM_STATE
        )
        for part in (X_train, X_valid, X_test):
            if "_dt_raw" in part.columns:
                part.drop(columns=["_dt_raw"], inplace=True)

    # Prune lists to existing columns
    categorical_cols = [c for c in categorical_cols if c in X_train.columns]
    numeric_cols = [c for c in numeric_cols if c in X_train.columns]

    print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")
    print("Categóricas:", categorical_cols)
    print("Numéricas  :", numeric_cols)

    preprocess = build_preprocess(categorical_cols, numeric_cols)
    X_train_p = preprocess.fit_transform(X_train)
    X_valid_p = preprocess.transform(X_valid)
    X_test_p  = preprocess.transform(X_test)

    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=2.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method=TREE_METHOD,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
        eval_metric="rmse"
    )

    model.fit(
        X_train_p, y_train,
        eval_set=[(X_valid_p, y_valid)],
        verbose=False
    )

    y_pred = model.predict(X_test_p)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    y_true = np.array(y_test)
    denom = np.where(y_true == 0, 1e-8, np.abs(y_true))
    mape_val = np.mean(np.abs((y_true - y_pred) / denom))

    print("\n== Métricas (Teste) ==")
    print(f"MAE :  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²  : {r2:.6f}")
    print(f"MAPE: {mape_val:.6f}")

    artifact = {
        "preprocess": preprocess,
        "model": model,
        "target": TARGET,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols
    }
    joblib.dump(artifact, "xgb_model.joblib")
    print("\nSalvo: xgb_model.joblib")

    if MAKE_PLOTS:
        try:
            plt.figure(figsize=(8,6))
            plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
            plt.tight_layout()
            plt.savefig("feature_importance.png", dpi=150)
            plt.close()

            plt.figure(figsize=(6,6))
            plt.scatter(y_test, y_pred, alpha=0.4)
            mn = float(min(np.min(y_test), np.min(y_pred)))
            mx = float(max(np.max(y_test), np.max(y_pred)))
            plt.plot([mn, mx], [mn, mx])
            plt.xlabel("True")
            plt.ylabel("Predicted")
            plt.tight_layout()
            plt.savefig("pred_vs_true.png", dpi=150)
            plt.close()

            print("Salvou: feature_importance.png, pred_vs_true.png")
        except Exception as e:
            print(f"Falha ao gerar plots: {e}")

if __name__ == "__main__":
    main()
