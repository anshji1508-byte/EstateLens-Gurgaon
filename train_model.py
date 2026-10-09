"""Train and persist the Gurgaon property price regression model.

The target is stored in crore rupees.  price_per_sqft is deliberately excluded:
it is calculated from price and would leak the answer into the model.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Cleaned" / "property_outlier_treated.csv"
ARTIFACT_DIR = ROOT / "Pkles"
MODEL_PATH = ARTIFACT_DIR / "property_price_model.pkl"
METRICS_PATH = ARTIFACT_DIR / "model_metrics.json"

TARGET = "price"
RAW_FEATURES = [
    "property_type", "sector", "area", "bedRoom", "bathroom",
    "balcony", "floorNum", "agePossession", "Pooja Room",
    "Study Room", "Store Room", "Servant Room", "Others", "furnishstatus",
    "super_built_up_area", "built_up_area", "carpet_area",
]
CATEGORICAL = ["property_type", "sector", "agePossession"]


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create only prediction-time features (no target-derived values)."""
    missing = [c for c in RAW_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    out = frame[RAW_FEATURES].copy()
    out["log_area"] = np.log1p(out["area"].clip(lower=1))
    beds = out["bedRoom"].clip(lower=1)
    out["area_per_bedroom"] = out["area"] / beds
    out["bathrooms_per_bedroom"] = out["bathroom"] / beds
    out["built_area_missing"] = out["built_up_area"].isna().astype(int)
    out["carpet_area_missing"] = out["carpet_area"].isna().astype(int)
    return out


def build_pipeline() -> Pipeline:
    all_features = RAW_FEATURES + [
        "log_area", "area_per_bedroom", "bathrooms_per_bedroom",
        "built_area_missing", "carpet_area_missing",
    ]
    numeric = [c for c in all_features if c not in CATEGORICAL]
    preprocessor = ColumnTransformer(
        transformers=[
            # Built-in sklearn encoder keeps the project installable without
            # the optional category_encoders package.
            ("categorical", TargetEncoder(
                smooth=10, cv=KFold(5, shuffle=True, random_state=42)
            ), CATEGORICAL),
            ("numeric", SimpleImputer(strategy="median"), numeric),
        ],
        remainder="drop",
    )
    model = XGBRegressor(
        n_estimators=1100,
        max_depth=6,
        learning_rate=0.025,
        subsample=0.80,
        colsample_bytree=0.85,
        min_child_weight=16,
        reg_alpha=0.20,
        reg_lambda=3.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", model)])


def main() -> None:
    data = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    X = make_features(data)
    y = np.log1p(data[TARGET])
    train_idx, test_idx = train_test_split(
        np.arange(len(data)), test_size=0.20, random_state=42
    )

    holdout_model = build_pipeline()
    holdout_model.fit(X.iloc[train_idx], y.iloc[train_idx])
    prediction = np.clip(np.expm1(holdout_model.predict(X.iloc[test_idx])), 0, None)
    actual = data[TARGET].iloc[test_idx]
    holdout = {
        "mae_crore": float(mean_absolute_error(actual, prediction)),
        "rmse_crore": float(np.sqrt(mean_squared_error(actual, prediction))),
        "r2": float(r2_score(actual, prediction)),
        "test_rows": int(len(test_idx)),
    }

    cv = cross_validate(
        build_pipeline(), X, y, cv=KFold(5, shuffle=True, random_state=42),
        scoring={"r2": "r2", "mae_log": "neg_mean_absolute_error"}, n_jobs=1,
    )
    cv_metrics = {
        "cv_r2_mean": float(cv["test_r2"].mean()),
        "cv_r2_std": float(cv["test_r2"].std()),
        "cv_log_mae_mean": float(-cv["test_mae_log"].mean()),
    }

    final_model = build_pipeline().fit(X, y)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    bundle = {
        "model": final_model,
        "raw_features": RAW_FEATURES,
        "metrics": {**holdout, **cv_metrics},
        "target": "price (crore INR)",
        "version": "1.0",
    }
    with MODEL_PATH.open("wb") as file:
        pickle.dump(bundle, file, protocol=pickle.HIGHEST_PROTOCOL)
    METRICS_PATH.write_text(json.dumps(bundle["metrics"], indent=2))
    print(json.dumps(bundle["metrics"], indent=2))
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
