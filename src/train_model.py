# src/train_model.py

from __future__ import annotations

import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def try_make_dataset(data_path: str, target: str, drop_cols: list[str] | None):
    """
    Try to use feature_engineering.py if available.
    If not available, return None so we can fall back.
    """
    try:
        # If src is a package, this import works when you run from repo root:
        # python -m src.train_model ...
        from src.feature_engineering import make_dataset  # type: ignore
    except Exception:
        try:
            # If you run as: python src/train_model.py ...
            from feature_engineering import make_dataset  # type: ignore
        except Exception:
            return None

    return make_dataset(data_path, target_col=target, drop_cols=drop_cols)


def baseline_raw_pipeline(df: pd.DataFrame, target: str, drop_cols: list[str] | None, task: str):
    """
    Baseline preprocessing for raw dataset:
    - Drop requested columns
    - Split X/y
    - ColumnTransformer:
        numeric: median impute
        categorical: most_frequent impute + onehot
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    target = target.strip().lower().replace(" ", "_")

    if drop_cols:
        drop_cols = [c.strip().lower().replace(" ", "_") for c in drop_cols]
        df = df.drop(columns=drop_cols, errors="ignore")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")

    y = df[target]
    X = df.drop(columns=[target])

    # Detect column types
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    if task == "classification":
        model = LogisticRegression(max_iter=2000)
    else:
        model = LinearRegression()

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return X, y, pipe


def train_and_eval(X, y, pipe, task: str):
    # Split
    if task == "classification":
        stratify = y if y.nunique() <= 20 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Fit
    pipe.fit(X_train, y_train)

    # Predict + metrics
    preds = pipe.predict(X_test)

    if task == "classification":
        acc = accuracy_score(y_test, preds)
        print(f"\nAccuracy: {acc:.4f}\n")
        print("Classification report:")
        print(classification_report(y_test, preds))
    else:
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)
        print(f"\nRMSE: {rmse:.4f}")
        print(f"R2:   {r2:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Baseline model training script.")
    parser.add_argument("--data", type=str, default="data/student_dropout_dataset_v3.csv", help="Path to CSV")
    parser.add_argument("--target", type=str, required=True, help="Target column name (exact column in CSV)")
    parser.add_argument("--task", type=str, choices=["classification", "regression"], required=True)
    parser.add_argument("--drop", nargs="*", default=None, help="Optional columns to drop")
    parser.add_argument("--use_tree", action="store_true", help="Use Decision Tree instead of linear baseline")
    args = parser.parse_args()

    # 1) Prefer feature_engineering.py if it exists
    fe_out = try_make_dataset(args.data, args.target, args.drop)

    if fe_out is not None:
        X, y = fe_out
        # If you have engineered features already, X should be numeric
        if args.task == "classification":
            model = DecisionTreeClassifier(random_state=42) if args.use_tree else LogisticRegression(max_iter=2000)
        else:
            model = DecisionTreeRegressor(random_state=42) if args.use_tree else LinearRegression()

        pipe = Pipeline(steps=[("model", model)])
        print("Using feature_engineering.py pipeline ✅")
        train_and_eval(X, y, pipe, args.task)
        return

    # 2) Fallback: raw CSV baseline preprocessing
    df = pd.read_csv(args.data)
    X, y, pipe = baseline_raw_pipeline(df, args.target, args.drop, args.task)

    # swap model if requested
    if args.use_tree:
        if args.task == "classification":
            pipe.set_params(model=DecisionTreeClassifier(random_state=42))
        else:
            pipe.set_params(model=DecisionTreeRegressor(random_state=42))

    print("feature_engineering.py not found yet — using raw baseline preprocessing ⚠️")
    train_and_eval(X, y, pipe, args.task)


if __name__ == "__main__":
    main()

#murshid is gay