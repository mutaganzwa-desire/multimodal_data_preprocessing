"""Product recommendation model: predicts the product category a customer buys.

Trains on ``datasets/merged_customer_recommendation_dataset.csv`` (transactions
joined with each customer's aggregated social-media profile). Only information
known *before* the purchase is used: the social-profile features and the visit
date. The purchase amount and the rating are excluded because they exist only
after the purchase.

Logistic Regression, Random Forest and XGBoost are compared with a most-frequent
-class baseline using 5-fold cross-validation repeated 5 times. Folds are grouped
by ``customer_id`` so a customer's purchases are never split between training
and testing. The model with the best macro F1 is saved to
``models/product_model.joblib``.

    python -m src.product_model         # train, evaluate and save
"""

from __future__ import annotations

import argparse
import datetime as dt
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.model_store import load_or_train, save

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = REPO_ROOT / "datasets" / "merged_customer_recommendation_dataset.csv"
MODEL_PATH = REPO_ROOT / "models" / "product_model.joblib"
SEED = 42
N_SPLITS, N_REPEATS = 5, 5
TARGET = "product_category"

PROFILE_FEATURES = [
    "has_social_profile", "social_record_count", "platform_count",
    "avg_engagement_score", "max_engagement_score",
    "avg_purchase_interest_score", "std_purchase_interest_score", "engagement_x_interest",
    "positive_sentiment_share", "neutral_sentiment_share", "negative_sentiment_share",
]
DATE_FEATURES = ["purchase_month", "purchase_dayofweek"]
CATEGORICAL = ["dominant_platform"]
FEATURES = PROFILE_FEATURES + DATE_FEATURES + CATEGORICAL


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", StandardScaler(), PROFILE_FEATURES + DATE_FEATURES),
        ("platform", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def candidate_models() -> dict:
    return {
        "baseline_most_frequent": DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(
            _preprocessor(), LogisticRegression(C=0.1, max_iter=5000, random_state=SEED)),
        "random_forest": make_pipeline(
            _preprocessor(), RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                    random_state=SEED, n_jobs=1)),
        "xgboost": make_pipeline(
            _preprocessor(), XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                                           random_state=SEED, n_jobs=1, verbosity=0)),
    }


def evaluate(model, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_classes: int):
    """Repeated grouped CV: mean/std of accuracy, macro F1, log loss; first-repeat OOF probs."""
    scores, first_oof = [], None
    for repeat in range(N_REPEATS):
        probs = np.zeros((len(X), n_classes))
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat)
        for train, test in cv.split(X, y, groups):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = clone(model).fit(X.iloc[train], y[train])
            probs[test] = fitted.predict_proba(X.iloc[test])
        pred = probs.argmax(axis=1)
        clipped = np.clip(probs, 1e-6, 1)
        clipped /= clipped.sum(axis=1, keepdims=True)
        scores.append({"accuracy": accuracy_score(y, pred),
                       "f1_macro": f1_score(y, pred, average="macro"),
                       "log_loss": log_loss(y, clipped, labels=range(n_classes))})
        first_oof = probs if first_oof is None else first_oof
    frame = pd.DataFrame(scores)
    summary = {f"{k}_mean": float(v) for k, v in frame.mean().items()}
    summary.update({f"{k}_std": float(v) for k, v in frame.std().items()})
    return summary, first_oof


@dataclass
class ProductModel:
    classes: list
    model_name: str
    pipeline: object
    profiles: pd.DataFrame       # latest profile features per customer_id
    metrics: dict
    sklearn_version: str = sklearn.__version__

    def features_for(self, customer_id: int, visit_date: dt.date | None = None) -> pd.DataFrame:
        """One feature row: the customer's social profile plus the visit date."""
        if customer_id not in self.profiles.index:
            raise KeyError(f"customer {customer_id} is not in the dataset")
        visit_date = visit_date or dt.date.today()
        row = self.profiles.loc[[customer_id], PROFILE_FEATURES + CATEGORICAL].copy()
        row["purchase_month"] = visit_date.month
        row["purchase_dayofweek"] = visit_date.weekday()
        return row[FEATURES]

    def recommend(self, customer_id: int, visit_date: dt.date | None = None) -> pd.Series:
        """Probability of each product category, most likely first."""
        probs = self.pipeline.predict_proba(self.features_for(customer_id, visit_date))[0]
        return pd.Series(probs, index=self.classes, name="probability").sort_values(ascending=False)


def train(data_csv: Path = DATA_CSV, verbose: bool = True) -> ProductModel:
    data = pd.read_csv(data_csv, parse_dates=["purchase_date"])
    classes = sorted(data[TARGET].unique())
    X, y = data[FEATURES], data[TARGET].map({c: i for i, c in enumerate(classes)}).to_numpy()
    groups = data["customer_id"].to_numpy()

    comparison, oof = {}, {}
    for name, model in candidate_models().items():
        comparison[name], oof[name] = evaluate(model, X, y, groups, len(classes))
    table = pd.DataFrame(comparison).T
    real = table.drop(index="baseline_most_frequent")
    model_name = real["f1_macro_mean"].idxmax()
    if verbose:
        print(f"[product] {N_REPEATS}x repeated {N_SPLITS}-fold CV grouped by customer_id:")
        cols = ["accuracy_mean", "accuracy_std", "f1_macro_mean", "f1_macro_std", "log_loss_mean"]
        print(table[cols].round(3).to_string())
        print(f"[product] selected: {model_name} (best macro F1)")

    pred = oof[model_name].argmax(axis=1)
    metrics = {
        "selected_model": model_name,
        "comparison": table.to_dict(orient="index"),
        "cv": comparison[model_name],
        "confusion_matrix": confusion_matrix(y, pred, labels=range(len(classes))).tolist(),
        "n_rows": len(data), "n_customers": int(data["customer_id"].nunique()),
        "class_counts": data[TARGET].value_counts().sort_index().to_dict(),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline = clone(candidate_models()[model_name]).fit(X, y)
    profiles = (data.sort_values("purchase_date").groupby("customer_id").last()[PROFILE_FEATURES + CATEGORICAL])
    return ProductModel(classes, model_name, pipeline, profiles, metrics)


def load(path: Path = MODEL_PATH) -> ProductModel:
    """Load the saved model, retraining it if it is missing or incompatible."""
    return load_or_train(path, train)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    # Import the module by its package name so the saved model refers to
    # src.product_model.ProductModel (not __main__), and loads from anywhere.
    from src import product_model

    model = product_model.train()
    save(model, args.output)
    cv = model.metrics["cv"]
    print(f"Saved {args.output.relative_to(REPO_ROOT)} | accuracy {cv['accuracy_mean']:.3f}, "
          f"F1 {cv['f1_macro_mean']:.3f}, log loss {cv['log_loss_mean']:.3f}")


if __name__ == "__main__":
    main()
