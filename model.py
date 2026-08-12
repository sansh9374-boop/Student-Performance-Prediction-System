"""Model training and prediction for student performance."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from .preprocess import build_preprocessor, load_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES

TARGET_LEVEL = "performance_level"
TARGET_SCORE = "final_score"
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def train_models(data_path, model_dir="models"):
    df = load_data(data_path)
    X = df[FEATURES]
    y_level = df[TARGET_LEVEL]
    y_score = df[TARGET_SCORE]

    X_train, X_test, y_train_level, y_test_level, y_train_score, y_test_score = train_test_split(
        X, y_level, y_score, test_size=0.2, random_state=42, stratify=y_level
    )

    clf = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=42, class_weight="balanced"
        )),
    ])
    dt = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", DecisionTreeClassifier(max_depth=8, random_state=42, class_weight="balanced")),
    ])
    reg = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", LinearRegression()),
    ])
    rf_reg = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=250, max_depth=12, random_state=42
        )),
    ])

    clf.fit(X_train, y_train_level)
    dt.fit(X_train, y_train_level)
    reg.fit(X_train, y_train_score)
    rf_reg.fit(X_train, y_train_score)

    pred_level = clf.predict(X_test)
    pred_dt = dt.predict(X_test)
    pred_score = reg.predict(X_test)
    pred_rf_score = rf_reg.predict(X_test)

    metrics = {
        "random_forest_accuracy": accuracy_score(y_test_level, pred_level),
        "decision_tree_accuracy": accuracy_score(y_test_level, pred_dt),
        "linear_regression_mae": mean_absolute_error(y_test_score, pred_score),
        "linear_regression_r2": r2_score(y_test_score, pred_score),
        "random_forest_regression_mae": mean_absolute_error(y_test_score, pred_rf_score),
        "random_forest_regression_r2": r2_score(y_test_score, pred_rf_score),
        "classification_report": classification_report(y_test_level, pred_level, output_dict=True),
    }

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_dir / "performance_classifier.joblib")
    joblib.dump(reg, model_dir / "score_regressor.joblib")
    joblib.dump(dt, model_dir / "decision_tree_classifier.joblib")
    joblib.dump(rf_reg, model_dir / "random_forest_regressor.joblib")
    return metrics

def load_models(model_dir="models"):
    model_dir = Path(model_dir)
    return (
        joblib.load(model_dir / "performance_classifier.joblib"),
        joblib.load(model_dir / "score_regressor.joblib"),
    )
