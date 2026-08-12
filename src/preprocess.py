"""Data loading and preprocessing utilities."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "attendance", "study_hours", "internal_marks", "participation",
    "previous_score", "assignment_completion", "weak_subjects"
]
CATEGORICAL_FEATURES = ["gender", "study_mode", "subject"]

def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates().copy()
    return df

def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
