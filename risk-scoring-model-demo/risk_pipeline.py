from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Optional
import os, pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

@dataclass(frozen=True)
class Schema:
    numeric: Sequence[str]
    categorical: Sequence[str]
    target: Optional[str] = None

def build_preprocess(schema: Schema) -> ColumnTransformer:
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric_pipe, list(schema.numeric)),("cat", categorical_pipe, list(schema.categorical))], remainder="drop")

def build_model(schema: Schema, estimator=None) -> Pipeline:
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    return Pipeline([("prep", build_preprocess(schema)),("clf", estimator)])

def fit_model(df_train: pd.DataFrame, schema: Schema, estimator=None) -> Pipeline:
    if not schema.target: raise ValueError("Schema.target must be set when training.")
    X = df_train[list(schema.numeric) + list(schema.categorical)]
    y = df_train[schema.target]
    pipe = build_model(schema, estimator)
    pipe.fit(X, y)
    return pipe

def save_model(pipe: Pipeline, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f: pickle.dump({"pipeline": pipe}, f)

def load_model(path: str) -> Pipeline:
    with open(path, "rb") as f: return pickle.load(f)["pipeline"]

def predict_proba(pipe: Pipeline, df: pd.DataFrame):
    proba = pipe.predict_proba(df)[:,1]
    return pd.Series(proba, index=df.index, name="default_proba")
