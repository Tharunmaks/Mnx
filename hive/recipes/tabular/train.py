"""Spreadsheet predictor: gradient boosting on a CSV, classification or regression.

Mnx Lab protocol: params.json + data/ in, output/ out, MNX_METRIC / MNX_RESULT lines.
"""
import glob
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

P = json.load(open("params.json"))
os.makedirs("output", exist_ok=True)
files = sorted(glob.glob("data/*.csv"))
if not files:
    sys.exit("No .csv file in the dataset")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
target = P.get("target", "")
if target not in df.columns:
    sys.exit(f"Column '{target}' not found. Columns: {', '.join(map(str, df.columns))}")
df = df.dropna(subset=[target])
y = df[target]
X = df.drop(columns=[target])

is_class = not pd.api.types.is_numeric_dtype(y) or (pd.api.types.is_integer_dtype(y) and y.nunique() <= 20)
task = "classification" if is_class else "regression"
cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
X[cat_cols] = X[cat_cols].astype(object)  # one text type for the encoder (pandas 2 and 3)
num_cols = [c for c in X.columns if c not in cat_cols]
print(f"{len(df)} rows, {len(X.columns)} input columns ({len(cat_cols)} text), task: {task}", flush=True)

pre = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1), cat_cols),
    ("num", "passthrough", num_cols),
])
cat_mask = [True] * len(cat_cols) + [False] * len(num_cols)
Model = HistGradientBoostingClassifier if is_class else HistGradientBoostingRegressor
est = Model(learning_rate=float(P.get("learning_rate", 0.1)), max_iter=10, warm_start=True,
            categorical_features=cat_mask or None, random_state=42)
pipe = Pipeline([("pre", pre), ("model", est)])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=float(P.get("test_split", 0.2)), random_state=42,
                                      stratify=y if is_class and y.value_counts().min() >= 2 else None)
total = int(P.get("iterations", 200))
step = max(10, total // 40)
done = 0
while done < total:
    done = min(total, done + step)
    pipe.set_params(model__max_iter=done)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    if is_class:
        print("MNX_METRIC " + json.dumps({"step": done, "val_accuracy": round(float(accuracy_score(yte, pred)), 4)}), flush=True)
    else:
        print("MNX_METRIC " + json.dumps({"step": done, "val_mae": round(float(mean_absolute_error(yte, pred)), 4),
                                          "val_r2": round(float(r2_score(yte, pred)), 4)}), flush=True)

pred = pipe.predict(Xte)
if is_class:
    result = {"task": task, "accuracy": round(float(accuracy_score(yte, pred)), 4)}
else:
    result = {"task": task, "mae": round(float(mean_absolute_error(yte, pred)), 4), "r2": round(float(r2_score(yte, pred)), 4)}
result.update({"rows": int(len(df)), "target": target, "inputs": list(map(str, X.columns))})
example = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in X.iloc[0].to_dict().items()}
joblib.dump({"pipeline": pipe, "task": task, "inputs": list(X.columns), "target": target, "example": example}, "output/model.joblib")
json.dump(result, open("output/metrics.json", "w"), indent=2)
print("MNX_RESULT " + json.dumps(result), flush=True)
