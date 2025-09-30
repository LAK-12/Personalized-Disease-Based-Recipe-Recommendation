"""
Train a simple logistic regression using weak labels from rules.
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump
import yaml

DATA_CSV = os.getenv("RECIPES_CSV", "data/recipes_sample.csv")
RULES_PATH = os.getenv("RULES_PATH", "src/rules/nutrition_rules.yaml")
MODEL_OUT = os.getenv("MODEL_PATH", "data/model_lr.pkl")
TARGET = os.getenv("TARGET_CONDITION", "diabetes")  # diabetes | hypertension | celiac | kidney_friendly

rules = yaml.safe_load(open(RULES_PATH))
df = pd.read_csv(DATA_CSV)

df.columns = [c.strip().lower().replace(" ", "_").replace("(g)", "").replace("g_", "g_") for c in df.columns]

if "sugar_g" not in df.columns:
    df["sugar_g"] = 0.0

required = ["calories","carbs_g","sugar_g","protein_g","fat_g","fiber_g","sodium_mg","prep_minutes"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")

def label_row(r):
    if TARGET == "diabetes":
        return int(
            (r["carbs_g"] <= rules["conditions"]["diabetes"]["max_carbs_g_per_serving"])
            and (r["fiber_g"] >= rules["conditions"]["diabetes"]["min_fiber_g"])
            and (r["sugar_g"] <= rules["conditions"]["diabetes"]["max_sugar_g_per_serving"])
        )
    if TARGET == "hypertension":
        return int(r["sodium_mg"] <= rules["conditions"]["hypertension"]["max_sodium_mg"])
    if TARGET == "celiac":
        text = str(r["ingredients_text"]).lower()
        bad = set(rules["conditions"]["celiac"]["exclude_terms"])
        return int(not any(t in text for t in bad))
    if TARGET == "kidney_friendly":
        return int(r["sodium_mg"] <= rules["conditions"]["kidney_friendly"].get("max_sodium_mg", 600))
    return 1

df["label"] = df.apply(label_row, axis=1)

feature_cols = ["calories","carbs_g","sugar_g","protein_g","fat_g","fiber_g","sodium_mg","prep_minutes"]
X = df[feature_cols].fillna(0.0).values
y = df["label"].astype(int).values

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000))
])

# robust split for tiny datasets
unique, counts = np.unique(y, return_counts=True)
min_count = counts.min() if len(counts) > 0 else 0

if len(unique) < 2 or min_count < 2 or len(y) < 8:
    print(f"[INFO] Tiny or imbalanced dataset (classes={dict(zip(unique, counts))}).")
    print("[INFO] Training on ALL data and skipping hold-out AUC.")
    pipe.fit(X, y)
    auc = float("nan")
else:
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)
    if len(np.unique(y_test)) > 1:
        probs = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
    else:
        auc = float("nan")

print(f"AUC for target='{TARGET}': {auc if not np.isnan(auc) else 'n/a (tiny dataset)'}")

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
dump(pipe, MODEL_OUT)
print(f"Saved model to {MODEL_OUT}")
