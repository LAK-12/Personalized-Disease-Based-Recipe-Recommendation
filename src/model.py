from joblib import load
import numpy as np
import os

def load_model(path: str):
    if not os.path.exists(path):
        return None
    try:
        return load(path)
    except Exception:
        return None

def predict_probability(model, X: np.ndarray) -> float:
    if model is None:
        return 0.5
    try:
        return float(model.predict_proba(X)[:,1][0])
    except Exception:
        return 0.5
