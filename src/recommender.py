from __future__ import annotations
import os
import faiss
import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from .utils import tokenize_ingredients, contains_any, pantry_coverage, assign_badges, GLUTEN_TERMS, PROCESSED_MEATS
from .model import load_model, predict_probability

DATA_CSV = os.getenv("RECIPES_CSV", "data/recipes_sample.csv")
FAISS_PATH = os.getenv("FAISS_INDEX", "data/embeddings.faiss")
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RULES_PATH = os.getenv("RULES_PATH", "src/rules/nutrition_rules.yaml")
MODEL_PATH = os.getenv("MODEL_PATH", "data/model_lr.pkl")

_df = None
_index = None
_model = None
_rules = None
_clf = None




def _load_rules() -> Dict:
  global _rules
  if _rules is None:
    with open(RULES_PATH, "r") as f:
      _rules = yaml.safe_load(f)
  return _rules




def _load_df() -> pd.DataFrame:
  global _df
  if _df is None:
    _df = pd.read_csv(DATA_CSV)
  return _df




def _load_embedder() -> SentenceTransformer:
  global _model
  if _model is None:
    _model = SentenceTransformer(MODEL_NAME)
  return _model

def _load_index() -> faiss.Index:
  global _index
  if _index is None:
    if not os.path.exists(FAISS_PATH):
      raise FileNotFoundError(f"FAISS index not found at {FAISS_PATH}. Run scripts/build_index.py")
    _index = faiss.read_index(FAISS_PATH)
  return _index




def _load_clf():
  global _clf
  if _clf is None:
    _clf = load_model(MODEL_PATH)
  return _clf




def _hard_filter(df: pd.DataFrame, 
                 conditions: List[str], allergies: List[str], dislikes: List[str], rules: Dict) -> pd.DataFrame:

  filtered = df.copy()
  allergies_l = {a.lower() for a in allergies}
  dislikes_l = {d.lower() for d in dislikes}


# allergies & dislikes exclusion
  mask_allergies = filtered["ingredients_text"].str.lower().apply(lambda t: not any(a in t for a in allergies_l))
  mask_dislikes = filtered["ingredients_text"].str.lower().apply(lambda t: not any(d in t for d in dislikes_l))
  filtered = filtered[mask_allergies & mask_dislikes]


# condition-specific
  conds = {c.lower() for c in conditions}
  if "celiac" in conds:
    filtered = filtered[~filtered["ingredients_text"].str.lower().apply(lambda t: contains_any(t, GLUTEN_TERMS))]


  if "hypertension" in conds:
    filtered = filtered[filtered["sodium_mg"] <= rules["conditions"]["hypertension"]["max_sodium_mg"]]


  if "diabetes" in conds:
    filtered = filtered[
        (filtered["carbs_g"] <= rules["conditions"]["diabetes"]["max_carbs_g_per_serving"])
        & (filtered["fiber_g"] >= rules["conditions"]["diabetes"]["min_fiber_g"])
        & (filtered["sugar_g"] <= rules["conditions"]["diabetes"]["max_sugar_g_per_serving"])
    ]



  if "kidney_friendly" in conds:
    if "max_sodium_mg" in rules["conditions"]["kidney_friendly"]:
      filtered = filtered[filtered["sodium_mg"] <= rules["conditions"]["kidney_friendly"]["max_sodium_mg"]]
      
  return filtered

def _embed_texts(texts: List[str]) -> np.ndarray:
  model = _load_embedder()
  emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
  return emb.astype("float32")


def _vector_search(pantry_text: str, k: int = 25) -> Tuple[np.ndarray, np.ndarray]:
  index = _load_index()
  q = _embed_texts([pantry_text])
  scores, idx = index.search(q, k)
  return scores[0], idx[0]

def _nutrition_score(row: Dict, conditions: List[str], rules: Dict) -> float:
  score = 1.0
  conds = {c.lower() for c in conditions}
  if "diabetes" in conds:
    carbs = row.get("carbs_g", 0)
    fiber = row.get("fiber_g", 0)
    max_c = rules["conditions"]["diabetes"]["max_carbs_g_per_serving"]
    score *= max(0.0, 1 - max(0, (carbs - max_c)) / (max_c + 1))
    score *= min(1.0, 0.5 + fiber/20.0)
  if "hypertension" in conds:
    sodium = row.get("sodium_mg", 0)
    max_s = rules["conditions"]["hypertension"]["max_sodium_mg"]
    score *= max(0.0, 1 - max(0, (sodium - max_s)) / (max_s + 1))
    if contains_any(row.get("ingredients_text",""), PROCESSED_MEATS):
      score *= (1 - rules["conditions"]["hypertension"]["processed_meat_penalty"])
  if "kidney_friendly" in conds:
    sodium = row.get("sodium_mg", 0)
    score *= max(0.2, 1 - sodium/1000.0)
  return float(max(0.0, min(1.0, score)))

def _prep_score(minutes: float) -> float:
  if not minutes or minutes <= 0:
    return 1.0
  return float(1.0 / (1.0 + (minutes/60.0)))

def recommend(
    conditions: List[str],
    allergies: List[str],
    dislikes: List[str],
    pantry: List[str],
    top_k: int = 10,
    must_use_pantry: bool = True,
    min_pantry_coverage: float = 0.0,
) -> List[Dict]:
    # load rules/data
    rules = _load_rules()
    df = _load_df()

    # Step 1: hard filter
    filtered = _hard_filter(df, conditions, allergies, dislikes, rules)
    if filtered.empty:
        return []

    # Step 2: vector search candidates
    pantry_text = ", ".join(pantry).strip()
    if pantry_text:
        scores, idx = _vector_search(pantry_text, k=min(50, len(df)))
        candidates = _load_df().iloc[idx]
        cand = candidates.merge(filtered, on=list(df.columns), how="inner")
        if cand.empty:
            cand = filtered.copy()
    else:
        cand = filtered.copy()

    # --- Pantry filters (exact match and optional coverage) ---
    if pantry:
        pantry_l = [p.strip().lower() for p in pantry if p.strip()]

        if must_use_pantry:
            # Require at least ONE pantry token to appear in the ingredients text
            cand = cand[cand["ingredients_text"].str.lower().apply(
                lambda t: any(p in t for p in pantry_l)
            )]

        if min_pantry_coverage and min_pantry_coverage > 0.0:
            # Enforce semantic coverage threshold (proportion of ingredients covered)
            def _cov(row):
                return pantry_coverage(pantry, tokenize_ingredients(row["ingredients_text"]))
            cand = cand[cand.apply(_cov, axis=1) >= float(min_pantry_coverage)]

        if cand.empty:
            # Strict behavior: if filters removed everything, return no results
            return []

    # Step 3: ML model (optional but enabled by default)
    clf = _load_clf()

    w = rules["weights"]
    rows = []

    for _, r in cand.iterrows():
        ings = tokenize_ingredients(r["ingredients_text"])
        pantry_score = pantry_coverage(pantry, ings) if pantry else 0.5
        nutrition = _nutrition_score(r.to_dict(), conditions, rules)
        prep = _prep_score(r.get("prep_minutes", 20))
        popularity = 0.5

        # model features (simple, interpretable)
        X = np.array([[
          r.get("calories",0), r.get("carbs_g",0), r.get("sugar_g",0),
          r.get("protein_g",0), r.get("fat_g",0), r.get("fiber_g",0),
          r.get("sodium_mg",0), r.get("prep_minutes",0)
        ]], dtype=float)

        ml_prob = float(predict_probability(clf, X)) if clf is not None else 0.5

        final = (w["nutrition"]*nutrition + w["pantry"]*pantry_score + w["prep"]*prep + w["popularity"]*popularity)
        # blend in ML as tie-breaker / bonus
        final = 0.85*final + 0.15*ml_prob

        rows.append({
            "id": r["id"],
            "title": r["title"],
            "ingredients_text": r["ingredients_text"],
            "instructions": r["instructions"],
            "calories": r.get("calories", None),
            "carbs_g": r.get("carbs_g", None),
            "protein_g": r.get("protein_g", None),
            "fat_g": r.get("fat_g", None),
            "fiber_g": r.get("fiber_g", None),
            "sodium_mg": r.get("sodium_mg", None),
            "prep_minutes": r.get("prep_minutes", None),
            "badges": assign_badges(r.to_dict(), rules),
            "why": f"nutrition={nutrition:.2f}, pantry={pantry_score:.2f}, prep={prep:.2f}, ml={ml_prob:.2f}",
            "score": final,
        })

    ranked = sorted(rows, key=lambda x: x["score"], reverse=True)[:top_k]
    return ranked
