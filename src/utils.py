import re
from typing import List, Dict

TERM_SPLIT = re.compile(r"[;,]+\s*") 

GLUTEN_TERMS = {"wheat","barley","rye","bulgur","semolina","couscous","spelt","farro"}
PROCESSED_MEATS = {"bacon","sausage","salami","ham","hot dog"}

def tokenize_ingredients(ingredients_text: str) -> List[str]:
    parts = [p.strip().lower() for p in TERM_SPLIT.split(ingredients_text)]
    return [p for p in parts if p]

def contains_any(text: str, terms: set[str]) -> bool:
    t = text.lower()
    return any(term in t for term in terms)

def pantry_coverage(pantry: List[str], ingredients: List[str]) -> float:
    if not ingredients:
        return 0.0
    pset = {p.strip().lower() for p in pantry}
    count = sum(1 for ing in ingredients if any(tok in ing for tok in pset))
    return count / max(1, len(ingredients))

def assign_badges(row: Dict, rules: Dict) -> List[str]:
    badges = []
    if row.get("sodium_mg", 0) <= rules["badges"]["low_sodium"]["threshold_mg"]:
        badges.append("Low Sodium")
    if row.get("carbs_g", 0) <= rules["badges"]["low_carb"]["threshold_g"]:
        badges.append("Low Carb")
    if row.get("fiber_g", 0) >= rules["badges"]["high_fiber"]["threshold_g"]:
        badges.append("High Fiber")
    if not contains_any(row.get("ingredients_text",""), set(rules["badges"]["gluten_free"]["requires_no_terms"])):
        badges.append("Gluten-Free")
    return badges
