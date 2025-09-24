from sentence_transformers import SentenceTransformer
import pandas as pd, numpy as np, faiss, os

DATA_CSV = "data/recipes_sample.csv"
FAISS_PATH = "data/embeddings.faiss"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)
df = pd.read_csv(DATA_CSV)
texts = df["ingredients_text"].fillna("").tolist()

model = SentenceTransformer(MODEL_NAME)
emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True).astype("float32")

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
faiss.write_index(index, FAISS_PATH)
print(f"Built FAISS index at {FAISS_PATH}")
