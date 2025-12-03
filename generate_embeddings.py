import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Model configuration
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # BGE model optimized for retrieval

def main():
    print("[INFO] Loading dataset: faculty_ready.csv")
    df = pd.read_csv("data/faculty_ready.csv", encoding="utf-8")

    if "profile_text" not in df.columns:
        raise ValueError("The file must contain a 'profile_text' column.")

    # Extract texts
    texts = df["profile_text"].fillna("").tolist()
    names = df["name"].tolist()

    print(f"[INFO] Loaded {len(texts)} faculty profiles.")

    print(f"[INFO] Loading SentenceTransformer model: {EMBEDDING_MODEL}")
    print("[INFO] This model is optimized for retrieval tasks (1024 dimensions)")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("[INFO] Generating embeddings with batch processing...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    # Verify embeddings are normalized
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"[INFO] Embedding normalization check - Mean norm: {norms.mean():.4f}, Std: {norms.std():.4f}")
    # Tolerance of 1e-3 allows for minor floating-point precision issues
    # while still catching significant normalization problems
    if not np.allclose(norms, 1.0, atol=1e-3):
        print("[WARNING] Embeddings not properly normalized, applying normalization...")
        embeddings = normalize(embeddings)
    else:
        print("[INFO] ✓ Embeddings are properly normalized")

    print("[INFO] Saving embedding matrix...")
    np.save("embeddings.npy", embeddings)

    print("[INFO] Saving faculty IDs...")
    with open("faculty_ids.json", "w", encoding="utf-8") as f:
        json.dump(names, f, indent=4)

    print("[INFO] Saving faculty texts (for retrieval)...")
    with open("faculty_texts.json", "w", encoding="utf-8") as f:
        json.dump(texts, f, indent=4)

    print("\n[DONE] Embeddings generated and saved successfully!")
    print("Files created:")
    print(" - embeddings.npy")
    print(" - faculty_ids.json")
    print(" - faculty_texts.json")

if __name__ == "__main__":
    main()
