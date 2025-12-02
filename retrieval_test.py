from sentence_transformers import SentenceTransformer
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

# Load everything
E = np.load("embeddings.npy")
ids = json.load(open("faculty_ids.json"))
texts = json.load(open("faculty_texts.json"))

model = SentenceTransformer("all-MiniLM-L6-v2")

query = "I want to work on blockchain security and differential privacy"
q_emb = model.encode([query])

scores = cosine_similarity(q_emb, E)[0]

top = scores.argsort()[::-1][:5]

for i in top:
    print(ids[i], round(scores[i], 3))
