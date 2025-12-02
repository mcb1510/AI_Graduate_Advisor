from sentence_transformers import SentenceTransformer
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

# Load everything
E = np.load("embeddings.npy")
ids = json.load(open("faculty_ids.json"))
texts = json.load(open("faculty_texts.json"))

model = SentenceTransformer("all-MiniLM-L6-v2")

# Test multiple queries to see score patterns
test_queries = [
    "I want to work on blockchain security and differential privacy",
    "machine learning and AI research",
    "software engineering and testing",
    "human computer interaction"
]

print("=" * 80)
print("RETRIEVAL QUALITY TEST")
print("=" * 80)

for query in test_queries:
    print(f"\n🔍 Query: '{query}'")
    print("-" * 80)
    
    q_emb = model.encode([query])
    scores = cosine_similarity(q_emb, E)[0]
    
    # Get top 5
    top = scores.argsort()[::-1][:5]
    
    print(f"{'Rank':<6} {'Faculty Name':<30} {'Score':<10} {'Quality'}")
    print("-" * 80)
    
    for rank, i in enumerate(top, 1):
        score = scores[i]
        # Classify score quality
        if score >= 0.4:
            quality = "✅ GOOD"
        elif score >= 0.3:
            quality = "⚠️  FAIR"
        else:
            quality = "❌ POOR"
        
        print(f"{rank:<6} {ids[i]:<30} {score:.4f}    {quality}")
    
    # Show what the best match actually says
    best_idx = top[0]
    print(f"\n📄 Best Match Profile Preview:")
    print(f"   {texts[best_idx][:200]}...")
    print()

print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY:")
print("=" * 80)
print("Score Interpretation:")
print("  ✅ >= 0.40: Good match (strong semantic similarity)")
print("  ⚠️  0.30-0.39: Fair match (moderate similarity)")
print("  ❌ < 0.30: Poor match (weak/irrelevant)")
print("\nIf all scores are < 0.3, the embeddings may need regeneration.")
print("=" * 80)