# RAG Improvements for BSU Graduate Advisor AI

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Phase 1 Improvements](#phase-1-improvements)
4. [Query Processing Pipeline](#query-processing-pipeline)
5. [Embedding Regeneration Guide](#embedding-regeneration-guide)
6. [Evaluation Methodology](#evaluation-methodology)
7. [Future Roadmap](#future-roadmap)

## Overview

This document provides detailed technical information about the Retrieval-Augmented Generation (RAG) improvements implemented for the BSU Graduate Advisor AI system.

### Problems Addressed

1. **Weak embedding model**: Original model (all-MiniLM-L6-v2, 384 dimensions) had poor semantic understanding
2. **No query expansion**: Single-term queries missed faculty with synonymous research areas
3. **Fixed top-k retrieval**: Always returned 3 results regardless of confidence
4. **No evaluation metrics**: Could not measure or track retrieval quality
5. **Inconsistent results**: Many queries had low relevance scores (<0.3)

### Solution Approach

Phase 1 focuses on foundational improvements that provide immediate measurable benefits:
- Upgrade to state-of-the-art embedding model
- Implement domain-specific query expansion
- Add adaptive retrieval with confidence thresholds
- Create comprehensive evaluation framework

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query Input                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              QueryProcessor (Query Expansion)                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Domain-specific synonyms:                             │  │
│  │ • AI → artificial intelligence, machine learning, ... │  │
│  │ • security → cybersecurity, privacy, cryptography...  │  │
│  │ • HCI → human computer interaction, UX, ...          │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │ Expanded Query
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           Embedding Model (BAAI/bge-large-en-v1.5)          │
│                   1024-dimensional vectors                   │
└────────────────────────┬────────────────────────────────────┘
                         │ Query Embedding
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Similarity Computation (Cosine)                 │
│           Query vs. Faculty Profile Embeddings               │
└────────────────────────┬────────────────────────────────────┘
                         │ Similarity Scores
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Adaptive Retrieval (Score Filtering)              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Filter: score >= 0.30 (confidence threshold)      │  │
│  │ 2. Score gap: if gap > 0.15, return only top result  │  │
│  │ 3. Return: 1-5 results based on confidence           │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │ Filtered Results
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Response Generation (LLM)                  │
│              Using retrieved faculty as context              │
└─────────────────────────────────────────────────────────────┘
```

## Phase 1 Improvements

### 1. Embedding Model Upgrade

**Original Model**: `sentence-transformers/all-MiniLM-L6-v2`
- Dimensions: 384
- General-purpose sentence embeddings
- Performance: Good for basic similarity tasks

**New Model**: `BAAI/bge-large-en-v1.5`
- Dimensions: 1024
- Specifically optimized for retrieval tasks
- Performance: State-of-the-art on BEIR benchmark
- Better semantic understanding of technical terms

**Implementation Details**:

```python
# In generate_embeddings.py
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
embeddings = model.encode(texts, batch_size=32, 
                         show_progress_bar=True, 
                         normalize_embeddings=True)
```

**Key Changes**:
- Increased batch size from 16 to 32 for efficiency
- Built-in normalization via `normalize_embeddings=True`
- Verification step to ensure proper normalization

### 2. Query Expansion

**Motivation**: Users often use shorthand or different terminology than faculty profiles. Query expansion bridges this vocabulary gap.

**Implementation**:

```python
class QueryProcessor:
    def __init__(self):
        self.synonyms = {
            'ai': ['artificial intelligence', 'machine learning', 
                   'deep learning', 'neural networks', 'transformers'],
            'security': ['cybersecurity', 'privacy', 'cryptography', 
                        'network security', 'secure systems'],
            # ... more mappings
        }
    
    def expand_query(self, query: str) -> str:
        # Finds matching terms and adds top 3 synonyms
        # Combines with original query
```

**Coverage**: 8+ research domains including:
- AI/ML, Security, HCI, Systems, NLP, Vision, Blockchain, Databases

**Example Expansions**:
- Input: "I want to work on AI"
- Output: "I want to work on AI artificial intelligence machine learning deep learning"

### 3. Adaptive Top-K Selection

**Problem**: Fixed k=3 means:
- Returning weak matches when only 1-2 strong candidates exist
- Missing good candidates when many faculty match

**Solution**: `adaptive_retrieve()` method with:

1. **Initial Retrieval**: Get top 10 candidates
2. **Confidence Filtering**: Keep only score >= 0.30
3. **Score Gap Analysis**: 
   - If gap(rank1, rank2) > 0.15 → return only rank 1
   - Indicates clear best match
4. **Bounded Results**: Return 1-5 results based on confidence

**Code**:

```python
def adaptive_retrieve(self, query, min_results=1, max_results=5, 
                     confidence_threshold=0.30, score_gap_threshold=0.15):
    # Get top 10 candidates
    candidates = get_top_k_candidates(query, k=10)
    
    # Filter by threshold
    filtered = [c for c in candidates if c['score'] >= confidence_threshold]
    
    # Check score gap
    if len(filtered) >= 2:
        gap = filtered[0]['score'] - filtered[1]['score']
        if gap > score_gap_threshold:
            return [filtered[0]]
    
    # Return bounded results
    return filtered[:max_results]
```

### 4. Evaluation Framework

**Components**:

1. **Test Query Suite**: 12 diverse queries covering:
   - Blockchain & privacy
   - ML & computer vision
   - HCI with children
   - NLP, software testing, graph ML
   - Social networks, cybersecurity
   - Parallel computing, LLMs
   - Quantum ML, program analysis

2. **Metrics**:
   - **MRR (Mean Reciprocal Rank)**: `1/rank` of first relevant result
   - **Recall@K**: % of relevant results found in top K
   - **Precision@K**: % of top K results that are relevant
   - **Score Distribution**: Statistical analysis of scores
   - **Coverage**: % of queries with score >= threshold

3. **Reporting**:
   - Detailed markdown reports
   - Side-by-side comparisons
   - Query-level debugging info

## Query Processing Pipeline

### Step-by-Step Example

**User Query**: "I want to work on AI and security"

1. **Input Processing**:
   ```
   Original: "I want to work on AI and security"
   ```

2. **Query Expansion**:
   ```
   Matches found:
   - "ai" → ['artificial intelligence', 'machine learning', 'deep learning']
   - "security" → ['cybersecurity', 'privacy', 'cryptography']
   
   Expanded: "I want to work on AI and security artificial intelligence 
              machine learning deep learning cybersecurity privacy cryptography"
   ```

3. **Embedding Generation**:
   ```
   Model: BAAI/bge-large-en-v1.5
   Output: 1024-dimensional vector (L2-normalized)
   ```

4. **Similarity Computation**:
   ```
   Cosine similarity with all 26 faculty embeddings
   Results (example):
   - Gaby Dagher: 0.68
   - Jyh-haw Yeh: 0.52
   - Yu Zhang: 0.35
   - Jun Zhuang: 0.29
   - ...
   ```

5. **Adaptive Filtering**:
   ```
   Threshold check (>= 0.30):
   ✓ Gaby Dagher: 0.68
   ✓ Jyh-haw Yeh: 0.52
   ✓ Yu Zhang: 0.35
   ✗ Jun Zhuang: 0.29 (filtered out)
   
   Score gap check:
   Gap = 0.68 - 0.52 = 0.16 > 0.15
   → Large gap detected, but 3 results pass threshold
   → Keep top 3 results
   
   Final results: 3 faculty
   ```

6. **Context Injection**:
   ```
   Build prompt with:
   - Faculty 1: Gaby Dagher (0.68) + profile
   - Faculty 2: Jyh-haw Yeh (0.52) + profile  
   - Faculty 3: Yu Zhang (0.35) + profile
   ```

7. **LLM Generation**:
   ```
   Generate response using Groq API with Llama 3
   Response focused on these 3 faculty and their relevance
   ```

## Embedding Regeneration Guide

### When to Regenerate

- After updating faculty profiles in `data/faculty_ready.csv`
- When switching embedding models
- If embeddings become corrupted

### Steps

1. **Prepare Data**:
   ```bash
   # Ensure faculty_ready.csv is up-to-date
   ls data/faculty_ready.csv
   ```

2. **Check Model Availability**:
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer("BAAI/bge-large-en-v1.5")
   ```

3. **Generate Embeddings**:
   ```bash
   python generate_embeddings.py
   ```

4. **Verify Output**:
   ```bash
   python check_embeddings.py
   ```
   
   Expected output:
   ```
   Embedding matrix shape: (26, 1024)
   IDs count: 26
   Texts count: 26
   ```

5. **Test Retrieval**:
   ```bash
   python retrieval_scores_tests.py
   ```
   
   Check that scores are >= 0.30 for relevant queries

### Troubleshooting

**Problem**: Model download fails
- **Solution**: Check internet connectivity or use local cache

**Problem**: Out of memory
- **Solution**: Reduce batch_size in generate_embeddings.py

**Problem**: Low scores after regeneration
- **Solution**: Ensure model name matches between generate_embeddings.py and response_engine.py

## Evaluation Methodology

### Test Query Design

Each test query has:
- **Query**: Natural language question
- **Expected**: List of relevant faculty names
- **Must Include**: Critical faculty that MUST appear in results

**Example**:
```python
{
    'query': 'I want to work on blockchain and privacy',
    'expected': ['Gaby Dagher', 'Jyh-haw Yeh'],
    'must_include': ['Gaby Dagher']
}
```

### Metric Calculations

**MRR (Mean Reciprocal Rank)**:
```
For each query, find rank R of first relevant result
RR = 1/R (or 0 if none found)
MRR = average of all RR scores

Example:
Query 1: First relevant at rank 1 → RR = 1.0
Query 2: First relevant at rank 2 → RR = 0.5
Query 3: No relevant results → RR = 0.0
MRR = (1.0 + 0.5 + 0.0) / 3 = 0.50
```

**Recall@3**:
```
For each query:
  Recall = (# relevant in top 3) / (# total relevant)

Example:
Expected: [A, B, C]
Retrieved: [A, D, B]
Recall = 2/3 = 0.667
```

**Precision@3**:
```
For each query:
  Precision = (# relevant in top 3) / 3

Example:
Expected: [A, B, C]
Retrieved: [A, D, B]
Precision = 2/3 = 0.667
```

### Running Evaluation

**Quick Test**:
```bash
python retrieval_scores_tests.py
```

**Full Evaluation**:
```python
from response_engine import ResponseEngine
from rag_evaluator import RAGEvaluator

evaluator = RAGEvaluator()
engine = ResponseEngine()
results = evaluator.evaluate(engine)
report = evaluator.generate_report(results)
```

**Model Comparison**:
```python
# Compare old vs new
old_engine = ResponseEngine()  # with old embeddings
new_engine = ResponseEngine()  # with new embeddings
comparison = evaluator.compare_models(old_engine, new_engine)
```

### Success Criteria

Phase 1 targets:
- ✅ MRR >= 0.60
- ✅ Recall@3 >= 0.50
- ✅ Coverage >= 0.80 (80% of queries have score >= 0.30)
- ✅ Average score increase of 0.10-0.15

## Future Roadmap

### Phase 2: Enhanced Retrieval

1. **Hybrid Search**:
   - Combine dense (embedding) and sparse (BM25) retrieval
   - Reciprocal Rank Fusion for result merging

2. **Re-ranking**:
   - Use cross-encoder for final ranking
   - Models like `BAAI/bge-reranker-large`

3. **Query Classification**:
   - Detect query intent (find advisor vs. general info)
   - Route to appropriate retrieval strategy

### Phase 3: Advanced Features

1. **User Feedback Loop**:
   - Collect relevance feedback
   - Fine-tune embedding model

2. **Multi-turn Context**:
   - Track conversation state
   - Use previous queries for context

3. **Explainability**:
   - Highlight matching terms
   - Show why each faculty was retrieved

4. **Personalization**:
   - User profile (background, interests)
   - Personalized ranking

### Phase 4: Production Optimization

1. **Performance**:
   - Vector database (FAISS, Qdrant)
   - Caching frequently queried embeddings

2. **Monitoring**:
   - Track retrieval quality metrics
   - A/B testing framework

3. **Scalability**:
   - Support for 100+ faculty
   - Multi-department support

## References

- **BAAI/bge-large-en-v1.5**: [HuggingFace](https://huggingface.co/BAAI/bge-large-en-v1.5)
- **BEIR Benchmark**: [Paper](https://arxiv.org/abs/2104.08663)
- **Sentence Transformers**: [Documentation](https://www.sbert.net/)
- **RAG Survey**: [Paper](https://arxiv.org/abs/2312.10997)

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: CS Department, Boise State University
