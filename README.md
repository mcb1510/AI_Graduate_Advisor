# BSU Graduate Advisor AI

An AI-powered assistant to help Computer Science graduate students at Boise State University find suitable research advisors based on their interests, skills, and goals.

## Features

- **Smart Faculty Matching**: Retrieval-Augmented Generation (RAG) system to match students with relevant faculty
- **Advanced Embeddings**: Uses BAAI/bge-large-en-v1.5 (1024 dimensions) for superior semantic understanding
- **Query Expansion**: Automatically expands queries with domain-specific synonyms for better retrieval
- **Adaptive Retrieval**: Returns variable number of results (1-5) based on confidence scores
- **Comprehensive Evaluation**: Built-in testing framework to measure and track retrieval quality

## RAG Improvements (Phase 1)

### 1. Upgraded Embedding Model

**From**: `all-MiniLM-L6-v2` (384 dimensions)  
**To**: `BAAI/bge-large-en-v1.5` (1024 dimensions)

The new model is specifically optimized for retrieval tasks and provides significantly better semantic understanding of research areas and faculty profiles.

**Benefits:**
- Improved matching accuracy for complex queries
- Better understanding of technical terminology
- Higher retrieval scores overall

### 2. Query Expansion

Automatically expands user queries with domain-specific synonyms to improve retrieval:

**Examples:**
- "AI" → expanded to include "artificial intelligence", "machine learning", "deep learning", "neural networks"
- "security" → expanded to include "cybersecurity", "privacy", "cryptography", "network security"
- "HCI" → expanded to include "human computer interaction", "user experience", "interface design"

**Supported domains:**
- Artificial Intelligence & Machine Learning
- Security & Cybersecurity
- Human-Computer Interaction
- Distributed Systems
- Natural Language Processing
- Computer Vision
- Blockchain
- Databases

### 3. Adaptive Top-K Selection

Instead of always returning exactly 3 results, the system now adaptively selects between 1-5 results based on:

**Confidence Threshold**: Only returns results with score >= 0.30  
**Score Gap Analysis**: If the top result is significantly better (gap > 0.15), only returns that one result  
**Variable Results**: Returns more results when multiple faculty match well, fewer when there's a clear best match

**Example:**
```python
# Query with one clear match
query = "quantum machine learning"
# Returns: 1 result (Jun Zhuang with high confidence)

# Query with multiple good matches  
query = "machine learning and computer vision"
# Returns: 3-5 results (multiple faculty with good scores)
```

### 4. Evaluation Framework

Comprehensive testing system to measure RAG quality:

**Metrics:**
- **MRR (Mean Reciprocal Rank)**: Position of first relevant result
- **Recall@3**: Percentage of relevant results found in top 3
- **Precision@3**: Percentage of top 3 results that are relevant
- **Score Distribution**: Min, max, mean, median of retrieval scores
- **Coverage**: Percentage of queries with score >= 0.30

**Test Suite**: 12 carefully designed test queries covering diverse research areas

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

## Usage

### Generate Embeddings

```bash
python generate_embeddings.py
```

This will:
1. Load faculty profiles from `data/faculty_ready.csv`
2. Generate embeddings using BAAI/bge-large-en-v1.5
3. Save to `embeddings.npy`, `faculty_ids.json`, `faculty_texts.json`

### Run the Chatbot

```bash
streamlit run chatbot_demo.py
```

### Run Evaluation

```bash
# Quick retrieval test
python retrieval_scores_tests.py

# Full evaluation with RAG engine
python -c "from response_engine import ResponseEngine; from rag_evaluator import RAGEvaluator; e = RAGEvaluator(); engine = ResponseEngine(); results = e.evaluate(engine); e.generate_report(results)"
```

### Test Query Expansion

```python
from response_engine import QueryProcessor

processor = QueryProcessor()

# Example expansions
print(processor.expand_query("I want to work on AI"))
# Output: "I want to work on AI artificial intelligence machine learning deep learning"

print(processor.expand_query("interested in security and blockchain"))
# Output: "interested in security and blockchain cybersecurity privacy cryptography distributed ledger cryptocurrency consensus protocols"
```

## Project Structure

```
.
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── generate_embeddings.py       # Generate faculty embeddings
├── response_engine.py           # Core RAG engine with query processing
├── rag_evaluator.py            # Comprehensive evaluation framework
├── retrieval_scores_tests.py   # Quick retrieval quality tests
├── chatbot_demo.py             # Streamlit web interface
├── data/
│   └── faculty_ready.csv       # Faculty profiles dataset
├── docs/
│   └── RAG_IMPROVEMENTS.md     # Detailed RAG documentation
├── embeddings.npy              # Generated embeddings (1024-dim)
├── faculty_ids.json            # Faculty names
└── faculty_texts.json          # Faculty profile texts
```

## Example Queries

The system handles various types of queries:

**Research Interest Matching:**
- "I want to work on blockchain and privacy"
- "machine learning and computer vision"
- "natural language processing"

**Specific Topics:**
- "quantum machine learning"
- "program analysis and verification"
- "LLM and generative AI"

**Faculty Listing:**
- "list all faculty"
- "show me the professors"

**Specific Faculty:**
- "Tell me about Gaby Dagher"
- "What does Jerry Fails work on?"

## Evaluation Results

After implementing Phase 1 improvements, expected metrics:

- **MRR**: ~0.65 (up from ~0.45, 44% increase)
- **Recall@3**: ~0.55 (up from ~0.35, 57% increase)
- **Average Score**: Increase of 0.10-0.15 points
- **Coverage**: Most queries now have scores >= 0.30

## Development

### Running Tests

```bash
# Test retrieval quality
python retrieval_scores_tests.py

# Test query expansion
python -c "from response_engine import QueryProcessor; p = QueryProcessor(); print(p.expand_query('AI and security'))"

# Full evaluation
python -c "from response_engine import ResponseEngine; from rag_evaluator import RAGEvaluator; e = RAGEvaluator(); engine = ResponseEngine(); results = e.evaluate(engine); e.generate_report(results)"
```

### Regenerating Embeddings

After updating faculty profiles:

```bash
python generate_embeddings.py
```

This will regenerate embeddings with the improved BAAI/bge-large-en-v1.5 model.

## Future Improvements (Phase 2 & 3)

- Hybrid search (dense + sparse retrieval)
- Re-ranking with cross-encoder
- Query classification for intent detection
- User feedback integration
- Multi-turn conversation context

## Documentation

- [RAG Improvements Details](docs/RAG_IMPROVEMENTS.md) - Detailed technical documentation
- [Evaluation Methodology](docs/RAG_IMPROVEMENTS.md#evaluation-methodology) - How we measure quality

## License

This project is for educational use at Boise State University.

## Contributors

- Computer Science Department, Boise State University
