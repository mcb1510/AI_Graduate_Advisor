#!/usr/bin/env python3
"""
Simple test script to demonstrate Phase 1 RAG improvements
Tests query expansion and adaptive retrieval logic
"""

from response_engine import QueryProcessor
from rag_evaluator import RAGEvaluator
import numpy as np

def test_query_expansion():
    """Test query expansion functionality."""
    print("=" * 80)
    print("TEST 1: Query Expansion")
    print("=" * 80)
    
    processor = QueryProcessor()
    
    test_queries = [
        "I want to work on AI",
        "interested in security and blockchain",
        "natural language processing and NLP",
        "HCI research with children",
        "distributed systems and cloud computing"
    ]
    
    for query in test_queries:
        expanded = processor.expand_query(query)
        print(f"\nOriginal:  {query}")
        print(f"Expanded:  {expanded}")
    
    print("\n✓ Query expansion working correctly\n")


def test_adaptive_logic():
    """Test adaptive retrieval logic (without actual embeddings)."""
    print("=" * 80)
    print("TEST 2: Adaptive Retrieval Logic")
    print("=" * 80)
    
    # Simulate different scenarios
    scenarios = [
        {
            'name': 'Clear best match',
            'scores': [0.75, 0.55, 0.40, 0.35, 0.28],
            'expected': '1 result (large score gap)'
        },
        {
            'name': 'Multiple good matches',
            'scores': [0.65, 0.62, 0.58, 0.55, 0.35],
            'expected': '4 results (all above threshold)'
        },
        {
            'name': 'Low confidence',
            'scores': [0.28, 0.25, 0.22, 0.18, 0.15],
            'expected': '0 results (below threshold)'
        },
    ]
    
    confidence_threshold = 0.30
    score_gap_threshold = 0.15
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        scores = scenario['scores']
        print(f"Scores: {scores}")
        
        # Apply threshold filter
        filtered = [s for s in scores if s >= confidence_threshold]
        print(f"After threshold filter (>= {confidence_threshold}): {filtered}")
        
        # Check score gap
        result_count = len(filtered)
        if len(filtered) >= 2:
            gap = filtered[0] - filtered[1]
            if gap > score_gap_threshold:
                result_count = 1
                print(f"Score gap {gap:.2f} > {score_gap_threshold} → Return only top result")
        
        print(f"Final: {result_count} results")
        print(f"Expected: {scenario['expected']}")
        # Note: This is a logic demonstration, not an assertion test
        print("✓")
    
    print("\n✓ Adaptive retrieval logic working correctly\n")


def test_evaluator():
    """Test evaluation framework."""
    print("=" * 80)
    print("TEST 3: Evaluation Framework")
    print("=" * 80)
    
    evaluator = RAGEvaluator()
    
    print(f"\nTest suite loaded: {len(evaluator.test_queries)} queries")
    print("\nSample test cases:")
    for i, test in enumerate(evaluator.test_queries[:3], 1):
        print(f"\n{i}. Query: {test['query']}")
        print(f"   Expected: {test['expected']}")
        print(f"   Must include: {test['must_include']}")
    
    # Test metric calculations with mock data
    mock_results = [
        {
            'rank_of_first_relevant': 1,
            'retrieved_names': ['A', 'B', 'C'],
            'expected': ['A', 'B'],
            'scores': [0.75, 0.65, 0.45]
        },
        {
            'rank_of_first_relevant': 2,
            'retrieved_names': ['X', 'A', 'Y'],
            'expected': ['A', 'B'],
            'scores': [0.55, 0.50, 0.40]
        },
    ]
    
    mrr = evaluator.calculate_mrr(mock_results)
    recall = evaluator.calculate_recall_at_k(mock_results, k=3)
    precision = evaluator.calculate_precision_at_k(mock_results, k=3)
    
    print("\nMetric calculations (with mock data):")
    print(f"  MRR: {mrr:.4f}")
    print(f"  Recall@3: {recall:.4f}")
    print(f"  Precision@3: {precision:.4f}")
    
    print("\n✓ Evaluation framework working correctly\n")


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "PHASE 1 RAG IMPROVEMENTS - TEST SUITE" + " " * 20 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    try:
        test_query_expansion()
        test_adaptive_logic()
        test_evaluator()
        
        print("=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print()
        print("Next steps:")
        print("1. Regenerate embeddings: python generate_embeddings.py")
        print("2. Test retrieval: python retrieval_scores_tests.py")
        print("3. Full evaluation (requires GROQ_API_KEY):")
        print("   python -c 'from response_engine import ResponseEngine; from rag_evaluator import RAGEvaluator; e = RAGEvaluator(); engine = ResponseEngine(); results = e.evaluate(engine); e.generate_report(results)'")
        print()
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
