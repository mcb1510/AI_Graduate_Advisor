#!/usr/bin/env python3
"""
Quick evaluation script for RAG improvements
Makes it easier to run comprehensive evaluation
"""

import sys
import os

def main():
    """Run comprehensive RAG evaluation."""
    print("=" * 80)
    print("RAG COMPREHENSIVE EVALUATION")
    print("=" * 80)
    print()
    
    # Check for GROQ_API_KEY
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not found in environment")
        print("Please create a .env file with your GROQ_API_KEY")
        print()
        print("Example .env file:")
        print("GROQ_API_KEY=your_key_here")
        sys.exit(1)
    
    print("[1/3] Importing modules...")
    try:
        from response_engine import ResponseEngine
        from rag_evaluator import RAGEvaluator
    except ImportError as e:
        print(f"ERROR: Failed to import required modules: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    print("[2/3] Initializing RAG engine...")
    try:
        engine = ResponseEngine()
    except Exception as e:
        print(f"ERROR: Failed to initialize RAG engine: {e}")
        print("\nPossible issues:")
        print("  - Missing embeddings.npy, faculty_ids.json, or faculty_texts.json")
        print("  - Run: python generate_embeddings.py")
        print("  - Embedding model not available (BAAI/bge-large-en-v1.5)")
        sys.exit(1)
    
    print("[3/3] Running evaluation suite...")
    evaluator = RAGEvaluator()
    
    try:
        results = evaluator.evaluate(engine)
    except Exception as e:
        print(f"ERROR: Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("GENERATING REPORT")
    print("=" * 80)
    
    report = evaluator.generate_report(results, filename="evaluation_report.md")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    metrics = results['metrics']
    print(f"  MRR: {metrics['mrr']:.4f}")
    print(f"  Recall@3: {metrics['recall_at_3']:.4f}")
    print(f"  Precision@3: {metrics['precision_at_3']:.4f}")
    print(f"  Coverage: {metrics['coverage']:.4f}")
    print(f"  Must-Include Success Rate: {metrics['must_include_success_rate']:.4f}")
    print()
    print("Full report saved to: evaluation_report.md")
    print()


if __name__ == "__main__":
    main()
