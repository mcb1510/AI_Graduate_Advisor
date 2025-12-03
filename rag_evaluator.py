"""
RAG Evaluation Framework for BSU Graduate Advisor AI
Comprehensive testing and evaluation system for RAG improvements
"""

import json
import numpy as np
from typing import List, Dict, Any
import statistics


class RAGEvaluator:
    """
    Comprehensive evaluation system for RAG retrieval quality.
    Measures MRR, Recall@K, Precision@K, and other metrics.
    """
    
    def __init__(self):
        """Initialize evaluator with test query suite."""
        self.test_queries = [
            {
                'query': 'I want to work on blockchain and privacy',
                'expected': ['Gaby Dagher', 'Jyh-haw Yeh'],
                'must_include': ['Gaby Dagher']
            },
            {
                'query': 'machine learning and computer vision',
                'expected': ['Yu Zhang', 'Xinyi Zhou', 'Tim Andersen'],
                'must_include': ['Yu Zhang']
            },
            {
                'query': 'human computer interaction with children',
                'expected': ['Jerry Alan Fails'],
                'must_include': ['Jerry Alan Fails']
            },
            {
                'query': 'natural language processing',
                'expected': ['Casey Kennington', 'Eric Henderson', 'Xinyi Zhou'],
                'must_include': ['Casey Kennington']
            },
            {
                'query': 'software engineering and testing',
                'expected': ['Elena Sherman', 'Bogdan Dit', 'Jim Buffenbarger'],
                'must_include': ['Elena Sherman']
            },
            {
                'query': 'graph machine learning',
                'expected': ['Jun Zhuang', 'Edoardo Serra'],
                'must_include': ['Jun Zhuang']
            },
            {
                'query': 'social network analysis',
                'expected': ['Francesca Spezzano', 'Edoardo Serra'],
                'must_include': ['Francesca Spezzano']
            },
            {
                'query': 'cybersecurity education',
                'expected': ['Jyh-haw Yeh', 'Shane Panter'],
                'must_include': ['Jyh-haw Yeh']
            },
            {
                'query': 'parallel computing and HPC',
                'expected': ['Max Taylor', 'Amit Jain', 'Steven Cutchin'],
                'must_include': ['Max Taylor']
            },
            {
                'query': 'LLM and generative AI',
                'expected': ['Xinyi Zhou', 'Gaby Dagher'],
                'must_include': ['Xinyi Zhou']
            },
            {
                'query': 'quantum machine learning',
                'expected': ['Jun Zhuang'],
                'must_include': ['Jun Zhuang']
            },
            {
                'query': 'program analysis and verification',
                'expected': ['Elena Sherman'],
                'must_include': ['Elena Sherman']
            }
        ]
    
    def calculate_mrr(self, results: List[Dict]) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        Args:
            results: List of evaluation results with 'rank_of_first_relevant' key
            
        Returns:
            MRR score (0.0 to 1.0)
        """
        reciprocal_ranks = []
        for result in results:
            rank = result.get('rank_of_first_relevant', 0)
            if rank > 0:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        
        return statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def calculate_recall_at_k(self, results: List[Dict], k: int = 3) -> float:
        """
        Calculate Recall@K: percentage of relevant results found in top K.
        
        Args:
            results: List of evaluation results
            k: Number of top results to consider
            
        Returns:
            Recall@K score (0.0 to 1.0)
        """
        recalls = []
        for result in results:
            retrieved_names = result.get('retrieved_names', [])[:k]
            expected = result.get('expected', [])
            
            if not expected:
                continue
            
            # Count how many expected items were found
            found = sum(1 for name in expected if name in retrieved_names)
            recall = found / len(expected)
            recalls.append(recall)
        
        return statistics.mean(recalls) if recalls else 0.0
    
    def calculate_precision_at_k(self, results: List[Dict], k: int = 3) -> float:
        """
        Calculate Precision@K: percentage of top K results that are relevant.
        
        Args:
            results: List of evaluation results
            k: Number of top results to consider
            
        Returns:
            Precision@K score (0.0 to 1.0)
        """
        precisions = []
        for result in results:
            retrieved_names = result.get('retrieved_names', [])[:k]
            expected = result.get('expected', [])
            
            if not retrieved_names:
                continue
            
            # Count how many retrieved items are relevant
            relevant = sum(1 for name in retrieved_names if name in expected)
            precision = relevant / len(retrieved_names)
            precisions.append(precision)
        
        return statistics.mean(precisions) if precisions else 0.0
    
    def calculate_score_distribution(self, results: List[Dict]) -> Dict[str, float]:
        """
        Calculate score distribution statistics.
        
        Args:
            results: List of evaluation results with 'scores' key
            
        Returns:
            Dictionary with min, max, mean, median scores
        """
        all_scores = []
        for result in results:
            scores = result.get('scores', [])
            all_scores.extend(scores)
        
        if not all_scores:
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'median': 0.0}
        
        return {
            'min': min(all_scores),
            'max': max(all_scores),
            'mean': statistics.mean(all_scores),
            'median': statistics.median(all_scores)
        }
    
    def calculate_coverage(self, results: List[Dict], threshold: float = 0.30) -> float:
        """
        Calculate coverage: percentage of queries with at least one result >= threshold.
        
        Args:
            results: List of evaluation results with 'scores' key
            threshold: Minimum score threshold
            
        Returns:
            Coverage percentage (0.0 to 1.0)
        """
        covered = 0
        for result in results:
            scores = result.get('scores', [])
            if any(score >= threshold for score in scores):
                covered += 1
        
        return covered / len(results) if results else 0.0
    
    def evaluate(self, rag_engine, use_adaptive: bool = True) -> Dict[str, Any]:
        """
        Run full evaluation suite on a RAG engine.
        
        Args:
            rag_engine: ResponseEngine instance to evaluate
            use_adaptive: Whether to use adaptive retrieval
            
        Returns:
            Dictionary with all evaluation metrics
        """
        print("=" * 80)
        print("RAG EVALUATION SUITE")
        print("=" * 80)
        
        results = []
        
        for i, test_case in enumerate(self.test_queries, 1):
            query = test_case['query']
            expected = test_case['expected']
            must_include = test_case['must_include']
            
            print(f"\n[{i}/{len(self.test_queries)}] Query: '{query}'")
            
            # Retrieve results
            if use_adaptive:
                retrieved = rag_engine.adaptive_retrieve(query)
            else:
                retrieved = rag_engine.retrieve_faculty(query, top_k=5)
            
            # Extract names and scores
            retrieved_names = [r['name'] for r in retrieved]
            scores = [r['score'] for r in retrieved]
            
            # Find rank of first relevant result
            rank_of_first_relevant = 0
            for rank, name in enumerate(retrieved_names, 1):
                if name in must_include:
                    rank_of_first_relevant = rank
                    break
            
            # Check if must_include is present
            must_include_found = all(name in retrieved_names for name in must_include)
            
            # Display results
            print(f"  Retrieved {len(retrieved)} results:")
            for j, (name, score) in enumerate(zip(retrieved_names, scores), 1):
                relevant = "✓" if name in expected else "✗"
                highlight = "⭐" if name in must_include else " "
                print(f"    {j}. {highlight} {name:<30} Score: {score:.4f} {relevant}")
            
            print(f"  Must-include found: {'✓' if must_include_found else '✗'}")
            print(f"  First relevant at rank: {rank_of_first_relevant if rank_of_first_relevant > 0 else 'N/A'}")
            
            # Store result
            results.append({
                'query': query,
                'expected': expected,
                'must_include': must_include,
                'retrieved_names': retrieved_names,
                'scores': scores,
                'rank_of_first_relevant': rank_of_first_relevant,
                'must_include_found': must_include_found
            })
        
        # Calculate metrics
        metrics = {
            'mrr': self.calculate_mrr(results),
            'recall_at_3': self.calculate_recall_at_k(results, k=3),
            'precision_at_3': self.calculate_precision_at_k(results, k=3),
            'score_distribution': self.calculate_score_distribution(results),
            'coverage': self.calculate_coverage(results, threshold=0.30),
            'must_include_success_rate': sum(1 for r in results if r['must_include_found']) / len(results)
        }
        
        return {
            'metrics': metrics,
            'results': results
        }
    
    def compare_models(self, old_engine, new_engine) -> Dict[str, Any]:
        """
        A/B comparison between old and new RAG engines.
        
        Args:
            old_engine: Original ResponseEngine
            new_engine: Improved ResponseEngine
            
        Returns:
            Dictionary with comparison metrics
        """
        print("\n" + "=" * 80)
        print("MODEL COMPARISON: OLD vs NEW")
        print("=" * 80)
        
        print("\n[1/2] Evaluating OLD model...")
        old_results = self.evaluate(old_engine, use_adaptive=False)
        
        print("\n[2/2] Evaluating NEW model...")
        new_results = self.evaluate(new_engine, use_adaptive=True)
        
        comparison = {
            'old_metrics': old_results['metrics'],
            'new_metrics': new_results['metrics'],
            'improvements': {}
        }
        
        # Calculate improvements
        old_metrics = old_results['metrics']
        new_metrics = new_results['metrics']
        
        for metric in ['mrr', 'recall_at_3', 'precision_at_3', 'coverage']:
            old_val = old_metrics[metric]
            new_val = new_metrics[metric]
            improvement = ((new_val - old_val) / old_val * 100) if old_val > 0 else 0
            comparison['improvements'][metric] = {
                'old': old_val,
                'new': new_val,
                'change': new_val - old_val,
                'percent_change': improvement
            }
        
        return comparison
    
    def generate_report(self, evaluation_results: Dict[str, Any], 
                       filename: str = "evaluation_report.md") -> str:
        """
        Generate markdown report with evaluation metrics.
        
        Args:
            evaluation_results: Results from evaluate() method
            filename: Output filename
            
        Returns:
            Markdown report as string
        """
        metrics = evaluation_results['metrics']
        results = evaluation_results['results']
        
        report = []
        report.append("# RAG Evaluation Report")
        report.append("")
        report.append("## Summary Metrics")
        report.append("")
        report.append(f"- **MRR (Mean Reciprocal Rank)**: {metrics['mrr']:.4f}")
        report.append(f"- **Recall@3**: {metrics['recall_at_3']:.4f}")
        report.append(f"- **Precision@3**: {metrics['precision_at_3']:.4f}")
        report.append(f"- **Coverage (score >= 0.30)**: {metrics['coverage']:.4f}")
        report.append(f"- **Must-Include Success Rate**: {metrics['must_include_success_rate']:.4f}")
        report.append("")
        
        report.append("## Score Distribution")
        report.append("")
        dist = metrics['score_distribution']
        report.append(f"- **Min**: {dist['min']:.4f}")
        report.append(f"- **Max**: {dist['max']:.4f}")
        report.append(f"- **Mean**: {dist['mean']:.4f}")
        report.append(f"- **Median**: {dist['median']:.4f}")
        report.append("")
        
        report.append("## Detailed Results")
        report.append("")
        
        for i, result in enumerate(results, 1):
            report.append(f"### Query {i}: {result['query']}")
            report.append("")
            report.append(f"**Expected**: {', '.join(result['expected'])}")
            report.append("")
            report.append(f"**Must Include**: {', '.join(result['must_include'])}")
            report.append("")
            report.append("**Retrieved:**")
            report.append("")
            for j, (name, score) in enumerate(zip(result['retrieved_names'], result['scores']), 1):
                relevant = "✓" if name in result['expected'] else "✗"
                report.append(f"{j}. {name} (score: {score:.4f}) {relevant}")
            report.append("")
            report.append(f"**Status**: {'✓ Success' if result['must_include_found'] else '✗ Failed'}")
            report.append("")
        
        report_text = "\n".join(report)
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n[INFO] Report saved to {filename}")
        
        return report_text


def main():
    """Run evaluation with example usage."""
    print("RAG Evaluator - Example Usage")
    print("=" * 80)
    print("\nTo use this evaluator:")
    print("1. Import: from rag_evaluator import RAGEvaluator")
    print("2. Create evaluator: evaluator = RAGEvaluator()")
    print("3. Load your RAG engine: engine = ResponseEngine()")
    print("4. Run evaluation: results = evaluator.evaluate(engine)")
    print("5. Generate report: evaluator.generate_report(results)")
    print("\nFor comparison:")
    print("   comparison = evaluator.compare_models(old_engine, new_engine)")


if __name__ == "__main__":
    main()
