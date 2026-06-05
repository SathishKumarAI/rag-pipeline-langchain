"""
Evaluation utilities for RAG pipeline.

WHAT: Provides evaluation metrics for RAG systems: hit rate, recall at k, answer quality
      (BLEU/ROUGE-like), and latency tracking.
WHY: Essential for measuring RAG system performance and identifying areas for improvement.
HOW: Implements standard information retrieval and NLP metrics, designed to be runnable
     from tests and notebooks.

Usage:
    from rag.evaluation import RAGEvaluator
    
    evaluator = RAGEvaluator()
    metrics = evaluator.evaluate(
        queries=["What is RAG?"],
        ground_truth=["RAG is Retrieval-Augmented Generation"],
        predictions=["RAG stands for Retrieval-Augmented Generation"]
    )
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class RAGEvaluator:
    """
    Evaluates RAG pipeline performance.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def evaluate(
        self,
        queries: List[str],
        ground_truth: List[str],
        predictions: List[str],
        retrieved_docs_list: Optional[List[List[Dict[str, Any]]]] = None,
        relevant_docs_list: Optional[List[List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate RAG predictions.
        
        Args:
            queries: List of queries
            ground_truth: List of ground truth answers
            predictions: List of predicted answers
            retrieved_docs_list: List of retrieved documents for each query (optional)
            relevant_docs_list: List of relevant document IDs for each query (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if len(queries) != len(ground_truth) or len(queries) != len(predictions):
            raise ValueError("queries, ground_truth, and predictions must have same length")
        
        metrics = {
            "num_queries": len(queries),
            "answer_quality": {},
            "retrieval_quality": {},
        }
        
        # Answer quality metrics
        metrics["answer_quality"] = self._compute_answer_metrics(
            ground_truth, predictions
        )
        
        # Retrieval quality metrics (if provided)
        if retrieved_docs_list and relevant_docs_list:
            metrics["retrieval_quality"] = self._compute_retrieval_metrics(
                retrieved_docs_list, relevant_docs_list
            )
        
        return metrics
    
    def _compute_answer_metrics(
        self,
        ground_truth: List[str],
        predictions: List[str],
    ) -> Dict[str, float]:
        """Compute answer quality metrics."""
        # Simple metrics (can be enhanced with BLEU/ROUGE)
        exact_matches = sum(1 for gt, pred in zip(ground_truth, predictions) if gt.lower() == pred.lower())
        exact_match_rate = exact_matches / len(ground_truth) if ground_truth else 0.0
        
        # Token overlap (simple F1-like metric)
        token_overlaps = []
        for gt, pred in zip(ground_truth, predictions):
            gt_tokens = set(gt.lower().split())
            pred_tokens = set(pred.lower().split())
            if not gt_tokens or not pred_tokens:
                token_overlaps.append(0.0)
                continue
            
            intersection = gt_tokens & pred_tokens
            precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
            recall = len(intersection) / len(gt_tokens) if gt_tokens else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            token_overlaps.append(f1)
        
        avg_token_overlap = sum(token_overlaps) / len(token_overlaps) if token_overlaps else 0.0
        
        return {
            "exact_match_rate": exact_match_rate,
            "avg_token_overlap_f1": avg_token_overlap,
        }
    
    def _compute_retrieval_metrics(
        self,
        retrieved_docs_list: List[List[Dict[str, Any]]],
        relevant_docs_list: List[List[str]],
    ) -> Dict[str, float]:
        """Compute retrieval quality metrics."""
        hit_rates = []
        recall_at_k = []
        
        for retrieved_docs, relevant_doc_ids in zip(retrieved_docs_list, relevant_docs_list):
            retrieved_ids = [doc.get("id") or doc.get("metadata", {}).get("id", "") for doc in retrieved_docs]
            
            # Hit rate: at least one relevant doc retrieved
            has_hit = any(rid in relevant_doc_ids for rid in retrieved_ids)
            hit_rates.append(1.0 if has_hit else 0.0)
            
            # Recall at k: fraction of relevant docs retrieved
            relevant_retrieved = sum(1 for rid in retrieved_ids if rid in relevant_doc_ids)
            recall = relevant_retrieved / len(relevant_doc_ids) if relevant_doc_ids else 0.0
            recall_at_k.append(recall)
        
        return {
            "hit_rate": sum(hit_rates) / len(hit_rates) if hit_rates else 0.0,
            "avg_recall_at_k": sum(recall_at_k) / len(recall_at_k) if recall_at_k else 0.0,
        }
    
    def compute_latency_stats(
        self,
        latencies: List[float],
    ) -> Dict[str, float]:
        """
        Compute latency statistics.
        
        Args:
            latencies: List of latency measurements in seconds
            
        Returns:
            Dictionary with latency statistics
        """
        if not latencies:
            return {}
        
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        return {
            "mean": sum(latencies) / n,
            "median": sorted_latencies[n // 2],
            "p95": sorted_latencies[int(n * 0.95)] if n > 0 else 0.0,
            "p99": sorted_latencies[int(n * 0.99)] if n > 0 else 0.0,
            "min": min(latencies),
            "max": max(latencies),
        }
    
    def evaluate_retrieval_at_k(
        self,
        queries: List[str],
        retrieved_docs_list: List[List[Dict[str, Any]]],
        relevant_docs_list: List[List[str]],
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[str, Dict[int, float]]:
        """
        Evaluate retrieval at different k values.
        
        Args:
            queries: List of queries
            retrieved_docs_list: List of retrieved documents for each query
            relevant_docs_list: List of relevant document IDs for each query
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary mapping metric names to k-value dictionaries
        """
        results = {
            "recall_at_k": {k: [] for k in k_values},
            "precision_at_k": {k: [] for k in k_values},
        }
        
        for retrieved_docs, relevant_doc_ids in zip(retrieved_docs_list, relevant_docs_list):
            retrieved_ids = [doc.get("id") or doc.get("metadata", {}).get("id", "") for doc in retrieved_docs]
            
            for k in k_values:
                top_k_ids = retrieved_ids[:k]
                
                # Recall at k
                relevant_retrieved = sum(1 for rid in top_k_ids if rid in relevant_doc_ids)
                recall = relevant_retrieved / len(relevant_doc_ids) if relevant_doc_ids else 0.0
                results["recall_at_k"][k].append(recall)
                
                # Precision at k
                precision = relevant_retrieved / k if k > 0 else 0.0
                results["precision_at_k"][k].append(precision)
        
        # Average over queries
        for metric_name in results:
            for k in k_values:
                values = results[metric_name][k]
                results[metric_name][k] = sum(values) / len(values) if values else 0.0
        
        return results

