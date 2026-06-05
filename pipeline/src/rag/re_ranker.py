"""
Re-ranker for improving retrieval results.

WHAT: Re-ranks retrieved documents using more sophisticated scoring (e.g., cross-encoder
      models, query-document similarity, etc.).
WHY: Initial retrieval may not perfectly rank results. Re-ranking can improve relevance
     by using more expensive but accurate scoring methods.
HOW: Takes initial retrieval results and re-scores them using a cross-encoder or other
     re-ranking model, then returns re-ordered results.

Usage:
    from rag.re_ranker import ReRanker
    
    reranker = ReRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranked = reranker.rerank(query, initial_results)
"""

from typing import Any, Dict, List, Optional

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class ReRanker:
    """
    Re-ranks retrieved documents for better relevance.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
    ):
        """
        Initialize re-ranker.
        
        Args:
            model: Re-ranking model name (if None, uses simple scoring)
            top_k: Number of results to return after re-ranking
        """
        self.model = model
        self.top_k = top_k
        self._model_instance = None
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents for a query.
        
        Args:
            query: Query string
            documents: List of document dictionaries with "text" and "score"
            
        Returns:
            Re-ranked list of documents
        """
        if not documents:
            return []
        
        logger.debug("Re-ranking documents", query=query[:100], num_docs=len(documents))
        
        # If no model specified, use simple score-based re-ranking (already sorted)
        if self.model is None:
            # Documents are already sorted by score, just return top_k
            if self.top_k:
                return documents[:self.top_k]
            return documents
        
        # Use cross-encoder model for re-ranking
        scores = self._compute_rerank_scores(query, documents)
        
        # Combine original scores with re-rank scores
        for doc, rerank_score in zip(documents, scores):
            # Weighted combination (can be tuned)
            original_score = doc.get("score", 0.0)
            combined_score = 0.7 * original_score + 0.3 * rerank_score
            doc["rerank_score"] = rerank_score
            doc["score"] = combined_score
        
        # Re-sort by combined score
        documents.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top_k
        if self.top_k:
            return documents[:self.top_k]
        return documents
    
    def _compute_rerank_scores(
        self,
        query: str,
        documents: List[Dict[str, Any]],
    ) -> List[float]:
        """
        Compute re-ranking scores using a cross-encoder model.
        
        Args:
            query: Query string
            documents: List of documents
            
        Returns:
            List of re-ranking scores
        """
        # Placeholder: would use sentence-transformers cross-encoder
        # For now, return uniform scores (no re-ranking effect)
        try:
            from sentence_transformers import CrossEncoder
            
            if self._model_instance is None:
                self._model_instance = CrossEncoder(self.model)
            
            pairs = [(query, doc["text"]) for doc in documents]
            scores = self._model_instance.predict(pairs)
            return scores.tolist()
        except ImportError:
            logger.warning("sentence-transformers not available, skipping re-ranking")
            return [0.5] * len(documents)
        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}, using original scores")
            return [0.5] * len(documents)

