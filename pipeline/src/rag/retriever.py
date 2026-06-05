"""
Retriever for vector and keyword-based document retrieval.

WHAT: Retrieves relevant documents using vector similarity search and optionally keyword search.
WHY: Provides a unified interface for retrieval that can combine multiple search strategies
     (vector similarity, keyword/BM25, hybrid) for better results.
HOW: Uses vector store for similarity search, optionally combines with keyword search,
     and returns ranked results.

Usage:
    from rag.retriever import Retriever
    
    retriever = Retriever(vector_store=store, top_k=5)
    results = retriever.retrieve("What is RAG?")
"""

from typing import Any, Callable, Dict, List, Optional

from utils.logging_utils import get_logger
from utils.embeddings import generate_embeddings
from utils.validation import validate_query
from .storage.vector_store import VectorStore

logger = get_logger(__name__)


class Retriever:
    """
    Retrieves relevant documents for a query.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: str = "text-embedding-ada-002",
        embedding_provider: str = "openai",
        top_k: int = 5,
        use_keyword_search: bool = False,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
    ):
        """
        Initialize retriever.

        Args:
            vector_store: Vector store instance
            embedding_model: Embedding model name
            embedding_provider: Embedding provider
            top_k: Number of documents to retrieve
            use_keyword_search: If True, also use keyword search (not yet implemented)
            embed_fn: Optional callable that turns a query string into an
                embedding vector. When provided it overrides
                embedding_provider/model — useful for tests or custom embedders
                without a live provider.
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.top_k = top_k
        self.use_keyword_search = use_keyword_search
        self.embed_fn = embed_fn
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            top_k: Number of results (overrides instance default)
            filter: Optional metadata filter
            
        Returns:
            List of document dictionaries with "text", "metadata", and "score"
        """
        validate_query(query)
        
        top_k = top_k or self.top_k
        
        logger.debug("Retrieving documents", query=query[:100], top_k=top_k)
        
        # Generate query embedding (injected embedder takes precedence)
        if self.embed_fn is not None:
            query_embedding = self.embed_fn(query)
        else:
            query_embedding = generate_embeddings(
                query,
                model=self.embedding_model,
                provider=self.embedding_provider,
            )
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter=filter,
        )
        
        logger.debug("Retrieved documents", num_results=len(results))
        return results
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with similarity scores.
        
        Same as retrieve() but explicitly includes scores in the result.
        """
        return self.retrieve(query, top_k=top_k, filter=filter)

