"""
End-to-end RAG inference pipeline.

WHAT: Orchestrates the complete RAG pipeline: query → retrieval → prompt building →
      LLM generation → answer.
WHY: Provides a single entry point for RAG queries that handles the entire pipeline
     and returns structured results with trace information.
HOW: Uses Retriever, ReRanker, PromptBuilder, and LLM utilities to process queries
     and generate answers.

Usage:
    from rag.inference import RAGInference
    
    inference = RAGInference(index_path="./index", top_k=5)
    result = inference.query("What is RAG?")
    print(result["answer"])
"""

from typing import Any, Dict, List, Optional

from utils.logging_utils import get_logger
from utils.llm_utils import call_llm
from utils.validation import validate_query
from .retriever import Retriever
from .re_ranker import ReRanker
from .prompt_builder import PromptBuilder
from .storage.vector_store import get_vector_store

logger = get_logger(__name__)


class RAGInference:
    """
    End-to-end RAG inference pipeline.
    """
    
    def __init__(
        self,
        index_path: Optional[str] = None,
        retriever: Optional[Retriever] = None,
        top_k: int = 5,
        use_reranking: bool = False,
        reranker_model: Optional[str] = None,
        llm_model: str = "gpt-4",
        llm_provider: str = "openai",
        prompt_style: str = "standard",
        **kwargs: Any,
    ):
        """
        Initialize RAG inference pipeline.
        
        Args:
            index_path: Path to vector index (if retriever not provided)
            retriever: Retriever instance (if index_path not provided)
            top_k: Number of documents to retrieve
            use_reranking: If True, use re-ranking
            reranker_model: Re-ranking model name
            llm_model: LLM model name
            llm_provider: LLM provider
            prompt_style: Prompt style ("standard", "concise", "verbose", "json")
            **kwargs: Additional arguments for retriever/LLM
        """
        # Initialize retriever
        if retriever is not None:
            self.retriever = retriever
        elif index_path is not None:
            vector_store = get_vector_store("in_memory", index_path=index_path)
            self.retriever = Retriever(
                vector_store=vector_store,
                top_k=top_k,
                **kwargs,
            )
        else:
            raise ValueError("Either index_path or retriever must be provided")
        
        # Initialize re-ranker (optional)
        self.use_reranking = use_reranking
        if use_reranking:
            self.reranker = ReRanker(model=reranker_model, top_k=top_k)
        else:
            self.reranker = None
        
        # Initialize prompt builder
        self.prompt_builder = PromptBuilder(style=prompt_style)
        
        # LLM settings
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        self.llm_kwargs = kwargs
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run a RAG query end-to-end.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve (overrides instance default)
            filter: Optional metadata filter
            
        Returns:
            Dictionary with:
                - answer: Generated answer
                - retrieved_docs: List of retrieved documents
                - prompt: Final prompt sent to LLM
                - model: Model used
                - metadata: Additional metadata (latency, etc.)
        """
        validate_query(query)
        
        logger.info("Running RAG query", query=query[:100])
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k, filter=filter)
        
        if not retrieved_docs:
            logger.warning("No documents retrieved for query", query=query[:100])
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "retrieved_docs": [],
                "prompt": "",
                "model": self.llm_model,
                "metadata": {"retrieved_count": 0},
            }
        
        logger.debug("Retrieved documents", num_docs=len(retrieved_docs))
        
        # Step 2: Re-rank (optional)
        if self.use_reranking and self.reranker:
            retrieved_docs = self.reranker.rerank(query, retrieved_docs)
            logger.debug("Re-ranked documents", num_docs=len(retrieved_docs))
        
        # Step 3: Build prompt
        prompt = self.prompt_builder.build(query, retrieved_docs)
        logger.debug("Built prompt", prompt_length=len(prompt))
        
        # Step 4: Generate answer using LLM
        try:
            answer = call_llm(
                prompt=prompt,
                model=self.llm_model,
                provider=self.llm_provider,
                **self.llm_kwargs,
            )
            logger.info("Generated answer", answer_length=len(answer))
        except Exception as e:
            logger.error("LLM call failed", error=str(e))
            answer = f"Error generating answer: {str(e)}"
        
        # Step 5: Build result
        result = {
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "prompt": prompt,
            "model": self.llm_model,
            "metadata": {
                "retrieved_count": len(retrieved_docs),
                "reranked": self.use_reranking,
            },
        }
        
        return result

