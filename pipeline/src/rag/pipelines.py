"""
High-level RAG pipeline orchestration.

WHAT: Provides top-level functions to build indexes and run RAG queries.
WHY: Simplifies the RAG workflow by providing easy-to-use functions that orchestrate
     the entire pipeline (ETL → index → query).
HOW: Wraps lower-level modules (IndexBuilder, RAGInference) to provide a simple interface
     for common RAG operations.

Usage:
    from rag.pipelines import build_index, run_rag_query
    
    # Build index from documents
    index = build_index(
        documents=["doc1.txt", "doc2.txt"],
        output_path="./index"
    )
    
    # Run a query
    result = run_rag_query(
        query="What is RAG?",
        index_path="./index"
    )
"""

from typing import Any, Dict, List, Optional
from pathlib import Path

from utils.logging_utils import get_logger
from .index_builder import IndexBuilder
from .inference import RAGInference

logger = get_logger(__name__)


def build_index(
    documents: List[str],
    output_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model: str = "text-embedding-ada-002",
    embedding_provider: str = "openai",
    vector_store_type: str = "pinecone",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Build a vector index from raw documents.
    
    This is a high-level function that orchestrates the entire indexing pipeline:
    1. Load documents
    2. Clean and chunk text
    3. Generate embeddings
    4. Store in vector store
    
    Args:
        documents: List of document paths or text content
        output_path: Path to save the index
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        embedding_model: Embedding model name
        embedding_provider: Embedding provider ("openai", "sentence-transformers", etc.)
        vector_store_type: Vector store type ("pinecone", "faiss", "chroma", etc.)
        **kwargs: Additional arguments passed to IndexBuilder
        
    Returns:
        Dictionary with index metadata
        
    Example:
        index = build_index(
            documents=["doc1.pdf", "doc2.txt"],
            output_path="./my_index",
            chunk_size=500
        )
    """
    logger.info(
        "Building index",
        num_documents=len(documents),
        output_path=output_path,
        chunk_size=chunk_size,
    )
    
    builder = IndexBuilder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        vector_store_type=vector_store_type,
        **kwargs,
    )
    
    index_metadata = builder.build(documents, output_path)
    
    logger.info("Index built successfully", index_path=output_path)
    return index_metadata


def run_rag_query(
    query: str,
    index_path: Optional[str] = None,
    retriever: Optional[Any] = None,
    top_k: int = 5,
    llm_model: str = "gpt-4",
    llm_provider: str = "openai",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run a RAG query end-to-end.
    
    This function orchestrates the entire RAG pipeline:
    1. Retrieve relevant chunks
    2. Build prompt with context
    3. Generate answer using LLM
    4. Return answer with trace
    
    Args:
        query: User query
        index_path: Path to the index (if retriever not provided)
        retriever: Retriever instance (if index_path not provided)
        top_k: Number of chunks to retrieve
        llm_model: LLM model name
        llm_provider: LLM provider ("openai", "bedrock", etc.)
        **kwargs: Additional arguments passed to RAGInference
        
    Returns:
        Dictionary with:
            - answer: Generated answer
            - retrieved_docs: List of retrieved documents
            - prompt: Final prompt sent to LLM
            - model: Model used
            - metadata: Additional metadata
            
    Example:
        result = run_rag_query(
            query="What is RAG?",
            index_path="./my_index",
            top_k=5
        )
        print(result["answer"])
    """
    logger.info("Running RAG query", query=query[:100], top_k=top_k)
    
    inference = RAGInference(
        index_path=index_path,
        retriever=retriever,
        top_k=top_k,
        llm_model=llm_model,
        llm_provider=llm_provider,
        **kwargs,
    )
    
    result = inference.query(query)
    
    logger.info("RAG query completed", answer_length=len(result.get("answer", "")))
    return result

