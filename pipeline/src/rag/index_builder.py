"""
Index builder for creating vector indexes from raw documents.

WHAT: Builds a searchable vector index from raw documents by loading, cleaning, chunking,
      embedding, and storing documents.
WHY: Provides a reusable component for building indexes that can be used in batch
     processing, ETL pipelines, and one-off indexing tasks.
HOW: Orchestrates the indexing pipeline: document loading → cleaning → chunking →
      embedding → vector store storage. Uses utility modules for each step.

Usage:
    from rag.index_builder import IndexBuilder
    
    builder = IndexBuilder(
        chunk_size=500,
        embedding_model="text-embedding-ada-002"
    )
    metadata = builder.build(["doc1.pdf", "doc2.txt"], "./index")
"""

from typing import Any, Dict, List, Optional
from pathlib import Path

from utils.logging_utils import get_logger
from utils.io_utils import read_text, ensure_dir
from utils.rag_preprocessing import clean_text, chunk_text, enrich_metadata, add_chunk_index
from utils.embeddings import batch_embeddings
from utils.validation import validate_document, validate_batch_chunks
from .storage.vector_store import VectorStore, get_vector_store

logger = get_logger(__name__)


class IndexBuilder:
    """
    Builds vector indexes from raw documents.
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        chunk_strategy: str = "fixed_size",
        embedding_model: str = "text-embedding-ada-002",
        embedding_provider: str = "openai",
        vector_store_type: str = "pinecone",
        batch_size: int = 100,
        **kwargs: Any,
    ):
        """
        Initialize index builder.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            chunk_strategy: Chunking strategy ("fixed_size", "recursive", "sentence")
            embedding_model: Embedding model name
            embedding_provider: Embedding provider
            vector_store_type: Vector store type
            batch_size: Batch size for embedding generation
            **kwargs: Additional arguments for vector store
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.batch_size = batch_size
        self.vector_store_type = vector_store_type
        self.vector_store_kwargs = kwargs
    
    def build(
        self,
        documents: List[str],
        output_path: str,
    ) -> Dict[str, Any]:
        """
        Build index from documents.
        
        Args:
            documents: List of document paths or text content
            output_path: Path to save the index
            
        Returns:
            Dictionary with index metadata
        """
        logger.info(
            "Starting index build",
            num_documents=len(documents),
            output_path=output_path,
        )
        
        # Ensure output directory exists
        output_path = Path(output_path)
        ensure_dir(output_path)
        
        # Step 1: Load and process documents
        all_chunks = []
        for i, doc_input in enumerate(documents):
            logger.debug("Processing document", doc_index=i, doc_input=doc_input[:50])
            
            # Load document (if it's a path)
            if Path(doc_input).exists():
                text = read_text(doc_input)
                source = str(doc_input)
            else:
                # Assume it's raw text
                text = doc_input
                source = f"text_{i}"
            
            # Clean text
            cleaned_text = clean_text(text)
            
            # Chunk text
            text_chunks = chunk_text(
                cleaned_text,
                strategy=self.chunk_strategy,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            
            # Convert to chunk dictionaries
            for j, chunk_text_content in enumerate(text_chunks):
                chunk = {
                    "text": chunk_text_content,
                    "metadata": {
                        "source": source,
                        "chunk_index": j,
                        "total_chunks": len(text_chunks),
                    },
                }
                
                # Enrich metadata
                chunk = enrich_metadata(chunk, source=source)
                all_chunks.append(chunk)
        
        logger.info("Documents processed", total_chunks=len(all_chunks))
        
        # Validate chunks
        validate_batch_chunks(all_chunks)
        
        # Step 2: Generate embeddings
        logger.info("Generating embeddings", num_chunks=len(all_chunks))
        texts = [chunk["text"] for chunk in all_chunks]
        embeddings = batch_embeddings(
            texts,
            model=self.embedding_model,
            provider=self.embedding_provider,
            batch_size=self.batch_size,
        )
        
        # Add embeddings to chunks
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk["embedding"] = embedding
        
        logger.info("Embeddings generated", num_embeddings=len(embeddings))
        
        # Step 3: Store in vector store
        logger.info("Storing in vector store", vector_store_type=self.vector_store_type)
        vector_store = get_vector_store(
            store_type=self.vector_store_type,
            index_path=output_path,
            **self.vector_store_kwargs,
        )
        
        # Store chunks
        vector_store.add_documents(all_chunks)
        
        # Build index metadata
        metadata = {
            "num_documents": len(documents),
            "num_chunks": len(all_chunks),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "embedding_provider": self.embedding_provider,
            "vector_store_type": self.vector_store_type,
            "output_path": str(output_path),
        }
        
        logger.info("Index build completed", **metadata)
        return metadata

