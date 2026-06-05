"""
Abstract vector store interface for the RAG pipeline.

WHAT: Defines a common interface for vector stores (Pinecone, FAISS, Chroma, etc.).
WHY: Allows the RAG pipeline to work with different vector stores without code changes.
HOW: Defines an abstract base class that concrete implementations must follow.

Usage:
    from rag.storage.vector_store import get_vector_store
    
    store = get_vector_store("pinecone", index_path="./index")
    store.add_documents(chunks)
    results = store.search(query_embedding, top_k=5)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path


class VectorStore(ABC):
    """
    Abstract interface for vector stores.
    """
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with "text", "embedding", and "metadata"
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of document dictionaries with "text", "metadata", and "score"
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by ID.
        
        Args:
            ids: List of document IDs to delete
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the store."""
        pass


class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory vector store for testing.
    """
    
    def __init__(self):
        """Initialize in-memory store."""
        self.documents: List[Dict[str, Any]] = []
        self.next_id = 0
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to in-memory store."""
        for doc in documents:
            if "id" not in doc:
                doc["id"] = str(self.next_id)
                self.next_id += 1
            self.documents.append(doc)
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search using cosine similarity."""
        import numpy as np
        
        query_vec = np.array(query_embedding)
        results = []
        
        for doc in self.documents:
            # Apply filter if provided
            if filter:
                metadata = doc.get("metadata", {})
                if not all(metadata.get(k) == v for k, v in filter.items()):
                    continue
            
            # Compute cosine similarity
            doc_vec = np.array(doc["embedding"])
            similarity = np.dot(query_vec, doc_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            )
            
            results.append({
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
                "score": float(similarity),
            })
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        self.documents = [d for d in self.documents if d.get("id") not in ids]
    
    def clear(self) -> None:
        """Clear all documents."""
        self.documents = []
        self.next_id = 0


def get_vector_store(
    store_type: str,
    index_path: Optional[str] = None,
    **kwargs: Any,
) -> VectorStore:
    """
    Get a vector store instance.
    
    Args:
        store_type: Store type ("pinecone", "faiss", "chroma", "in_memory")
        index_path: Path to index (for file-based stores)
        **kwargs: Additional arguments for store initialization
        
    Returns:
        VectorStore instance
    """
    if store_type == "in_memory":
        return InMemoryVectorStore()
    elif store_type == "faiss":
        from .faiss_store import FAISSVectorStore
        return FAISSVectorStore(index_path=index_path, **kwargs)
    elif store_type == "chroma":
        from .chroma_store import ChromaVectorStore
        return ChromaVectorStore(index_path=index_path, **kwargs)
    elif store_type == "pinecone":
        # Pinecone is a hosted service (needs API key + network); not wired for
        # local/offline use. Add a pinecone_store.py adapter when deploying.
        raise NotImplementedError(
            "Pinecone store is not implemented. Use 'faiss' or 'chroma' for "
            "local use, or add a PineconeVectorStore adapter for hosted deploys."
        )
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")

