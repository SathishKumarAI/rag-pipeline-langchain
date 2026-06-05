"""
RAG preprocessing utilities for text cleaning, chunking, and metadata enrichment.

WHAT: Text cleaning (HTML stripping, normalization), chunking strategies (fixed-size,
      recursive, semantic), and metadata enrichment.
WHY: Clean, well-chunked text with rich metadata is essential for good RAG performance.
HOW: Provides multiple chunking strategies, text cleaning functions, and utilities
     to enrich chunks with metadata (source, page, section, timestamps).

Usage:
    from utils.rag_preprocessing import clean_text, chunk_text, enrich_metadata
    
    cleaned = clean_text("<html>text</html>")
    chunks = chunk_text(cleaned, strategy="fixed_size", chunk_size=500)
    enriched = enrich_metadata(chunks[0], source="doc.pdf", page=1)
"""

import re
import html
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


def clean_text(
    text: str,
    remove_html: bool = True,
    normalize_whitespace: bool = True,
    remove_urls: bool = False,
    remove_emails: bool = False,
) -> str:
    """
    Clean text by removing HTML, normalizing whitespace, etc.
    
    Args:
        text: Input text
        remove_html: If True, strip HTML tags
        normalize_whitespace: If True, normalize whitespace
        remove_urls: If True, remove URLs
        remove_emails: If True, remove email addresses
        
    Returns:
        Cleaned text
    """
    cleaned = text
    
    # Remove HTML
    if remove_html:
        if BeautifulSoup is not None:
            soup = BeautifulSoup(cleaned, "html.parser")
            cleaned = soup.get_text()
        else:
            # Fallback: basic regex-based HTML removal
            cleaned = re.sub(r"<[^>]+>", "", cleaned)
            cleaned = html.unescape(cleaned)
    
    # Remove URLs
    if remove_urls:
        cleaned = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            cleaned,
        )
    
    # Remove emails
    if remove_emails:
        cleaned = re.sub(r"\S+@\S+", "", cleaned)
    
    # Normalize whitespace
    if normalize_whitespace:
        # Replace multiple whitespace with single space
        cleaned = re.sub(r"\s+", " ", cleaned)
        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()
    
    return cleaned


def chunk_text(
    text: str,
    strategy: str = "fixed_size",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """
    Chunk text using the specified strategy.
    
    Args:
        text: Text to chunk
        strategy: Chunking strategy ("fixed_size", "recursive", "sentence")
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        separators: List of separators for recursive chunking
        
    Returns:
        List of text chunks
    """
    if strategy == "fixed_size":
        return _chunk_fixed_size(text, chunk_size, chunk_overlap)
    elif strategy == "recursive":
        return _chunk_recursive(text, chunk_size, chunk_overlap, separators)
    elif strategy == "sentence":
        return _chunk_by_sentence(text, chunk_size, chunk_overlap)
    else:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")


def _chunk_fixed_size(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Chunk text into fixed-size chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap
    
    return chunks


def _chunk_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: Optional[List[str]],
) -> List[str]:
    """Recursively chunk text by separators."""
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]
    
    if len(text) <= chunk_size:
        return [text]
    
    # Try each separator in order
    for separator in separators:
        if separator == "":
            # Last resort: fixed-size chunking
            return _chunk_fixed_size(text, chunk_size, chunk_overlap)
        
        if separator in text:
            splits = text.split(separator)
            chunks = []
            current_chunk = ""
            
            for split in splits:
                # Check if adding this split would exceed chunk_size
                test_chunk = current_chunk + separator + split if current_chunk else split
                
                if len(test_chunk) <= chunk_size:
                    current_chunk = test_chunk
                else:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = split
            
            # Add last chunk
            if current_chunk:
                chunks.append(current_chunk)
            
            # If we successfully chunked, return chunks
            if len(chunks) > 1:
                return chunks
    
    # Fallback to fixed-size
    return _chunk_fixed_size(text, chunk_size, chunk_overlap)


def _chunk_by_sentence(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Chunk text by sentences."""
    # Simple sentence splitting (can be improved with NLTK/spaCy)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        test_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        if len(test_chunk) <= chunk_size:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def enrich_metadata(
    chunk: Dict[str, Any],
    source: Optional[str] = None,
    page: Optional[int] = None,
    section: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Enrich chunk metadata with additional information.
    
    Args:
        chunk: Chunk dictionary (must have "metadata" key)
        source: Source document path/URL
        page: Page number
        section: Section name/title
        timestamp: Timestamp when chunk was created
        **kwargs: Additional metadata fields
        
    Returns:
        Chunk with enriched metadata
    """
    if "metadata" not in chunk:
        chunk["metadata"] = {}
    
    metadata = chunk["metadata"]
    
    if source is not None:
        metadata["source"] = source
    
    if page is not None:
        metadata["page"] = page
    
    if section is not None:
        metadata["section"] = section
    
    if timestamp is not None:
        metadata["timestamp"] = timestamp.isoformat()
    elif "timestamp" not in metadata:
        metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    # Add any additional metadata
    metadata.update(kwargs)
    
    return chunk


def add_chunk_index(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add chunk index to each chunk's metadata.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Chunks with added index metadata
    """
    for i, chunk in enumerate(chunks):
        if "metadata" not in chunk:
            chunk["metadata"] = {}
        chunk["metadata"]["chunk_index"] = i
        chunk["metadata"]["total_chunks"] = len(chunks)
    
    return chunks

