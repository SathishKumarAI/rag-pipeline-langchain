"""
Prompt builder for RAG queries.

WHAT: Builds prompts for LLM queries by combining system instructions, retrieved context,
      and user queries.
WHY: Standardizes prompt formatting and makes it easy to customize prompt styles
     (concise, verbose, JSON-structured, etc.).
HOW: Provides template-based prompt building with support for citations, sources,
     and different output formats.

Usage:
    from rag.prompt_builder import PromptBuilder
    
    builder = PromptBuilder(style="concise")
    prompt = builder.build(query="What is RAG?", context_docs=retrieved_docs)
"""

from typing import Any, Dict, List, Optional

from utils.logging_utils import get_logger
from utils.llm_utils import build_rag_prompt

logger = get_logger(__name__)


class PromptBuilder:
    """
    Builds prompts for RAG queries.
    """
    
    def __init__(
        self,
        style: str = "standard",
        include_sources: bool = True,
        max_context_length: int = 2000,
    ):
        """
        Initialize prompt builder.
        
        Args:
            style: Prompt style ("standard", "concise", "verbose", "json")
            include_sources: If True, include source citations
            max_context_length: Maximum length of context to include
        """
        self.style = style
        self.include_sources = include_sources
        self.max_context_length = max_context_length
    
    def build(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        system_instructions: Optional[str] = None,
    ) -> str:
        """
        Build a RAG prompt.
        
        Args:
            query: User query
            context_docs: List of retrieved documents with "text" and "metadata"
            system_instructions: Optional system instructions
            
        Returns:
            Formatted prompt string
        """
        logger.debug("Building prompt", query=query[:100], num_docs=len(context_docs))
        
        # Build context from documents
        context = self._build_context(context_docs)
        
        # Get system instructions
        if system_instructions is None:
            system_instructions = self._get_default_instructions()
        
        # Build prompt based on style
        if self.style == "concise":
            return self._build_concise_prompt(query, context, system_instructions)
        elif self.style == "verbose":
            return self._build_verbose_prompt(query, context, system_instructions)
        elif self.style == "json":
            return self._build_json_prompt(query, context, system_instructions)
        else:
            return self._build_standard_prompt(query, context, system_instructions)
    
    def _build_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Build context string from documents."""
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(context_docs):
            text = doc["text"]
            metadata = doc.get("metadata", {})
            
            # Add source citation if enabled
            if self.include_sources:
                source = metadata.get("source", f"Document {i+1}")
                text_with_source = f"[Source: {source}]\n{text}"
            else:
                text_with_source = text
            
            # Check if adding this doc would exceed max length
            if current_length + len(text_with_source) > self.max_context_length:
                break
            
            context_parts.append(text_with_source)
            current_length += len(text_with_source)
        
        return "\n\n---\n\n".join(context_parts)
    
    def _get_default_instructions(self) -> str:
        """Get default system instructions."""
        return (
            "You are a helpful assistant that answers questions based on the provided context. "
            "Use only the information from the context to answer. If the context doesn't contain "
            "enough information, say so. Include citations when referencing specific sources."
        )
    
    def _build_standard_prompt(
        self,
        query: str,
        context: str,
        system_instructions: str,
    ) -> str:
        """Build standard RAG prompt."""
        return build_rag_prompt(query, context, system_instructions)
    
    def _build_concise_prompt(
        self,
        query: str,
        context: str,
        system_instructions: str,
    ) -> str:
        """Build concise prompt."""
        return f"Context:\n{context}\n\nQ: {query}\nA:"
    
    def _build_verbose_prompt(
        self,
        query: str,
        context: str,
        system_instructions: str,
    ) -> str:
        """Build verbose prompt with detailed instructions."""
        prompt = f"{system_instructions}\n\n"
        prompt += "=" * 50 + "\n"
        prompt += "CONTEXT DOCUMENTS:\n"
        prompt += "=" * 50 + "\n\n"
        prompt += context
        prompt += "\n\n" + "=" * 50 + "\n"
        prompt += "USER QUESTION:\n"
        prompt += "=" * 50 + "\n\n"
        prompt += query
        prompt += "\n\n" + "=" * 50 + "\n"
        prompt += "INSTRUCTIONS:\n"
        prompt += "- Answer based solely on the context provided above.\n"
        prompt += "- If the context doesn't contain enough information, state that clearly.\n"
        prompt += "- Include source citations when referencing specific documents.\n"
        prompt += "- Be thorough and accurate.\n\n"
        prompt += "ANSWER:\n"
        return prompt
    
    def _build_json_prompt(
        self,
        query: str,
        context: str,
        system_instructions: str,
    ) -> str:
        """Build prompt requesting JSON-structured output."""
        prompt = f"{system_instructions}\n\n"
        prompt += f"Context:\n{context}\n\n"
        prompt += f"Question: {query}\n\n"
        prompt += "Please provide your answer in the following JSON format:\n"
        prompt += '{\n  "answer": "your answer here",\n  "sources": ["source1", "source2"],\n  "confidence": 0.95\n}\n'
        return prompt

