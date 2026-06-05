"""
LLM utility functions for the RAG pipeline.

WHAT: Wrapper functions to call LLMs (OpenAI, AWS Bedrock, local models) with retry logic,
      timeout handling, and prompt formatting.
WHY: Provides a unified interface for different LLM providers, handles common patterns
     like retries and timeouts, and standardizes prompt formatting.
HOW: Abstracts LLM calls behind a simple interface, implements exponential backoff retries,
     and provides prompt templating utilities.

Usage:
    from utils.llm_utils import call_llm, format_prompt, retry_with_backoff
    
    response = call_llm(
        prompt="What is RAG?",
        model="gpt-4",
        provider="openai"
    )
    
    template = "Context: {context}\nQuestion: {question}"
    prompt = format_prompt(template, context="...", question="...")
"""

import json
import time
from typing import Any, Callable, Dict, List, Optional, Union
from functools import wraps
try:
    import openai
except ImportError:
    openai = None
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = Exception


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator to retry a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        max_delay: Maximum delay in seconds
        exceptions: Tuple of exceptions to catch and retry on
        
    Returns:
        Decorated function
        
    Example:
        @retry_with_backoff(max_retries=3, initial_delay=1.0)
        def call_api():
            # API call that might fail
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(min(delay, max_delay))
                        delay *= backoff_factor
                    else:
                        raise
            
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


def format_prompt(template: str, **kwargs: Any) -> str:
    """
    Format a prompt template with keyword arguments.
    
    Args:
        template: Prompt template string (supports {variable} placeholders)
        **kwargs: Variables to substitute in template
        
    Returns:
        Formatted prompt string
        
    Example:
        template = "Context: {context}\nQuestion: {question}"
        prompt = format_prompt(template, context="...", question="What is RAG?")
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing template variable: {e}")


def call_llm(
    prompt: str,
    model: str = "gpt-4",
    provider: str = "openai",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: float = 60.0,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """
    Call an LLM with the given prompt.
    
    Args:
        prompt: Input prompt
        model: Model name/identifier
        provider: Provider name ("openai", "bedrock", "local")
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        api_key: API key (if not provided, uses environment/config)
        **kwargs: Additional provider-specific arguments
        
    Returns:
        Generated text response
        
    Raises:
        ValueError: If provider is not supported
        TimeoutError: If request times out
        Exception: If LLM call fails
    """
    if provider == "openai":
        return _call_openai(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            api_key=api_key,
            **kwargs,
        )
    elif provider == "bedrock":
        return _call_bedrock(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs,
        )
    elif provider == "local":
        return _call_local(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


@retry_with_backoff(
    max_retries=3,
    # Guard: openai may be None when the package isn't installed, and this
    # decorator is evaluated at import time. Fall back to a generic Exception.
    exceptions=(openai.OpenAIError,) if openai is not None else (Exception,),
)
def _call_openai(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    timeout: float,
    api_key: Optional[str],
    **kwargs: Any,
) -> str:
    """Call OpenAI API."""
    client = openai.OpenAI(api_key=api_key, timeout=timeout)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    
    return response.choices[0].message.content


def _call_bedrock(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    timeout: float,
    **kwargs: Any,
) -> str:
    """Call AWS Bedrock API."""
    bedrock = boto3.client("bedrock-runtime")
    
    # Determine model ID format
    if "anthropic" in model.lower() or "claude" in model.lower():
        body = {
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
            "max_tokens_to_sample": max_tokens or 1024,
            "temperature": temperature,
            **kwargs,
        }
        model_id = model if model.startswith("anthropic") else f"anthropic.claude-v2"
    else:
        # Default to text generation format
        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens or 1024,
                "temperature": temperature,
            },
            **kwargs,
        }
        model_id = model
    
    try:
        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
        )
        result = json.loads(response["body"].read())
        
        if "completion" in result:
            return result["completion"]
        elif "generatedText" in result:
            return result["generatedText"]
        else:
            raise ValueError(f"Unexpected response format: {result}")
    except ClientError as e:
        raise Exception(f"Bedrock API error: {e}")


def _call_local(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    timeout: float,
    **kwargs: Any,
) -> str:
    """
    Call a local LLM (placeholder - implement based on your local setup).
    
    This is a placeholder that should be implemented based on your local
    LLM setup (e.g., llama.cpp, vLLM, etc.).
    """
    raise NotImplementedError("Local LLM calls not yet implemented")


def build_system_prompt(
    instructions: str,
    context: Optional[str] = None,
    examples: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Build a system prompt with instructions, context, and examples.
    
    Args:
        instructions: System instructions
        context: Additional context
        examples: List of example dicts with "input" and "output" keys
        
    Returns:
        Formatted system prompt
    """
    parts = [instructions]
    
    if context:
        parts.append(f"\nContext:\n{context}")
    
    if examples:
        parts.append("\nExamples:")
        for i, example in enumerate(examples, 1):
            parts.append(f"\nExample {i}:")
            parts.append(f"Input: {example.get('input', '')}")
            parts.append(f"Output: {example.get('output', '')}")
    
    return "\n".join(parts)


def build_rag_prompt(
    query: str,
    context: str,
    system_instructions: Optional[str] = None,
) -> str:
    """
    Build a RAG prompt with query and retrieved context.
    
    Args:
        query: User query
        context: Retrieved context/documentation
        system_instructions: Optional system instructions
        
    Returns:
        Formatted RAG prompt
    """
    if system_instructions:
        prompt = f"{system_instructions}\n\n"
    else:
        prompt = "Answer the question based on the following context.\n\n"
    
    prompt += f"Context:\n{context}\n\n"
    prompt += f"Question: {query}\n\n"
    prompt += "Answer:"
    
    return prompt

