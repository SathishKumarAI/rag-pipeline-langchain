# Gaps and Future Improvements

> **Update (2026-06):** Several High-priority gaps were closed in the
> refactor effort. See [`BUILD_LOG.md`](BUILD_LOG.md), the
> [ADRs](adr/), and [`research/rag_best_practices.md`](research/rag_best_practices.md)
> for the full reasoning, branch trail, and research-driven roadmap.
>
> **Done:** real FAISS + Chroma vector stores (OSS adapters, ADR-0001) ·
> typed error hierarchy + tenacity retry (ADR-0002) · lazy package imports
> (ADR-0003) · LF normalization (ADR-0004) · repaired packaging/imports so the
> test suite runs (50 passed, 2 skipped).

## Known Gaps

### 1. Vector Store Implementations

**Status**: ✅ FAISS and Chroma implemented (OSS adapters, see ADR-0001);
Pinecone still pending (hosted, raises explicit NotImplementedError).

**Missing:**
- Full Pinecone integration (hosted; needs API key + network)
- Connection pooling and retry logic for hosted backends
- Batch operations optimization

**Priority**: Medium (local stores done; Pinecone only for hosted deploy)

**Estimated Effort**: 2-3 days (Pinecone)

### 2. Advanced Re-ranking

**Status**: Basic re-ranking implemented

**Missing:**
- Multiple re-ranking models support
- Learning-to-rank approaches
- Query-dependent re-ranking
- Re-ranking with metadata

**Priority**: Medium

**Estimated Effort**: 3-5 days

### 3. Evaluation Metrics

**Status**: Basic metrics implemented

**Missing:**
- BLEU/ROUGE scores
- Semantic similarity metrics
- Human evaluation framework
- A/B testing infrastructure

**Priority**: Medium

**Estimated Effort**: 2-3 days

### 4. Multi-vector Search

**Status**: Not implemented

**Missing:**
- Multiple embedding models per document
- Hybrid search (vector + keyword)
- Query expansion
- Multi-modal retrieval

**Priority**: Low

**Estimated Effort**: 1-2 weeks

### 5. Caching Layer

**Status**: Not implemented

**Missing:**
- Query result caching
- Embedding caching
- LLM response caching
- Cache invalidation strategies

**Priority**: Medium

**Estimated Effort**: 3-5 days

## Technical Debt

### 1. Error Handling

**Status**: ◐ Partially done (ADR-0002). A typed `RAGError` hierarchy
(`utils/exceptions.py`) and a tenacity-based `retry_on_transient`
(`utils/resilience.py`) now exist and are applied to the OpenAI embedding calls.

**Remaining**: thread the typed exceptions through the rest of the modules
(loaders, retriever, inference) and retire the legacy `llm_utils.retry_with_backoff`.

**Priority**: Medium

### 2. Configuration Management

**Issue**: Some hardcoded values remain

**Improvement**: Move all configuration to config system

**Priority**: Low

### 3. Documentation

**Issue**: Some modules lack detailed docstrings

**Improvement**: Add comprehensive docstrings, examples, type hints

**Priority**: Low

### 4. Type Hints

**Issue**: Not all functions have complete type hints

**Improvement**: Add type hints throughout codebase

**Priority**: Low

## Future Enhancements

### 1. Async/Streaming Support

**Description**: Add async/await support for concurrent operations and streaming responses

**Benefits:**
- Better performance for batch operations
- Real-time streaming answers
- Improved user experience

**Estimated Effort**: 1-2 weeks

**Suggested Implementation:**
```python
async def async_retrieve(query: str) -> List[Dict]:
    # Async retrieval
    pass

async def stream_answer(query: str):
    # Stream answer tokens
    async for token in generate_stream(query):
        yield token
```

### 2. Advanced Observability

**Description**: Add distributed tracing, metrics, and monitoring

**Benefits:**
- Better debugging
- Performance monitoring
- Production insights

**Estimated Effort**: 1 week

**Suggested Tools:**
- OpenTelemetry for tracing
- Prometheus for metrics
- Grafana for dashboards

### 3. Multi-tenant Support

**Description**: Support multiple tenants with isolated indexes

**Benefits:**
- SaaS deployment ready
- Better resource management
- Security isolation

**Estimated Effort**: 2-3 weeks

**Suggested Implementation:**
```python
class MultiTenantRAG:
    def __init__(self):
        self.tenants = {}
    
    def get_index(self, tenant_id: str):
        return self.tenants[tenant_id]
```

### 4. Query Rewriting and Expansion

**Description**: Improve queries before retrieval

**Benefits:**
- Better retrieval accuracy
- Handle ambiguous queries
- Support for complex questions

**Estimated Effort**: 1 week

**Suggested Implementation:**
```python
def rewrite_query(query: str) -> str:
    # Expand query with synonyms
    # Disambiguate entities
    # Add context
    pass
```

### 5. Advanced Chunking Strategies

**Description**: Semantic chunking, hierarchical chunking

**Benefits:**
- Better chunk boundaries
- Preserve document structure
- Improve retrieval quality

**Estimated Effort**: 1 week

**Suggested Implementation:**
```python
def semantic_chunk(text: str, model: str) -> List[str]:
    # Use LLM to identify semantic boundaries
    pass
```

## Performance Considerations

### Current Limitations

1. **Index Size**: In-memory stores limited by RAM
2. **Latency**: Sequential processing in some paths
3. **Batch Sizes**: Fixed batch sizes may not be optimal
4. **Embedding Generation**: No parallelization

### Optimization Opportunities

1. **Parallel Processing**: Use multiprocessing for batch operations
2. **Index Sharding**: Split large indexes across multiple stores
3. **Embedding Caching**: Cache frequently used embeddings
4. **Lazy Loading**: Load documents on-demand

## Scaling Considerations

### Current Assumptions

- Medium-scale production (thousands of documents)
- Single-machine deployment
- Synchronous processing

### Scaling Challenges

1. **Large Document Collections**: Need distributed indexing
2. **High Query Volume**: Need load balancing and caching
3. **Real-time Updates**: Need incremental indexing
4. **Multi-region**: Need replication strategies

## Security Considerations

### Current State

- Basic input validation
- No authentication/authorization
- API keys in configuration

### Improvements Needed

1. **Authentication**: Add user authentication
2. **Authorization**: Role-based access control
3. **API Key Management**: Secure key storage
4. **Input Sanitization**: Prevent injection attacks
5. **Rate Limiting**: Prevent abuse

## Suggested Roadmap

### Phase 1 (Immediate - 1 month)
- Complete vector store implementations
- Add comprehensive error handling
- Improve evaluation metrics

### Phase 2 (Short-term - 2-3 months)
- Add caching layer
- Implement async/streaming support
- Add advanced re-ranking

### Phase 3 (Medium-term - 4-6 months)
- Multi-vector search
- Query rewriting
- Advanced observability

### Phase 4 (Long-term - 6+ months)
- Multi-tenant support
- Distributed indexing
- Advanced security features

## Contributing

When adding new features:

1. **Document Assumptions**: Add to this file
2. **Add Tests**: Ensure >80% coverage
3. **Update Documentation**: Keep docs in sync
4. **Follow Patterns**: Use existing code patterns
5. **Consider Performance**: Profile critical paths

## Questions and Decisions

### Open Questions

1. Should we support multiple embedding models simultaneously?
2. What's the preferred vector store for production?
3. How should we handle document updates/deletions?
4. What's the target latency for queries?

### Design Decisions Needed

1. Caching strategy (LRU, TTL, etc.)
2. Re-ranking default behavior (always on/off)
3. Chunking strategy defaults
4. Error recovery strategies

