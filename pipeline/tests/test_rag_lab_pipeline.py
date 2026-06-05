"""End-to-end smoke test for the SimpleRAGPipeline."""

from pathlib import Path

from rag.models import Document
from chunking.fixed_window import FixedWindowChunker
from rag_pipeline.lab_pipeline import RAGLabConfig, SimpleRAGPipeline


class DummyEmbedder:
    """Deterministic, dependency-free embedder for tests."""

    model_name = "dummy"

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        # Simple character-code based vector
        return [float(len(text)), float(sum(ord(c) for c in text) % 1000)]


def test_simple_rag_pipeline_index_and_query(tmp_path):
    # Create a small synthetic document
    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("RAG stands for Retrieval-Augmented Generation.", encoding="utf-8")

    config = RAGLabConfig(data_root=tmp_path, chunk_size=50, chunk_overlap=10)
    pipeline = SimpleRAGPipeline(
        config=config,
        embedder=DummyEmbedder(),
        chunker=FixedWindowChunker(size=50, overlap=10),
    )

    stats = pipeline.index_directory(exts=[".txt"])
    assert stats["num_files"] == 1
    assert stats["num_chunks"] >= 1

    result = pipeline.run_query("What does RAG stand for?", top_k=1)
    assert result["retrieved"]
    assert "Question:" in result["prompt"]


