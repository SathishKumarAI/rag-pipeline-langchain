"""Document loaders for the multi-format RAG lab.

Each loader converts a concrete file type (PDF, DOCX, CSV, etc.)
into one or more `rag.models.Document` instances.

Importing this package registers every available loader via the
``@register_loader`` decorator so ``get_loader_for_path`` can resolve a file
by suffix. Loaders whose optional third-party dependency is missing (e.g.
python-docx, beautifulsoup4, Pillow) are skipped gracefully rather than
breaking the whole registry.
"""

from .base import DocumentLoader, get_loader_for_path, register_loader  # noqa: F401

# Import side effect: each module's @register_loader populates the registry.
# stdlib-only loaders import unconditionally.
from . import text_loader  # noqa: F401
from . import json_loader  # noqa: F401
from . import csv_loader  # noqa: F401

# Optional loaders pull third-party deps; skip if not installed so one missing
# dependency only disables that single format.
for _optional in ("pdf_loader", "docx_loader", "html_loader", "image_loader"):
    try:
        __import__(f"{__name__}.{_optional}", fromlist=[_optional])
    except ImportError:
        pass

__all__ = ["DocumentLoader", "get_loader_for_path", "register_loader"]
