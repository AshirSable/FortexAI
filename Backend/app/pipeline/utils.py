"""
Small shared helper for pipeline phases that need code from
Backend/semantic_search/. That folder is not a normal Python package (it has
no __init__.py, and its files import each other like `from embedder import
embed_text`), so its files only import cleanly when the semantic_search
folder itself is on sys.path. This file adds it to sys.path once, so other
pipeline phases can reuse it (e.g. autoencoder.py needs the same embeddings
semantic_search.py uses) without copy-pasting this path trick everywhere.

We don't modify anything inside Backend/semantic_search/ - we just import it.
"""

import sys
from pathlib import Path

# Backend/semantic_search
SEMANTIC_SEARCH_DIR = Path(__file__).resolve().parents[2] / "semantic_search"


def add_semantic_search_to_path():
    path_str = str(SEMANTIC_SEARCH_DIR)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def embed_text(text: str):
    """Embeds text the same way Backend/semantic_search does (nomic-embed-text via Ollama)."""
    add_semantic_search_to_path()
    from embedder import embed_text as _embed_text  # semantic_search/embedder.py

    return _embed_text(text)
