"""
Shared helpers for the two on-disk stores behind Stage 2: a FAISS index
plus a SQLite sidecar for "Known Attacks" and another pair for "Allowed
Prompts" (the same two collections named in FortexAI's architecture
diagram).
"""

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from embedder import embed_text

STORE_DIR = Path(__file__).resolve().parent / "store"
STORE_DIR.mkdir(exist_ok=True)

KNOWN_ATTACKS_INDEX_PATH = STORE_DIR / "known_attacks.faiss"
KNOWN_ATTACKS_DB_PATH = STORE_DIR / "known_attacks.sqlite"

ALLOWED_PROMPTS_INDEX_PATH = STORE_DIR / "allowed_prompts.faiss"
ALLOWED_PROMPTS_DB_PATH = STORE_DIR / "allowed_prompts.sqlite"

EMBEDDING_DIMENSION = 768  # nomic-embed-text always returns 768-dim vectors


# turns a raw embedding vector into a unit-length vector, so FAISS inner product search behaves like cosine similarity
def normalize_vector(vector):
    array = np.array(vector, dtype="float32")
    norm = np.linalg.norm(array)
    if norm > 0:
        array = array / norm
    return array


# loads a FAISS index from disk if one already exists there, otherwise creates a fresh empty cosine-similarity index
def load_or_create_index(index_path):
    if index_path.exists():
        return faiss.read_index(str(index_path))
    flat_index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    return faiss.IndexIDMap2(
        flat_index
    )  # lets us tag each vector with our own SQLite row id


# writes a FAISS index to disk so it survives between runs
def save_index(index, index_path):
    faiss.write_index(index, str(index_path))


# opens the SQLite sidecar file for one index, creating its "prompts" table and a lookup index on prompt_hash the first time
def connect_metadata_db(db_path):
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=WAL")  # faster commits, still crash-safe
    connection.execute("""
        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY,
            prompt_text TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            added_at TEXT NOT NULL
        )
        """)
    # without this, checking for a duplicate prompt does a full table scan that gets
    # slower as the table grows (it was the biggest bottleneck in early testing)
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_prompts_hash ON prompts(prompt_hash)"
    )
    connection.commit()
    return connection


# hashes prompt text so exact duplicates can be spotted before embedding them a second time
def hash_prompt(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# inserts a prompt's text into the metadata table if it isn't already there; returns (row_id, was_newly_inserted).
# commit=False lets a bulk caller batch many inserts into one commit instead of committing every single row
def insert_prompt_row(db_connection, prompt_text, commit=True):
    prompt_hash = hash_prompt(prompt_text)

    existing_row = db_connection.execute(
        "SELECT id FROM prompts WHERE prompt_hash = ?", (prompt_hash,)
    ).fetchone()
    if existing_row is not None:
        return existing_row[0], False

    added_at = datetime.now(timezone.utc).isoformat()
    cursor = db_connection.execute(
        "INSERT INTO prompts (prompt_text, prompt_hash, added_at) VALUES (?, ?, ?)",
        (prompt_text, prompt_hash, added_at),
    )
    if commit:
        db_connection.commit()
    return cursor.lastrowid, True


# embeds one prompt and adds the resulting vector to the FAISS index under the given SQLite row id
def add_vector_to_index(index, row_id, prompt_text):
    vector = normalize_vector(embed_text(prompt_text))
    index.add_with_ids(vector.reshape(1, -1), np.array([row_id], dtype="int64"))


# adds several already-embedded vectors to the FAISS index in one call, each tagged with its matching SQLite row id
def add_vectors_to_index(index, row_ids, vectors):
    normalized = np.stack([normalize_vector(v) for v in vectors])
    index.add_with_ids(normalized, np.array(row_ids, dtype="int64"))
