"""
Seeds the two Stage 2 indexes from the existing labelled prompt corpus:
every row with label == 1 (attack) goes into the Known Attacks index,
every row with label == 0 (benign) goes into the Allowed Prompts index.

Prompts are embedded in small batches (one Ollama request per batch,
not one per prompt) since that's much faster than embedding one prompt
at a time. Progress is checkpointed to disk every CHECKPOINT_EVERY rows,
so an interrupted run never loses more than one checkpoint's worth of
work, and can just be re-run to pick up where it left off.

Run this once:
    python3 build_index.py
"""

import time
from pathlib import Path

import pandas as pd

from embedder import embed_texts
from storage import (
    ALLOWED_PROMPTS_DB_PATH,
    ALLOWED_PROMPTS_INDEX_PATH,
    KNOWN_ATTACKS_DB_PATH,
    KNOWN_ATTACKS_INDEX_PATH,
    add_vectors_to_index,
    connect_metadata_db,
    insert_prompt_row,
    load_or_create_index,
    save_index,
)

# the labelled corpus lives in the repo-root Data/ folder, outside Backend/
CORPUS_PATH = Path(__file__).resolve().parents[2] / "Data" / "final_data_cleaner.csv"

ATTACK_LABEL = 1  # label used in the corpus for malicious prompts (0 means benign)
PROGRESS_EVERY = 5000  # print a progress line every N rows
EMBED_BATCH_SIZE = 64  # prompts sent to Ollama per request
CHECKPOINT_EVERY = 20000  # rows between saving both indexes + committing both databases


# embeds every pending (index, row_id, text) entry in one batched Ollama call and adds the vectors to their matching index
def flush_pending_batch(pending):
    if not pending:
        return

    texts = [text for _, _, text in pending]
    vectors = embed_texts(texts)

    # one batch can contain a mix of attack and benign rows, so group by which index each vector belongs to
    grouped_by_index = {}
    for (index, row_id, _), vector in zip(pending, vectors):
        row_ids, vecs = grouped_by_index.setdefault(id(index), (index, [], []))[1:]
        row_ids.append(row_id)
        vecs.append(vector)

    for index, row_ids, vecs in grouped_by_index.values():
        add_vectors_to_index(index, row_ids, vecs)

    pending.clear()


# saves both FAISS indexes and commits both SQLite databases, so progress made so far is safe on disk
def checkpoint(attack_index, attack_db, benign_index, benign_db):
    save_index(attack_index, KNOWN_ATTACKS_INDEX_PATH)
    save_index(benign_index, ALLOWED_PROMPTS_INDEX_PATH)
    attack_db.commit()
    benign_db.commit()


# reads the labelled corpus and writes every prompt into the Known Attacks or Allowed Prompts store based on its label
def build_indexes():
    if not CORPUS_PATH.is_file():
        raise FileNotFoundError(
            f"Could not find the labelled corpus at {CORPUS_PATH}. "
            "Update CORPUS_PATH to point at the labelled training data (must have 'text' and 'label' columns)."
        )

    dataframe = pd.read_csv(CORPUS_PATH)
    total_rows = len(dataframe)

    attack_index = load_or_create_index(KNOWN_ATTACKS_INDEX_PATH)
    attack_db = connect_metadata_db(KNOWN_ATTACKS_DB_PATH)

    benign_index = load_or_create_index(ALLOWED_PROMPTS_INDEX_PATH)
    benign_db = connect_metadata_db(ALLOWED_PROMPTS_DB_PATH)

    pending = []  # (index, row_id, prompt_text) waiting to be embedded together
    rows_since_checkpoint = 0
    started_at = time.time()

    for row_number, row in enumerate(dataframe.itertuples(index=False), start=1):
        if not isinstance(row.text, str) or not row.text.strip():
            print(f"Skipping row {row_number}: text is missing/empty")
            continue

        if row.label == ATTACK_LABEL:
            target_index, target_db = attack_index, attack_db
        else:
            target_index, target_db = benign_index, benign_db

        row_id, was_new = insert_prompt_row(target_db, row.text, commit=False)
        if was_new:
            pending.append((target_index, row_id, row.text))

        if len(pending) >= EMBED_BATCH_SIZE:
            flush_pending_batch(pending)

        rows_since_checkpoint += 1
        if rows_since_checkpoint >= CHECKPOINT_EVERY:
            flush_pending_batch(pending)  # never checkpoint with embedded vectors still unsaved
            checkpoint(attack_index, attack_db, benign_index, benign_db)
            rows_since_checkpoint = 0

        if row_number % PROGRESS_EVERY == 0 or row_number == total_rows:
            elapsed_seconds = time.time() - started_at
            print(f"Processed {row_number}/{total_rows} prompts ({elapsed_seconds:.1f}s elapsed)")

    flush_pending_batch(pending)
    checkpoint(attack_index, attack_db, benign_index, benign_db)
    attack_db.close()
    benign_db.close()

    print(
        f"Done. Known Attacks vectors: {attack_index.ntotal}, "
        f"Allowed Prompts vectors: {benign_index.ntotal}"
    )


if __name__ == "__main__":
    build_indexes()
