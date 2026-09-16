"""
Stage 2 of FortexAI's detection cascade: checks an incoming prompt against
the Known Attacks and Allowed Prompts indexes before the autoencoder runs,
so exact-or-near repeats of already-confirmed prompts skip the rest of the
cascade.
"""

from embedder import embed_text
from storage import (
    ALLOWED_PROMPTS_DB_PATH,
    ALLOWED_PROMPTS_INDEX_PATH,
    KNOWN_ATTACKS_DB_PATH,
    KNOWN_ATTACKS_INDEX_PATH,
    add_vector_to_index,
    connect_metadata_db,
    insert_prompt_row,
    load_or_create_index,
    normalize_vector,
    save_index,
)

# below this cosine similarity, a nearest neighbor doesn't count as a real match
SIMILARITY_THRESHOLD = 0.90


# finds the single closest vector in a FAISS index and looks up its original text; returns (similarity, matched_text), or (0.0, None) if the index is empty
def find_nearest_neighbor(query_vector, index, db_connection):
    if index.ntotal == 0:
        return 0.0, None

    similarities, ids = index.search(query_vector.reshape(1, -1), 1)
    similarity = float(similarities[0][0])
    matched_id = int(ids[0][0])

    row = db_connection.execute(
        "SELECT prompt_text FROM prompts WHERE id = ?", (matched_id,)
    ).fetchone()
    matched_text = row[0] if row else None

    return similarity, matched_text


# embeds the input text, checks it against both indexes, and reports whether it matches a known attack, a known-benign prompt, or neither
def search_prompt(text):
    query_vector = normalize_vector(embed_text(text))

    attack_index = load_or_create_index(KNOWN_ATTACKS_INDEX_PATH)
    attack_db = connect_metadata_db(KNOWN_ATTACKS_DB_PATH)
    attack_similarity, attack_text = find_nearest_neighbor(query_vector, attack_index, attack_db)
    attack_db.close()

    benign_index = load_or_create_index(ALLOWED_PROMPTS_INDEX_PATH)
    benign_db = connect_metadata_db(ALLOWED_PROMPTS_DB_PATH)
    benign_similarity, benign_text = find_nearest_neighbor(query_vector, benign_index, benign_db)
    benign_db.close()

    if attack_similarity >= SIMILARITY_THRESHOLD and attack_similarity >= benign_similarity:
        return {"match": "attack", "similarity": attack_similarity, "matched_text": attack_text}

    if benign_similarity >= SIMILARITY_THRESHOLD and benign_similarity > attack_similarity:
        return {"match": "benign", "similarity": benign_similarity, "matched_text": benign_text}

    return {
        "match": "not_found",
        "similarity": max(attack_similarity, benign_similarity),
        "matched_text": None,
    }


# embeds a confirmed prompt, stores it in the matching index (attack or benign), and immediately re-saves that index to disk; call this once a verdict has been confirmed
def add_confirmed(text, label):
    if label not in ("attack", "benign"):
        raise ValueError("label must be 'attack' or 'benign'")

    if label == "attack":
        index_path, db_path = KNOWN_ATTACKS_INDEX_PATH, KNOWN_ATTACKS_DB_PATH
    else:
        index_path, db_path = ALLOWED_PROMPTS_INDEX_PATH, ALLOWED_PROMPTS_DB_PATH

    index = load_or_create_index(index_path)
    db_connection = connect_metadata_db(db_path)

    row_id, was_new = insert_prompt_row(db_connection, text)
    if was_new:
        add_vector_to_index(index, row_id, text)
        save_index(index, index_path)

    db_connection.close()
    return row_id
