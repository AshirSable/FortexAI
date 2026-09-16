# Semantic Search (Stage 2)

Standalone module for Stage 2 of FortexAI's prompt-injection detection cascade.
Before the autoencoder runs, it checks an incoming prompt against two known
collections — **Known Attacks** and **Allowed Prompts** — using vector
similarity search. A near-duplicate of an already-confirmed prompt is caught
immediately, without a model call.

This folder is fully isolated: it doesn't import from or modify `ml_factory`,
`miniBERTs`, `LLM_Judge`, or `app/main.py`. It reuses the same embedding
model (`nomic-embed-text` via Ollama) that `ml_factory` uses, so vectors here
live in the same embedding space.

## Files

- **`embedder.py`** — turns a piece of text into an embedding vector by
  calling the local Ollama server with the `nomic-embed-text` model (the
  same model and prefix convention `ml_factory/utils` uses).
- **`storage.py`** — shared helpers for the on-disk stores: creating/loading
  a FAISS index, normalizing vectors for cosine similarity, and reading/
  writing the SQLite sidecar that maps a vector ID back to its prompt text.
- **`build_index.py`** — one-time seed script. Reads the labelled training
  corpus (`Data/final_data_cleaner.csv`, columns `text`/`label`), embeds
  each prompt, and stores it in the Known Attacks index (label `1`) or the
  Allowed Prompts index (label `0`). Safe to re-run — it skips prompts it
  has already stored.
- **`search.py`** — the query interface:
  - `search_prompt(text)` embeds the input and searches both indexes,
    returning `{"match": "attack" | "benign" | "not_found", "similarity": float, "matched_text": str | None}`.
    A match only counts if its cosine similarity is at or above
    `SIMILARITY_THRESHOLD` (0.90 by default, set at the top of the file).
  - `add_confirmed(text, label)` embeds a confirmed prompt (`label` is
    `"attack"` or `"benign"`), adds it to the right index, appends it to the
    matching SQLite sidecar, and re-saves the index to disk immediately.
    Not wired into anything yet — call it later once a verdict is confirmed.
- **`demo.py`** — small runnable example. Builds the indexes if they don't
  exist yet, then runs a few example prompts through `search_prompt()` and
  prints the results.
- **`store/`** — where the FAISS index files (`known_attacks.faiss`,
  `allowed_prompts.faiss`) and their SQLite sidecars
  (`known_attacks.sqlite`, `allowed_prompts.sqlite`) are written.

## How to run

1. Make sure Ollama is running locally with `nomic-embed-text` pulled:
   ```
   ollama pull nomic-embed-text
   ```
2. Build the indexes (only needs to be done once; re-running skips prompts
   already stored):
   ```
   python3 build_index.py
   ```
3. Run the demo:
   ```
   python3 demo.py
   ```

`search.py`'s two functions (`search_prompt`, `add_confirmed`) can then be
imported by whatever wires Stage 2 into the real cascade later — that
integration is a separate, later step and is not done here.
