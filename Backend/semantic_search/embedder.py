"""
Turns prompt text into embedding vectors using the same Ollama model
(nomic-embed-text) that Backend/ml_factory already uses, so Stage 2's
vectors live in the same embedding space as the autoencoder's.
"""

import ollama

# same model ml_factory/training_scripts uses for the autoencoder embeddings
EMBEDDING_MODEL = "nomic-embed-text"

# nomic-embed-text expects this prefix on every embedded string (same default
# ml_factory/utils/embedding_text uses)
PREFIX_TEXT = "search_document: "

# one shared sync client talking to the local Ollama server
_client = ollama.Client(host="http://localhost:11434")


# sends one piece of text to Ollama and returns its embedding vector as a plain list of floats
def embed_text(text):
    response = _client.embed(model=EMBEDDING_MODEL, input=PREFIX_TEXT + text)
    return response["embeddings"][0]


# sends a list of texts to Ollama in a single request and returns one embedding vector per text, in the same order (much faster than calling embed_text in a loop when embedding many prompts at once)
def embed_texts(texts):
    response = _client.embed(model=EMBEDDING_MODEL, input=[PREFIX_TEXT + t for t in texts])
    return response["embeddings"]
