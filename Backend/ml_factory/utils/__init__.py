import ollama
from typing import List

EMBEDDING_CLIENT = ollama.AsyncClient(host='http://localhost:11434')

async def embedding_text(model: str, text: List[str], item_id:int, prefix_text: str = "search_document: ") -> List:
    try:
        response = await EMBEDDING_CLIENT.embed(model, [prefix_text + t for t in text])
        print(f"Success: Processed Item: {item_id}")

        return response['embeddings']
    except Exception as e:
        print(f"Error on Item id: {item_id}, error: {e}")
        return []

