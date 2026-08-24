import numpy as np
import torch
from torch.utils.data import Dataset
import polars as pl
import asyncio
from pathlib import Path


async def precompute_features_ae(
    dataset_records: pl.LazyFrame,
    embed_fn,
    embedding_model_name: str,
    out_path: str | Path,
    batch_size: int = 32,
    chunk_size: int = 2,
    max_concurrent: int = 5,
):
    all_ids, all_embeddings, all_labels = [], [], []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_embed(texts, batch_id):
        async with semaphore:
            return await embed_fn(embedding_model_name, texts, batch_id)

    async def flush(tasks, pending_meta):
        results = await asyncio.gather(*tasks)
        for (ids_b, texts_b, labels_b), emb_b in zip(pending_meta, results):
            for item_id, text, label, emb in zip(ids_b, texts_b, labels_b, emb_b):
                if emb is None:
                    continue  # skip failed embeddings, keeps arrays shape-consistent
                all_ids.append(item_id)
                all_embeddings.append(emb)
                all_labels.append(label)

    tasks, pending_meta = [], []
    for batch_id, rec in enumerate(dataset_records.collect_batches(chunk_size=chunk_size)):
        texts = rec["text"].to_list()
        ids = rec["id"].to_list()
        labels = rec["label"].to_list()

        tasks.append(bounded_embed(texts, batch_id))
        pending_meta.append((ids, texts, labels))

        if len(tasks) >= batch_size:
            await flush(tasks, pending_meta)
            tasks, pending_meta = [], []

    if tasks:
        await flush(tasks, pending_meta)

    np.savez(
        out_path,
        ids=np.array(all_ids),
        embeddings=np.stack(all_embeddings),
        labels=np.array(all_labels),
    )
    print(f"Saved {len(all_ids)} items to {out_path}")


class PromptFeatureDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.ids = [str(i) for i in data["ids"]]
        self.features = torch.from_numpy(data["embeddings"]).float()
        self.labels = torch.from_numpy(data["labels"]).long()
        self.id_to_pos = {item_id: pos for pos, item_id in enumerate(self.ids)}
    @property
    def item_ids(self):
        return self.ids

    def ids_to_positions(self, id_list):
        return [self.id_to_pos[str(i)] for i in id_list]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
