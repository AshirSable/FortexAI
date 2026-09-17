import asyncio
from pathlib import Path

import numpy as np
import polars as pl
from datasets.utils.py_utils import Literal

from ml_factory.utils.structural_extractor import (
    StructuralExtractor,
)


async def precompute_features_ae(
    dataset_records: pl.LazyFrame,
    embed_fn,
    embedding_model_name: str,
    out_path: str | Path,
    batch_size: int = 32,
    chunk_size: int = 2,
    max_concurrent: int = 5,
    extra_columns: list[str] | None = None,
    structural_extractor: StructuralExtractor | None = None,
    save_n: Literal["all"] | int = "all",
):
    extra_columns = extra_columns or []
    all_ids, all_embeddings, all_labels, all_structural = [], [], [], []
    all_extra = {col: [] for col in extra_columns}
    part_paths = []
    part_idx = 0

    semaphore = asyncio.Semaphore(max_concurrent)

    def write_part():
        nonlocal all_ids, all_embeddings, all_labels, all_structural, part_paths, part_idx, all_extra

        if not all_ids:
            return

        if save_n == "all":
            part_path = out_path
        else:
            part_path = Path(out_path).with_suffix(f".part{part_idx}.npz")

        save_dict = {
            "ids": np.array(all_ids),
            "embeddings": np.stack(all_embeddings),
            "labels": np.array(all_labels),
        }

        if structural_extractor is not None:
            save_dict["structural"] = np.array(all_structural, dtype=np.float32)

        for col in extra_columns:
            save_dict[col] = np.array(all_extra[col], dtype=object)

        np.savez(part_path, **save_dict)

        part_paths.append(part_path)

        print(f"Wrote {len(all_ids)} items to {part_path}")
        all_ids, all_embeddings, all_labels, all_structural = [], [], [], []
        all_extra = {col: [] for col in extra_columns}
        if save_n != "all":
            part_idx += 1

    async def bounded_embed(texts, batch_id):
        async with semaphore:
            return await embed_fn(embedding_model_name, texts, batch_id)

    async def flush(tasks, pending_meta):
        results = await asyncio.gather(*tasks)
        for idx_g, ((ids_b, texts_b, labels_b, extra_b), emb_b) in enumerate(
            zip(pending_meta, results)
        ):
            successful_texts = []
            for idx, (item_id, text, label, emb) in enumerate(
                zip(ids_b, texts_b, labels_b, emb_b)
            ):
                if emb is None:
                    continue
                all_ids.append(item_id)
                all_embeddings.append(emb)
                all_labels.append(label)
                if structural_extractor is not None:
                    successful_texts.append(text)
                for col in extra_columns:
                    all_extra[col].append(extra_b[col][idx])

            if structural_extractor is not None and successful_texts:
                print(
                    f"Processed {len(successful_texts)} batch Item Id: {idx_g:<5}, extracting structural data from texts",
                    end="\r",
                )
                struct_batch = structural_extractor.structural_fn_batch(
                    successful_texts
                )
                all_structural.extend(struct_batch)

        if save_n != "all" and len(all_ids) >= save_n:
            write_part()

    tasks, pending_meta = [], []
    for batch_id, rec in enumerate(
        dataset_records.collect_batches(chunk_size=chunk_size)
    ):
        texts = rec["text"].to_list()
        ids = rec["id"].to_list()
        labels = rec["label"].to_list()
        extra_b = {col: rec[col].to_list() for col in extra_columns}
        tasks.append(bounded_embed(texts, batch_id))
        pending_meta.append((ids, texts, labels, extra_b))

        if len(tasks) >= batch_size:
            await flush(tasks, pending_meta)
            tasks, pending_meta = [], []

    if tasks:
        await flush(tasks, pending_meta)

    write_part()

    if save_n == "all":
        print(f"Wrote the entire file, in path {out_path}")

    else:
        print(
            f"Wrote {len(part_paths)} part files. Concatenate with merge_parts() before use."
        )

    return part_paths
