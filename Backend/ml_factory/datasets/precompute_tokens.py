from pathlib import Path
from typing import Literal, Optional

import numpy as np
from transformers import PreTrainedTokenizerBase

from ml_factory.datasets.errors import FileFormatNotValid
from ml_factory.datasets.tokenizer import TokenizerConfig
from ml_factory.datasets.utils import (
    CHUNK_ID_NAME,
    ID_NAME,
    LABELS_NAME,
    NUM_CHUNK_NAME,
    TEXT_NAME,
    TOKENIZED_NAME,
    read_file_to_lazy,
)
from ml_factory.utils import give_id_to_data

ACCEPTABLE_TYPES_READ = {".csv", ".parquet"}

ACCEPTABLE_TYPES_WRITE = {".npz"}


def precompute_tokens(
    dataset_path: Path | str,
    out_path: Path | str,
    tokenizer: type[PreTrainedTokenizerBase],
    pretrained_tokenizer: Optional[str] = None,
    chunk_size: int = 256,
    write_id_to_dataset: bool = True,
    save_bytes: Literal["all"] | int = "all",
    token_chunk: bool = False,
    stride: int = 0,
    max_length: Optional[int] = None,
    _text_col_name: Optional[str] = None,
    _id_col_name: Optional[str] = None,
    _label_col_name: Optional[str] = None,
):
    """
    IMPORTANT: allows tokenizer chunking
    needs token_chunk = True

    this would add 2 more columns in npz file
    - chunk_idx: 0 indexed positions of this chunk within its original text's chunks
    - num_chunks: total chunk count for that original text

    """
    dataset_path = Path(dataset_path)
    out_path = Path(out_path)
    _text_col_name = _text_col_name or "text"
    _id_col_name = _id_col_name or "id"
    _label_col_name = _label_col_name or "label"

    if dataset_path.suffix not in ACCEPTABLE_TYPES_READ:
        raise FileFormatNotValid(
            dataset_path.suffix, formats_available=ACCEPTABLE_TYPES_READ
        )

    if out_path.suffix != "" and out_path.suffix not in ACCEPTABLE_TYPES_WRITE:
        raise FileFormatNotValid(
            out_path.suffix, formats_available=ACCEPTABLE_TYPES_WRITE
        )

    lazy_frame = read_file_to_lazy(dataset_path)

    if lazy_frame is None:
        raise FileFormatNotValid(
            dataset_path.suffix, formats_available=ACCEPTABLE_TYPES_READ
        )

    if write_id_to_dataset:
        lazy_frame = give_id_to_data(
            lazy_frame, text_col=_text_col_name, _id_col_name=_id_col_name
        )

    if pretrained_tokenizer is not None:
        tokenized = tokenizer.from_pretrained(pretrained_tokenizer)
    else:
        tokenized = tokenizer()

    #########################################

    all_ids, all_texts, all_labels, all_tokenized = [], [], [], []
    all_chunks_idx, all_num_chunks = [], []

    part_paths = []
    part_idx = 0

    def write_part():
        nonlocal all_ids, all_texts, all_labels, all_tokenized, part_paths, part_idx, all_chunks_idx, all_num_chunks

        if not all_ids:
            return

        if save_bytes == "all":
            part_path = out_path

        else:
            part_path = out_path.with_suffix(f".part{part_idx}.npz")

        save_dict = {
            ID_NAME: np.array(all_ids, dtype=np.str_),
            TOKENIZED_NAME: np.stack(all_tokenized, dtype=np.int64),
            TEXT_NAME: np.array(all_texts, np.str_),
            LABELS_NAME: np.array(all_labels, np.int64),
        }

        if token_chunk:
            save_dict[CHUNK_ID_NAME] = np.array(all_chunks_idx, dtype=np.int64)
            save_dict[NUM_CHUNK_NAME] = np.array(all_num_chunks, dtype=np.int64)

        np.savez(part_path, **save_dict)

        part_paths.append(part_path)

        print(f"Wrote {len(all_ids)} items to {part_path}")

        all_ids, all_tokenized, all_labels, all_texts = [], [], [], []
        all_chunks_idx, all_num_chunks = [], []

        if save_bytes != "all":
            part_idx += 1

    for record_idx, rec in enumerate(lazy_frame.collect_batches(chunk_size=chunk_size)):
        texts = rec[_text_col_name].to_list()
        ids = rec[_id_col_name].to_list()
        labels = rec[_label_col_name].to_list()

        enc = tokenized(
            texts,
            padding="max_length",
            truncation=True,
            return_tensors="np",
            return_overflowing_tokens=token_chunk,
            stride=stride,
            max_length=max_length,
            save_bytes=2000,
        )
        input_ids = enc["input_ids"]
        if token_chunk:
            sample_map = enc["overflow_to_sample_mapping"]
            total_per_sample = {}

            for s in sample_map:
                total_per_sample[s] = total_per_sample.get(s, 0) + 1

            running_chunk_idx = {}

            for row_idx, sample_idx in enumerate(sample_map):
                running_chunk_idx[sample_idx] = (
                    running_chunk_idx.get(sample_idx, -1) + 1
                )

                all_ids.append(ids[sample_idx])
                all_texts.append(texts[sample_idx])
                all_labels.append(labels[sample_idx])
                all_tokenized.append(input_ids[row_idx])
                all_chunks_idx.append(running_chunk_idx[sample_idx])
                all_num_chunks.append(total_per_sample[sample_idx])

        else:
            all_ids.extend(ids)
            all_texts.extend(texts)
            all_labels.extend(labels)
            all_tokenized.extend(input_ids)

        print(f"Written Record Chunk[={chunk_size}]: {record_idx}", end="\r")

        if save_bytes != "all" and sum(a.nbytes for a in all_tokenized) >= save_bytes:
            write_part()

    write_part()

    config = TokenizerConfig.from_tokenizer(tokenized)
    config.max_length = max_length
    json_path = config.save(out_path)

    if save_bytes == "all":
        print(f"Wrote the entire file, in path {out_path}")
    else:
        print(
            f"Wrote {len(part_paths)} part files. Concatenate with merge_parts() before use."
        )

    print(f"Saved tokenizer config path, in path {json_path}")

    return part_paths
