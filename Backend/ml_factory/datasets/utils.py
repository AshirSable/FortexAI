from pathlib import Path
from typing import Optional

import polars as pl

TOKENIZED_NAME = "tokenized"
LABELS_NAME = "labels"
ID_NAME = "ids"
CHUNK_ID_NAME = "chunk_idx"
NUM_CHUNK_NAME = "num_chunks"
TEXT_NAME = "text"
EMBEDDING_NAME = "embeddings"
STRUCTURAL_NAME = "structural"


def read_file_to_lazy(file: Path | str) -> Optional[pl.LazyFrame]:
    file = Path(file)

    if file.suffix == ".csv":
        return pl.scan_csv(file)

    elif file.suffix == ".parquet":
        return pl.scan_parquet(file)

    else:
        return None
