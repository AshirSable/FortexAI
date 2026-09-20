import hashlib
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import ollama
import polars as pl
import torch

EMBEDDING_CLIENT = ollama.AsyncClient(host="http://localhost:11434")


async def embedding_text(
    model: str, text: list[str], item_id: int, prefix_text: str = "search_document: "
) -> list:
    try:
        response = await EMBEDDING_CLIENT.embed(model, [prefix_text + t for t in text])
        print(f"Success: Processed Item: {item_id:<100}", end="\r")

        return response["embeddings"]
    except Exception as e:
        print(f"Error on Item id: {item_id}, error: {e}")
        return []


def give_id_to_data[T: (pl.LazyFrame, pl.DataFrame)](
    df: T, text_col: str = "text", _id_col_name: str = "id"
) -> T:
    return df.with_columns(
        pl.col(text_col)
        .map_elements(
            lambda t: hashlib.sha256(t.encode()).hexdigest(), return_dtype=pl.Utf8
        )
        .alias(_id_col_name)
    )


def merge_parts_to_dir(part_paths: list[Path], out_dir: str | Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    total_rows = 0
    keys = None
    shapes = {}
    dtypes = {}

    for path in part_paths:
        with np.load(path, allow_pickle=True) as data:
            if keys is None:
                keys = data.files
                for k in keys:
                    arr = data[k]
                    shapes[k] = arr.shape[1:]
                    dtypes[k] = arr.dtype
            total_rows += data[keys[0]].shape[0]

    memmaps = {}
    for k in keys:
        full_shape = (total_rows, *shapes[k])
        memmaps[k] = np.lib.format.open_memmap(
            out_dir / f"{k}.npy", mode="w+", dtype=dtypes[k], shape=full_shape
        )

    write_pos = 0
    for path in part_paths:
        with np.load(path, allow_pickle=True) as data:
            n = data[keys[0]].shape[0]
            for k in keys:
                memmaps[k][write_pos : write_pos + n] = data[k]
            write_pos += n

    for m in memmaps.values():
        m.flush()

    for path in part_paths:
        path.unlink()

    print(f"Merged {len(part_paths)} parts into directory {out_dir}")
    return out_dir


def merge_parts(part_paths: list[Path], out_path: str | Path):
    out_path = Path(out_path)

    total_rows = 0
    keys = None
    shapes = {}  # key -> shape of one row (excluding the leading N dimension)
    dtypes = {}

    for path in part_paths:
        with np.load(path, allow_pickle=True) as data:
            if keys is None:
                keys = data.files
                for k in keys:
                    arr = data[k]
                    shapes[k] = arr.shape[1:]  # per-row shape
                    dtypes[k] = arr.dtype
            total_rows += data[keys[0]].shape[0]

    tmp_dir = out_path.parent / f".{out_path.stem}_merge_tmp"
    tmp_dir.mkdir(exist_ok=True)

    memmaps = {}
    for k in keys:
        full_shape = (total_rows, *shapes[k])
        memmaps[k] = np.lib.format.open_memmap(
            tmp_dir / f"{k}.npy", mode="w+", dtype=dtypes[k], shape=full_shape
        )

    write_pos = 0
    for path in part_paths:
        with np.load(path, allow_pickle=True) as data:
            n = data[keys[0]].shape[0]
            for k in keys:
                memmaps[k][write_pos : write_pos + n] = data[k]
            write_pos += n

    for m in memmaps.values():
        m.flush()

    save_dict = {k: memmaps[k] for k in keys}
    np.savez(out_path, **save_dict)

    for k in keys:
        (tmp_dir / f"{k}.npy").unlink()
    tmp_dir.rmdir()

    for path in part_paths:
        path.unlink()

    print(f"Merged {len(part_paths)} parts into {out_path}")


def merge_parts_simple(part_paths: list[Path], out_path: str | Path):
    part_paths = [Path(p) for p in part_paths]

    merged: dict[str, list[np.ndarray]] = {}

    for path in part_paths:
        with np.load(path, allow_pickle=True) as data:
            for key in data.files:
                merged.setdefault(key, []).append(data[key])
            n_items = len(data[data.files[0]])
        print(f"Loaded {n_items} items from {path}")

    final = {key: np.concatenate(arrays, axis=0) for key, arrays in merged.items()}

    np.savez(out_path, **final)

    total = len(next(iter(final.values())))
    print(f"Wrote {total} merged items to {out_path}")

    return final


def parameters_in_dictionary(*args):
    return {str(a.__name__): a for a in args}


@dataclass
class CONFIG:
    loss_map: dict
    include_loss: dict
    metrics_map: dict
    include_metrics: list
    best_metric: str
    greater_is_better: bool

    batch_size: int = 200
    seed: int = 3123
    train_ratio: float = 0.7
    test_ratio: float = 0.15
    validation_ratio: float = 0.15
    epoch: int = 200
    attack_ratio: float = 0.2
    ratio_sampler: bool = False
    embedding_model: str = "nomic-embed-text"
    ae_embedding_size: int = 64

    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    def _serialize_value(self, val):
        if isinstance(val, partial):
            fn_name = getattr(val.func, "__name__", str(val.func))
            return f"partial({fn_name}, {val.keywords})"
        elif callable(val):
            return getattr(val, "__name__", str(val))
        elif isinstance(val, dict):
            return {k: self._serialize_value(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [self._serialize_value(v) for v in val]
        return val

    def get_dict(self) -> dict:
        raw_dict = asdict(self)
        return self._serialize_value(raw_dict)


from ml_factory.utils.model_saver import ModelSaver  # # pyright: ignore[]
from ml_factory.utils.parameter_tracker import Tracker  # pyright: ignore[]

__all__ = [
    "ModelSaver",
    "Tracker",
]
