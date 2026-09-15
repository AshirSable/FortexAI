from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ml_factory.datasets.tokenizer import TokenizerConfig
from ml_factory.datasets.utils import (
    CHUNK_ID_NAME,
    EMBEDDING_NAME,
    ID_NAME,
    LABELS_NAME,
    NUM_CHUNK_NAME,
    STRUCTURAL_NAME,
    TEXT_NAME,
    TOKENIZED_NAME,
)


@dataclass(frozen=True)
class PromptBERTResult:
    tokenized: torch.Tensor
    label: torch.Tensor
    doc_id: str
    text: str
    chunk_idx: int
    num_chunks: int
    attention_masks: torch.Tensor


class PromptBERTDataset(Dataset):
    def __init__(self, path: Path, config_path: Optional[Path] = None):

        if path.is_file() and path.suffix == ".npz":
            self.__make_npz_file(path, config_path)

        elif path.exists() and path.is_dir():
            self.__make_dir_file(path, config_path)

        else:
            raise ValueError(
                f"Could Not import the data, got paths {path=} {config_path=}"
            )

    def __make_dir_file(self, path: Path, config_path: Optional[Path] = None):
        data_dir = Path(path)

        self.tokenized = np.load(data_dir / f"{TOKENIZED_NAME}.npy", mmap_mode="r")
        self.labels = np.load(data_dir / f"{LABELS_NAME}.npy", mmap_mode="r")
        self.ids = np.load(data_dir / f"{ID_NAME}.npy", mmap_mode="r")
        self.text = np.load(data_dir / f"{TEXT_NAME}.npy", mmap_mode="r")
        has_chunks = (data_dir / f"{CHUNK_ID_NAME}.npy").exists() and (
            data_dir / f"{NUM_CHUNK_NAME}.npy"
        ).exists()

        self.has_chunks = has_chunks

        if has_chunks:
            self.chunk_idx = np.load(data_dir / f"{CHUNK_ID_NAME}.npy", mmap_mode="r")
            self.num_chunks = np.load(data_dir / f"{NUM_CHUNK_NAME}.npy", mmap_mode="r")

        self.ids_str = [str(i) for i in self.ids]
        self.id_to_pos = defaultdict(list)

        for pos, item_id in enumerate(self.ids_str):
            self.id_to_pos[item_id].append(pos)

        self.config = TokenizerConfig.load(config_path or path)
        self.attention_mask = self.tokenized != self.config.pad_token_id

    def __make_npz_file(self, npz_path: Path, config_path: Optional[Path] = None):

        data = np.load(npz_path, allow_pickle=True)

        has_chunks = False
        chunk_idx = None
        num_chunks = None
        if CHUNK_ID_NAME in data and NUM_CHUNK_NAME in data:
            has_chunks = True
            num_chunks = data[NUM_CHUNK_NAME]
            chunk_idx = data[CHUNK_ID_NAME]

        self.ids = data[ID_NAME]
        self.ids_str = [str(i) for i in data[ID_NAME]]
        self.tokenized = torch.from_numpy(data[TOKENIZED_NAME]).long()
        self.text = data[TEXT_NAME]
        self.labels = torch.from_numpy(data[LABELS_NAME]).long()
        self.chunk_idx = chunk_idx
        self.num_chunks = num_chunks
        self.has_chunks = has_chunks
        self.id_to_pos = defaultdict(list)

        for pos, item_id in enumerate(self.ids_str):
            self.id_to_pos[item_id].append(pos)

        self.config = TokenizerConfig.load(config_path or npz_path)
        self.attention_mask = (self.tokenized != self.config.pad_token_id).long()

    @property
    def item_ids(self):
        return self.ids_str

    def __len__(self):
        return len(self.ids)

    def ids_to_position(self, id_list):
        positions = []
        for i in id_list:
            positions.extend(self.id_to_pos[str(i)])

        return positions

    def __getitem__(self, index) -> PromptBERTResult:

        return PromptBERTResult(
            tokenized=torch.tensor(self.tokenized[index]).long(),
            label=torch.tensor(self.labels[index]).long(),
            doc_id=self.item_ids[index],
            chunk_idx=int(self.chunk_idx[index]) if self.has_chunks else 0,
            num_chunks=int(self.num_chunks[index]) if self.has_chunks else 0,
            attention_masks=torch.tensor(self.attention_mask[index]).long(),
            text=self.text[index],
        )


class PromptFeatureDataset(Dataset):
    def __init__(self, path: Path | str):
        path = Path(path)
        if path.is_file() and path.suffix == ".npz":
            self.__make_npz_data(path)

        elif path.exists() and path.is_dir():
            self.__make_dir_data(path)

    def __make_npz_data(self, npz_path: Path) -> None:
        data = np.load(npz_path, allow_pickle=True)
        self.ids = data[ID_NAME]
        self.ids_str = [str(i) for i in data[ID_NAME]]
        self.embeddings = torch.from_numpy(data[EMBEDDING_NAME]).float()
        self.labels = torch.from_numpy(data[LABELS_NAME]).long()

        if STRUCTURAL_NAME in data.files:
            self.structural = torch.from_numpy(data[STRUCTURAL_NAME]).float()
            self.features = torch.cat([self.embeddings, self.structural], dim=1)

        else:
            self.structural = None
            self.features = self.embeddings

        self.id_to_pos = {item_id: pos for pos, item_id in enumerate(self.ids)}
        reserved = {ID_NAME, EMBEDDING_NAME, LABELS_NAME, STRUCTURAL_NAME}
        self.extra = {key: data[key] for key in data.files if key not in reserved}

    def __make_dir_data(self, data_dir: Path) -> None:

        self.ids = np.load(data_dir / f"{ID_NAME}.npy", mmap_mode="r")
        self.embeddings = np.load(data_dir / f"{EMBEDDING_NAME}.npy", mmap_mode="r")
        self.labels = np.load(data_dir / f"{LABELS_NAME}.npy", mmap_mode="r")
        has_structural = (data_dir / f"{STRUCTURAL_NAME}.npy").exists()

        self.has_structural = has_structural

        if has_structural:
            self.structural = np.load(
                data_dir / f"{STRUCTURAL_NAME}.npy", mmap_mode="r"
            )
            self.features = np.concatenate([self.embeddings, self.structural], axis=1)

        else:
            self.structural = None
            self.features = self.embeddings

        self.ids_str = [str(i) for i in self.ids]
        self.id_to_pos = {item_id: pos for pos, item_id in enumerate(self.ids)}

    @property
    def item_ids(self):
        return self.ids_str

    def ids_to_positions(self, id_list):
        return [self.id_to_pos[str(i)] for i in id_list]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
