from pathlib import Path
from typing import Optional
from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Optional, Self
import json


@dataclass()
class TokenizerConfig:
    name_or_path: Optional[str] = None
    vocab_size: Optional[int] = None
    model_max_length: Optional[int] = None

    pad_token: Optional[str] = None
    pad_token_id: Optional[int] = None
    eos_token: Optional[str] = None
    eos_token_id: Optional[int] = None
    bos_token: Optional[str] = None
    bos_token_id: Optional[int] = None
    unk_token: Optional[str] = None
    unk_token_id: Optional[int] = None
    cls_token: Optional[str] = None
    cls_token_id: Optional[int] = None
    sep_token: Optional[str] = None
    sep_token_id: Optional[int] = None
    max_length: Optional[int] = None
    is_fast: Optional[bool] = None
    padding_side: Optional[str] = None

    @classmethod
    def from_tokenizer(cls, tokenized: PreTrainedTokenizerBase) -> Self:
        field_names = {f.name for f in fields(cls)}
        kwargs = {}
        for name in field_names:
            kwargs[name] = getattr(tokenized, name, None)
        return cls(**kwargs)

    def save(self, out_path: Path | str) -> Path:
        out_path = Path(out_path)

        if out_path.is_file() and out_path.suffix != ".json":
            meta_path = out_path.with_suffix(".tokenizer_meta.json")
        elif out_path.suffix == "":
            meta_path = out_path / "tokenizer_meta.json"
        else:
            meta_path = out_path

        out_path.mkdir(exist_ok=True, parents=True)

        with open(meta_path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return meta_path

    @classmethod
    def load(cls, out_path: Path | str) -> Self:
        out_path = Path(out_path)

        if out_path.is_file():
            meta_path = (
                out_path
                if out_path.suffix == ".json"
                else out_path.with_suffix(".tokenizer_meta.json")
            )
        else:
            meta_path = out_path / "tokenizer_meta.json"
        with open(meta_path) as f:
            data = json.load(f)
        field_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)
