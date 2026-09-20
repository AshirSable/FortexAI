from pathlib import Path

import torch
import torch.nn as nn
from torch.func import functional_call, stack_module_state
from transformers import AutoConfig, AutoModelForSequenceClassification


class EnsembleBERT(nn.Module):
    def __init__(self, models_num: int):
        super().__init__()
        self.models_num = models_num
        self.models: nn.ModuleDict = nn.ModuleDict()

        self.models_name: list[str] = []

        # populated by _build_vmap_ensemble, after _load_models
        self._stacked_params: dict[str, torch.Tensor] | None = None
        self._stacked_buffers: dict[str, torch.Tensor] | None = None
        self._meta_model: nn.Module | None = None
        self.device: str | torch.device | None = None

    def _load_models(
        self,
        model_folder: Path,
        _fallback_model: str,
        device: str | torch.device = "cuda",
    ):
        self.device = device
        config_file_exists = (model_folder / "config.json").exists()
        base_config = AutoConfig.from_pretrained(
            model_folder if config_file_exists else _fallback_model
        )
        for idx in range(1, self.models_num + 1):
            file_name = model_folder / f"bert_{idx}.pt"
            name = file_name.stem
            model = AutoModelForSequenceClassification.from_config(base_config)
            state_dict = torch.load(file_name, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            self.models[name] = model
            self.models_name.append(name)

        self._build_vmap_ensemble()

    def _build_vmap_ensemble(self):
        ordered_models = [self.models[name] for name in self.models_name]

        params, buffers = stack_module_state(ordered_models)
        self._stacked_params = params
        self._stacked_buffers = buffers

        # meta device: holds shape/structure only, no real weights/memory
        self._meta_model = ordered_models[0].to("meta")

    def _functional_forward(self, params, buffers, input_ids, attention_mask):
        out = functional_call(
            self._meta_model,
            (params, buffers),
            args=(input_ids,),
            kwargs={"attention_mask": attention_mask},
        )
        return out.logits

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self._stacked_params is None:
            raise RuntimeError("Call _load_models() before running inference.")

        logits = torch.vmap(self._functional_forward, in_dims=(0, 0, None, None))(
            self._stacked_params, self._stacked_buffers, input_ids, attention_mask
        )

        return logits

    def predict_proba(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Softmax over class dim, averaged across ensemble members."""
        logits = self.forward(input_ids, attention_mask)  # [n_models, batch, n_classes]
        probs = torch.softmax(logits, dim=-1)
        return probs.mean(dim=0)  # [batch, n_classes]

    @classmethod
    def _determine_num_models(cls, path: Path):
        return len(list(path.glob("bert_*.pt")))

    @classmethod
    def _load_from_file(
        cls,
        model_folder: Path,
        fallback_model: str,
        device: str | torch.device = "cuda",
    ) -> "EnsembleBERT":
        num_models = cls._determine_num_models(model_folder)
        instance = cls(num_models)
        instance._load_models(
            model_folder, _fallback_model=fallback_model, device=device
        )
        return instance
