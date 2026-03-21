from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_DENSE_EMBEDDING_MODEL = "BAAI/bge-m3"


class BGEEmbeddingEncoder:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_DENSE_EMBEDDING_MODEL,
        modelscope_model_id: str | None = None,
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 512,
        local_model_dir: str | Path | None = None,
        use_modelscope_download: bool = False,
    ) -> None:
        self.model_name = model_name
        self.modelscope_model_id = modelscope_model_id
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.local_model_dir = (
            Path(local_model_dir) if local_model_dir is not None else None
        )
        self.use_modelscope_download = use_modelscope_download
        self._tokenizer = None
        self._model = None
        self._torch = None

    def encode(self, texts: list[str]):
        tokenizer, model, torch = self._load_backend()
        import numpy as np

        all_vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded)
            hidden = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            masked_hidden = hidden * attention_mask
            summed = masked_hidden.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)
            embeddings = summed / counts
            embeddings = embeddings.detach().cpu().float().numpy()
            all_vectors.append(embeddings.astype("float32"))
        if not all_vectors:
            return np.zeros((0, 1), dtype="float32")
        matrix = np.vstack(all_vectors).astype("float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _load_backend(self) -> tuple[Any, Any, Any]:
        if (
            self._tokenizer is not None
            and self._model is not None
            and self._torch is not None
        ):
            return self._tokenizer, self._model, self._torch

        import torch
        from transformers import AutoModel, AutoTokenizer

        model_path = self._resolve_model_path()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        resolved_device = _resolve_device(self.device, torch)
        model.to(resolved_device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        return tokenizer, model, torch

    def _resolve_model_path(self) -> str:
        if self.local_model_dir is not None:
            return str(self.local_model_dir)
        if self.use_modelscope_download:
            from modelscope import snapshot_download

            return snapshot_download(self.modelscope_model_id or self.model_name)
        return self.model_name


def _resolve_device(device: str, torch: Any) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device
