from __future__ import annotations

import pytest

from legal_rag.generation.llm_client import _resolve_device


def test_resolve_device_cpu() -> None:
    assert _resolve_device("cpu") == "cpu"


def test_resolve_device_raises_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        _resolve_device("cuda")
