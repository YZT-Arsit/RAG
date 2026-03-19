from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol
from urllib import request


@dataclass(slots=True)
class LLMResponse:
    content: str


class LLMClient(Protocol):
    def complete(self, prompt: str) -> LLMResponse: ...


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key_env: str,
        model_name: str,
        temperature: float,
        timeout_seconds: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.model_name = model_name
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def complete(self, prompt: str) -> LLMResponse:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            msg = f"Missing API key in environment variable: {self.api_key_env}"
            raise RuntimeError(msg)

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        return LLMResponse(content=content)


class LocalTransformersClient:
    def __init__(
        self,
        *,
        model_name: str | None,
        modelscope_model_id: str | None,
        local_model_dir: Path | None,
        use_modelscope_download: bool,
        device: str,
        temperature: float,
        max_new_tokens: int,
        timeout_seconds: int,
    ) -> None:
        self.model_name = model_name
        self.modelscope_model_id = modelscope_model_id
        self.local_model_dir = local_model_dir
        self.use_modelscope_download = use_modelscope_download
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.timeout_seconds = timeout_seconds

    def complete(self, prompt: str) -> LLMResponse:
        model_dir = self._resolve_model_dir()
        tokenizer, model, resolved_device = _load_local_model(
            str(model_dir), self.device
        )
        prompt_text = _build_model_input(tokenizer, prompt)
        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = {name: value.to(resolved_device) for name, value in inputs.items()}

        import torch

        generation_kwargs: dict[str, object] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)
        prompt_tokens = inputs["input_ids"].shape[-1]
        response_ids = output_ids[0][prompt_tokens:]
        content = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        return LLMResponse(content=content)

    def _resolve_model_dir(self) -> Path:
        if self.local_model_dir is not None:
            return self.local_model_dir
        if self.use_modelscope_download and self.modelscope_model_id:
            return _download_modelscope_model(self.modelscope_model_id)
        msg = "Local transformers backend requires llm_local_model_dir or llm_modelscope_model_id with llm_use_modelscope_download=true."
        raise RuntimeError(msg)


def _build_model_input(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(rendered, str):
                return rendered
        except Exception:
            pass
    return prompt


@lru_cache(maxsize=2)
def _download_modelscope_model(model_id: str) -> Path:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ModuleNotFoundError as exc:
        msg = "ModelScope is required for automatic local model download."
        raise RuntimeError(msg) from exc
    model_dir = snapshot_download(model_id)
    return Path(model_dir)


@lru_cache(maxsize=2)
def _load_local_model(model_dir: str, requested_device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = _resolve_device(requested_device)
    torch_dtype = torch.float16 if resolved_device in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.to(resolved_device)
    model.eval()
    return tokenizer, model, resolved_device


def _resolve_device(requested_device: str) -> str:
    import torch

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            msg = "Requested llm_device=cuda but CUDA is not available."
            raise RuntimeError(msg)
        return "cuda"
    if requested_device == "mps":
        if not torch.backends.mps.is_available():
            msg = "Requested llm_device=mps but MPS is not available."
            raise RuntimeError(msg)
        return "mps"
    if requested_device == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
