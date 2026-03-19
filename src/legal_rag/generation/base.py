from __future__ import annotations

from typing import Protocol

from legal_rag.schemas.generation import GenerationInput, GroundedAnswer


class BaseGenerator(Protocol):
    def generate(self, generation_input: GenerationInput) -> GroundedAnswer:
        """Generate a grounded answer from processed retrieval results."""
