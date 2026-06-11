"""Direct RAG orchestration (implemented in Task 4)."""
from __future__ import annotations

from typing import Literal


async def run_direct_rag_turn(*_args, **_kwargs) -> Literal["handled", "skipped"]:
    raise NotImplementedError("rag_pipeline.run_direct_rag_turn is not implemented yet")


__all__ = ["run_direct_rag_turn"]
