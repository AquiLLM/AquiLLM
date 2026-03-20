"""
Embedding providers and utilities.

This module provides:
- Local OpenAI-compatible embedding (via vLLM or other endpoints)
- Cohere embedding (requires client injection)
- Multimodal embedding support
- Embedding dimension fitting

For the Django-integrated interface, use aquillm.utils.get_embedding().
"""

from .config import (
    get_local_embed_config,
    get_target_dims,
    max_embed_input_chars,
    is_context_limit_error,
    extract_context_limit_tokens,
)
from .local import (
    get_embedding_via_local_openai,
    get_embeddings_via_local_openai,
)
from .cohere import (
    get_embedding_via_cohere,
    get_embeddings_via_cohere,
)
from .multimodal import (
    get_multimodal_embedding_via_vllm_pooling,
)
from .utils import (
    fit_embedding_dims,
)

__all__ = [
    # Config
    'get_local_embed_config',
    'get_target_dims',
    'max_embed_input_chars',
    'is_context_limit_error',
    'extract_context_limit_tokens',
    # Local
    'get_embedding_via_local_openai',
    'get_embeddings_via_local_openai',
    # Cohere
    'get_embedding_via_cohere',
    'get_embeddings_via_cohere',
    # Multimodal
    'get_multimodal_embedding_via_vllm_pooling',
    # Utils
    'fit_embedding_dims',
]
