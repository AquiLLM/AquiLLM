"""LLM provider interfaces."""
from .base import LLMInterface
from .claude import ClaudeInterface
from .openai import OpenAIInterface, gpt_enc
from .gemini import GeminiInterface


def get_provider(provider_name: str, **kwargs) -> LLMInterface:
    """
    Factory function to get an LLM provider by name.
    
    Args:
        provider_name: One of 'claude', 'openai', 'gemini'
        **kwargs: Provider-specific arguments (client, model, etc.)
    
    Returns:
        An LLMInterface instance
    """
    providers = {
        'claude': ClaudeInterface,
        'openai': OpenAIInterface,
        'gemini': GeminiInterface,
    }
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(providers.keys())}")
    return providers[provider_name](**kwargs)


__all__ = [
    'LLMInterface', 
    'ClaudeInterface', 
    'OpenAIInterface', 
    'GeminiInterface', 
    'get_provider',
    'gpt_enc',
]
