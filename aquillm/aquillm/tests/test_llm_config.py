"""
Tests for LLM configuration and initialization.

Validates that different LLM providers (cloud and local via Ollama) are correctly
configured and initialized based on the LLM_CHOICE environment variable.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, Mock
from aquillm.llm import OpenAIInterface, ClaudeInterface


class TestOllamaLLMConfiguration:
    """Test suite for Ollama local LLM model configuration."""

    @patch('aquillm.apps.getenv')
    @patch('aquillm.apps.openai.AsyncOpenAI')
    @patch('aquillm.apps.cohere.Client')
    @patch('aquillm.apps.anthropic.AsyncAnthropic')
    @patch('aquillm.apps.anthropic.Anthropic')
    def test_gemma3_ollama_model_initialization(
        self, mock_anthropic, mock_async_anthropic, mock_cohere, mock_openai, mock_getenv
    ):
        """Test that GEMMA3 model initializes with correct Ollama configuration."""
        # Set up mock environment
        def getenv_side_effect(key, default=None):
            if key == 'LLM_CHOICE':
                return 'GEMMA3'
            return default
        
        mock_getenv.side_effect = getenv_side_effect
        
        # Mock the OpenAI client
        mock_ollama_client = Mock()
        mock_openai.return_value = mock_ollama_client
        
        # Import and initialize the app config
        from aquillm.apps import AquillmConfig
        config = AquillmConfig('aquillm', None)
        config.ready()

        # Verify that the LLM interface is an OpenAIInterface
        assert isinstance(config.llm_interface, OpenAIInterface)
        
        # Verify that the model name is correct
        assert config.llm_interface.base_args['model'] == 'ebdm/gemma3-enhanced:12b'
        
        # Verify OpenAI client was called with Ollama base URL
        mock_openai.assert_called_with(base_url='http://ollama:11434/v1/')

    @patch('aquillm.apps.getenv')
    @patch('aquillm.apps.openai.AsyncOpenAI')
    @patch('aquillm.apps.cohere.Client')
    @patch('aquillm.apps.anthropic.AsyncAnthropic')
    @patch('aquillm.apps.anthropic.Anthropic')
    def test_llama32_ollama_model_initialization(
        self, mock_anthropic, mock_async_anthropic, mock_cohere, mock_openai, mock_getenv
    ):
        """Test that LLAMA3.2 model initializes with correct Ollama configuration."""
        def getenv_side_effect(key, default=None):
            if key == 'LLM_CHOICE':
                return 'LLAMA3.2'
            return default
        
        mock_getenv.side_effect = getenv_side_effect
        mock_openai.return_value = Mock()
        
        from aquillm.apps import AquillmConfig
        config = AquillmConfig('aquillm', None)
        config.ready()

        assert isinstance(config.llm_interface, OpenAIInterface)
        assert config.llm_interface.base_args['model'] == 'llama3.2'
        mock_openai.assert_called_with(base_url='http://ollama:11434/v1/')

    @patch('aquillm.apps.getenv')
    @patch('aquillm.apps.openai.AsyncOpenAI')
    @patch('aquillm.apps.cohere.Client')
    @patch('aquillm.apps.anthropic.AsyncAnthropic')
    @patch('aquillm.apps.anthropic.Anthropic')
    def test_gpt_oss_ollama_model_initialization(
        self, mock_anthropic, mock_async_anthropic, mock_cohere, mock_openai, mock_getenv
    ):
        """Test that GPT-OSS model initializes with correct Ollama configuration."""
        def getenv_side_effect(key, default=None):
            if key == 'LLM_CHOICE':
                return 'GPT-OSS'
            return default
        
        mock_getenv.side_effect = getenv_side_effect
        mock_openai.return_value = Mock()
        
        from aquillm.apps import AquillmConfig
        config = AquillmConfig('aquillm', None)
        config.ready()

        assert isinstance(config.llm_interface, OpenAIInterface)
        assert config.llm_interface.base_args['model'] == 'gpt-oss:120b'
        mock_openai.assert_called_with(base_url='http://ollama:11434/v1/')

    @patch('aquillm.apps.getenv')
    @patch('aquillm.apps.openai.AsyncOpenAI')
    @patch('aquillm.apps.cohere.Client')
    @patch('aquillm.apps.anthropic.AsyncAnthropic')
    @patch('aquillm.apps.anthropic.Anthropic')
    def test_all_ollama_models_use_openai_interface(
        self, mock_anthropic, mock_async_anthropic, mock_cohere, mock_openai, mock_getenv
    ):
        """Test that all Ollama models use OpenAIInterface."""
        ollama_models = ['GEMMA3', 'LLAMA3.2', 'GPT-OSS']
        
        for model_choice in ollama_models:
            def getenv_side_effect(key, default=None):
                if key == 'LLM_CHOICE':
                    return model_choice
                return default
            
            mock_getenv.side_effect = getenv_side_effect
            mock_openai.return_value = Mock()
            
            from aquillm.apps import AquillmConfig
            config = AquillmConfig('aquillm', None)
            config.ready()
            
            assert isinstance(config.llm_interface, OpenAIInterface), \
                f"{model_choice} should use OpenAIInterface"

    @patch('aquillm.apps.getenv')
    @patch('aquillm.apps.openai.AsyncOpenAI')
    @patch('aquillm.apps.cohere.Client')
    @patch('aquillm.apps.anthropic.AsyncAnthropic')
    @patch('aquillm.apps.anthropic.Anthropic')
    def test_claude_uses_claude_interface(
        self, mock_anthropic, mock_async_anthropic, mock_cohere, mock_openai, mock_getenv
    ):
        """Test that CLAUDE choice uses ClaudeInterface, not OpenAIInterface."""
        def getenv_side_effect(key, default=None):
            if key == 'LLM_CHOICE':
                return 'CLAUDE'
            return default
        
        mock_getenv.side_effect = getenv_side_effect
        mock_async_anthropic.return_value = Mock()
        
        from aquillm.apps import AquillmConfig
        config = AquillmConfig('aquillm', None)
        config.ready()

        assert isinstance(config.llm_interface, ClaudeInterface)
        assert not isinstance(config.llm_interface, OpenAIInterface)

    @patch('aquillm.apps.getenv')
    @patch('aquillm.apps.openai.AsyncOpenAI')
    @patch('aquillm.apps.cohere.Client')
    @patch('aquillm.apps.anthropic.AsyncAnthropic')
    @patch('aquillm.apps.anthropic.Anthropic')
    def test_openai_uses_openai_interface(
        self, mock_anthropic, mock_async_anthropic, mock_cohere, mock_openai, mock_getenv
    ):
        """Test that OPENAI choice uses OpenAIInterface with GPT-4 model."""
        def getenv_side_effect(key, default=None):
            if key == 'LLM_CHOICE':
                return 'OPENAI'
            return default
        
        mock_getenv.side_effect = getenv_side_effect
        mock_openai.return_value = Mock()
        
        from aquillm.apps import AquillmConfig
        config = AquillmConfig('aquillm', None)
        config.ready()

        assert isinstance(config.llm_interface, OpenAIInterface)
        assert config.llm_interface.base_args['model'] == 'gpt-4o'

    @patch('aquillm.apps.getenv')
    @patch('aquillm.apps.openai.AsyncOpenAI')
    @patch('aquillm.apps.cohere.Client')
    @patch('aquillm.apps.anthropic.AsyncAnthropic')
    @patch('aquillm.apps.anthropic.Anthropic')
    def test_invalid_llm_choice_raises_error(
        self, mock_anthropic, mock_async_anthropic, mock_cohere, mock_openai, mock_getenv
    ):
        """Test that an invalid LLM_CHOICE raises a ValueError."""
        def getenv_side_effect(key, default=None):
            if key == 'LLM_CHOICE':
                return 'INVALID_MODEL'
            return default
        
        mock_getenv.side_effect = getenv_side_effect
        
        from aquillm.apps import AquillmConfig
        config = AquillmConfig('aquillm', None)
        
        with pytest.raises(ValueError, match="Invalid LLM choice"):
            config.ready()

    @patch('aquillm.apps.getenv')
    @patch('aquillm.apps.openai.AsyncOpenAI')
    @patch('aquillm.apps.cohere.Client')
    @patch('aquillm.apps.anthropic.AsyncAnthropic')
    @patch('aquillm.apps.anthropic.Anthropic')
    def test_default_llm_is_claude(
        self, mock_anthropic, mock_async_anthropic, mock_cohere, mock_openai, mock_getenv
    ):
        """Test that when LLM_CHOICE is not set, it defaults to CLAUDE."""
        # Return None for LLM_CHOICE to trigger default
        def getenv_side_effect(key, default=None):
            if key == 'LLM_CHOICE':
                return default  # Use the default value
            return default
        
        mock_getenv.side_effect = getenv_side_effect
        mock_async_anthropic.return_value = Mock()
        
        from aquillm.apps import AquillmConfig
        config = AquillmConfig('aquillm', None)
        config.ready()

        assert isinstance(config.llm_interface, ClaudeInterface)


class TestLLMInterfaceAttributes:
    """Test that LLM interfaces have the expected attributes."""

    @patch('aquillm.apps.getenv')
    @patch('aquillm.apps.openai.AsyncOpenAI')
    @patch('aquillm.apps.cohere.Client')
    @patch('aquillm.apps.anthropic.AsyncAnthropic')
    @patch('aquillm.apps.anthropic.Anthropic')
    def test_ollama_interface_has_client(
        self, mock_anthropic, mock_async_anthropic, mock_cohere, mock_openai, mock_getenv
    ):
        """Test that Ollama-based interfaces have a client attribute."""
        def getenv_side_effect(key, default=None):
            if key == 'LLM_CHOICE':
                return 'GEMMA3'
            return default
        
        mock_getenv.side_effect = getenv_side_effect
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        from aquillm.apps import AquillmConfig
        config = AquillmConfig('aquillm', None)
        config.ready()

        assert hasattr(config.llm_interface, 'client')
        assert config.llm_interface.client is not None

    @patch('aquillm.apps.getenv')
    @patch('aquillm.apps.openai.AsyncOpenAI')
    @patch('aquillm.apps.cohere.Client')
    @patch('aquillm.apps.anthropic.AsyncAnthropic')
    @patch('aquillm.apps.anthropic.Anthropic')
    def test_ollama_interface_has_base_args(
        self, mock_anthropic, mock_async_anthropic, mock_cohere, mock_openai, mock_getenv
    ):
        """Test that Ollama-based interfaces have base_args with model."""
        def getenv_side_effect(key, default=None):
            if key == 'LLM_CHOICE':
                return 'LLAMA3.2'
            return default
        
        mock_getenv.side_effect = getenv_side_effect
        mock_openai.return_value = Mock()
        
        from aquillm.apps import AquillmConfig
        config = AquillmConfig('aquillm', None)
        config.ready()

        assert hasattr(config.llm_interface, 'base_args')
        assert 'model' in config.llm_interface.base_args
        assert config.llm_interface.base_args['model'] == 'llama3.2'

    @patch('aquillm.apps.getenv')
    @patch('aquillm.apps.openai.AsyncOpenAI')
    @patch('aquillm.apps.cohere.Client')
    @patch('aquillm.apps.anthropic.AsyncAnthropic')
    @patch('aquillm.apps.anthropic.Anthropic')
    def test_ollama_models_use_correct_endpoint(
        self, mock_anthropic, mock_async_anthropic, mock_cohere, mock_openai, mock_getenv
    ):
        """Test that all Ollama models use http://ollama:11434/v1/ as base_url."""
        ollama_models = ['GEMMA3', 'LLAMA3.2', 'GPT-OSS']
        expected_base_url = 'http://ollama:11434/v1/'
        
        for model_choice in ollama_models:
            # Reset mock
            mock_openai.reset_mock()
            
            def getenv_side_effect(key, default=None):
                if key == 'LLM_CHOICE':
                    return model_choice
                return default
            
            mock_getenv.side_effect = getenv_side_effect
            mock_openai.return_value = Mock()
            
            from aquillm.apps import AquillmConfig
            config = AquillmConfig('aquillm', None)
            config.ready()
            
            # Verify OpenAI client was called with correct base_url
            mock_openai.assert_called_with(base_url=expected_base_url)
