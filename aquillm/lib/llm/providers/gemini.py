"""Google Gemini LLM interface."""
from typing import Optional, override
import uuid
import logging
import base64

from google.genai import types as genai_types

from ..types.messages import UserMessage, ToolMessage, AssistantMessage
from ..types.conversation import Conversation
from ..types.response import LLMResponse
from .base import LLMInterface


logger = logging.getLogger(__name__)


class GeminiInterface(LLMInterface):
    """
    LLM interface for Google Gemini models.
    Translates between the app's internal message/tool format and the google-genai SDK format.
    """
    base_args: dict = {'model': 'gemini-2.5-flash'}

    @override
    def __init__(self, google_client, model: str = 'gemini-2.5-flash'):
        """Store the Gemini client and which model to use."""
        self.client = google_client
        self.base_args = {'model': model}

    def _transform_tools(self, tools: list[dict]) -> genai_types.Tool:
        """
        Convert tool definitions from the app's internal format (Anthropic-style dicts)
        into a Gemini Tool object containing FunctionDeclarations.
        The input_schema passes through directly since Gemini accepts the same JSON schema format.
        """
        return genai_types.Tool(
            function_declarations=[
                genai_types.FunctionDeclaration(
                    name=tool['name'],
                    description=tool['description'],
                    parametersJsonSchema=tool['input_schema'],
                ) for tool in tools
            ]
        )

    def _convert_messages(self, messages: list[dict]) -> list[genai_types.Content]:
        """
        Convert a list of rendered message dicts into Gemini Content objects using plain text.
        Used only by token_count(), where approximate counts are acceptable.
        For actual API calls, use _convert_pydantic_messages() instead.
        """
        contents = []
        for msg in messages:
            role = 'model' if msg['role'] == 'assistant' else 'user'
            contents.append(genai_types.Content(
                role=role,
                parts=[genai_types.Part.from_text(text=msg['content'])]
            ))
        return contents

    def _convert_pydantic_messages(self, messages: list) -> list[genai_types.Content]:
        """
        Convert Pydantic message objects into Gemini Content objects, using proper
        FunctionCall and FunctionResponse parts for tool call history.
        """
        contents = []
        for msg in messages:
            if isinstance(msg, AssistantMessage):
                parts = []
                if msg.content and msg.content.strip() and msg.content != "** Empty Message, tool call **":
                    parts.append(genai_types.Part.from_text(text=msg.content))
                if msg.tool_call_id:
                    parts.append(genai_types.Part(
                        function_call=genai_types.FunctionCall(
                            name=msg.tool_call_name,
                            args=msg.tool_call_input or {},
                        )
                    ))
                if not parts:
                    parts = [genai_types.Part.from_text(text=msg.content or '')]
                contents.append(genai_types.Content(role='model', parts=parts))
            elif isinstance(msg, ToolMessage):
                parts = [genai_types.Part(
                    function_response=genai_types.FunctionResponse(
                        name=msg.tool_name,
                        response={'output': msg.content},
                    )
                )]
                
                if msg.has_images():
                    for img in msg.get_images():
                        data_url = img.get("image_data_url", "")
                        if data_url and data_url.startswith("data:"):
                            try:
                                header, b64_data = data_url.split(",", 1)
                                mime_type = header.split(":")[1].split(";")[0]
                                image_bytes = base64.b64decode(b64_data)
                                parts.append(genai_types.Part.from_bytes(
                                    data=image_bytes,
                                    mime_type=mime_type
                                ))
                                if img.get("caption"):
                                    parts.append(genai_types.Part.from_text(
                                        text=f"[Image {img.get('result_index', '?')}: {img.get('title', 'Image')}]"
                                    ))
                            except Exception as e:
                                logger.warning("Failed to parse image data URL for Gemini: %s", e)
                
                contents.append(genai_types.Content(role='user', parts=parts))
            else:
                # UserMessage
                contents.append(genai_types.Content(
                    role='user',
                    parts=[genai_types.Part.from_text(text=msg.content or '')]
                ))
        return contents

    def _build_tool_config(self, tool_choice: dict) -> genai_types.ToolConfig:
        """
        Convert the app's tool_choice setting into a Gemini ToolConfig object.
        """
        mode_map = {'auto': 'AUTO', 'any': 'ANY', 'tool': 'ANY'}
        mode = mode_map.get(tool_choice['type'], 'AUTO')
        if tool_choice['type'] == 'tool':
            allowed = [tool_choice['name']]
        else:
            allowed = None
        return genai_types.ToolConfig(
            functionCallingConfig=genai_types.FunctionCallingConfig(
                mode=mode,
                allowedFunctionNames=allowed
            )
        )

    @override
    async def get_message(self, *args, **kwargs) -> LLMResponse:
        """
        Main method: send the conversation to Gemini and return a standardised LLMResponse.
        """
        kwargs.pop('stream_callback', None)
        kwargs.pop('stream_message_uuid', None)
        system = kwargs.pop('system')
        messages = kwargs.pop('messages')
        messages_pydantic = kwargs.pop('messages_pydantic', None)
        max_tokens = kwargs.pop('max_tokens')
        tools = kwargs.pop('tools', None)
        tool_choice = kwargs.pop('tool_choice', None)
        thinking_budget = kwargs.pop('thinking_budget', None)

        if messages_pydantic is not None:
            contents = self._convert_pydantic_messages(messages_pydantic)
        else:
            contents = self._convert_messages(messages)

        if tools:
            gemini_tools = [self._transform_tools(tools)]
            tool_config = self._build_tool_config(tool_choice)
        else:
            gemini_tools = None
            tool_config = None

        thinking_config = genai_types.ThinkingConfig(thinkingBudget=thinking_budget) if thinking_budget is not None else None
        config = genai_types.GenerateContentConfig(
            systemInstruction=system,
            maxOutputTokens=max_tokens,
            tools=gemini_tools,
            toolConfig=tool_config,
            thinkingConfig=thinking_config,
        )

        response = await self.client.aio.models.generate_content(
            model=self.base_args['model'],
            contents=contents,
            config=config,
        )

        function_calls = response.function_calls
        if function_calls:
            fc = function_calls[0]
            if fc.id:
                tool_call_id = fc.id
            else:
                tool_call_id = str(uuid.uuid4())
            if fc.args:
                tool_call_input = fc.args
            else:
                tool_call_input = {}
            tool_call = {
                'tool_call_id': tool_call_id,
                'tool_call_name': fc.name,
                'tool_call_input': tool_call_input,
            }
            stop_reason = 'tool_use'
        else:
            tool_call = {}
            stop_reason = 'end_turn'

        usage = response.usage_metadata
        if usage:
            input_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0
        else:
            input_tokens = 0
            output_tokens = 0

        try:
            text = response.text
        except (ValueError, AttributeError):
            text = None

        return LLMResponse(
            text=text,
            tool_call=tool_call,
            stop_reason=stop_reason,
            input_usage=input_tokens,
            output_usage=output_tokens,
            model=self.base_args['model'],
        )

    @override
    async def token_count(self, conversation: Conversation, new_message: Optional[str] = None) -> int:
        """
        Count the total tokens in the conversation using Gemini's token counting API.
        """
        messages_for_bot = []
        for message in conversation:
            if isinstance(message, ToolMessage) and message.for_whom == 'user':
                pass
            else:
                messages_for_bot.append(message)

        if new_message:
            new_user_message = UserMessage(content=new_message)
            all_messages = messages_for_bot + [new_user_message]
        else:
            all_messages = messages_for_bot

        rendered = []
        for message in all_messages:
            rendered.append(message.render(include={'role', 'content'}))
        contents = self._convert_messages(rendered)

        response = await self.client.aio.models.count_tokens(
            model=self.base_args['model'],
            contents=contents,
        )

        if response.total_tokens:
            return response.total_tokens
        else:
            return 0


__all__ = ['GeminiInterface']
