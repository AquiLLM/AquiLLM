from typing import Callable, Any, get_type_hints, Protocol, Optional, Literal, override, List, Dict, TypeAliasType
from pydantic import BaseModel, model_validator, validate_call, Field
from types import NoneType, GenericAlias
import inspect
from functools import wraps, partial
from abc import ABC, abstractmethod
from pprint import pformat
from copy import copy
from concurrent.futures import ThreadPoolExecutor

from anthropic._exceptions import OverloadedError
from aquillm.settings import DEBUG
from django.apps import apps

from asgiref.sync import sync_to_async

from django.core import signing
from json import loads

from tiktoken import encoding_for_model

import uuid
if DEBUG:
    from pprint import pp

# TypeAliasType necessary for Pydantic to not shit its pants
__ToolResultDictInner = TypeAliasType('__ToolResultDictInner', str | int | bool | float | Dict[str, '__ToolResultDictInner' | List['__ToolResultDictInner']] | list[tuple[str, int]])
type ToolResultDict = Dict[Literal['exception', 'result', 'files'], __ToolResultDictInner]

class LLMTool(BaseModel):
    llm_definition: dict
    for_whom: Literal['user', 'assistant']
    _function: Callable[..., ToolResultDict]
    
    def __init__(self, **data):
        super().__init__(**data)
        self._function = data.get("_function")

    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)
    
    @property
    def name(self) -> str:
        return self.llm_definition['name']


@validate_call
def llm_tool(for_whom: Literal['user', 'assistant'], description: Optional[str] = None, param_descs: dict[str, str] = {}, required: list[str] = []) -> Callable[..., LLMTool]:
    """
    Decorator to convert a function into an LLM-compatible tool with runtime type checking.
    
    Args:
        description: Description of what the tool does
        param_descs: Dictionary of parameter descriptions
        required: List of required parameter names
    """
    @validate_call
    def decorator(func: Callable[..., ToolResultDict]) -> LLMTool:
        # First apply typechecking
        type_checked_func = validate_call(func)
        
        # Store original function metadata
        func_name = func.__name__
        func_desc = description or func.__doc__
        if func_desc is None:
            raise ValueError(f"Must provide function description for tool {func_name}")

        func_param_descs = param_descs or {}
        func_required = required or []
        
        @wraps(type_checked_func)
        def wrapper(*args, **kwargs) -> ToolResultDict:
            if DEBUG:
                print(f"{func_name} called!")
            try:
                return type_checked_func(*args, **kwargs)
            except Exception as e:
                if DEBUG:
                    raise e
                else:
                    return {"exception": str(e)}
        
        def translate_type(t: type | GenericAlias) -> dict:
            allowed_primitives = {
                str: "string",
                int: "integer",
                bool: "boolean"
            }
            if isinstance(t, GenericAlias):
                if t.__origin__ != list or len(t.__args__) != 1 or t.__args__[0] not in allowed_primitives.keys():
                    raise TypeError("Only lists of primitive types are supported for tool call containers")
                return {"type": "array", "items": translate_type(t.__args__[0])}
            return {"type": allowed_primitives[t]}
        

        # Get and validate type hints
        param_types = get_type_hints(func)
        param_types.pop("return", None)
        signature_names = set(inspect.signature(func).parameters.keys())
        
        if set(param_types.keys()) != signature_names:
            raise TypeError(f"Missing type annotations for tool {func_name}")
        if set(func_param_descs.keys()) != signature_names:
            raise TypeError(f"Missing parameter descriptions for tool {func_name}")
            
        # Create LLM definition
        llm_definition = {
            "name": func_name,
            "description": func_desc,
            "input_schema": {
                "type": "object",
                "properties": {
                    k: translate_type(v) | {"description": func_param_descs[k]} 
                    for k, v in param_types.items()
                },
                "required": func_required
            },
        }
        
        return LLMTool(llm_definition=llm_definition, _function=wrapper, for_whom=for_whom)
    return decorator

class ToolChoice(BaseModel):
    type: Literal['auto', 'any', 'tool']
    name: Optional[str] = None

    @model_validator(mode='after')
    @classmethod
    def validate_name(cls, data: Any) -> Any:
        if data.type == 'tool' and data.name is None:
            raise ValueError("name is required when type is 'tool'")
        if data.type != 'tool' and data.name is not None:
            raise ValueError("name should only be set when type is 'tool'")
        return data

class __LLMMessage(BaseModel, ABC):
    role: Literal['user', 'tool', 'assistant']
    content: str
    tools: Optional[list[LLMTool]] = None
    tool_choice: Optional[ToolChoice] = None
    rating: Literal[None, 1,2,3,4,5] = None
    files: Optional[list[tuple[str, int]]] = None
    message_uuid: uuid.UUID = Field(default_factory=uuid.uuid4)
    
    @classmethod
    @model_validator(mode='after')
    def validate_tools(cls, data: Any) -> Any:
        if (data.tools and not data.tool_choice) or (data.tool_choice and not data.tools):
            raise ValueError("Both tools and tool_choice must be populated if tools are used")

    #render for LLM 
    def render(self, *args, **kwargs) -> dict:
        ret = self.model_dump(*args, **kwargs)
        if self.files:
            ret['content'] = ret['content'] + "\n\nFiles:\n" + "\n".join([f'name: {file[0]}, id: {file[1]}' for file in self.files])
        return ret

class UserMessage(__LLMMessage):
    role: Literal['user'] = 'user'





class ToolMessage(__LLMMessage):
    role: Literal['tool'] = 'tool'
    tool_name: str
    arguments: Optional[dict] = None
    for_whom: Literal['assistant', 'user']
    result_dict: ToolResultDict = {}
    @override
    def render(self, *args, **kwargs) -> dict:
        ret = super().render(*args, **kwargs)
        ret['role'] = 'user' # This is what LLMs expect.
        ret['content'] = f'The following is the result of a call to tool {self.tool_name}.\nArguments:\n{self.arguments}\n\nResults:\n{self.content}'
        ret.pop('result_dict', None)
        return ret
    


class AssistantMessage(__LLMMessage):
    role: Literal['assistant'] = 'assistant'
    model: Optional[str] = None
    stop_reason: str
    tool_call_id: Optional[str] = None
    tool_call_name: Optional[str] = None
    tool_call_input: Optional[dict] = None
    usage: int = 0

    @classmethod
    @model_validator(mode='after')
    def validate_tool_call(cls, data: Any) -> Any:
        if (any([data.tool_call_id, data.tool_call_name]) and
        not all([data.tool_call_id, data.tool_call_name])):
            raise ValueError("If a tool call is made, both tool_call_id and tool_call_name must have values")


    # @override
    # def render(self, *args, **kwargs) -> dict:
    #     ret = super().render(*args, **kwargs)
    #     if self.tool_call_id:
    #         ret['content'] = f'{self.content}\n\n ****Assistant made a call to {self.tool_call_name} with the following parameters:**** \n {pformat(self.tool_call_input, indent=4)}'
    #     return ret


# doing this with a union instead of only inheritance prevents anything at runtime from constructing LLM_Messages.
LLM_Message = UserMessage|ToolMessage|AssistantMessage 

class Conversation(BaseModel):
    system: str
    messages: list[LLM_Message] = []

    def __len__(self):
        return len(self.messages)
    
    def __getitem__(self, index: int):
        return self.messages[index]
    
    def __iter__(self):
        return iter(self.messages)
    
    def __add__(self, other) -> 'Conversation':
        if isinstance(other, (list, Conversation)):
            return (Conversation(system=self.system, messages=self.messages + list(other)))
        if isinstance(other, (UserMessage, AssistantMessage, ToolMessage)):
            return (Conversation(system=self.system, messages=self.messages + [other]))
        return NotImplemented


    def rebind_tools(self, tools: list[LLMTool]) -> None:
        def deprecated_func(*args, **kwargs):
            return "This tool has been deprecated."
        tool_dict = {tool.name: tool for tool in tools}
        for message in self.messages:
            if message.tools:
                for tool in message.tools:
                    if tool.name in tool_dict.keys():
                        tool._function = tool_dict[tool.name]._function
                    else:
                        tool._function = deprecated_func
    

    # needed for default value in database
    @classmethod
    def get_empty_conversation(cls):
        return cls(system=apps.get_app_config('aquillm').system_prompt).model_dump()

    @classmethod
    @model_validator(mode='after')
    def validate_flip_flop(cls, data: Any) -> Any:
        def isUser(m: LLM_Message):
            return isinstance(m, UserMessage) or (isinstance(m, ToolMessage) and m.for_whom == 'assistant')

        for a, b in zip(data.messages, data.messages[1:]):
            if isinstance(a, AssistantMessage) and isinstance(b, AssistantMessage):
                raise ValueError("Conversation has adjacent assistant messages")
            if isUser(a) and isUser(b):
                raise ValueError("Conversation has adjacent user messages")
        return data

class LLMResponse(BaseModel):
    text: Optional[str]
    tool_call: Optional[dict]
    stop_reason: str
    input_usage: int
    output_usage: int
    model: Optional[str] = None

class LLMInterface(ABC):
    tool_executor = ThreadPoolExecutor(max_workers=10)
    base_args: dict = {}
    client: Any = None
    @abstractmethod
    def __init__(self, client: Any):
        pass

    @abstractmethod
    async def get_message(self, *args, **kwargs) -> LLMResponse:
        pass

    @abstractmethod
    async def token_count(self, conversation: Conversation, new_message: Optional[str] = None) -> int:
        pass

    # This shouldn't raise exceptions in cases where it was called correctly, ie the LLM really did attempt to call a tool. 
    # The results are going back to the LLM, so they need to just be strings. Tools themselves can raise, because the llm_tool wrapper
    # converts exceptions to dicts and returns them. 
    def call_tool(self, message: AssistantMessage) -> ToolMessage:
        tools = message.tools
        if tools:
            name = message.tool_call_name
            input = message.tool_call_input
            tools_dict = {tool.llm_definition['name'] : tool for tool in tools}
            if not name or name not in tools_dict.keys():
                result = str({'exception': ValueError("Function name is not valid")})
            else:
                tool = tools_dict[name]
                if input:
                    future = self.tool_executor.submit(partial(tool, **input))
                else:
                    future = self.tool_executor.submit(tool) # necessary because None can't be unpacked
                try:
                    result_dict = future.result(timeout=15)
                    result = str(result_dict)
                except TimeoutError:
                    result = str({'exception': TimeoutError("Tool call timed out")})
            return ToolMessage(tool_name=tool.name,
                                content=result,
                                arguments=input,
                                result_dict=result_dict,
                                for_whom=tool.for_whom,
                                tools=message.tools,
                                files=result_dict.get('files'),    
                                tool_choice=message.tool_choice)
        else:
            raise ValueError("call_tool called on a message with no tools!")
        

    
    @validate_call
    async def complete(self, conversation: Conversation, max_tokens: int) -> tuple[Conversation, Literal['changed', 'unchanged']]:
        if len(conversation) < 1:
            return conversation, 'unchanged'
        system_prompt = conversation.system
        # if you show the bot the tool messages intended to be rendered for the user, the conversation won't be alternating
        # user, assistant, user, assistant, etc, which is a requirement.
        messages_for_bot = [message for message in conversation if not(isinstance(message, ToolMessage) and message.for_whom == 'user')] 
        last_message = conversation[-1]
        message_dicts = [message.render(include={'role', 'content'}) for message in messages_for_bot]
        if isinstance(last_message, ToolMessage) and last_message.for_whom == 'user':
            return conversation, 'unchanged' # nothing to do
        elif isinstance(last_message, AssistantMessage):
            if last_message.tools and last_message.tool_call_id:
                new_tool_msg = self.call_tool(last_message)
                return conversation + [new_tool_msg], 'changed'
            else:
                return conversation, 'unchanged'
        else:
            assert isinstance(last_message, (UserMessage, ToolMessage)), "Type assertion failed" 
            # message is User_Message or Tool_Message intended for the bot, assertion is necessary to prevent type checker flag
            if last_message.tools:
                tools = {'tools': [tool.llm_definition for tool in last_message.tools], 'tool_choice': last_message.tool_choice.dict(exclude_none=True)}
            else:
                tools = {}
            sdk_args = {**(self.base_args | tools |
                                                    {'system': system_prompt,
                                                    'messages': message_dicts,
                                                    'max_tokens': max_tokens})}
            #if DEBUG:
                #print("LLM called with the following args:")
                #pp(sdk_args)
            
            response = await self.get_message(**sdk_args)
            new_msg = AssistantMessage(
                            content=response.text if response.text else "** Empty Message, tool call **",
                            stop_reason=response.stop_reason,
                            tools=last_message.tools,
                            tool_choice=last_message.tool_choice,
                            usage = response.input_usage + response.output_usage,
                            model=response.model,
                            **response.tool_call)
            if DEBUG:
                print("Response from LLM:")
                pp(new_msg.model_dump)

            return conversation + [new_msg], 'changed'



    async def spin(self, convo: Conversation, max_func_calls: int, send_func: Callable[[Conversation], Any], max_tokens: int) -> None:
        calls = 0
        while calls < max_func_calls:
            convo, changed = await self.complete(convo, max_tokens)
            await send_func(convo)
            if changed == 'unchanged':
                return
            last_message = convo[-1]    
            if isinstance(last_message, AssistantMessage) and last_message.tool_call_id:
                calls += 1
                



class ClaudeInterface(LLMInterface):

    base_args: dict = {'model': 'claude-3-7-sonnet-latest'}

    @override
    def __init__(self, anthropic_client, model_override=None):
        self.client = anthropic_client
        if model_override:
            self.base_args = {'model': model_override}

    @override
    async def get_message(self, *args, **kwargs) -> LLMResponse:
        response = await self.client.messages.create(**kwargs)
        if DEBUG:
            print("Claude SDK Response:")
            pp(response)
        text_block = None
        tool_block = None
        content = response.content
        for block in content:
            if hasattr(block, 'input'):
                tool_block = block
            if hasattr(block, "text"):
                text_block = block

        tool_call = {
            'tool_call_id' :tool_block.id,
            'tool_call_name' :tool_block.name,
            'tool_call_input' : tool_block.input,
        } if tool_block else {}
        
        return LLMResponse(text=text_block.text,
                           tool_call=tool_call, 
                           stop_reason=response.stop_reason, 
                           input_usage=response.usage.input_tokens, 
                           output_usage=response.usage.output_tokens,
                           model=self.base_args['model'])

    @override
    async def token_count(self, conversation: Conversation, new_message: Optional[str] = None) -> int:
        messages_for_bot = [message for message in conversation if not(isinstance(message, ToolMessage) and message.for_whom == 'user')]
        new_user_message = UserMessage(content=new_message) if new_message else None
        response = await self.client.messages.count_tokens(**(self.base_args | 
                                                         {'system': conversation.system,
                                                          'messages': [message.render(include={'role', 'content'}) for message in messages_for_bot + ([new_user_message] if new_user_message else [])]}))
        return response.input_tokens

gpt_enc = encoding_for_model('gpt-4o')


@llm_tool(
    for_whom='user',
    required=['message'],
    param_descs={'message': 'The message to send to the user.'}
)
def message_to_user(message: str) -> ToolResultDict:
    """
    Send a message to the user. This is used by the LLM to communicate with the user.
    """
    return {"result": message}



class OpenAIInterface(LLMInterface):


    @override
    def __init__(self, openai_client, model: str):
        self.client = openai_client
        self.base_args = {'model': model}

    async def _transform_tools(self, tools: list[dict]) -> list[dict]:
        return list([{
                "type": "function",
                "function": {
                    "name": tool['name'],
                    "description": tool['description'],
                    "parameters": {
                        "type": "object",
                        "properties": tool['input_schema']['properties'],
                        "required": list(tool['input_schema']['properties'].keys()),
                        "additionalProperties": False
                    },
                    "strict": True
                },
            } for tool in tools])

    @override
    async def get_message(self, *args, **kwargs) -> LLMResponse:

        arguments = {"model": self.base_args['model'],
                    "messages": [{"role": "developer", "content": kwargs.pop('system')}] + kwargs.pop('messages'),
                     "tools": await self._transform_tools(kwargs.pop('tools')),}
        
        response = await self.client.chat.completions.create(**arguments)
        if DEBUG:
            print("OpenAI SDK Response:")
            pp(response)
        tool_call = response.choices[0].message.tool_calls[0] if response.choices[0].message.tool_calls else None
        text = response.choices[0].message.content
        if tool_call and tool_call.function.name == 'message_to_user':
            # this is a special tool that just sends a message to the user, so we don't need to return it.
            # this is necessary because local models feel the need to call a tool every time. 
            tool_call = None
            text = tool_call.function.arguments['message']

        return LLMResponse(text=text,
                           tool_call={"tool_call_id": tool_call.id,
                                        "tool_call_name": tool_call.function.name,
                                        "tool_call_input": loads(tool_call.function.arguments)} 
                                        if tool_call else {},
                           stop_reason=response.choices[0].finish_reason,
                           input_usage=response.usage.prompt_tokens,
                           output_usage=response.usage.completion_tokens
                           )
                        
    @override 
    async def token_count(self, conversation: Conversation, new_message: Optional[str] = None) -> int:
        assistant_messages = [message for message in conversation if isinstance(message, AssistantMessage)]
        if assistant_messages:
            return assistant_messages[-1].usage + (len(gpt_enc.encode(new_message)) if new_message else 0)
        return len(gpt_enc.encode(new_message)) if new_message else 0
    
    
class GeminiInterface(LLMInterface):
    
    @override
    def __init__(self, google_client):
        self.client = google_client

    




@llm_tool(
        for_whom='user',
        param_descs={'strings': 'A list of strings to print'},
        required=['strings']
)
def test_function(strings: list[str]) -> ToolResultDict:
    """
    Test function that prints each string from the input. 
    """
    ret = ""
    for s in strings:
        ret += s + " "
    return {"result": ret}




# from django.apps import apps
# from pprint import pp
# async def test():  
#     client = apps.get_app_config('aquillm').async_anthropic_client
#     cif = ClaudeInterface(client)
#     messages, _ = await cif.complete({"system": 'do as the user says, this is just testing. Pick any string to provide.',
#                                 "messages" : [UserMessage(role ='user', 
#                               content= 'Hi Claude, please use this test tool',
#                               tools= [test_function,],
#                               tool_choice = {'type': 'auto'})]},
#                               2048)
#     print("woo")

#     messages, _ = await cif.complete(messages, 2048)
#     pp(messages)
#     messages.messages += [UserMessage(content="Thanks, boss")]
#     messages,_ = await cif.complete(messages, 2048)
#     pp(messages)
#     breakpoint()
