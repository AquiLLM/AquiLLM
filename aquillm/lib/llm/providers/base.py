"""Base LLM interface class."""
from typing import Callable, Any, Awaitable, Optional, Literal
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from os import getenv
from json import dumps
import re
import uuid

from pydantic import validate_call

from ..types.messages import UserMessage, ToolMessage, AssistantMessage, LLM_Message
from ..types.conversation import Conversation
from ..types.tools import LLMTool, dump_tool_choice
from ..types.response import LLMResponse


try:
    from aquillm.settings import DEBUG
except ImportError:
    DEBUG = False

if DEBUG:
    from pprint import pp


class LLMInterface(ABC):
    """Abstract base class for LLM provider interfaces."""
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

    @staticmethod
    def _looks_like_deferred_tool_intent(text: Optional[str]) -> bool:
        """
        Heuristic: detect when the model says it will search/look something up
        instead of actually issuing a tool call in this turn.
        """
        if not text:
            return False
        normalized = re.sub(r"[\\u2018\\u2019]", chr(39), text.lower())
        cues = (
            "i'll search",
            "i will search",
            "i'll search for",
            "let me search",
            "i can search",
            "i'm going to search",
            "i'll look through",
            "i will look through",
            "i'll look up",
            "i will look up",
            "let me look up",
            "i'll check",
            "i will check",
            "i'll read the papers",
            "i will read the papers",
        )
        return any(cue in normalized for cue in cues)

    @staticmethod
    def _extractive_fallback_enabled() -> bool:
        return getenv("LLM_ALLOW_EXTRACTIVE_FALLBACK", "0").strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _first_sentence(text: str, max_chars: int = 260) -> str:
        cleaned = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", text or "")
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            return ""
        match = re.search(r"(.+?[.!?])(\s|$)", cleaned)
        candidate = match.group(1) if match else cleaned
        if len(candidate) > max_chars:
            return candidate[:max_chars].rstrip() + "..."
        return candidate

    @staticmethod
    def _is_useful_fallback_sentence(text: str) -> bool:
        candidate = (text or "").strip()
        if not candidate:
            return False
        if not re.match(r'^[A-Z"(]', candidate):
            return False
        words = re.findall(r"[A-Za-z0-9]+", candidate)
        if len(words) < 10:
            return False
        alpha_chars = len(re.findall(r"[A-Za-z]", candidate))
        if alpha_chars < 40:
            return False
        digit_chars = len(re.findall(r"\d", candidate))
        if digit_chars / max(1, len(candidate)) > 0.08:
            return False
        upper_chars = len(re.findall(r"[A-Z]", candidate))
        if alpha_chars and (upper_chars / alpha_chars) > 0.45:
            return False
        bad_tokens = ("mmlu", "bbh", "gsm8k", "triviaqa", "humaneval", "mbpp", "cmath")
        lowered = candidate.lower()
        if sum(1 for token in bad_tokens if token in lowered) >= 2:
            return False
        return True

    @staticmethod
    def _is_high_quality_summary(text: str) -> bool:
        candidate = (text or "").strip()
        if len(candidate) < 220:
            return False
        lowered = candidate.lower()
        if lowered.startswith("here are the key points from the retrieved passages"):
            return False
        if "i retrieved supporting passages but could not generate a final answer" in lowered:
            return False
        if "please retry and i will provide a direct summary" in lowered:
            return False
        bad_tokens = ("mmlu", "bbh", "gsm8k", "triviaqa", "humaneval", "mbpp", "cmath")
        if sum(1 for token in bad_tokens if token in lowered) >= 5:
            return False
        bullet_count = candidate.count("\n- ") + candidate.count("\n* ")
        sentence_count = len(re.findall(r"[.!?](?:\s|$)", candidate))
        return bullet_count >= 3 or sentence_count >= 5

    @staticmethod
    def _sanitize_data_urls_for_llm_text(text: str) -> str:
        return re.sub(
            r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+",
            "[image data url redacted for context budget]",
            text or "",
            flags=re.IGNORECASE,
        )

    @classmethod
    def _serialize_tool_result_for_llm(cls, result_dict: Any) -> str:
        """
        Build tool result text for LLM context while excluding private transport keys
        (e.g. _images blobs) and redacting inline base64 data URLs.
        """
        if isinstance(result_dict, dict):
            visible = {
                key: value
                for key, value in result_dict.items()
                if not str(key).startswith("_")
            }
            serialized = dumps(visible, ensure_ascii=False, default=str)
            return cls._sanitize_data_urls_for_llm_text(serialized)
        return cls._sanitize_data_urls_for_llm_text(str(result_dict))

    @classmethod
    def _select_evidence_snippet(cls, text: str, max_chars: int = 420) -> str:
        cleaned = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", text or "")
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            return ""
        sentence_candidates = re.split(r"(?<=[.!?])\s+", cleaned)
        good: list[str] = []
        for sentence in sentence_candidates:
            s = sentence.strip()
            if not cls._is_useful_fallback_sentence(s):
                continue
            good.append(s)
            joined = " ".join(good)
            if len(joined) >= max_chars or len(good) >= 2:
                break
        snippet = " ".join(good).strip()
        if not snippet:
            first = cls._first_sentence(cleaned, max_chars=max_chars)
            if cls._is_useful_fallback_sentence(first):
                snippet = first
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."
        return snippet

    @classmethod
    def _extract_recent_tool_evidence(
        cls,
        conversation: Conversation,
        max_snippets: int = 10,
        max_chars_per_snippet: int = 420,
    ) -> tuple[str, list[tuple[str, str]]]:
        latest_user_query = ""
        for msg in reversed(conversation.messages):
            if isinstance(msg, UserMessage):
                latest_user_query = (msg.content or "").strip()
                if latest_user_query:
                    break

        title_re = re.compile(r"--\s*(.*?)\s*chunk\s*#:", flags=re.IGNORECASE)
        snippets: list[tuple[str, str]] = []
        seen_keys: set[str] = set()

        tool_messages = [
            msg for msg in reversed(conversation.messages)
            if isinstance(msg, ToolMessage) and msg.for_whom == 'assistant'
        ]

        for tool_msg in tool_messages[:4]:
            result_dict = tool_msg.result_dict if isinstance(tool_msg.result_dict, dict) else {}
            payload = result_dict.get("result")
            entries: list[tuple[str, str]] = []
            if isinstance(payload, dict):
                entries = [(str(k), str(v)) for k, v in list(payload.items())[:12]]
            elif isinstance(payload, str):
                entries = [(tool_msg.tool_name, payload)]
            for key_text, raw_text in entries:
                source = tool_msg.tool_name
                title_match = title_re.search(key_text)
                if title_match and title_match.group(1).strip():
                    source = title_match.group(1).strip()
                snippet = cls._select_evidence_snippet(raw_text, max_chars=max_chars_per_snippet)
                if not snippet:
                    continue
                dedupe_key = snippet[:180]
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                snippets.append((source, snippet))
                if len(snippets) >= max_snippets:
                    break
            if len(snippets) >= max_snippets:
                break

        return latest_user_query, snippets

    async def _generate_compact_tool_summary(
        self,
        conversation: Conversation,
        max_tokens: int,
    ) -> Optional[str]:
        latest_user_query, evidence = self._extract_recent_tool_evidence(conversation)
        if not evidence:
            return None

        evidence_lines = "\n".join(
            [f"{i+1}. [{source}] {snippet}" for i, (source, snippet) in enumerate(evidence[:8])]
        )
        user_request = latest_user_query or "Summarize the key points from these retrieved document excerpts."
        summary_prompt = (
            f"User request: {user_request}\n\n"
            "You are given evidence snippets retrieved from documents.\n"
            "Write a complete, direct answer for the user.\n"
            "Requirements:\n"
            "- Provide 4 to 8 concise key points.\n"
            "- Use normal readable prose and bullets.\n"
            "- Do not mention tools, retrieval, or internal system behavior.\n"
            "- Prefer conceptual conclusions and practical takeaways over raw benchmark tables.\n"
            "- If numbers are included, explain what they mean.\n"
            "- If evidence is incomplete, state what is missing in one sentence.\n\n"
            f"Evidence snippets:\n{evidence_lines}"
        )
        summary_max_tokens = max(512, min(max_tokens + 256, 1400))
        attempt_prompt = summary_prompt
        best_text: Optional[str] = None
        for _ in range(3):
            try:
                summary_response = await self.get_message(
                    system="You summarize technical evidence into a clear final answer.",
                    messages=[{"role": "user", "content": attempt_prompt}],
                    messages_pydantic=[UserMessage(content=attempt_prompt)],
                    max_tokens=summary_max_tokens,
                )
            except Exception:
                continue

            text = (summary_response.text or "").strip()
            if not text or self._looks_like_deferred_tool_intent(text):
                continue
            best_text = text
            stop_reason_normalized = str(summary_response.stop_reason or "").strip().lower()
            if (
                stop_reason_normalized in {"length", "max_tokens"}
                or self._looks_cut_off(text)
                or not self._is_high_quality_summary(text)
            ):
                attempt_prompt = (
                    f"{summary_prompt}\n\n"
                    "The previous draft was incomplete or low quality. "
                    "Rewrite the answer as a polished final response with coherent bullets and clear conclusions.\n\n"
                    f"Draft to improve:\n{text}"
                )
                continue
            return text
        if best_text and self._is_high_quality_summary(best_text):
            return best_text
        if best_text and len(best_text.strip()) >= 140 and not self._looks_like_deferred_tool_intent(best_text):
            return best_text.strip()
        return None

    @staticmethod
    def _looks_cut_off(text: str) -> bool:
        cleaned = (text or "").rstrip()
        if not cleaned:
            return False
        if cleaned.endswith(("...", "…")):
            return True
        if cleaned[-1] in ".!?)]}\"'":
            return False
        return True

    @staticmethod
    def _continue_on_cutoff_enabled() -> bool:
        return getenv("LLM_CONTINUE_ON_CUTOFF", "1").strip().lower() in ("1", "true", "yes", "on")

    async def _continue_cutoff_response(
        self,
        *,
        system_prompt: str,
        message_dicts: list[dict],
        messages_for_bot: list[LLM_Message],
        partial_text: str,
        max_tokens: int,
    ) -> Optional[LLMResponse]:
        if not partial_text.strip():
            return None
        continuation_prompt = (
            "Continue the previous assistant response exactly where it stopped. "
            "Do not restart, do not repeat prior points, and keep the same structure/tone."
        )
        continuation_messages = message_dicts + [
            {"role": "assistant", "content": partial_text},
            {"role": "user", "content": continuation_prompt},
        ]
        continuation_messages_pydantic: list[LLM_Message] = messages_for_bot + [
            AssistantMessage(content=partial_text, stop_reason='max_tokens'),
            UserMessage(content=continuation_prompt),
        ]
        try:
            return await self.get_message(
                **(self.base_args | {
                    'system': system_prompt,
                    'messages': continuation_messages,
                    'messages_pydantic': continuation_messages_pydantic,
                    'max_tokens': max_tokens,
                })
            )
        except Exception:
            return None

    @classmethod
    def _synthesize_from_recent_tool_results(cls, conversation: Conversation) -> Optional[str]:
        tool_messages = [
            msg for msg in reversed(conversation.messages)
            if isinstance(msg, ToolMessage) and msg.for_whom == 'assistant'
        ]
        if not tool_messages:
            return None

        bullets: list[str] = []
        seen: set[str] = set()
        source_titles: list[str] = []
        title_re = re.compile(r"--\s*(.*?)\s*chunk\s*#:", flags=re.IGNORECASE)

        for tool_msg in tool_messages[:3]:
            result_dict = tool_msg.result_dict if isinstance(tool_msg.result_dict, dict) else {}
            payload = result_dict.get("result")
            if isinstance(payload, dict):
                for k, v in list(payload.items())[:8]:
                    key_text = str(k)
                    val_text = str(v)
                    title_match = title_re.search(key_text)
                    if title_match:
                        title = title_match.group(1).strip()
                        if title and title not in source_titles:
                            source_titles.append(title)
                    sentence = cls._first_sentence(val_text)
                    if sentence and cls._is_useful_fallback_sentence(sentence) and sentence not in seen:
                        seen.add(sentence)
                        bullets.append(sentence)
                    if len(bullets) >= 6:
                        break
            elif isinstance(payload, str):
                sentence = cls._first_sentence(payload)
                if sentence and cls._is_useful_fallback_sentence(sentence) and sentence not in seen:
                    seen.add(sentence)
                    bullets.append(sentence)
            if len(bullets) >= 6:
                break

        if not bullets:
            return None

        header = "Here are the key points from the retrieved passages:"
        bullet_lines = [f"- {point}" for point in bullets[:5]]
        if source_titles:
            sources = ", ".join(source_titles[:4])
            return f"{header}\n" + "\n".join(bullet_lines) + f"\n\nSources consulted: {sources}"
        return f"{header}\n" + "\n".join(bullet_lines)

    def call_tool(self, message: AssistantMessage) -> ToolMessage:
        """Execute a tool call from an assistant message."""
        tools = message.tools
        if tools:
            name = message.tool_call_name
            input = message.tool_call_input
            tools_dict = {tool.llm_definition['name']: tool for tool in tools}
            tool_name = name or "invalid_tool"
            for_whom: Literal['assistant', 'user'] = 'assistant'
            result_dict: dict = {"exception": "Tool call failed before execution"}
            if not name or name not in tools_dict.keys():
                result_dict = {'exception': "Function name is not valid"}
                result = self._serialize_tool_result_for_llm(result_dict)
            else:
                tool = tools_dict[name]
                tool_name = tool.name
                for_whom = tool.for_whom
                if input:
                    future = self.tool_executor.submit(partial(tool, **input))
                else:
                    future = self.tool_executor.submit(tool)
                try:
                    tool_timeout_s = float(getenv("TOOL_CALL_TIMEOUT_SECONDS", "10"))
                    result_dict = future.result(timeout=tool_timeout_s)
                    result = self._serialize_tool_result_for_llm(result_dict)
                except TimeoutError:
                    result_dict = {'exception': "Tool call timed out"}
                    result = self._serialize_tool_result_for_llm(result_dict)
                except Exception as e:
                    if DEBUG:
                        raise
                    result_dict = {'exception': str(e)}
                    result = self._serialize_tool_result_for_llm(result_dict)
            return ToolMessage(
                tool_name=tool_name,
                content=result,
                arguments=input,
                result_dict=result_dict,
                for_whom=for_whom,
                tools=message.tools,
                files=result_dict.get('files') if isinstance(result_dict, dict) else None,
                tool_choice=message.tool_choice
            )
        else:
            raise ValueError("call_tool called on a message with no tools!")

    @validate_call
    async def complete(
        self,
        conversation: Conversation,
        max_tokens: int,
        stream_func: Optional[Callable[[dict], Awaitable[Any]]] = None,
    ) -> tuple[Conversation, Literal['changed', 'unchanged']]:
        """Complete a conversation by getting the next message from the LLM."""
        if len(conversation) < 1:
            return conversation, 'unchanged'
        system_prompt = conversation.system
        messages_for_bot = [message for message in conversation if not(isinstance(message, ToolMessage) and message.for_whom == 'user')] 
        last_message = conversation[-1]
        message_dicts = [message.render(include={'role', 'content'}) for message in messages_for_bot]
        if isinstance(last_message, ToolMessage) and last_message.for_whom == 'user':
            return conversation, 'unchanged'
        elif isinstance(last_message, AssistantMessage):
            if last_message.tools and last_message.tool_call_id:
                new_tool_msg = self.call_tool(last_message)
                return conversation + [new_tool_msg], 'changed'
            else:
                return conversation, 'unchanged'
        else:
            assert isinstance(last_message, (UserMessage, ToolMessage)), "Type assertion failed" 
            is_post_tool_result_turn = isinstance(last_message, ToolMessage) and last_message.for_whom == 'assistant'
            try:
                tool_step_max_tokens = max(128, int(getenv("LLM_TOOL_STEP_MAX_TOKENS", "512")))
            except Exception:
                tool_step_max_tokens = 512
            try:
                post_tool_max_tokens = max(256, int(getenv("LLM_POST_TOOL_MAX_TOKENS", "1024")))
            except Exception:
                post_tool_max_tokens = 1024
            request_max_tokens = max_tokens
            if isinstance(last_message, UserMessage) and last_message.tools:
                request_max_tokens = min(max_tokens, tool_step_max_tokens)
            elif is_post_tool_result_turn:
                request_max_tokens = min(max_tokens, post_tool_max_tokens)
            if last_message.tools:
                tools = {
                    'tools': [tool.llm_definition for tool in last_message.tools],
                    'tool_choice': dump_tool_choice(last_message.tool_choice),
                }
            else:
                tools = {}
            stream_message_uuid = str(uuid.uuid4())
            sdk_args = {**(self.base_args | tools |
                                                    {'system': system_prompt,
                                                    'messages': message_dicts,
                                                    'messages_pydantic': messages_for_bot,
                                                    'max_tokens': request_max_tokens,
                                                    'stream_callback': stream_func,
                                                    'stream_message_uuid': stream_message_uuid})}
            
            response = await self.get_message(**sdk_args)
            should_force_tool_retry = (
                bool(last_message.tools)
                and bool(last_message.tool_choice)
                and last_message.tool_choice.type == 'auto'
                and not response.tool_call
                and self._looks_like_deferred_tool_intent(response.text)
            )
            if should_force_tool_retry:
                retry_args = sdk_args | {'tool_choice': {'type': 'any'}}
                response = await self.get_message(**retry_args)

            if is_post_tool_result_turn:
                response_text_for_retry = (response.text or "").strip()
                response_has_tool_call = bool(response.tool_call)
                needs_final_synthesis_retry = (
                    (not response_has_tool_call and not response_text_for_retry)
                    or (not response_has_tool_call and self._looks_like_deferred_tool_intent(response.text))
                )
                if needs_final_synthesis_retry:
                    finalize_prompt = (
                        "Use the tool results above to answer the user's last request directly. "
                        "Do not call tools. Return a complete final answer in plain text."
                    )
                    finalize_messages = message_dicts + [{"role": "user", "content": finalize_prompt}]
                    finalize_pydantic_messages = messages_for_bot + [UserMessage(content=finalize_prompt)]
                    finalize_args = self.base_args | {
                        'system': system_prompt,
                        'messages': finalize_messages,
                        'messages_pydantic': finalize_pydantic_messages,
                        'max_tokens': min(max_tokens, post_tool_max_tokens),
                        'stream_callback': stream_func,
                        'stream_message_uuid': stream_message_uuid,
                    }
                    response = await self.get_message(**finalize_args)
                post_finalize_text = (response.text or "").strip()
                if (not response.tool_call) and (
                    (not post_finalize_text)
                    or self._looks_like_deferred_tool_intent(post_finalize_text)
                ):
                    compact_summary = await self._generate_compact_tool_summary(conversation, max_tokens)
                    if compact_summary:
                        response = LLMResponse(
                            text=compact_summary,
                            tool_call={},
                            stop_reason='stop',
                            input_usage=response.input_usage,
                            output_usage=response.output_usage,
                            model=response.model,
                            message_uuid=response.message_uuid,
                        )

            allowed_tool_names = {tool.name for tool in (last_message.tools or [])}
            response_text = response.text if response.text else ""
            response_tool_call = response.tool_call or {}

            if response_tool_call:
                called_tool_name = response_tool_call.get("tool_call_name")
                if (not allowed_tool_names) or (called_tool_name not in allowed_tool_names):
                    response_tool_call = {}
                    if not response_text.strip():
                        compact_summary = await self._generate_compact_tool_summary(conversation, max_tokens)
                        response_text = compact_summary or (
                            self._synthesize_from_recent_tool_results(conversation) if self._extractive_fallback_enabled() else None
                        ) or (
                            "I completed retrieval but received an unusable tool-call payload. "
                            "Please retry and I will provide a full summary."
                        )

            if (not response_tool_call) and (not response_text.strip()):
                compact_summary = await self._generate_compact_tool_summary(conversation, max_tokens)
                response_text = compact_summary or (
                    self._synthesize_from_recent_tool_results(conversation) if self._extractive_fallback_enabled() else None
                ) or (
                    "I retrieved supporting passages but could not generate a final answer. "
                    "Please retry and I will provide a full summary."
                )

            stop_reason_normalized = str(response.stop_reason or "").strip().lower()
            if (
                (not response_tool_call)
                and response_text.strip()
                and stop_reason_normalized in {"length", "max_tokens"}
                and self._looks_cut_off(response_text)
            ):
                continuation_response: Optional[LLMResponse] = None
                continuation_text = ""
                if self._continue_on_cutoff_enabled():
                    continuation_response = await self._continue_cutoff_response(
                        system_prompt=system_prompt,
                        message_dicts=message_dicts,
                        messages_for_bot=messages_for_bot,
                        partial_text=response_text,
                        max_tokens=max_tokens,
                    )
                    continuation_text = (continuation_response.text or "").strip() if continuation_response else ""
                if continuation_text and not self._looks_like_deferred_tool_intent(continuation_text):
                    separator = "\n" if response_text and not response_text.endswith(("\n", " ")) else ""
                    response_text = f"{response_text.rstrip()}{separator}{continuation_text}"
                    response = continuation_response
                else:
                    compact_summary = await self._generate_compact_tool_summary(conversation, max_tokens)
                    if compact_summary:
                        response_text = compact_summary
                    elif self._extractive_fallback_enabled():
                        synthesized = self._synthesize_from_recent_tool_results(conversation)
                        if synthesized:
                            response_text = synthesized
            new_msg = AssistantMessage(
                            content=response_text,
                            stop_reason=response.stop_reason,
                            tools=last_message.tools,
                            tool_choice=last_message.tool_choice,
                            usage=response.input_usage + response.output_usage,
                            model=response.model,
                            message_uuid=response.message_uuid or uuid.uuid4(),
                            **response_tool_call)
            if DEBUG:
                print("Response from LLM:")
                pp(new_msg.model_dump())

            return conversation + [new_msg], 'changed'

    async def spin(
        self,
        convo: Conversation,
        max_func_calls: int,
        send_func: Callable[[Conversation], Any],
        max_tokens: int,
        stream_func: Optional[Callable[[dict], Awaitable[Any]]] = None,
    ) -> None:
        """Spin the conversation, executing tool calls until complete."""
        calls = 0
        repeated_tool_call_count = 0
        prev_tool_signature: Optional[str] = None
        tool_name_call_counts: dict[str, int] = {}
        try:
            repeat_break_threshold = max(1, int(getenv("LLM_REPEAT_TOOL_BREAK_THRESHOLD", "3")))
        except Exception:
            repeat_break_threshold = 3
        try:
            max_calls_per_tool_name = max(1, int(getenv("LLM_MAX_CALLS_PER_TOOL_NAME", "2")))
        except Exception:
            max_calls_per_tool_name = 2
        while calls < max_func_calls:
            convo, changed = await self.complete(convo, max_tokens, stream_func=stream_func)
            await send_func(convo)
            if changed == 'unchanged':
                return
            last_message = convo[-1]
            if isinstance(last_message, AssistantMessage) and last_message.tool_call_id:
                sig = f"{last_message.tool_call_name}|{dumps(last_message.tool_call_input or {}, sort_keys=True, ensure_ascii=True)}"
                if sig == prev_tool_signature:
                    repeated_tool_call_count += 1
                else:
                    repeated_tool_call_count = 0
                prev_tool_signature = sig
                tool_name = str(last_message.tool_call_name or "").strip().lower()
                if tool_name:
                    tool_name_call_counts[tool_name] = tool_name_call_counts.get(tool_name, 0) + 1
                    if tool_name_call_counts[tool_name] >= max_calls_per_tool_name:
                        break
                if repeated_tool_call_count >= repeat_break_threshold:
                    break
                calls += 1
        last = convo[-1]
        if isinstance(last, AssistantMessage) and last.tool_call_id:
            convo, _ = await self.complete(convo, max_tokens, stream_func=stream_func)
            await send_func(convo)
            convo[-1].tools = None
            convo[-1].tool_choice = None
            convo, _ = await self.complete(convo, max_tokens, stream_func=stream_func)
            await send_func(convo)


__all__ = ['LLMInterface']
