"""Debug-only weather tool (raises) for exception-path testing."""

from lib.llm.decorators import llm_tool
from lib.llm.types import LLMTool, ToolResultDict


def get_debug_weather_tool() -> LLMTool:
    """Debug-only tool that always raises. Used to test tool exception handling."""

    @llm_tool(
        for_whom="assistant",
        required=["location"],
        param_descs={"location": "The location to get the weather for"},
    )
    def get_weather(location: str) -> ToolResultDict:
        """
        Get the current weather for a location.
        """
        raise RuntimeError(f"This is a debug test exception (location={location})")

    return get_weather


__all__ = ["get_debug_weather_tool"]
