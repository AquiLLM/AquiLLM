from .exceptions import FeedbackQLError, FeedbackQLFieldError, FeedbackQLSyntaxError
from .executor import execute
from .parser import parse

__all__ = [
    'parse',
    'execute',
    'run',
    'FeedbackQLError',
    'FeedbackQLSyntaxError',
    'FeedbackQLFieldError',
]


def run(query_string: str) -> list[dict]:
    """Parse and execute a FeedbackQL query string, returning a list of result dicts."""
    return execute(parse(query_string))
