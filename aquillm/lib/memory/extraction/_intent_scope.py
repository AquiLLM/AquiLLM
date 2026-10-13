"""Enclosing unquoted example scope cannot authorize a persistence directive."""
import re

_EXAMPLE_SCOPE = re.compile(r'(?i)\b(?:if|hypothetical(?:ly)?|suppose|imagine|example|discussion)\b|would happen')


def directive_is_example(text, start, protected, after=0):
    # A previously proven affirmative payload is its own closed clause. Quotes
    # and fences elsewhere do not turn a separate user directive into an example.
    context = text[after:start]
    for left, right in protected:
        left, right = max(after, left), min(start, right)
        if left < right:
            context = context[:left - after] + ' ' * (right - left) + context[right - after:]
    return bool(_EXAMPLE_SCOPE.search(context))
