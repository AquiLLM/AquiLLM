"""
Custom exceptions for FeedbackQL.

We use two distinct error types so the dashboard can show the user a helpful
message depending on what went wrong:

  - FeedbackQLSyntaxError: the query text itself is malformed
      e.g. "messages | wher rating == 5"  (typo in clause name)
           "messages | limit -1"           (invalid value)

  - FeedbackQLFieldError: the query references a field we don't allow
      e.g. "messages | where password == 'secret'"

Both inherit from FeedbackQLError so callers can catch either with one except
if they don't need to distinguish between them.
"""


class FeedbackQLError(Exception):
    """Base class for all FeedbackQL errors."""
    pass


class FeedbackQLSyntaxError(FeedbackQLError):
    """Raised when the query string cannot be parsed — bad grammar or structure."""
    pass


class FeedbackQLFieldError(FeedbackQLError):
    """Raised when the query references a field not in the allowed whitelist."""
    pass
