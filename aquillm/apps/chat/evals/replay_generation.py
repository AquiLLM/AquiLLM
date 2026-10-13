"""Deny generation while retaining safe provider context-policy identity."""


class GenerationDenied(RuntimeError):
    pass


class GenerationGuard:
    def __new__(cls, interface):
        if cls is GenerationGuard:
            # Match the existing capacity policy's closed provider families.
            provider = type(interface).__name__.lower()
            cls = (
                GeminiGenerationGuard
                if "gemini" in provider
                else ClaudeGenerationGuard
                if "claude" in provider
                else cls
            )
        return object.__new__(cls)

    def __init__(self, interface):
        self._interface = interface
        self.attempted = False
        # Proposal observation needs metadata, never the provider's nested clients.
        self.base_args = {"model": "unknown"}

    def __getattr__(self, name):
        value = getattr(self._interface, name)
        if callable(value):

            def denied(*args, **kwargs):
                self.attempted = True
                raise GenerationDenied("generation_prohibited")

            return denied
        # Never expose nested clients that could bypass the interface guard.
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise GenerationDenied("interface_attribute_prohibited")


class GeminiGenerationGuard(GenerationGuard):
    pass


class ClaudeGenerationGuard(GenerationGuard):
    pass
