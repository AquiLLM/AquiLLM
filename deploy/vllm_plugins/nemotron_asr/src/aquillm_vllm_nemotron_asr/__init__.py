from .compat import install_compatibility_hook
from .constants import ARCHITECTURE, MODEL_CLASS


def register() -> None:
    """Register the Nemotron ASR architecture with vLLM's model registry."""
    install_compatibility_hook()
    from vllm.model_executor.models.registry import ModelRegistry

    existing_model = ModelRegistry.models.get(ARCHITECTURE)
    if existing_model is not None:
        module_name, class_name = MODEL_CLASS.split(":", maxsplit=1)
        existing_module_name = getattr(existing_model, "module_name", None)
        existing_class_name = getattr(existing_model, "class_name", None)
        if (existing_module_name, existing_class_name) == (module_name, class_name):
            return

        existing_model_class = getattr(existing_model, "model_cls", None)
        if isinstance(existing_model_class, type):
            existing_registration = (
                f"{existing_model_class.__module__}.{existing_model_class.__qualname__}"
            )
        elif isinstance(existing_module_name, str) and isinstance(
            existing_class_name, str
        ):
            existing_registration = f"{existing_module_name}:{existing_class_name}"
        else:
            existing_registration = type(existing_model).__qualname__

        raise RuntimeError(
            f"{ARCHITECTURE} is already registered to "
            f"{existing_registration}; cannot register {MODEL_CLASS}"
        )

    ModelRegistry.register_model(ARCHITECTURE, MODEL_CLASS)
