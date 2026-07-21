from .constants import ARCHITECTURE, MODEL_CLASS


def register() -> None:
    """Register the Nemotron ASR architecture with vLLM's model registry."""
    from vllm.model_executor.models.registry import ModelRegistry

    existing_model = ModelRegistry.models.get(ARCHITECTURE)
    if existing_model is not None:
        module_name, class_name = MODEL_CLASS.split(":", maxsplit=1)
        if (
            existing_model.module_name == module_name
            and existing_model.class_name == class_name
        ):
            return
        raise RuntimeError(
            f"{ARCHITECTURE} is already registered to "
            f"{existing_model.module_name}:{existing_model.class_name}"
        )

    ModelRegistry.register_model(ARCHITECTURE, MODEL_CLASS)
