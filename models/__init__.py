"""Model registry. Keys match the model names used in output file names."""

from importlib import import_module

_REGISTRY = {
    "gpt-5.4-2026-03-05": ("models.providers", "GPT5_4"),
    "gemini-3-flash": ("models.gemini", "Gemini3Flash"),
    "deepseek-v4-pro": ("models.providers", "DeepSeekV4Pro"),
    "gemma-4-31b-it": ("models.providers", "Gemma4"),
    "hcx-seed-thinking-32b": ("models.providers", "HyperClovaX"),
    "qwen3.5-27b": ("models.providers", "Qwen3_5"),
}

MODEL_NAMES = list(_REGISTRY)


def get_model(name: str):
    """Return the model class registered under `name` (imported lazily so
    that only the SDK of the selected provider needs to be installed)."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_NAMES)}")
    module, cls = _REGISTRY[name]
    return getattr(import_module(module), cls)
