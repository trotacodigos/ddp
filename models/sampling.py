"""Per-provider sampling parameters.

All runs use temperature=1.0, top_p=0.95, seed=42 (Section 4.2).
`to_kwargs()` returns the request arguments expected by each provider.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 1024
    stop: Optional[List[str]] = None
    seed: int = 42

    def to_kwargs(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, dic: dict):
        return cls(**dic)


@dataclass
class ThinkingMixin:
    thinking: bool = False

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_kwargs()
        kwargs.pop("thinking", None)
        if self.thinking:
            kwargs["max_tokens"] = 32_768
            kwargs.pop("temperature", None)
        return kwargs


@dataclass
class OpenAIParams(ThinkingMixin, SamplingParams):
    """OpenAI Chat Completions (SDK)."""

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_kwargs()
        if self.thinking:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        return kwargs


@dataclass
class OpenAIChatParams(SamplingParams):
    """OpenAI-compatible HTTP endpoints served with vLLM."""
    repetition_penalty: Optional[float] = None

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_kwargs()
        kwargs["do_sample"] = (self.temperature or 0.0) > 0.0
        return kwargs


@dataclass
class QwenParams(ThinkingMixin, OpenAIChatParams):
    min_p: float = 0.0
    presence_penalty: float = 1.5

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_kwargs()
        if self.thinking:
            kwargs.update(temperature=1.0, top_p=0.95)
        return kwargs


@dataclass
class HyperClovaXParams(ThinkingMixin, OpenAIChatParams):
    thinking_token_budget: int = 4000

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_kwargs()
        kwargs.pop("thinking_token_budget", None)
        kwargs.pop("do_sample", None)
        if self.thinking:
            kwargs["extra_body"] = {
                "thinking_token_budget": self.thinking_token_budget,
                "chat_template_kwargs": {"thinking": True},
            }
        else:
            kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": False}}
        return kwargs


@dataclass
class DeepseekParams(SamplingParams):
    """DeepSeek API (https://api-docs.deepseek.com/)."""
    thinking: bool = False

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_kwargs()
        kwargs.pop("thinking", None)
        # Merged into the top level of the request body by OpenAIChatModel.
        kwargs["extra_body"] = {"thinking": self.thinking, "reasoning_effort": "medium"}
        return kwargs


@dataclass
class Gemini3Params(SamplingParams):
    """Gemini 3.x (thinking is controlled by thinking_level)."""
    thinking_level: str = "medium"  # minimal | low | medium | high

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_kwargs()
        kwargs["max_output_tokens"] = kwargs.pop("max_tokens")
        return kwargs
