"""The six models evaluated in the paper (Table 2).

Open-weight models are served locally with vLLM through an
OpenAI-compatible endpoint (default http://localhost:8000/v1).
"""

from openai import OpenAI

from models.base import BaseModel
from models.openai_compat import OpenAIChatModel, OpenAIModel
from models.sampling import (
    DeepseekParams,
    HyperClovaXParams,
    OpenAIChatParams,
    OpenAIParams,
    QwenParams,
)
from models.tools import extract_token_usage, get_keys, strip_thinking


class GPT5_4(OpenAIModel):
    def __init__(self, sampling_params=None):
        super().__init__(
            "gpt-5.4-2026-03-05", "gpt-5.4-2026-03-05",
            sampling_params or OpenAIParams(thinking=True),
        )


class DeepSeekV4Pro(OpenAIChatModel):
    def __init__(self, sampling_params=None):
        super().__init__(
            "deepseek-v4-pro", "deepseek-v4-pro",
            endpoint="https://api.deepseek.com/chat/completions",
            sampling_params=sampling_params or DeepseekParams(thinking=False),
            strip_thinking=True,
        )

    def _headers(self):
        return {"Authorization": f"Bearer {get_keys('DEEPSEEK_API_KEYS')[0]}"}


class Gemma4(OpenAIChatModel):
    # https://huggingface.co/google/gemma-4-31B-it
    def __init__(self, endpoint="http://localhost:8000/v1/chat/completions", sampling_params=None):
        super().__init__(
            "gemma-4-31b-it", "google/gemma-4-31B-it", endpoint=endpoint,
            sampling_params=sampling_params or OpenAIChatParams(),
        )


class Qwen3_5(OpenAIChatModel):
    # https://huggingface.co/Qwen/Qwen3.5-27B
    def __init__(self, endpoint="http://localhost:8000/v1/chat/completions", sampling_params=None):
        super().__init__(
            "qwen3.5-27b", "qwen3.5-27b", endpoint=endpoint,
            sampling_params=sampling_params or QwenParams(),
        )

    def _extra_payload(self):
        if getattr(self.sampling_params, "thinking", False):
            return {}
        return {"chat_template_kwargs": {"enable_thinking": False}}


class HyperClovaX(BaseModel):
    # https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B
    def __init__(self, endpoint="http://localhost:8000/v1", sampling_params=None):
        super().__init__(
            "hcx-seed-thinking-32b", "hcx-seed-thinking-32b",
            sampling_params or HyperClovaXParams(thinking=False),
        )
        self.client = OpenAI(base_url=endpoint, api_key=get_keys("OPENAI_API_KEYS")[0])

    def _complete(self, system, user):
        # The chat template has no system role.
        merged = f"{system}\n{user}" if system else user
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": merged}],
            **self.sampling_params.to_kwargs(),
        )
        content = strip_thinking(resp.choices[0].message.content or "")
        return (content.strip(), *extract_token_usage(getattr(resp, "usage", None)))
