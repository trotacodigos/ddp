import io
import json
import time

import requests
from openai import OpenAI

from models.base import BaseModel
from models.sampling import OpenAIChatParams, OpenAIParams
from models.tools import extract_token_usage, get_keys, strip_thinking


def _messages(system: str | None, user: str, merge_system: bool = False) -> list[dict]:
    if merge_system or system is None:
        return [{"role": "user", "content": f"{system}\n{user}" if system else user}]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class OpenAIModel(BaseModel):
    """OpenAI SDK. `generate_batch` uses the OpenAI Batch API."""

    POLL_INTERVAL: int = 30

    def __init__(self, name: str, model_id: str, sampling_params: OpenAIParams | dict | None = None):
        super().__init__(name, model_id, sampling_params or OpenAIParams())
        self.client = OpenAI(api_key=get_keys("OPENAI_API_KEYS")[0])

    def _complete(self, system, user):
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=_messages(system, user),
            **self.sampling_params.to_kwargs(),
        )
        text = (resp.choices[0].message.content or "").strip()
        return (text, *extract_token_usage(getattr(resp, "usage", None)))

    def generate_batch(self, prompts):
        body = "\n".join(
            json.dumps({
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_id,
                    "messages": _messages(system, user),
                    **self.sampling_params.to_kwargs(),
                },
            })
            for i, (system, user) in enumerate(prompts)
        ).encode("utf-8")

        file_obj = self.client.files.create(
            file=("batch_input.jsonl", io.BytesIO(body), "application/jsonl"), purpose="batch"
        )
        batch = self.client.batches.create(
            input_file_id=file_obj.id, endpoint="/v1/chat/completions", completion_window="24h"
        )
        print(f"[OpenAI batch] submitted {len(prompts)} requests -> {batch.id}")

        while True:
            time.sleep(self.POLL_INTERVAL)
            status = self.client.batches.retrieve(batch.id)
            c = status.request_counts
            print(f"[OpenAI batch] completed={c.completed} failed={c.failed} total={c.total}")
            if status.status in ("completed", "failed", "expired", "cancelled"):
                break

        results: list[tuple[str, dict] | None] = [None] * len(prompts)
        if status.status != "completed" or not status.output_file_id:
            print(f"[OpenAI batch] ended with status={status.status}")
            return results
        if status.error_file_id:
            for line in self.client.files.content(status.error_file_id).text.splitlines():
                print(f"[OpenAI batch] error: {line}")

        # Results are matched by custom_id, not by order.
        for line in self.client.files.content(status.output_file_id).text.splitlines():
            item = json.loads(line)
            idx = int(item["custom_id"])
            if item.get("error"):
                print(f"[OpenAI batch] index {idx} failed: {item['error']}")
                continue
            resp_body = item["response"]["body"]
            text = (resp_body["choices"][0]["message"]["content"] or "").strip()
            in_tok, out_tok = extract_token_usage(resp_body.get("usage", {}))
            metrics = {
                "model_name": self.name,
                "input_token": int(in_tok or 0),
                "output_token": int(out_tok or 0),
                "latency_sec": None,  # not exposed by the Batch API
                "tps": None,
            }
            self._log(metrics)
            results[idx] = (text, metrics)
        return results


class OpenAIChatModel(BaseModel):
    """Plain HTTP client for OpenAI-compatible endpoints (vLLM, DeepSeek)."""

    BATCH_WORKERS = 4

    def __init__(
        self,
        name: str,
        model_id: str,
        endpoint: str = "http://localhost:8000/v1/chat/completions",
        sampling_params: OpenAIChatParams | dict | None = None,
        strip_thinking: bool = False,
        merge_system_prompt: bool = False,
    ):
        super().__init__(name, model_id, sampling_params or OpenAIChatParams())
        self.endpoint = endpoint
        self.strip_thinking = strip_thinking
        self.merge_system_prompt = merge_system_prompt

    def _complete(self, system, user):
        kwargs = self.sampling_params.to_kwargs()
        # `extra_body` is an OpenAI-SDK convention; over raw HTTP its fields
        # belong at the top level of the request body.
        extra = kwargs.pop("extra_body", {})
        payload = {
            "model": self.model_id,
            "messages": _messages(system, user, self.merge_system_prompt),
            **kwargs,
            **extra,
            **self._extra_payload(),
        }
        response = requests.post(self.endpoint, json=payload, headers=self._headers(), timeout=3600)
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"] or ""
        if self.strip_thinking:
            content = strip_thinking(content)
        return (content.strip(), *extract_token_usage(data.get("usage", {})))

    def _extra_payload(self) -> dict:
        return {}

    def _headers(self) -> dict:
        return {}
