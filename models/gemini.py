import concurrent.futures
import time

from google import genai
from google.genai import types

from models.base import BaseModel
from models.sampling import Gemini3Params
from models.tools import get_keys

_TERMINAL_STATES = {
    types.JobState.JOB_STATE_SUCCEEDED,
    types.JobState.JOB_STATE_FAILED,
    types.JobState.JOB_STATE_CANCELLED,
    types.JobState.JOB_STATE_EXPIRED,
    types.JobState.JOB_STATE_PARTIALLY_SUCCEEDED,
}


def _build_config(system: str | None, sampling_kwargs: dict) -> types.GenerateContentConfig:
    kwargs = dict(sampling_kwargs)
    thinking_level = kwargs.pop("thinking_level", None)
    thinking_config = types.ThinkingConfig(thinking_level=thinking_level) if thinking_level else None
    return types.GenerateContentConfig(system_instruction=system, thinking_config=thinking_config, **kwargs)


class GeminiModel(BaseModel):
    """Google Gen AI SDK.

    By default `generate_batch` sends real-time requests in parallel, which
    keeps every output tied to its prompt. `use_batch_api=True` switches to
    the Batch API with inlined requests; results are then mapped back by
    position, so check the alignment of the outputs before using them.
    """

    POLL_INTERVAL: int = 30
    BATCH_CHUNK_SIZE: int = 500
    MAX_CONCURRENT_BATCHES: int = 8

    def __init__(self, name: str, model_id: str, sampling_params=None, use_batch_api: bool = False):
        super().__init__(name, model_id, sampling_params or Gemini3Params())
        self.client = genai.Client(api_key=get_keys("GEMINI_API_KEYS")[0])
        self.use_batch_api = use_batch_api

    def _complete(self, system, user):
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=user,
            config=_build_config(system, self.sampling_params.to_kwargs()),
        )
        usage = response.usage_metadata
        return (
            (response.text or "").strip(),
            getattr(usage, "prompt_token_count", None),
            getattr(usage, "candidates_token_count", None),
        )

    def generate_batch(self, prompts):
        if not self.use_batch_api:
            return super().generate_batch(prompts)
        chunks = [(i, prompts[i:i + self.BATCH_CHUNK_SIZE]) for i in range(0, len(prompts), self.BATCH_CHUNK_SIZE)]
        by_offset = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_CONCURRENT_BATCHES) as ex:
            futures = {ex.submit(self._batch_api, c): o for o, c in chunks}
            for f in concurrent.futures.as_completed(futures):
                try:
                    by_offset[futures[f]] = f.result()
                except Exception as e:
                    print(f"[Gemini batch] chunk at {futures[f]} failed: {e!r}")
                    by_offset[futures[f]] = [None] * len(dict(chunks)[futures[f]])
        return [r for o, _ in chunks for r in by_offset[o]]

    def _batch_api(self, prompts):
        requests = [
            types.InlinedRequest(
                model=self.model_id,
                contents=user,
                config=_build_config(system, self.sampling_params.to_kwargs()),
            )
            for system, user in prompts
        ]
        batch = self.client.batches.create(model=self.model_id, src=requests)
        print(f"[Gemini batch] submitted {len(prompts)} requests -> {batch.name}")
        while True:
            time.sleep(self.POLL_INTERVAL)
            batch = self.client.batches.get(name=batch.name)
            if batch.state in _TERMINAL_STATES:
                break

        results = [None] * len(prompts)
        ok = (types.JobState.JOB_STATE_SUCCEEDED, types.JobState.JOB_STATE_PARTIALLY_SUCCEEDED)
        if batch.state not in ok or not batch.dest or not batch.dest.inlined_responses:
            print(f"[Gemini batch] ended with state={batch.state}")
            return results
        responses = batch.dest.inlined_responses
        if len(responses) != len(prompts):
            print(f"[Gemini batch] got {len(responses)} responses for {len(prompts)} prompts; discarding")
            return results

        for idx, inlined in enumerate(responses):
            if inlined.error:
                print(f"[Gemini batch] index {idx} failed: {inlined.error}")
                continue
            resp = inlined.response
            usage = resp.usage_metadata
            metrics = {
                "model_name": self.name,
                "input_token": int(getattr(usage, "prompt_token_count", 0) or 0),
                "output_token": int(getattr(usage, "candidates_token_count", 0) or 0),
                "latency_sec": None,
                "tps": None,
            }
            self._log(metrics)
            results[idx] = ((resp.text or "").strip(), metrics)
        return results


class Gemini3Flash(GeminiModel):
    def __init__(self, sampling_params=None, use_batch_api: bool = False):
        super().__init__("gemini-3-flash", "gemini-3-flash-preview", sampling_params, use_batch_api)
