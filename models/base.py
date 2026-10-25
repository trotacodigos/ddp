import copy
import csv
import dataclasses
import os
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from models.sampling import SamplingParams
from models.tools import rough_token_count


class BaseModel(ABC):
    """Common interface. Subclasses implement `_complete(system, user)`."""

    BATCH_WORKERS: int = 32

    def __init__(
        self,
        name: str,
        model_id: str,
        sampling_params: dict | SamplingParams | None,
        log_path: str | None = "data/output/inference_log.csv",
    ):
        self.name = name
        self.model_id = model_id
        self.sampling_params = (
            sampling_params
            if isinstance(sampling_params, SamplingParams)
            else SamplingParams.from_dict(sampling_params or {})
        )
        self.log_path = log_path
        self._csv_lock = threading.Lock()

    @abstractmethod
    def _complete(self, system: str, user: str) -> tuple[str, int | None, int | None]:
        """Return (text, input_tokens, output_tokens)."""

    def generate(self, system: str, user: str) -> tuple[str, dict]:
        t0 = time.perf_counter()
        text, in_token, out_token = self._complete(system, user)
        latency = time.perf_counter() - t0

        in_token = rough_token_count(user) if in_token is None else in_token
        out_token = rough_token_count(text) if out_token is None else out_token
        metrics = {
            "model_name": self.name,
            "input_token": int(in_token),
            "output_token": int(out_token),
            "latency_sec": round(latency, 4),
            "tps": round(out_token / latency, 2) if latency > 0 else 0.0,
        }
        self._log(metrics)
        return text, metrics

    def generate_batch(self, prompts: list[tuple[str, str]]) -> list[tuple[str, dict] | None]:
        """Parallel `generate`. The result list is aligned with `prompts`;
        failed items are None."""
        results: list[tuple[str, dict] | None] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=self.BATCH_WORKERS) as ex:
            futures = {ex.submit(self.generate, s, u): i for i, (s, u) in enumerate(prompts)}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=self.name):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    print(f"[{self.name}] index {idx} failed: {e}")
        return results

    def _log(self, metrics: dict):
        if not self.log_path:
            return
        with self._csv_lock:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            new = not os.path.isfile(self.log_path)
            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(metrics))
                if new:
                    writer.writeheader()
                writer.writerow(metrics)

    def with_sampling(self, **overrides):
        """Copy of the model with some sampling parameters changed."""
        new = copy.copy(self)
        new.sampling_params = dataclasses.replace(self.sampling_params, **overrides)
        return new
