"""Table 7: inference overhead per model and strategy, from the `usage`
field of the APE outputs.

Usage:
    python -m analysis.overhead --lang_pair en-ko_KR --out overhead_summary.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("data/output")
MODELS = {
    "deepseek-v4-pro": ("DeepSeek-V4", True),
    "gpt-5.4-2026-03-05": ("GPT-5.4", True),
    "gemini-3-flash": ("Gemini-3-flash", True),
    "gemma-4-31b-it": ("Gemma-4-31B", False),
    "hcx-seed-thinking-32b": ("HCX-32B", False),
    "qwen3.5-27b": ("Qwen3.5-27B", False),
}
STRATEGIES = ["seg", "seq", "rel", "exp", "full"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection_id", default="wmt24pp")
    ap.add_argument("--lang_pair", default="en-ko_KR")
    ap.add_argument("--out", default="overhead_summary.csv")
    args = ap.parse_args()

    rows = []
    for model, (name, is_api) in MODELS.items():
        for strategy in STRATEGIES:
            path = OUTPUT_DIR / f"doc-{strategy}" / f"{args.collection_id}_{args.lang_pair}_{model}.jsonl"
            if not path.exists():
                continue
            usage = pd.DataFrame([json.loads(l)["usage"] for l in open(path, encoding="utf-8")])
            if usage.empty:
                continue
            lat = usage["latency_sec"].dropna()
            rows.append({
                "model": name, "strategy": strategy, "n_segs": len(usage),
                "input_token_mean": usage["input_token"].mean().round(),
                "input_token_std": usage["input_token"].std(),
                "output_token_mean": usage["output_token"].mean().round(),
                "output_token_std": usage["output_token"].std(),
                "latency_mean": round(lat.mean(), 2) if len(lat) else None,
                "latency_std": round(lat.std(), 2) if len(lat) else None,
                "is_api": is_api,
            })

    df = pd.DataFrame(rows)
    base = df[df.strategy == "seg"].set_index("model")
    df["input_token_ratio"] = df.input_token_mean / df.model.map(base.input_token_mean)
    df["latency_ratio"] = df.latency_mean / df.model.map(base.latency_mean)
    df.sort_values(["model", "strategy"]).to_csv(args.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
