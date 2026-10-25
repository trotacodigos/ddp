"""Run automatic post-editing with one context-injection strategy.

Usage:
    python -m ape.inference --context seq --collection_id wmt24pp \
        --lang_pair en-ko_KR --model gpt-5.4-2026-03-05

Inputs (see README):
    seg / exp / full : data/input/original/{stem}.jsonl
    seq / rel        : data/input/doc-{seq,rel}/{stem}.jsonl
Output:
    data/output/doc-{context}/{collection_id}_{lang_pair}_{model}.jsonl

The run can be resumed: lines already present in the output are skipped.
"""

import argparse
import json

from tqdm import tqdm

from ape.prompter import CONTEXTS, build_prompt
from ape.utils import INPUT_DIR, OUTPUT_DIR, file_stem, read_jsonl, segment_key
from models import MODEL_NAMES, get_model

CHUNK = 500


def input_path(context: str, collection_id: str, lang_pair: str):
    sub = f"doc-{context}" if context in ("seq", "rel") else "original"
    return INPUT_DIR / sub / f"{file_stem(collection_id, lang_pair)}.jsonl"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", required=True, choices=CONTEXTS)
    ap.add_argument("--collection_id", required=True, choices=["wmt24pp", "wmt25"])
    ap.add_argument("--lang_pair", required=True, help="en-ko_KR or en-zh_CN")
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    args = ap.parse_args()

    entries = read_jsonl(input_path(args.context, args.collection_id, args.lang_pair))
    out_file = OUTPUT_DIR / f"doc-{args.context}" / f"{args.collection_id}_{args.lang_pair}_{args.model}.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            done = {segment_key(json.loads(l)) for l in f if l.strip()}

    items = []
    for entry in tqdm(entries, desc="Building prompts"):
        if segment_key(entry) in done:
            continue
        try:
            items.append((entry, *build_prompt(entry, args.context, args.lang_pair)))
        except Exception as e:
            print(f"Prompt error for doc_id={entry.get('doc_id')} seg_id={entry.get('seg_id')}: {e!r}")
    print(f"Total: {len(entries)}, done: {len(done)}, pending: {len(items)}")

    model = get_model(args.model)()
    with open(out_file, "a", encoding="utf-8") as f:
        for i in range(0, len(items), CHUNK):
            chunk = items[i:i + CHUNK]
            results = model.generate_batch([(system, user) for _, system, user in chunk])
            for (entry, _, _), result in zip(chunk, results):
                if result is None:
                    continue
                text, usage = result
                record = {
                    "collection_id": entry["collection_id"],
                    "doc_id": entry["doc_id"],
                    "seg_id": entry["seg_id"],
                    "system": entry.get("system"),
                    "tgt_seg": entry["tgt_seg"],
                    "mt_pe_seg": text,
                    "usage": {k: usage.get(k) for k in ("input_token", "output_token", "latency_sec")},
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[inference] chunk {i // CHUNK + 1} written to {out_file}")


if __name__ == "__main__":
    main()
