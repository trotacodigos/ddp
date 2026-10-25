"""APE_seq context: the k segment pairs immediately preceding each segment.

Segments with fewer than k predecessors are skipped.

Usage:
    python -m ape.context.seq --collection_id wmt24pp --lang_pair en-ko_KR
"""

import argparse

from ape.utils import INPUT_DIR, file_stem, group_documents, read_jsonl, write_jsonl

K = 5


def sequential_context(doc: list[dict], k: int = K) -> list[dict]:
    out = []
    for i in range(k, len(doc)):
        prev = doc[i - k:i]
        out.append({
            **doc[i],
            "seq": {
                "src_ctx": "\n".join(s["src_seg"] for s in prev),
                "tgt_ctx": "\n".join(s["tgt_seg"] for s in prev),
            },
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection_id", required=True)
    ap.add_argument("--lang_pair", required=True)
    ap.add_argument("--k", type=int, default=K)
    args = ap.parse_args()

    stem = file_stem(args.collection_id, args.lang_pair)
    docs = group_documents(read_jsonl(INPUT_DIR / "original" / f"{stem}.jsonl"))
    records = [r for doc in docs.values() for r in sequential_context(doc, args.k)]

    out = INPUT_DIR / "doc-seq" / f"{stem}.jsonl"
    write_jsonl(out, records)
    print(f"Saved {len(records)} records to {out}")


if __name__ == "__main__":
    main()
