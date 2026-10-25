"""Annotate a WMT-style JSONL file with segment-level DDP.

DDP is a source-side property, so it is computed once per document from the
unique source segments and then attached to every line (every system).

Usage:
    python -m ddp.annotate \
        --input  data/input/original/wmt24pp_enko_KR.jsonl \
        --output data/anchor_extraction/ddp_spacy/wmt24pp.jsonl \
        --ddp_map data/anchor_extraction/wmt24pp_map.json \
        --backend spacy
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from statistics import mean

from tqdm import tqdm

from ddp.core import compute_discourse_dependency
from ddp.extractors import get_extractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _seg_key(seg_id):
    try:
        return (0, int(seg_id))
    except (TypeError, ValueError):
        return (1, str(seg_id))


def unique_documents(lines: list[dict]) -> dict[str, list[dict]]:
    """{doc_id: [one line per unique seg_id, in document order]}.

    Multiple systems (or several outputs of one system) share the same
    source segment; keeping one line per (doc_id, seg_id) prevents the
    document from being processed with duplicated sentences.
    """
    docs: dict[str, dict[str, dict]] = defaultdict(dict)
    for line in lines:
        docs[line["doc_id"]].setdefault(str(line["seg_id"]), line)
    return {
        doc_id: sorted(segs.values(), key=lambda x: _seg_key(x["seg_id"]))
        for doc_id, segs in docs.items()
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="JSONL with doc_id, seg_id, src_seg")
    ap.add_argument("--output", required=True, help="input lines with a DDP field added")
    ap.add_argument("--ddp_map", default=None, help="optional {doc_id: {seg_id: ddp}} JSON")
    ap.add_argument("--backend", choices=["spacy", "spacy-sm", "flair"], default="spacy")
    ap.add_argument("--field", default="ddp", help="name of the DDP field in the output")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    docs = unique_documents(lines)
    logger.info(f"{len(lines)} lines, {len(docs)} documents, "
                f"{sum(len(v) for v in docs.values())} unique segments")

    extractor = get_extractor(args.backend)
    ddp_map: dict[str, dict[str, int]] = {}
    for doc_id, segs in tqdm(docs.items(), desc="DDP"):
        doc = compute_discourse_dependency(doc_id, [s["src_seg"] for s in segs], extractor)
        ddp_map[doc_id] = {str(s["seg_id"]): d.ddp for s, d in zip(segs, doc.seg_dds)}

    if args.ddp_map:
        Path(args.ddp_map).parent.mkdir(parents=True, exist_ok=True)
        with open(args.ddp_map, "w", encoding="utf-8") as f:
            json.dump(ddp_map, f, ensure_ascii=False, indent=2)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for line in lines:
            out = {**line, args.field: ddp_map[line["doc_id"]][str(line["seg_id"])]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(lines)} lines to {args.output}")

    by_domain: dict[str, list[int]] = defaultdict(list)
    for doc_id, segs in docs.items():
        for s in segs:
            by_domain[s.get("domain", "unknown")].append(ddp_map[doc_id][str(s["seg_id"])])
    for domain, vals in sorted(by_domain.items()):
        zero = 100 * sum(v == 0 for v in vals) / len(vals)
        logger.info(f"{domain:<10} n={len(vals):5d} mean={mean(vals):6.1f} max={max(vals):4d} %=0={zero:5.1f}")


if __name__ == "__main__":
    main()
