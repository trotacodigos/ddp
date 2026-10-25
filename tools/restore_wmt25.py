"""Restore the WMT25 texts that are not redistributed in this repository.

The WMT25 files here keep identifiers, DDP values and context indices only.
This script fills in the source, reference and system translations from the
official WMT25 General MT release and writes the files in place:

    data/anchor_extraction/ddp_{spacy,flair}/wmt25.jsonl   + src_seg, tgt_seg, ref_seg
    data/input/doc-seq/wmt25_{lp}.jsonl                    + src_seg, tgt_seg, seq.{src,tgt}_ctx
    data/input/doc-rel/wmt25_{lp}.jsonl                    + src_seg, tgt_seg, rel.{src,tgt}_ctx
    data/input/doc-full/wmt25_{lp}_{src,tgt}.json          (created from wmt25_{lp}_index.json)

A segment is a paragraph of the official document (`src_text` split on a
blank line); `para_idx` gives its position.

Usage:
    git clone https://github.com/wmt-conference/wmt25-general-mt
    python -m tools.restore_wmt25 --wmt25_dir wmt25-general-mt
"""

import argparse
import json
from pathlib import Path

DATA = Path("data")
LANG_PAIRS = {"enko_KR": "ko_KR", "enzh_CN": "zh_CN"}
# System names in this repository that differ from the official file names.
SYSTEM_NAMES = {
    "GPT-4": "GPT-4.1",
    "Gemini-2": "Gemini-2.5-Pro",
    "Llama-3": "Llama-3.1-8B",
    "Qwen2": "Qwen2.5-7B",
}
PARA_SEP = "\n\n"


class WMT25:
    def __init__(self, root: Path):
        data = root / "data"
        self.docs = {}  # (tgt_lang, doc_name) -> record
        for line in open(data / "wmt25-genmt.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["src_lang"] == "en" and r["tgt_lang"] in LANG_PAIRS.values():
                self.docs[(r["tgt_lang"], r["doc_id"].split("_#_")[2])] = r
        self.hyps = {}  # (tgt_lang, doc_name, system) -> document translation
        for f in sorted((data / "systems").glob("*.jsonl")):
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                lp, _, name = r["doc_id"].split("_#_")[:3]
                tgt = lp.split("-", 1)[1]
                if lp.startswith("en-") and tgt in LANG_PAIRS.values():
                    self.hyps[(tgt, name, f.stem)] = r["hypothesis"]

    def src(self, tgt, doc):
        return self.docs[(tgt, doc)]["src_text"]

    def ref(self, tgt, doc):
        return self.docs[(tgt, doc)]["refs"]["refA"]["ref"]

    def hyp(self, tgt, doc, system):
        return self.hyps[(tgt, doc, SYSTEM_NAMES.get(system, system))]

    def paras(self, text):
        return text.split(PARA_SEP)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def restore_segments(path: Path, wmt: WMT25, tgt: str, ctx_key: str | None, with_ref: bool):
    if not path.exists():
        return
    rows = read_jsonl(path)
    for r in rows:
        src = wmt.paras(wmt.src(tgt, r["doc_id"]))
        hyp = wmt.paras(wmt.hyp(tgt, r["doc_id"], r["system"]))
        i = r["para_idx"]
        r["src_seg"], r["tgt_seg"] = src[i], hyp[i]
        if with_ref:
            r["ref_seg"] = wmt.paras(wmt.ref(tgt, r["doc_id"]))[i]
        if ctx_key:
            idx = r[ctx_key]["idx"]
            r[ctx_key] = {
                "src_ctx": "\n".join(src[j] for j in idx),
                "tgt_ctx": "\n".join(hyp[j] for j in idx),
            }
    write_jsonl(path, rows)
    print(f"restored {path} ({len(rows)} lines)")


def restore_full_docs(wmt: WMT25, stem: str, tgt: str):
    out = DATA / "input" / "doc-full"
    index = out / f"wmt25_{stem}_index.json"  # {doc_id: [systems]}
    if not index.exists():
        return
    systems = json.load(open(index, encoding="utf-8"))
    src = {d: wmt.src(tgt, d) for d in systems}
    tgt_docs = {d: {s: wmt.hyp(tgt, d, s) for s in ss} for d, ss in systems.items()}
    json.dump(src, open(out / f"wmt25_{stem}_src.json", "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(tgt_docs, open(out / f"wmt25_{stem}_tgt.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"restored {out}/wmt25_{stem}_{{src,tgt}}.json ({len(src)} documents)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wmt25_dir", required=True, type=Path, help="clone of wmt-conference/wmt25-general-mt")
    args = ap.parse_args()

    wmt = WMT25(args.wmt25_dir)
    for backend in ("ddp_spacy", "ddp_flair"):
        restore_segments(DATA / "anchor_extraction" / backend / "wmt25.jsonl", wmt, "ko_KR", None, True)
    for stem, tgt in LANG_PAIRS.items():
        restore_segments(DATA / "input" / "doc-seq" / f"wmt25_{stem}.jsonl", wmt, tgt, "seq", False)
        restore_segments(DATA / "input" / "doc-rel" / f"wmt25_{stem}.jsonl", wmt, tgt, "rel", False)
        restore_full_docs(wmt, stem, tgt)


if __name__ == "__main__":
    main()
