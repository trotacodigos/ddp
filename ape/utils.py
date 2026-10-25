import json
import re
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("data")
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

LANG_NAMES = {"en": "English", "ko_KR": "Korean", "zh_CN": "Chinese"}
LANG_CODES = {v: k for k, v in LANG_NAMES.items()}


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path, records) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def file_stem(collection_id: str, lang_pair: str) -> str:
    """('wmt24pp', 'en-ko_KR') -> 'wmt24pp_enko_KR'"""
    src, tgt = lang_pair.split("-", 1)
    return f"{collection_id}_{LANG_CODES.get(src, src)}{LANG_CODES.get(tgt, tgt)}"


def segment_key(rec: dict) -> tuple:
    """Identifies one draft translation. seg_id alone is not unique because
    every system translates the same segment (and WMT24++ has two
    'Gemini-1' outputs per segment)."""
    return (str(rec["doc_id"]), str(rec["seg_id"]), rec.get("system"), rec["tgt_seg"])


def group_documents(entries: list[dict]) -> dict[tuple, list[dict]]:
    """Group segment lines into documents: {(system, variant, doc_id): [lines]}.

    A system may contribute more than one translation of the same segment
    (WMT24++ has two 'Gemini-1' outputs). The n-th occurrence of a segment
    goes to variant n, so each document holds one translation per segment.
    """
    seen: dict[tuple, int] = defaultdict(int)
    docs: dict[tuple, list[dict]] = defaultdict(list)
    for e in entries:
        k = (e.get("system"), e["doc_id"], str(e["seg_id"]))
        variant = seen[k]
        seen[k] += 1
        docs[(e.get("system"), variant, e["doc_id"])].append(e)
    for segs in docs.values():
        segs.sort(key=lambda x: int(x["seg_id"]))
    return dict(docs)


_PE = re.compile(r"<pe>(.*?)(?:</pe>|$)", re.S)


def extract_pe(text: str) -> str:
    """Return the text inside <pe>...</pe>; falls back to the raw text."""
    m = _PE.search(text or "")
    return (m.group(1) if m else (text or "")).replace("</pe>", "").strip()
