"""Prompt construction for the five context-injection strategies (Fig. 9)."""

import json
from functools import lru_cache
from pathlib import Path

from ape.utils import INPUT_DIR, file_stem, read_json, read_jsonl

CONTEXTS = ("seg", "seq", "rel", "exp", "full")
TEMPLATE_PATH = Path(__file__).with_name("templates.json")
OUTPUT_INSTRUCTION = "\n\nMUST output in this format:\n<pe>{tgt_lang} corrected sentence only</pe>"


@lru_cache(maxsize=None)
def _templates() -> dict:
    return json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _full_docs(collection_id: str, lang_pair: str) -> tuple[dict, dict]:
    stem = file_stem(collection_id, lang_pair)
    src = read_json(INPUT_DIR / "doc-full" / f"{stem}_src.json")  # {doc_id: text}
    tgt = read_json(INPUT_DIR / "doc-full" / f"{stem}_tgt.json")  # {doc_id: {system: text}}
    return src, tgt


@lru_cache(maxsize=None)
def _exp_annotations(collection_id: str) -> dict:
    # Annotations are source-side, so one file serves every language pair.
    records = read_jsonl(INPUT_DIR / "doc-exp" / f"{collection_id}.jsonl")
    return {k: v for rec in records for k, v in rec.items()}


def format_doc_info(ann: dict) -> str:
    """Render an APE_exp annotation as the [Document Information] block."""
    lines = []
    gd = ann.get("genre_and_domain", {})
    if gd.get("text_type"):
        lines.append(f"Text type: {gd['text_type']}")
    if gd.get("domain"):
        lines.append(f"Domain: {gd['domain']}")

    pr = ann.get("participant_relationships", {})
    if pr.get("identified") and pr.get("participants"):
        lines.append("Participants:")
        for p in pr["participants"]:
            role, desc = p.get("role", "unspecified"), p.get("description", "")
            lines.append(f"  - {role}: {desc}" if desc else f"  - {role}")
        if pr.get("relationship_type"):
            lines.append(f"Relationship: {pr['relationship_type']}")
    else:
        lines.append("Participants: not identifiable")

    rf = ann.get("register_and_formality", {})
    if rf.get("overall_register"):
        lines.append(f"Register: {rf['overall_register']}")
    if rf.get("notes"):
        lines.append(f"Register notes: {rf['notes']}")
    return "\n".join(lines)


def build_prompt(entry: dict, context: str, lang_pair: str) -> tuple[str, str]:
    """Return (system, user) for one segment."""
    if context not in CONTEXTS:
        raise ValueError(f"Unsupported context: {context}")

    fields = {k: entry[k] for k in ("src_lang", "tgt_lang", "src_seg", "tgt_seg")}

    if context == "full":
        src_docs, tgt_docs = _full_docs(entry["collection_id"], lang_pair)
        fields["src_doc"] = src_docs[entry["doc_id"]]
        # Draft translation of the document by the same system as tgt_seg.
        fields["tgt_doc"] = tgt_docs[entry["doc_id"]][entry["system"]]
    elif context == "exp":
        ann = _exp_annotations(entry["collection_id"])[entry["doc_id"]]
        fields["doc_summary"] = format_doc_info(ann)
    elif context in ("seq", "rel"):
        fields["src_ctx"] = entry[context]["src_ctx"]
        fields["tgt_ctx"] = entry[context]["tgt_ctx"]

    template = _templates()[f"ape-{context}"]
    system = template["system"].format(tgt_lang=entry["tgt_lang"])
    user = template["user"].format(**fields) + OUTPUT_INSTRUCTION.format(tgt_lang=entry["tgt_lang"])
    return system, user
