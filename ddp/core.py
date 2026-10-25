"""Discourse dependency (DDP) computation.

Distances are computed over sentences. A segment's DDP is the maximum
sentence-level DDP among the sentences it contains (Section 3.2).

Rule 1 (named entities)
    d = 0 for a first mention, otherwise (current sentence - earliest prior
    sentence that mentions the same entity).

Rule 2 (pronouns)
    A pronoun is anchored to the most recent prior sentence that contains a
    named entity (or, as a fallback, a head noun). When the antecedent is a
    named entity, the entity's own distance is accumulated. When no
    antecedent exists, d = 1.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


PRONOUN_GROUPS: dict[str, str] = {
    "he": "masc", "him": "masc", "his": "masc", "himself": "masc",
    "she": "fem", "her": "fem", "hers": "fem", "herself": "fem",
    "it": "neut", "its": "neut", "itself": "neut",
    "they": "plur", "them": "plur", "their": "plur",
    "theirs": "plur", "themselves": "plur",
}


def get_pronoun_group(text: str) -> Optional[str]:
    return PRONOUN_GROUPS.get(text.lower().strip(".,!?;:"))


def is_pronoun(text: str) -> bool:
    return get_pronoun_group(text) is not None


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Anchor:
    text: str
    sent_idx: int
    seg_idx: int
    source: str  # "ner" or "pronoun"
    dist: int    # 0 = first mention


@dataclass
class SentenceDD:
    sent_idx: int
    seg_idx: int
    text: str
    anchors: list[Anchor] = field(default_factory=list)
    ddp: int = 0


@dataclass
class SegmentDD:
    seg_idx: int
    text: str
    sentences: list[SentenceDD] = field(default_factory=list)
    ddp: int = 0


@dataclass
class DocumentDD:
    doc_id: str
    segments: list[str]
    seg_dds: list[SegmentDD] = field(default_factory=list)
    doc_ddp_mean: float = 0.0
    doc_ddp_max: int = 0


# ── Distance computation ─────────────────────────────────────────────────────

def compute_distances(
    sentences: list[str],
    sent_to_seg: list[int],
    ner_per_sent: dict[int, list[str]],
    pron_per_sent: dict[int, list[str]],
    noun_per_sent: dict[int, list[str]],
) -> list[SentenceDD]:
    """Compute anchor distances for every sentence of one document."""
    ner_history: dict[str, list[int]] = defaultdict(list)
    ner_sent_order: list[int] = []   # sentences that contain a named entity
    noun_sent_order: list[int] = []  # sentences that contain a head noun
    pron_seen: set[str] = set()      # pronoun groups seen so far
    sent_dds: list[SentenceDD] = []

    for sent_idx, sent_text in enumerate(sentences):
        anchors: list[Anchor] = []
        seg_idx = sent_to_seg[sent_idx]

        if noun_per_sent.get(sent_idx) and sent_idx not in noun_sent_order:
            noun_sent_order.append(sent_idx)

        # Rule 1: named entities
        for ent_text in ner_per_sent.get(sent_idx, []):
            prior = ner_history[ent_text.lower()]
            dist = 0 if not prior else sent_idx - min(prior)
            anchors.append(Anchor(ent_text, sent_idx, seg_idx, "ner", dist))

        # Rule 2: pronouns
        for pron_text in pron_per_sent.get(sent_idx, []):
            group = get_pronoun_group(pron_text)
            if group is None:
                continue

            prior_ner = [s for s in ner_sent_order if s < sent_idx]
            prior_noun = [s for s in noun_sent_order if s < sent_idx]
            recent_ner = max(prior_ner) if prior_ner else None
            recent_noun = max(prior_noun) if prior_noun else None

            if group in pron_seen:
                # Same group seen before: chain through the most recent entity.
                if recent_ner is not None:
                    ner_dist = 0
                    for ent in ner_per_sent.get(recent_ner, []):
                        prev = [s for s in ner_history[ent.lower()] if s < recent_ner]
                        if prev:
                            ner_dist = max(ner_dist, recent_ner - min(prev))
                    dist = (sent_idx - recent_ner) + ner_dist
                else:
                    dist = 1
            else:
                # First pronoun of this group. Feminine pronouns prefer a head
                # noun (e.g. "the manager"), other groups prefer a named entity.
                order = (recent_noun, recent_ner) if group == "fem" else (recent_ner, recent_noun)
                antecedent = next((s for s in order if s is not None), None)
                dist = sent_idx - antecedent if antecedent is not None else 1

            anchors.append(Anchor(pron_text, sent_idx, seg_idx, "pronoun", dist))
            pron_seen.add(group)

        # Update entity history after the anchors of this sentence are scored,
        # so that an entity cannot refer to itself.
        for ent_text in ner_per_sent.get(sent_idx, []):
            ner_history[ent_text.lower()].append(sent_idx)
            if sent_idx not in ner_sent_order:
                ner_sent_order.append(sent_idx)

        ddp = max((a.dist for a in anchors), default=0)
        sent_dds.append(SentenceDD(sent_idx, seg_idx, sent_text, anchors, ddp))

    return sent_dds


def aggregate_to_segments(segments: list[str], sent_dds: list[SentenceDD]) -> list[SegmentDD]:
    """Segment-level DDP = max over the segment's sentences (Eq. 4)."""
    seg_dds = [SegmentDD(seg_idx=i, text=seg) for i, seg in enumerate(segments)]
    for sd in sent_dds:
        seg_dds[sd.seg_idx].sentences.append(sd)
    for seg in seg_dds:
        seg.ddp = max((s.ddp for s in seg.sentences), default=0)
    return seg_dds


def compute_discourse_dependency(doc_id: str, segments: list[str], extractor) -> DocumentDD:
    """Run the full pipeline on one document.

    `segments` must be the document's unique source segments in order.
    `extractor` is an object implementing `split_into_sentences` and
    `extract_all` (see ddp/extractors.py).
    """
    sentences, sent_to_seg = extractor.split_into_sentences(segments)
    ner, pron, noun = extractor.extract_all(sentences)
    sent_dds = compute_distances(sentences, sent_to_seg, ner, pron, noun)
    seg_dds = aggregate_to_segments(segments, sent_dds)

    nonzero = [s.ddp for s in seg_dds if s.ddp > 0]
    return DocumentDD(
        doc_id=doc_id,
        segments=segments,
        seg_dds=seg_dds,
        doc_ddp_mean=sum(nonzero) / len(nonzero) if nonzero else 0.0,
        doc_ddp_max=max(nonzero, default=0),
    )
