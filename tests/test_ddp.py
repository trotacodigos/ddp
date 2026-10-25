"""Checks the worked example in Table 1 of the paper (no NLP model needed)."""

from ddp.core import aggregate_to_segments, compute_discourse_dependency, compute_distances

SENTENCES = [
    "John arrived.",
    "He looked tired.",
    "The manager greeted him.",
    "John smiled.",
    "She offered him coffee.",
]
NER = {0: ["John"], 3: ["John"]}
PRON = {1: ["He"], 2: ["him"], 4: ["She", "him"]}
NOUN = {2: ["manager"]}


def test_table1_example():
    sent_dds = compute_distances(SENTENCES, list(range(5)), NER, PRON, NOUN)
    assert [s.ddp for s in sent_dds] == [0, 1, 2, 3, 4]
    assert {a.text: a.dist for a in sent_dds[4].anchors} == {"She": 2, "him": 4}


def test_segment_takes_max_over_sentences():
    # Segment 0 = sentences 0-1, segment 1 = sentences 2-4.
    sent_dds = compute_distances(SENTENCES, [0, 0, 1, 1, 1], NER, PRON, NOUN)
    seg_dds = aggregate_to_segments(["s0", "s1"], sent_dds)
    assert [s.ddp for s in seg_dds] == [1, 4]


class _ManualExtractor:
    def split_into_sentences(self, segments):
        return segments, list(range(len(segments)))

    def extract_all(self, sentences):
        return NER, PRON, NOUN


def test_document_pipeline():
    doc = compute_discourse_dependency("example", SENTENCES, _ManualExtractor())
    assert [s.ddp for s in doc.seg_dds] == [0, 1, 2, 3, 4]
    assert doc.doc_ddp_max == 4
    assert doc.doc_ddp_mean == 2.5
