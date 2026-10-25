"""Anchor extractors: sentence splitting + NER / pronoun / head-noun tagging."""

from __future__ import annotations

from typing import Literal

import spacy

from ddp.core import is_pronoun

NER_LABELS = {"PERSON", "ORG", "GPE", "LOC"}
NOUN_DEPS = {"nsubj", "dobj", "attr", "nsubjpass", "pobj"}


class AnchorExtractor:
    """Interface shared by all backends."""

    def _split(self, text: str) -> list[str]:
        raise NotImplementedError

    def split_into_sentences(self, segments: list[str]) -> tuple[list[str], list[int]]:
        """Return (sentences, sent_to_seg).

        A segment with no detected sentence is kept as a single sentence so
        that sentence indices stay aligned with the document.
        """
        sentences, sent_to_seg = [], []
        for seg_idx, seg in enumerate(segments):
            sents = [s.strip() for s in self._split(seg) if s.strip()] or [seg]
            sentences.extend(sents)
            sent_to_seg.extend([seg_idx] * len(sents))
        return sentences, sent_to_seg

    def extract_all(self, sentences: list[str]) -> tuple[dict, dict, dict]:
        """Return (ner_per_sent, pron_per_sent, noun_per_sent)."""
        raise NotImplementedError


class SpacyExtractor(AnchorExtractor):
    """spaCy pipeline for every step (default: en_core_web_trf)."""

    def __init__(self, model: str = "en_core_web_trf"):
        self.nlp = spacy.load(model)

    def _split(self, text):
        return [s.text for s in self.nlp(text).sents]

    def extract_all(self, sentences):
        ner, pron, noun = {}, {}, {}
        for idx, sent in enumerate(sentences):
            doc = self.nlp(sent)
            ner[idx] = [e.text.strip() for e in doc.ents if e.label_ in NER_LABELS]
            pron[idx] = [
                t.text for t in doc
                if t.pos_ == "PRON" and is_pronoun(t.text)
                and not (t.text.lower() == "it" and t.dep_ == "expl")
            ]
            noun[idx] = [
                t.text for t in doc
                if t.pos_ == "NOUN" and t.dep_ in NOUN_DEPS and not t.ent_type_
            ]
        return ner, pron, noun


class FlairExtractor(AnchorExtractor):
    """Flair NER + Flair UPOS (ablation in Appendix A.3).

    Flair has no sentence splitter or dependency parser, so sentence
    splitting and the head-noun filter use spaCy en_core_web_sm.
    Expletive "it" cannot be filtered without a parse and is kept.
    """

    def __init__(
        self,
        ner_model: str = "flair/ner-english-ontonotes-large",
        pos_model: str = "flair/upos-english",
        sent_splitter: Literal["spacy", "nltk"] = "spacy",
    ):
        from flair.models import SequenceTagger

        self.ner_tagger = SequenceTagger.load(ner_model)
        self.pos_tagger = SequenceTagger.load(pos_model)
        self.sent_splitter = sent_splitter

        if sent_splitter == "spacy":
            self._sent_nlp = spacy.load(
                "en_core_web_sm",
                disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
            )
            if "senter" in self._sent_nlp.component_names:
                if "senter" not in self._sent_nlp.pipe_names:
                    self._sent_nlp.enable_pipe("senter")
            else:
                self._sent_nlp.add_pipe("senter")
        else:
            import nltk
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)

        self._dep_nlp = spacy.load("en_core_web_sm", disable=["ner"])

    def _split(self, text):
        if self.sent_splitter == "spacy":
            return [s.text for s in self._sent_nlp(text).sents]
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)

    def extract_all(self, sentences):
        from flair.data import Sentence

        ner, pron, noun = {}, {}, {}
        for idx, sent in enumerate(sentences):
            fs = Sentence(sent)
            self.ner_tagger.predict(fs)
            ner[idx] = [s.text.strip() for s in fs.get_spans("ner") if s.tag in NER_LABELS]

            fs_pos = Sentence(sent)
            self.pos_tagger.predict(fs_pos)
            pron[idx] = [
                t.text for t in fs_pos
                if t.get_label("upos").value == "PRON" and is_pronoun(t.text)
            ]

            ner_lower = {t.lower() for t in ner[idx]}
            noun[idx] = [
                t.text for t in self._dep_nlp(sent)
                if t.pos_ == "NOUN" and t.dep_ in NOUN_DEPS and t.text.lower() not in ner_lower
            ]
        return ner, pron, noun


def get_extractor(backend: str) -> AnchorExtractor:
    if backend == "spacy":
        return SpacyExtractor("en_core_web_trf")
    if backend == "spacy-sm":
        return SpacyExtractor("en_core_web_sm")
    if backend == "flair":
        return FlairExtractor()
    raise ValueError(f"Unknown backend: {backend}")
