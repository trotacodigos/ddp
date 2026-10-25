# Discourse Dependency (DDP)

Code and data for **[Discourse Dependency: A Continuous Criterion for Translation Difficulty](https://arxiv.org/abs/2609.04959)**
Ahrii Kim, Chanjun Park, Seong-heum Kim (Soongsil University)

DDP measures how far back into its document a segment must look to resolve the
entities and pronouns it contains. It is computed from the English source only,
so it does not depend on any MT system or quality metric.

This repository contains

- the DDP implementation (`ddp/`),
- DDP annotations for WMT24++ and WMT25,
- the five context-injection strategies for LLM post-editing (APE) used in the paper (`ape/`, `models/`),
- APE outputs and automatic metric scores for WMT24++ En–Ko and En–Zh, and the recoverable part of the En–Ko human rankings (see [`data/README.md`](data/README.md#human-rankings--evaluationhumanrankingsizeselectionjsonl)).

## How DDP is computed

Each document is split into sentences. For every anchor in sentence *i*:

| Anchor | Distance |
|---|---|
| Named entity (PERSON, ORG, GPE, LOC) | 0 on first mention, otherwise *i − j* where *j* is the earliest prior sentence mentioning the same entity |
| Pronoun (he/she/it/they and their forms) | distance to the most recent prior named entity or head noun; if it is a named entity, that entity's own distance is added; 1 if there is no antecedent |

Sentence DDP is the maximum anchor distance, and segment DDP is the maximum over
the segment's sentences. Expletive *it* is excluded. A high DDP means the
segment needs long-range context; a low DDP does not guarantee the segment is
easy (Section 3.2).

Example (Table 1 of the paper):

```
(1) John arrived.                ddp = 0
(2) He looked tired.             ddp = 1   He  -> John (1)
(3) The manager greeted him.     ddp = 2   him -> John (1)
(4) John smiled.                 ddp = 3   John -> John (1)
(5) She offered him coffee.      ddp = 4   She -> manager (3) = 2, him -> John (4) + 3 = 4
```

## Installation

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/trotacodigos/ddp.git
cd ddp
pip install -r requirements.txt
```

`requirements.txt` installs everything. To compute DDP only, `spacy`,
`en_core_web_trf` and `tqdm` are enough. The Flair backend additionally needs
`flair` and `nltk`.

## Computing DDP

### On your own documents

```python
from ddp import compute_discourse_dependency
from ddp.extractors import SpacyExtractor

segments = [
    "John arrived.",
    "He looked tired.",
    "The manager greeted him.",
    "John smiled.",
    "She offered him coffee.",
]
doc = compute_discourse_dependency("example", segments, SpacyExtractor())
print([seg.ddp for seg in doc.seg_dds])
```

Pass the segments of one document in order. Each `SegmentDD` also keeps its
sentences and anchors (`seg.sentences[k].anchors`) for inspection.

### On a WMT-style JSONL file

```bash
python -m ddp.annotate \
    --input  data/input/original/wmt24pp_enko_KR.jsonl \
    --output wmt24pp_ddp.jsonl \
    --ddp_map wmt24pp_map.json \
    --backend spacy          # spacy (en_core_web_trf) | spacy-sm | flair
```

Each line needs `doc_id`, `seg_id` and `src_seg`. DDP is computed once per
document from the unique source segments and copied to every line, so files
with several systems per segment are handled correctly. The script prints
per-domain statistics at the end.

### Tests

```bash
python -m pytest tests
```

The tests check the worked example above without loading any NLP model.

## Data

```
data/
├── anchor_extraction/          # DDP annotations
│   ├── ddp_spacy/{wmt24pp,wmt25}.jsonl
│   ├── ddp_flair/{wmt24pp,wmt25}.jsonl
│   └── {wmt24pp,wmt25}_map.json
├── input/                      # APE inputs
│   ├── doc-seq/  doc-rel/      # per-segment context (k = 5)
│   ├── doc-exp/                # document annotations for APE_exp
│   └── doc-full/               # full source / draft documents for APE_full
└── output/doc-{seg,seq,rel,exp,full}/   # APE outputs, one file per model
evaluation/
├── metrics/{tgt,hpe}/wmt24pp/{en-ko_KR,en-zh_CN}/   # segment-level metric scores
└── human/ranking/{size,selection}.jsonl           # human rankings (Section 4.4)
```

See [`data/README.md`](data/README.md) for field descriptions.

### WMT25 texts

The WMT25 files keep identifiers, DDP values and context indices but not the
source, reference or system translations, which are not redistributed here.
To restore them from the official release:

```bash
git clone https://github.com/wmt-conference/wmt25-general-mt
python -m tools.restore_wmt25 --wmt25_dir wmt25-general-mt
```

This fills in the text fields and recreates `data/input/doc-full/wmt25_*`.

## Reproducing the APE experiments

Five strategies (Section 4.3):

| Strategy | Context | Input file |
|---|---|---|
| `seg` | none | `data/input/original/` |
| `seq` | 5 preceding segment pairs | `data/input/doc-seq/` |
| `rel` | 5 most similar segment pairs (LaBSE) | `data/input/doc-rel/` |
| `exp` | genre, participants and register of the document | `data/input/original/` + `doc-exp/` |
| `full` | whole source and draft document | `data/input/original/` + `doc-full/` |

`data/input/original/{collection}_{lp}.jsonl` holds one line per segment and
system with `collection_id, doc_id, seg_id, domain, system, src_lang, tgt_lang,
src_seg, tgt_seg`. It is built from the public WMT24++ / WMT25 releases and is
not redistributed here; `data/anchor_extraction/ddp_spacy/wmt24pp.jsonl` has
the same fields for WMT24++ En–Ko.

**1. Build the context files** (the released ones are already in `data/input/`)

```bash
python -m ape.context.seq --collection_id wmt24pp --lang_pair en-ko_KR
python -m ape.context.rel --collection_id wmt24pp --lang_pair en-ko_KR
python -m ape.context.exp --collection_id wmt24pp   # annotations were reviewed by hand afterwards
```

**2. Run post-editing**

```bash
export OPENAI_API_KEYS=...     # GPT-5.4; also used as the key for the local vLLM server of HyperCLOVA X
export GEMINI_API_KEYS=...
export DEEPSEEK_API_KEYS=...

python -m ape.inference --context seq --collection_id wmt24pp \
    --lang_pair en-ko_KR --model gpt-5.4-2026-03-05
```

| `--model` | Served via |
|---|---|
| `gpt-5.4-2026-03-05` | OpenAI Batch API |
| `gemini-3-flash` | Google Gen AI API |
| `deepseek-v4-pro` | DeepSeek API |
| `gemma-4-31b-it`, `qwen3.5-27b`, `hcx-seed-thinking-32b` | vLLM, OpenAI-compatible server on `localhost:8000` |

Sampling: `temperature=1.0, top_p=0.95, seed=42`, at most 1,024 output tokens.
Prompt templates are in [`ape/templates.json`](ape/templates.json) (Figure 9).
Outputs go to `data/output/doc-{context}/`, and an interrupted run resumes
where it stopped. The post-edit is the text between `<pe>` and `</pe>`
(`ape.utils.extract_pe`).

## Figures

```bash
# Figure 1: DDP distribution in WMT24++ and WMT25
python -m analysis.plot_ddp_distribution --out ddp_distribution.pdf

# Figure 2: human preference across DDP (one rating file per assumption)
python -m analysis.plot_human_preference --ratings evaluation/human/ranking/size.jsonl --out pref_size.pdf
python -m analysis.plot_human_preference --ratings evaluation/human/ranking/selection.jsonl --out pref_selection.pdf
```

Ranks are turned into preference scores (higher = better; tied candidates get
the average of the positions they span), averaged per segment and smoothed
with LOWESS (`frac = 0.4`).

## Repository layout

```
ddp/        DDP computation (core rules, spaCy/Flair extractors, CLI)
ape/        prompts, context construction and APE inference
analysis/   scripts for the figures
models/     API / vLLM clients for the six evaluated LLMs
tools/      restore_wmt25.py (fills in WMT25 texts)
tests/      unit tests for the DDP rules
assets/     figures
```

## Citation

```bibtex
@article{kim2026discourse,
  title   = {Discourse Dependency: A Continuous Criterion for Translation Difficulty},
  author  = {Kim, Ahrii and Park, Chanjun and Kim, Seong-heum},
  journal = {arXiv preprint arXiv:2609.04959},
  year    = {2026}
}
```

## License

Code: MIT ([LICENSE](LICENSE)). Annotations, model outputs and human rankings:
CC BY 4.0.

WMT24++ source texts, references and post-edits come from
[google/wmt24pp](https://huggingface.co/datasets/google/wmt24pp) (Apache 2.0) and
are redistributed under that license. WMT25 data are not included; they are
available from the [WMT25 General MT task](https://www2.statmt.org/wmt25/translation-task.html)
for research use.
