# Data

All files are JSON Lines unless noted. `seg_id` is unique within a collection;
the same `(doc_id, seg_id)` appears once per MT system.

## DDP annotations — `anchor_extraction/`

| File | Content |
|---|---|
| `ddp_spacy/wmt24pp.jsonl` | WMT24++ En–Ko, one line per segment and system, DDP from spaCy `en_core_web_trf` in `ddp_spacy` |
| `ddp_spacy/wmt25.jsonl` | same for WMT25 En–Ko |
| `ddp_flair/{wmt24pp,wmt25}.jsonl` | DDP from the Flair backend in `ddp_flair` (Appendix A.3) |
| `wmt24pp_map.json` | `{doc_id: {seg_id: ddp}}`, spaCy backend, one value per source segment |
| `wmt25_map.json` | `{doc_id: {para_idx: ddp}}`, same for WMT25 |

Fields: `collection_id, doc_id, seg_id, bucket_id, domain, system, src_lang,
tgt_lang, src_seg, tgt_seg, ref_seg` (+ `human_pe_seg` for WMT24++) and the DDP
field. DDP is computed from the English source, so it applies to any target
language.

## APE inputs — `input/`

| Directory | Content |
|---|---|
| `doc-seq/{collection}_{lp}.jsonl` | segment line + `seq: {src_ctx, tgt_ctx}` with the 5 preceding segment pairs; the first 5 segments of each document are absent |
| `doc-rel/{collection}_{lp}.jsonl` | segment line + `rel: {src_ctx, tgt_ctx}` with the 5 most similar segment pairs; documents with ≤ 5 segments are absent |
| `doc-exp/{collection}.jsonl` | one line per document: `{doc_id: {genre_and_domain, participant_relationships, register_and_formality}}` |
| `doc-full/{collection}_{lp}_src.json` | `{doc_id: source document}` |
| `doc-full/{collection}_{lp}_tgt.json` | `{doc_id: {system: draft translation of the document}}` |

`lp` is `enko_KR` or `enzh_CN`.

## WMT25 files

WMT25 texts are not redistributed. In the WMT25 files, `src_seg`, `tgt_seg`
and `ref_seg` are removed, `para_idx` gives the segment's position in the
official document (`src_text` split on blank lines), and the `seq` / `rel`
contexts are stored as `{"idx": [para_idx, ...]}`. `doc-full/wmt25_{lp}_index.json`
lists the documents and systems. `python -m tools.restore_wmt25` rebuilds the
full files from the official release (see the main README); the restored files
are identical to the ones used in the experiments.

## APE outputs — `output/doc-{seg,seq,rel,exp,full}/`

File name: `{collection}_{lang_pair}_{model}.jsonl`.

| Field | Meaning |
|---|---|
| `doc_id`, `seg_id` | segment |
| `tgt_seg` | draft translation given to the model |
| `mt_pe_seg` | raw model output; the post-edit is inside `<pe>…</pe>` |
| `usage` | input / output tokens and latency |

The released outputs have no `system` field. Lines follow the order of the
corresponding input file; `doc-seq/system.jsonl` and `doc-rel/system.jsonl`
give the system of each line. Outputs produced with the current
`ape/inference.py` include `system` directly.

## Automatic metrics — `../evaluation/metrics/`

`{tgt,hpe}/wmt24pp/{lp}/{metric}.jsonl` with `{doc_id, seg_id, score}` for the
draft translations (`tgt`) and the professional post-edits (`hpe`). Metrics:
`bleu, dbleu, doc_comet, metricx, metricx_qe, slide, ter, xcomet, xcomet_kiwi`.
There is one line per segment and system, in the same order as
`anchor_extraction/ddp_spacy/wmt24pp.jsonl`.

## Human rankings — `../evaluation/human/ranking/{size,selection}.jsonl`

| File | Strategies compared | Judgments | Segments |
|---|---|---|---|
| `size.jsonl` | `seg`, `seq`, `full`, `hpe` | 288 | 139 |
| `selection.jsonl` | `seq`, `rel`, `exp`, `hpe` | 342 | 171 |

One line per judgment (one annotation task). `candidates` holds the four
translations shown to the annotator under letters A–D, `label_map` maps each
letter to its strategy, and `ranks` gives the rank of each letter (1 = best,
ties allowed). `model` is the LLM that produced the APE candidates and
`system` the MT system of the draft. `prev_src_seg` / `next_src_seg` are the
neighbouring source segments shown as context.

These files are the subset of the human evaluation that could be recovered
from the annotation exports. Annotator identities were not preserved, so
inter-annotator agreement cannot be computed from them, and the statistics
reported in the paper cannot be reproduced exactly. Some tasks were submitted
twice (37 in `size.jsonl`, 54 in `selection.jsonl`); both submissions are kept
and can be told apart by `task_id`.

## License

Annotations, outputs and rankings are released under CC BY 4.0. WMT24++ texts
are redistributed under Apache 2.0 (google/wmt24pp). WMT25 texts are not
included.
