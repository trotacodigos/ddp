"""APE_rel context: the k segment pairs most similar to each segment (LaBSE).

All other source segments of the document are encoded with LaBSE; the top-k
by cosine similarity are returned in document order (Appendix B.2).
Documents with k or fewer segments are skipped.

Usage:
    python -m ape.context.rel --collection_id wmt24pp --lang_pair en-ko_KR
"""

import argparse

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from ape.utils import INPUT_DIR, file_stem, group_documents, read_jsonl, write_jsonl

K = 5


class LaBSEEncoder:
    MODEL_NAME = "sentence-transformers/LaBSE"

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device).eval()

    @torch.no_grad()
    def encode(self, sentences: list[str], batch_size: int = 64) -> np.ndarray:
        """(N, D) L2-normalised [CLS] embeddings."""
        out = []
        for i in range(0, len(sentences), batch_size):
            enc = self.tokenizer(
                sentences[i:i + batch_size], padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(self.device)
            emb = self.model(**enc).last_hidden_state[:, 0, :]
            out.append(torch.nn.functional.normalize(emb, p=2, dim=1).cpu().float().numpy())
        return np.vstack(out)


def top_k_indices(embeddings: np.ndarray, query_idx: int, k: int) -> list[int]:
    """Indices of the k segments most similar to the query (query excluded),
    sorted by document position."""
    candidates = [i for i in range(len(embeddings)) if i != query_idx]
    sims = embeddings[candidates] @ embeddings[query_idx]
    ranked = sorted(zip(candidates, sims), key=lambda x: x[1], reverse=True)
    return sorted(i for i, _ in ranked[:k])


def relevant_context(doc: list[dict], encoder: LaBSEEncoder, k: int = K) -> list[dict]:
    emb = encoder.encode([s["src_seg"] for s in doc])
    out = []
    for i, seg in enumerate(doc):
        ctx = [doc[j] for j in top_k_indices(emb, i, k)]
        out.append({
            **seg,
            "rel": {
                "src_ctx": "\n".join(s["src_seg"] for s in ctx),
                "tgt_ctx": "\n".join(s["tgt_seg"] for s in ctx),
            },
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection_id", required=True)
    ap.add_argument("--lang_pair", required=True)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = ap.parse_args()

    stem = file_stem(args.collection_id, args.lang_pair)
    docs = group_documents(read_jsonl(INPUT_DIR / "original" / f"{stem}.jsonl"))
    encoder = LaBSEEncoder(args.device)

    records = []
    for (system, _, doc_id), doc in docs.items():
        if len(doc) <= args.k:
            print(f"[skip] system={system} doc_id={doc_id}: {len(doc)} segments <= k={args.k}")
            continue
        records.extend(relevant_context(doc, encoder, args.k))

    out = INPUT_DIR / "doc-rel" / f"{stem}.jsonl"
    write_jsonl(out, records)
    print(f"Saved {len(records)} records to {out}")


if __name__ == "__main__":
    main()
