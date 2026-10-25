"""APE_exp context: LLM annotation of genre, participants and register per document.

The released annotations in data/input/doc-exp/ were generated with this
script and then reviewed and corrected by a trained annotator (Appendix B.2).

Usage:
    python -m ape.context.exp --collection_id wmt24pp
"""

import argparse
import json
import logging
import time

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from ape.context.exp_prompt import build_prompt
from ape.utils import INPUT_DIR, file_stem, read_json, read_jsonl
from models.tools import get_keys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 1024
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0


def run_inference(client: OpenAI, model: str, system: str, user: str) -> str:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_completion_tokens=MAX_OUTPUT_TOKENS,
                temperature=0,
                response_format={"type": "json_object"},
            )
            choice = resp.choices[0]
            if choice.finish_reason == "length":
                logger.warning("Output hit max_completion_tokens; JSON may be truncated.")
            return (choice.message.content or "").strip()
        except (RateLimitError, APITimeoutError, APIError) as e:
            last_err = e
            delay = RETRY_BASE_DELAY * 2 ** (attempt - 1)
            logger.warning(f"API error ({attempt}/{MAX_RETRIES}): {e}. Retrying in {delay:.0f}s")
            time.sleep(delay)
    raise RuntimeError(f"Inference failed after {MAX_RETRIES} retries: {last_err}")


def parse_json_output(raw: str) -> dict:
    cleaned = "\n".join(l for l in raw.strip().splitlines() if not l.strip().startswith("```"))
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}")
        return {"_raw": raw, "_error": str(e)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection_id", required=True)
    ap.add_argument("--lang_pair", default="en-ko_KR",
                    help="any pair works; only the source document and domain are used")
    ap.add_argument("--model", default="gpt-5.4")
    args = ap.parse_args()

    stem = file_stem(args.collection_id, args.lang_pair)
    src_docs = read_json(INPUT_DIR / "doc-full" / f"{stem}_src.json")
    domains = {}
    for e in read_jsonl(INPUT_DIR / "original" / f"{stem}.jsonl"):
        domains.setdefault(str(e["doc_id"]), e["domain"])
    doc_ids = [d for d in domains if d in src_docs]

    out_file = INPUT_DIR / "doc-exp" / f"{args.collection_id}.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.update(json.loads(line).keys())

    client = OpenAI(api_key=get_keys("OPENAI_API_KEYS")[0])
    with open(out_file, "a", encoding="utf-8") as f:
        for i, doc_id in enumerate(doc_ids, 1):
            if doc_id in done:
                continue
            logger.info(f"[{i}/{len(doc_ids)}] {doc_id} ({domains[doc_id]})")
            system, user = build_prompt(domains[doc_id], src_docs[doc_id])
            try:
                parsed = parse_json_output(run_inference(client, args.model, system, user))
            except Exception as e:
                logger.error(f"{doc_id} failed: {e}")
                parsed = {"_error": str(e)}
            f.write(json.dumps({doc_id: parsed}, ensure_ascii=False) + "\n")
            f.flush()
    logger.info(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
