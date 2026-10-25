"""Figure 2 (a, b, d, e): human preference across DDP.

Ranks (1 = best, ties allowed) are converted to preference scores
(higher = better): tied candidates get the average of the positions they
span, then pref = 5 - rank, so the four scores of a segment sum to 10.
Scores are averaged per segment, then smoothed with LOWESS (frac = 0.4).

Usage:
    python -m analysis.plot_human_preference \
        --ratings evaluation/human/ranking/rater1.csv --out pref_size.pdf
"""

import argparse
import ast
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402
from statsmodels.nonparametric.smoothers_lowess import lowess  # noqa: E402

COLORS = {"seg": "#888780", "seq": "#378ADD", "full": "#1D9E75",
          "rel": "#7FB800", "exp": "#7B3FA0", "hpe": "#E8703A"}
LABELS = {"seg": "APEseg", "seq": "APEseq", "full": "APEfull",
          "rel": "APErel", "exp": "APEexp", "hpe": "HPE"}
PIVOT = "seq"  # APE_seq is the shared reference strategy (Section 4.3)


def load_preferences(paths: list[str], ddp_map_path: str) -> pd.DataFrame:
    """One row per (annotator, model, doc_id, seg_id) with a preference column per strategy."""
    ddp_map = json.load(open(ddp_map_path, encoding="utf-8"))
    records = []
    for path in paths:
        for _, row in pd.read_csv(path).iterrows():
            label_map = ast.literal_eval(row["label_map"])
            letters = sorted(label_map)
            ranks = rankdata([row[f"rank_{l.lower()}"] for l in letters], method="average")
            rec = {"annotator": row["annotator"], "model": row["model"],
                   "doc_id": row["doc_id"], "seg_id": str(row["seg_id"])}
            rec.update({label_map[l]: 5 - r for l, r in zip(letters, ranks)})
            records.append(rec)
    df = pd.DataFrame(records)
    df["ddp"] = [ddp_map[d][s] for d, s in zip(df["doc_id"], df["seg_id"])]
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratings", nargs="+", required=True, help="rater CSV file(s) of one assumption")
    ap.add_argument("--ddp_map", default="data/anchor_extraction/wmt24pp_map.json")
    ap.add_argument("--frac", type=float, default=0.4)
    ap.add_argument("--out", default="ddp_strategy_lowess.pdf")
    args = ap.parse_args()

    df = load_preferences(args.ratings, args.ddp_map)
    strategies = [c for c in ["seg", "rel", "exp", "full", "seq", "hpe"] if c in df.columns]
    seg = df.groupby(["doc_id", "seg_id", "ddp"])[strategies].mean().reset_index()
    print(f"{len(df)} judgments, {len(seg)} segments, strategies: {strategies}")
    x = seg["ddp"].to_numpy()

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.subplots_adjust(wspace=0.42)

    for s in strategies:
        y = seg[s].to_numpy()
        ax_a.scatter(x, y, color=COLORS[s], alpha=0.10, s=10)
        sm = lowess(y, x, frac=args.frac)
        ax_a.plot(sm[:, 0], sm[:, 1], color=COLORS[s], linewidth=2.2, label=LABELS[s])
    ax_a.set(xlabel="DDP score", ylabel="preference score (higher = better)", ylim=(1, 4),
             title="(a) Strategy preference")
    ax_a.legend(frameon=False, loc="lower right")

    for s in [s for s in strategies if s not in (PIVOT, "hpe")]:
        y = (seg[PIVOT] - seg[s]).to_numpy()
        ax_b.scatter(x, y, color=COLORS[s], alpha=0.10, s=10)
        sm = lowess(y, x, frac=args.frac)
        label = f"{LABELS[PIVOT]} − {LABELS[s]}"
        ax_b.plot(sm[:, 0], sm[:, 1], color=COLORS[s], linewidth=2.2, label=label)
        rho, p = spearmanr(x, y)
        print(f"{label}: Spearman rho={rho:.3f}, p={p:.4f}")
    ax_b.axhline(0, color="#444", linewidth=0.8, linestyle="--")
    ax_b.set(xlabel="DDP score", ylabel=f"preference diff (positive = {PIVOT} better)",
             title=f"(b) {LABELS[PIVOT]} vs. others")
    ax_b.legend(frameon=False, loc="lower right")

    if "hpe" in strategies:
        gap = (seg["hpe"] - seg[PIVOT]).to_numpy()
        rho, p = spearmanr(x, gap)
        print(f"HPE − {LABELS[PIVOT]} vs DDP: Spearman rho={rho:.3f}, p={p:.2e}, n={len(gap)}")

    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
