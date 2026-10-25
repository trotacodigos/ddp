"""Figure 1: DDP distribution per domain in WMT24++ and WMT25.

DDP is a source-side value, so each source segment is counted once. Domains
come from the `domain` field of the annotation files.

Usage:
    python -m analysis.plot_ddp_distribution --out assets/ddp_distribution.pdf
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

ANN_DIR = Path("data/anchor_extraction/ddp_spacy")
DOMAINS = ["news", "social", "literary"]
COLORS = {"news": "#378ADD", "social": "#888780", "literary": "#1D9E75"}
LSTYLES = {"news": "-", "social": "--", "literary": "-"}
LWS = {"news": 1.5, "social": 1.2, "literary": 1.8}


def load_scores(path: Path, field: str = "ddp_spacy") -> dict[str, list[int]]:
    """{domain: [ddp of each unique source segment]}.

    Segment ids are not always shared across systems (WMT25 numbers every
    line), and one system may appear twice per segment (WMT24++ Gemini-1),
    so the segments of a single system, deduplicated, are used.
    """
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    system = sorted({r["system"] for r in rows})[0]
    seen, scores = set(), defaultdict(list)
    for r in rows:
        key = (r["doc_id"], str(r["seg_id"]))
        if r["system"] != system or key in seen:
            continue
        seen.add(key)
        scores[r["domain"]].append(int(r[field]))
    return dict(scores)


def plot_cdf(ax, scores, xmax, title):
    xs = np.arange(0, xmax + 1)
    for d in DOMAINS:
        arr = np.array(scores[d])
        ax.plot(xs, [np.mean(arr <= x) * 100 for x in xs], LSTYLES[d], color=COLORS[d], linewidth=LWS[d])
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 103)
    ax.set_xlabel("DDP score")
    ax.set_ylabel("cumulative %")
    ax.set_title(title, fontsize=16, pad=4)
    ax.axhline(100, color="#ccc", linewidth=0.4, linestyle=":")


def mark_truncated(ax, scores, xmax):
    """Axis-break marks and a max label when the literary tail is cut off."""
    lit_max = max(scores["literary"])
    if lit_max <= xmax:
        return
    ax.spines["right"].set_visible(False)
    kw = dict(transform=ax.transAxes, color="#666", linewidth=0.8, clip_on=False)
    d = 0.015
    ax.plot((1 - d, 1 + d), (-d * 1.5, d * 1.5), **kw)
    ax.plot((1 - d, 1 + d), (-d * 1.5 + 0.03, d * 1.5 + 0.03), **kw)
    y = np.mean(np.array(scores["literary"]) <= xmax) * 100
    ax.annotate(f"(max={lit_max})", xy=(xmax, y), xytext=(5, 0), textcoords="offset points",
                fontsize=12, color=COLORS["literary"], va="center", annotation_clip=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wmt24pp", default=ANN_DIR / "wmt24pp.jsonl", type=Path)
    ap.add_argument("--wmt25", default=ANN_DIR / "wmt25.jsonl", type=Path)
    ap.add_argument("--xmax24", type=int, default=None, help="x range of (b); default: data max")
    ap.add_argument("--xmax25", type=int, default=250, help="x range of (c)")
    ap.add_argument("--out", default="ddp_distribution.pdf")
    args = ap.parse_args()

    s24, s25 = load_scores(args.wmt24pp), load_scores(args.wmt25)
    for name, sc in (("WMT24++", s24), ("WMT25", s25)):
        print(name, {d: f"n={len(sc[d])} mean={np.mean(sc[d]):.1f} max={max(sc[d])}" for d in DOMAINS})

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 12,
        "axes.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(10.5, 3.2))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1.2, 1.8], wspace=0.42,
                  left=0.07, right=0.97, top=0.88, bottom=0.22)
    ax_bar, ax24, ax25 = (fig.add_subplot(gs[i]) for i in range(3))

    # (a) mean DDP per domain
    x, w = np.arange(len(DOMAINS)), 0.32
    m24 = [np.mean(s24[d]) for d in DOMAINS]
    m25 = [np.mean(s25[d]) for d in DOMAINS]
    b24 = ax_bar.bar(x - w / 2, m24, w, facecolor="white", hatch="///",
                     edgecolor=[COLORS[d] for d in DOMAINS], linewidth=1.0)
    b25 = ax_bar.bar(x + w / 2, m25, w, color=[COLORS[d] for d in DOMAINS], alpha=0.85)
    for bars, vals, bold in ((b24, m24, False), (b25, m25, True)):
        for bar, v in zip(bars, vals):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=10,
                        fontweight="bold" if bold else "normal", color="#222" if bold else "#555")
    ax_bar.set_xticks(x, DOMAINS)
    ax_bar.set_ylabel("mean DDP")
    ax_bar.set_ylim(0, max(m24 + m25) * 1.45)
    ax_bar.set_title("(a) mean DDP", fontsize=16, pad=4)
    ax_bar.legend(handles=[
        mpatches.Patch(facecolor="white", hatch="///", edgecolor="#666", label="WMT24++"),
        mpatches.Patch(facecolor="#999", edgecolor="#666", label="WMT25"),
    ], frameon=False, loc="upper left")

    xmax24 = args.xmax24 or max(max(v) for v in s24.values())
    plot_cdf(ax24, s24, xmax24, "(b) WMT24++")
    mark_truncated(ax24, s24, xmax24)
    plot_cdf(ax25, s25, args.xmax25, "(c) WMT25")
    mark_truncated(ax25, s25, args.xmax25)

    handles = [plt.Line2D([0], [0], color=COLORS[d], linestyle=LSTYLES[d], linewidth=LWS[d], label=d)
               for d in DOMAINS]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=14, frameon=False,
               bbox_to_anchor=(0.63, -0.06))
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
