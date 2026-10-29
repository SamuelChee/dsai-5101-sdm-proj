"""Stage 7 - Describe disagreement on development data.

Descriptive analysis of repeated responses on the development partitions
(fitting + validation) in canonical coordinates (plan Stage 7). The internal
test partition is never loaded.

* Per development pair: n_e, k_e, q_e = k_e / n_e for the canonical photo i.
* 95% Wilson intervals under a binomial working assumption (a working
  assumption only - annotators are unidentified, so within-pair responses
  need not be independent).
* Separate within-cat / between-cat summaries.
* In-sample pair-constant loss  L_pair = sum_e n_e h(q_e) / sum_e n_e
  (0 log 0 := 0). This fits and evaluates each pair on the SAME responses:
  it is an in-sample description, NOT an oracle floor and NOT an
  unseen-pair benchmark.
* Illustrative-pair selection uses only pairs with >= 20 responses: one with
  the largest |q_e - 0.5| and one closest to 0.5, ties broken by larger n_e
  then lexicographic pair ID. If fewer than two pairs qualify, photo
  examples are OMITTED - the threshold is never lowered after seeing
  outcomes (plan Stage 7 decision rule).

Outputs
-------
tables/development_pair_summary.csv
figures/pair_proportions.pdf
tables/pair_benchmark.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from src.split import load_partition_rows

EXAMPLE_MIN_RESPONSES = 20
WILSON_Z = float(norm.ppf(0.975))
CAVEAT = (
    "In-sample pair-constant description fitted and evaluated on the same "
    "responses; not an oracle floor and not an unseen-pair benchmark. "
    "Wilson intervals assume a binomial working model; respondent identity "
    "is unknown, so within-pair independence is not established."
)


def wilson_interval(k: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """95% Wilson score interval for a binomial proportion."""
    z2 = WILSON_Z**2
    p = k / n
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (WILSON_Z * np.sqrt(p * (1 - p) / n + z2 / (4 * n**2))) / denom
    return centre - half, centre + half


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Bernoulli entropy in nats with 0 log 0 := 0."""
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p)
    mid = (p > 0) & (p < 1)
    out[mid] = -(p[mid] * np.log(p[mid]) + (1 - p[mid]) * np.log(1 - p[mid]))
    return out


def pair_constant_loss(summary: pd.DataFrame) -> float:
    """sum_e n_e h(q_e) / sum_e n_e over the given pair summary."""
    h = binary_entropy(summary["q_e"].to_numpy())
    return float((summary["n_e"] * h).sum() / summary["n_e"].sum())


def select_illustrative_pairs(summary: pd.DataFrame) -> dict:
    """Frozen example selection (>= 20 responses; ties by n_e, then pair ID)."""
    eligible = summary[summary["n_e"] >= EXAMPLE_MIN_RESPONSES]
    if len(eligible) < 2:
        return {
            "threshold_responses": EXAMPLE_MIN_RESPONSES,
            "n_pairs_meeting": int(len(eligible)),
            "action": "omitted",
            "rule": "fewer than two pairs meet the threshold; photo examples "
                    "are omitted rather than lowering the threshold after "
                    "seeing outcomes (plan Stage 7 decision rule)",
        }
    dev = eligible.assign(dev=(eligible["q_e"] - 0.5).abs())
    strongest = dev.sort_values(
        ["dev", "n_e", "pair_id"], ascending=[False, False, True]
    ).iloc[0]
    closest = dev.sort_values(
        ["dev", "n_e", "pair_id"], ascending=[True, False, True]
    ).iloc[0]
    return {
        "threshold_responses": EXAMPLE_MIN_RESPONSES,
        "n_pairs_meeting": int(len(eligible)),
        "action": "selected",
        "strongest_preference": strongest["pair_id"],
        "closest_to_even": closest["pair_id"],
        "caveat": "Selected by a frozen rule on development data; binomial "
                  "working assumption; not evidence about unseen pairs.",
    }


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run_disagreement(config_path: Path) -> int:
    root = config_path.resolve().parent
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    tables_dir = root / cfg["paths"]["tables_dir"]
    figures_dir = root / cfg["paths"].get("figures_dir", "figures")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Development data only: fitting + validation (test refused by the loader).
    frames = []
    for part in ("fitting", "validation"):
        part_rows = load_partition_rows(config_path, part)
        part_rows = part_rows.assign(partition=part)
        frames.append(part_rows)
    dev = pd.concat(frames, ignore_index=True)

    # Canonical pair summaries.
    summary = (
        dev.groupby(["pair_id", "partition", "comparison_type"], as_index=False)
        .agg(
            n_e=("y_canonical", "size"),
            k_e=("y_canonical", "sum"),
        )
    )
    summary["q_e"] = summary["k_e"] / summary["n_e"]
    lo, hi = wilson_interval(
        summary["k_e"].to_numpy(dtype=float),
        summary["n_e"].to_numpy(dtype=float),
    )
    summary["wilson_lo"] = lo
    summary["wilson_hi"] = hi
    summary = summary.sort_values(
        ["comparison_type", "n_e", "pair_id"]
    ).reset_index(drop=True)
    summary.to_csv(tables_dir / "development_pair_summary.csv", index=False)

    # In-sample pair-constant benchmark.
    benchmark_rows = []

    def bench(scope: str, sub: pd.DataFrame) -> None:
        benchmark_rows.append(
            {
                "scope": scope,
                "n_pairs": len(sub),
                "n_responses": int(sub["n_e"].sum()),
                "pair_constant_loss": pair_constant_loss(sub),
                "caveat": CAVEAT,
            }
        )

    bench("all_development", summary)
    for ctype, sub in summary.groupby("comparison_type"):
        bench(f"comparison_type={ctype}", sub)
    for part, subb in summary.groupby("partition"):
        bench(f"partition={part}", subb)
    benchmark = pd.DataFrame(benchmark_rows)
    benchmark.to_csv(tables_dir / "pair_benchmark.csv", index=False)

    # Frozen illustrative-pair selection.
    selection = select_illustrative_pairs(summary)
    with (root / cfg["paths"]["artifacts_dir"]
          / "pair_example_selection.json").open("w", encoding="utf-8") as fh:
        json.dump(selection, fh, indent=2)

    # Figure: canonical proportion vs response count, panels per comparison type.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    rng = np.random.default_rng(int(cfg["seeds"]["split"]))
    for ax, (ctype, sub) in zip(axes, summary.groupby("comparison_type")):
        for part, marker in (("fitting", "o"), ("validation", "^")):
            s = sub[sub["partition"] == part]
            x = s["n_e"] + rng.uniform(-0.06, 0.06, size=len(s))
            yerr = np.vstack([
                (s["q_e"] - s["wilson_lo"]).clip(lower=0.0),
                (s["wilson_hi"] - s["q_e"]).clip(lower=0.0),
            ])  # clip: Wilson bounds are within [0,1] up to fp rounding
            ax.errorbar(
                x, s["q_e"],
                yerr=yerr,
                fmt=marker, ms=3.5, elinewidth=0.5, capsize=0, alpha=0.45,
                color="#1565c0" if part == "fitting" else "#ef6c00",
                label=f"{part} pairs (n={len(s):,})",
            )
        ax.axhline(0.5, color="#9e9e9e", lw=0.8, ls=":")
        ax.set_xlabel("Responses per pair $n_e$")
        ax.set_title(f"{ctype} ({len(sub):,} pairs)")
        ax.set_xticks(sorted(summary["n_e"].unique()))
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Canonical choice proportion $q_e$ (95% Wilson)")
    fig.suptitle(
        "Development pairs: disagreement and support\n"
        f"Descriptive only - {selection['action']} photo examples "
        f"(threshold n_e >= {EXAMPLE_MIN_RESPONSES}); binomial working "
        "assumption; not an oracle floor",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(figures_dir / "pair_proportions.pdf")
    plt.close(fig)

    # Console summary.
    overall = benchmark_rows[0]
    print("Stage 7 disagreement analysis complete.")
    print(f"  development pairs: {len(summary):,} "
          f"({overall['n_responses']:,} responses)")
    print(f"  q_e range: [{summary['q_e'].min():.3f}, "
          f"{summary['q_e'].max():.3f}]; n_e range: "
          f"[{summary['n_e'].min()}, {summary['n_e'].max()}]")
    for row in benchmark_rows:
        print(f"  pair-constant loss [{row['scope']}]: "
              f"{row['pair_constant_loss']:.4f} "
              f"({row['n_pairs']:,} pairs)")
    print(f"  illustrative pairs: {selection['action']} "
          f"({selection['n_pairs_meeting']} pairs meet n_e >= "
          f"{selection['threshold_responses']})")
    print("  GO: every display identifies its partition and support; "
          "caveats attached.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 7 disagreement.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config.yaml",
    )
    args = parser.parse_args(argv)
    return run_disagreement(args.config)


if __name__ == "__main__":
    sys.exit(main())
