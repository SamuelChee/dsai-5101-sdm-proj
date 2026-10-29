"""Stage 4.5 - Structural presentation visuals.

Translates the frozen structural artifacts (split, coverage, fitting graph)
into audience-facing figures. These figures use only structural information:
partition assignments, component membership, and audited counts. No held-out
outcome is read anywhere in this module.

Decision rules implemented:
* every internal-test row receives exactly one coverage category, and the
  category counts must sum to the manifest denominator (asserted);
* displayed component statistics are reconciled against the fitting graph
  (asserted against photo_components.parquet);
* zero-count coverage categories are omitted from the bar;
* schematic and sampled views carry visible disclosure text;
* no raw IDs are shown - human-readable labels only.

Outputs
-------
figures/presentation_component_sizes.pdf
figures/presentation_coverage.pdf
figures/presentation_split_diagram.pdf
figures/presentation_graph_network.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Circle, FancyBboxPatch

NETWORK_SAMPLE_SIZE = 50
COVERAGE_LABELS = {
    "eligible_strict_g2": "Strict G2 (eligible)",
    "cross_component": "Known photos, different fitting components",
    "unseen_photo": "At least one photo unseen in fitting",
}
COVERAGE_COLORS = {
    "eligible_strict_g2": "#2e7d32",
    "cross_component": "#f9a825",
    "unseen_photo": "#c62828",
}


# ---------------------------------------------------------------------------
# Figure 1 - fitting component sizes
# ---------------------------------------------------------------------------


def fig_component_sizes(
    comp_sizes: pd.Series, figures_dir: Path
) -> None:
    sizes = comp_sizes.sort_values(ascending=False).reset_index(drop=True)
    n_components = len(sizes)
    total_photos = int(sizes.sum())
    largest_share = float(sizes.iloc[0] / total_photos)

    # Show up to five components individually, aggregate the rest.
    show = sizes.iloc[:5]
    rest = sizes.iloc[5:]
    labels = ["Main Component"] + [f"Component {k + 2}" for k in
                                   range(len(show) - 1)]
    values = list(show)
    if len(rest):
        labels.append(f"Remaining components ({len(rest)})")
        values.append(int(rest.sum()))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, values, color=["#1565c0"] * len(show)
                  + (["#90a4ae"] if len(rest) else []))
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Fitting photos in component")
    ax.set_title("Fitting comparison graph: component sizes")
    ax.text(
        0.99, 0.97,
        f"Total components: {n_components}\n"
        f"Total represented fitting photos: {total_photos:,}\n"
        f"Largest-component share: {largest_share:.1%}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="#eceff1", edgecolor="#b0bec5"),
    )
    ax.set_ylim(0, max(values) * 1.25)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    # Small, clearly labeled schematic explaining disconnectedness.
    inset = fig.add_axes((0.60, 0.40, 0.35, 0.30))
    rng = np.random.default_rng(0)
    for cx, cy, n in ((0.30, 0.55, 5), (0.78, 0.35, 3)):
        pts = np.column_stack(
            [cx + 0.10 * np.cos(np.linspace(0, 2 * np.pi, n, endpoint=False)),
             cy + 0.16 * np.sin(np.linspace(0, 2 * np.pi, n, endpoint=False))]
        )
        for a in range(n):
            for b in range(a + 1, n):
                inset.plot(*zip(pts[a], pts[b]), color="#78909c", lw=0.8,
                           zorder=1)
        inset.scatter(pts[:, 0], pts[:, 1], s=22, color="#1565c0", zorder=2)
    inset.text(0.5, -0.08, "SCHEMATIC - what two disconnected\n"
                           "components would look like (not data)",
               ha="center", va="top", fontsize=7, color="#b71c1c",
               transform=inset.transAxes)
    inset.set_xlim(0, 1.05)
    inset.set_ylim(0, 0.9)
    inset.axis("off")

    # Manual margins: tight_layout does not support the manually added inset.
    fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.20)
    fig.savefig(figures_dir / "presentation_component_sizes.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 - internal-test coverage (stacked count bar)
# ---------------------------------------------------------------------------


def fig_coverage(manifest: pd.DataFrame, figures_dir: Path) -> dict:
    test = manifest[manifest["partition"] == "test"]
    denominator = len(test)  # read directly from the frozen manifest
    counts = test["eligibility"].value_counts()

    # Stop/go: every test row exactly one category; counts sum to denominator.
    assert test["eligibility"].notna().all(), "uncategorized test row found"
    assert int(counts.sum()) == denominator, "coverage counts do not reconcile"

    # Omit zero-count categories rather than plotting empty segments.
    present = [c for c in COVERAGE_LABELS if counts.get(c, 0) > 0]
    other = int(counts.sum() - sum(counts.get(c, 0) for c in COVERAGE_LABELS))

    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    left = 0
    legend_entries = []
    for cat in present:
        n = int(counts[cat])
        ax.barh([0], [n], left=left, color=COVERAGE_COLORS[cat], height=0.55)
        ax.text(left + n / 2, 0, f"{n:,}\n({n / denominator:.1%})",
                ha="center", va="center", color="white", fontsize=10,
                fontweight="bold")
        legend_entries.append(f"{COVERAGE_LABELS[cat]}: {n:,} "
                              f"({n / denominator:.1%})")
        left += n
    if other:
        ax.barh([0], [other], left=left, color="#616161", height=0.55)
        legend_entries.append(f"Other documented exclusions: {other:,} "
                              f"({other / denominator:.1%})")
    ax.set_xlim(0, denominator)
    ax.set_yticks([])
    ax.set_xlabel(f"Internal-test rows (denominator from frozen manifest: "
                  f"{denominator:,})")
    ax.set_title("Internal-test coverage by eligibility category")
    ax.legend(legend_entries, loc="upper center",
              bbox_to_anchor=(0.5, -0.55), fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "presentation_coverage.pdf",
                bbox_inches="tight")
    plt.close(fig)
    return {cat: int(counts.get(cat, 0)) for cat in COVERAGE_LABELS}


# ---------------------------------------------------------------------------
# Figure 3 - pair-level split diagram (comparison cards)
# ---------------------------------------------------------------------------


def fig_split_diagram(
    manifest: pd.DataFrame, figures_dir: Path, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    pairs = (
        manifest.groupby(["partition", "pair_id"])["row_id"]
        .size()
        .rename("n_responses")
        .reset_index()
    )
    totals = (
        manifest.groupby("partition")
        .agg(n_rows=("row_id", "size"), n_pairs=("pair_id", "nunique"))
    )
    targets = {"fitting": "70%", "validation": "15%", "test": "15%"}
    titles = {"fitting": "Fitting", "validation": "Validation",
              "test": "Internal Test"}
    colors = {"fitting": "#1565c0", "validation": "#ef6c00",
              "test": "#2e7d32"}

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6))
    cards_per_panel = 10
    for ax, part in zip(axes, ("fitting", "validation", "test")):
        sub = pairs[pairs["partition"] == part]
        take = min(cards_per_panel, len(sub))
        sample = sub.iloc[rng.choice(len(sub), size=take, replace=False)]
        for k, (_, card) in enumerate(sample.iterrows()):
            x, y = (k % 2) * 0.52 + 0.03, 0.88 - (k // 2) * 0.19
            ax.add_patch(FancyBboxPatch(
                (x, y), 0.44, 0.15, boxstyle="round,pad=0.008",
                facecolor="white", edgecolor=colors[part], lw=1.4,
            ))
            for m in range(int(card["n_responses"])):
                ax.plot(x + 0.05 + m * 0.055, y + 0.075, marker="|",
                        color=colors[part], markersize=9, markeredgewidth=1.8)
        t = totals.loc[part]
        ax.set_title(
            f"{titles[part]}\ntarget \u2248 {targets[part]} of pairs  |  "
            f"actual: {t['n_pairs']:,} pairs, {t['n_rows']:,} rows",
            fontsize=10,
        )
        ax.set_xlim(0, 1.08)
        ax.set_ylim(0, 1.02)
        ax.axis("off")
    fig.suptitle(
        "Pair-level split: every response to a matchup stays together",
        fontsize=12,
    )
    fig.text(
        0.5, 0.015,
        "SCHEMATIC - each card is one unordered matchup, marks are its "
        "responses; card counts per panel are illustrative, panel labels "
        "show actual realized counts.",
        ha="center", fontsize=8, color="#b71c1c",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(figures_dir / "presentation_split_diagram.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 - sampled network view of the fitting photo graph
# ---------------------------------------------------------------------------


def fig_graph_network(
    rows: pd.DataFrame,
    manifest: pd.DataFrame,
    comp_sizes: pd.Series,
    photo_components: pd.DataFrame,
    figures_dir: Path,
    seed: int,
) -> None:
    fitting_ids = set(
        manifest.loc[manifest["partition"] == "fitting", "row_id"]
    )
    fitting = rows[rows["row_id"].isin(fitting_ids)]
    edges = pd.unique(
        fitting[["a_image", "b_image"]].apply(sorted, axis=1).map(tuple)
    )

    graph = nx.Graph()
    graph.add_edges_from(map(tuple, edges))

    comp_of = dict(
        zip(photo_components["photo"], photo_components["fitting_component"])
    )
    main_comp = comp_sizes.idxmax()
    main_nodes = np.array(sorted(
        p for p, c in comp_of.items() if c == main_comp
    ))
    island_comps = [c for c in comp_sizes.index if c != main_comp]
    island_comps = sorted(island_comps, key=lambda c: comp_sizes[c])

    # 1. Compute layout on the FULL main component
    main_graph = graph.subgraph(main_nodes)
    # Lower k value pulls nodes closer together (prevents explosion)
    pos_full = nx.spring_layout(main_graph, seed=seed, k=0.05)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    
    # 2. Draw ALL nodes and edges
    nx.draw_networkx_edges(main_graph, pos_full, ax=ax, alpha=0.2,
                           edge_color="#546e7a", width=0.5)
    nx.draw_networkx_nodes(main_graph, pos_full, ax=ax, node_size=5,
                           node_color="#1565c0", edgecolors="none")

    # Human-readable cluster label (fixed at top)
    ax.text(0.5, 0.95, "Main Component (strict-G2 eligible)",
            transform=ax.transAxes, ha="center", fontsize=12, 
            fontweight="bold", color="#0d47a1")

    # Actual smallest disconnected components
    n_islands_drawn = 0
    xs = [p[0] for p in pos_full.values()]
    ys = [p[1] for p in pos_full.values()]
    
    for comp in island_comps[:3]:
        members = sorted(p for p, c in comp_of.items() if c == comp)
        island = graph.subgraph(members)
        if island.number_of_nodes() == 0:
            continue
        ipos = nx.spring_layout(island, seed=seed)
        # Place the island to the right of the main cluster.
        shift = np.array([max(xs) + 1.2, np.mean(ys)])
        ipos = {n: p + shift + np.array([0, n_islands_drawn * 0.9])
                for n, p in ipos.items()}
        nx.draw_networkx_edges(island, ipos, ax=ax, alpha=0.7,
                               edge_color="#c62828", width=1.2)
        nx.draw_networkx_nodes(island, ipos, ax=ax, node_size=55,
                               node_color="#c62828", edgecolors="white",
                               linewidths=0.6)
        pts = np.array(list(ipos.values()))
        centre, radius = pts.mean(axis=0), pts.std() * 2.2 + 0.25
        ax.add_patch(Circle(centre, radius, fill=False, ls="--",
                            ec="#c62828", lw=1.4))
        ax.text(centre[0], centre[1] + radius + 0.06,
                "Actual Disconnected Islands (excluded)",
                ha="center", fontsize=9, color="#b71c1c")
        n_islands_drawn += 1

    if n_islands_drawn == 0:
        ax.text(
            0.99, 0.02,
            "No disconnected components exist in this data:\n"
            f"all {len(main_nodes):,} fitting photos share one component.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            color="#1b5e20",
            bbox=dict(boxstyle="round", facecolor="#e8f5e9",
                      edgecolor="#a5d6a7"),
        )

    ax.text(
        0.01, 0.01,
        f"FULL GRAPH - all {main_graph.number_of_nodes():,} main-component nodes shown.",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
        color="#b71c1c",
    )
    ax.set_title("Fitting photo-comparison network (full data)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures_dir / "presentation_graph_network.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run_presentation_figures(config_path: Path) -> int:
    root = config_path.resolve().parent
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    artifacts_dir = root / cfg["paths"]["artifacts_dir"]
    figures_dir = root / cfg["paths"].get("figures_dir", "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seeds"]["split"])

    manifest = pd.read_parquet(artifacts_dir / "split_manifest.parquet")
    rows = pd.read_parquet(artifacts_dir / "rows_canonical.parquet")
    rows = rows[rows["source"] == "train"]
    photo_components = pd.read_parquet(
        artifacts_dir / "photo_components.parquet"
    )

    comp_sizes = (
        photo_components.groupby("fitting_component")["photo"]
        .size()
        .rename("n_photos")
    )

    # Reconcile displayed component statistics with the fitting graph (stop/go).
    summary = pd.read_csv(root / cfg["paths"]["tables_dir"]
                          / "graph_summary.csv")
    photo_summary = summary[summary["graph_level"] == "photo"].set_index(
        "metric"
    )["value"]
    assert int(photo_summary["n_components"]) == len(comp_sizes), (
        "component count disagrees with graph_summary.csv"
    )
    assert int(photo_summary["largest_component_nodes"]) == int(
        comp_sizes.max()
    ), "largest component size disagrees with graph_summary.csv"

    fig_component_sizes(comp_sizes, figures_dir)
    coverage_counts = fig_coverage(manifest, figures_dir)
    fig_split_diagram(manifest, figures_dir, seed)
    fig_graph_network(rows, manifest, comp_sizes, photo_components,
                      figures_dir, seed)

    print("Stage 4.5 presentation figures written.")
    print(f"  components: {len(comp_sizes)} total, largest "
          f"{int(comp_sizes.max()):,} photos "
          f"({comp_sizes.max() / comp_sizes.sum():.1%})")
    print(f"  internal-test coverage counts: {coverage_counts}")
    print("  figures/presentation_component_sizes.pdf")
    print("  figures/presentation_coverage.pdf")
    print("  figures/presentation_split_diagram.pdf")
    print("  figures/presentation_graph_network.pdf")
    print("  GO: categories reconcile with the manifest; component statistics "
          "reconcile with the fitting graph; disclosures included.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 4.5 structural presentation visuals."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config.yaml",
    )
    args = parser.parse_args(argv)
    return run_presentation_figures(args.config)


if __name__ == "__main__":
    sys.exit(main())
