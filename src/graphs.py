"""Stage 4 - Build fitting graphs and freeze strict-G2 eligibility.

Uses only the fitting partition and the split manifest (plan Stage 4):

* photo-level and cat-level undirected comparison graphs;
* unique-opponent counts reported separately from response counts;
* directed winner-to-loser graphs from fitting outcomes;
* connected-component and directed strong-connectivity (SCC) status;
* strict-G2 eligibility for held-out rows: both photos appear in fitting
  AND belong to the same fitting-photo component;
* cross-component, unseen-photo, and excluded rows tabulated separately.

Eligibility is a structural property of the split and the fitting graph only;
no held-out outcome is read anywhere in this module.

Outputs
-------
tables/graph_summary.csv
tables/split_coverage.csv
figures/graph_support.pdf
artifacts/photo_components.parquet   photo -> fitting component id (Stage 6 rankings)
artifacts/eligibility.json           strict-G2 counts and adequacy labels
artifacts/split_manifest.parquet     updated in place with eligibility flags
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
import scipy.sparse as sp
import scipy.sparse.csgraph as cs
import yaml

# Plan Stage 4 decision-rule thresholds (operational rules, not power claims).
STRICT_VALIDATION_MIN_RESPONSES = 200
STRICT_VALIDATION_MIN_PAIRS = 50
STRICT_TEST_MIN_RESPONSES = 500
STRICT_TEST_MIN_PAIRS = 100


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_undirected_graph(
    left: pd.Series, right: pd.Series
) -> tuple[np.ndarray, list[str], sp.csr_matrix]:
    """Undirected graph over unique node labels; self-loops are dropped.

    Returns (node array, edge list as index pairs, adjacency matrix).
    """
    nodes = np.sort(pd.unique(pd.concat([left, right])))
    index = {node: k for k, node in enumerate(nodes)}
    edges = {
        (index[a], index[b])
        for a, b in zip(left, right)
        if a != b
    }
    edges = {(min(e), max(e)) for e in edges}
    if edges:
        ii, jj = map(np.array, zip(*edges))
        data = np.ones(len(ii))
        adj = sp.coo_matrix(
            (data, (ii, jj)), shape=(len(nodes), len(nodes))
        ).tocsr()
        adj = adj + adj.T
    else:
        adj = sp.csr_matrix((len(nodes), len(nodes)))
    return nodes, sorted(edges), adj


def build_directed_graph(
    winners: pd.Series, losers: pd.Series, nodes: np.ndarray
) -> sp.csr_matrix:
    """Directed winner-to-loser adjacency over the given node set."""
    index = {node: k for k, node in enumerate(nodes)}
    edges = {
        (index[w], index[l]) for w, l in zip(winners, losers) if w != l
    }
    if not edges:
        return sp.csr_matrix((len(nodes), len(nodes)))
    ii, jj = map(np.array, zip(*edges))
    return sp.coo_matrix(
        (np.ones(len(ii)), (ii, jj)), shape=(len(nodes), len(nodes))
    ).tocsr()


def graph_metrics(
    level: str,
    nodes: np.ndarray,
    n_unique_edges: int,
    n_responses: int,
    adj: sp.csr_matrix,
    responses_per_node: pd.Series,
    directed: sp.csr_matrix,
) -> list[dict]:
    n_components, labels = cs.connected_components(adj, directed=False)
    comp_sizes = np.bincount(labels, minlength=n_components)
    n_scc, scc_labels = cs.connected_components(
        directed, directed=True, connection="strong"
    )
    scc_sizes = np.bincount(scc_labels, minlength=n_scc)
    opponents = pd.Series(np.diff(adj.indptr), index=nodes)  # unique opponents

    def s(series: pd.Series) -> dict:
        return {
            "min": float(series.min()),
            "median": float(series.median()),
            "mean": float(series.mean()),
            "max": float(series.max()),
        }

    flat = {
        "n_nodes": len(nodes),
        "n_unique_edges": n_unique_edges,
        "n_responses_incident": n_responses,
        "unique_opponents_min": s(opponents)["min"],
        "unique_opponents_median": s(opponents)["median"],
        "unique_opponents_mean": s(opponents)["mean"],
        "unique_opponents_max": s(opponents)["max"],
        "responses_per_node_min": s(responses_per_node)["min"],
        "responses_per_node_median": s(responses_per_node)["median"],
        "responses_per_node_mean": s(responses_per_node)["mean"],
        "responses_per_node_max": s(responses_per_node)["max"],
        "n_components": int(n_components),
        "largest_component_nodes": int(comp_sizes.max()),
        "largest_component_share": float(comp_sizes.max() / len(nodes)),
        "n_singleton_components": int((comp_sizes == 1).sum()),
        "directed_n_scc": int(n_scc),
        "directed_largest_scc_nodes": int(scc_sizes.max()),
        "directed_largest_scc_share": float(scc_sizes.max() / len(nodes)),
    }
    return [{"graph_level": level, "metric": k, "value": v}
            for k, v in flat.items()]


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run_graphs(config_path: Path) -> int:
    root = config_path.resolve().parent
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    artifacts_dir = root / cfg["paths"]["artifacts_dir"]
    tables_dir = root / cfg["paths"]["tables_dir"]
    figures_dir = root / cfg["paths"].get("figures_dir", "figures")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(artifacts_dir / "split_manifest.parquet")
    rows = pd.read_parquet(artifacts_dir / "rows_canonical.parquet")
    rows = rows[rows["source"] == "train"].copy()

    fitting_ids = set(
        manifest.loc[manifest["partition"] == "fitting", "row_id"]
    )
    fitting = rows[rows["row_id"].isin(fitting_ids)]

    # --- Undirected graphs on fitting rows -----------------------------------
    photo_nodes, photo_edges, photo_adj = build_undirected_graph(
        fitting["a_image"], fitting["b_image"]
    )
    cat_nodes, cat_edges, cat_adj = build_undirected_graph(
        fitting["a_cat"], fitting["b_cat"]
    )

    # --- Directed winner-to-loser graphs from fitting outcomes ----------------
    y = fitting["y_displayed"].astype(int).to_numpy()
    photo_winners = fitting["a_image"].where(
        pd.Series(y == 1, index=fitting.index), fitting["b_image"]
    )
    photo_losers = fitting["b_image"].where(
        pd.Series(y == 1, index=fitting.index), fitting["a_image"]
    )
    cat_winners = fitting["a_cat"].where(
        pd.Series(y == 1, index=fitting.index), fitting["b_cat"]
    )
    cat_losers = fitting["b_cat"].where(
        pd.Series(y == 1, index=fitting.index), fitting["a_cat"]
    )
    photo_digraph = build_directed_graph(photo_winners, photo_losers, photo_nodes)
    cat_digraph = build_directed_graph(cat_winners, cat_losers, cat_nodes)

    # --- Unique opponents vs response counts ----------------------------------
    photo_responses = pd.concat(
        [fitting["a_image"], fitting["b_image"]]
    ).value_counts()
    cat_responses = pd.concat(
        [fitting["a_cat"], fitting["b_cat"]]
    ).value_counts()

    summary_rows = graph_metrics(
        "photo", photo_nodes, len(photo_edges), len(fitting),
        photo_adj, photo_responses.reindex(photo_nodes), photo_digraph,
    ) + graph_metrics(
        "cat", cat_nodes, len(cat_edges), len(fitting),
        cat_adj, cat_responses.reindex(cat_nodes), cat_digraph,
    )
    pd.DataFrame(summary_rows).to_csv(tables_dir / "graph_summary.csv",
                                      index=False)

    # --- Photo components and strict-G2 eligibility ---------------------------
    n_components, comp_labels = cs.connected_components(photo_adj,
                                                        directed=False)
    component_of = dict(zip(photo_nodes, comp_labels))
    pd.DataFrame(
        {"photo": photo_nodes, "fitting_component": comp_labels}
    ).to_parquet(artifacts_dir / "photo_components.parquet", index=False)

    in_fitting = set(photo_nodes)

    def eligibility(a_img: str, b_img: str) -> str:
        if a_img not in in_fitting or b_img not in in_fitting:
            return "unseen_photo"
        if component_of[a_img] != component_of[b_img]:
            return "cross_component"
        return "eligible_strict_g2"

    held = rows.merge(
        manifest[["row_id", "partition"]], on="row_id", validate="one_to_one"
    )
    held = held[held["partition"].isin(["validation", "test"])].copy()
    held["eligibility"] = [
        eligibility(a, b) for a, b in zip(held["a_image"], held["b_image"])
    ]
    held["eligible_strict_g2"] = held["eligibility"] == "eligible_strict_g2"

    # --- Freeze flags into the manifest ---------------------------------------
    flags = held.set_index("row_id")[["eligibility", "eligible_strict_g2"]]
    manifest["eligibility"] = manifest["row_id"].map(flags["eligibility"])
    manifest["eligible_strict_g2"] = (
        manifest["row_id"].map(flags["eligible_strict_g2"])
    )
    manifest.loc[manifest["partition"] == "fitting", "eligibility"] = (
        "fitting_partition"
    )
    manifest["eligible_strict_g2"] = (
        manifest["eligible_strict_g2"].fillna(False).astype(bool)
    )
    manifest.to_parquet(artifacts_dir / "split_manifest.parquet", index=False)

    # --- Coverage tabulation ----------------------------------------------------
    coverage = (
        held.groupby(["partition", "eligibility"])
        .agg(n_rows=("row_id", "size"), n_pairs=("pair_id", "nunique"))
        .reset_index()
    )
    by_type = (
        held.groupby(["partition", "eligibility", "comparison_type"])
        .agg(n_rows=("row_id", "size"), n_pairs=("pair_id", "nunique"))
        .reset_index()
    )
    coverage.to_csv(tables_dir / "split_coverage.csv", index=False)
    by_type.to_csv(tables_dir / "split_coverage_by_type.csv", index=False)

    # --- Adequacy labels (plan Stage 4 decision rules) --------------------------
    def strict_counts(partition: str) -> tuple[int, int]:
        sub = held[
            (held["partition"] == partition) & held["eligible_strict_g2"]
        ]
        return len(sub), sub["pair_id"].nunique()

    val_rows, val_pairs = strict_counts("validation")
    test_rows, test_pairs = strict_counts("test")
    labels = {
        "strict_validation": {
            "n_responses": int(val_rows),
            "n_pairs": int(val_pairs),
            "label": "adequate"
            if val_rows >= STRICT_VALIDATION_MIN_RESPONSES
            and val_pairs >= STRICT_VALIDATION_MIN_PAIRS
            else "limited model-selection evidence",
        },
        "strict_test": {
            "n_responses": int(test_rows),
            "n_pairs": int(test_pairs),
            "label": "adequate"
            if test_rows >= STRICT_TEST_MIN_RESPONSES
            and test_pairs >= STRICT_TEST_MIN_PAIRS
            else "primary result exploratory",
        },
        "g2_claim_allowed": bool(val_rows > 0 and test_rows > 0),
        "frozen_without_heldout_outcomes": True,
        "note": "This strict-G2 row set is used for all primary model "
                "comparisons (plan Stage 4).",
    }
    with (artifacts_dir / "eligibility.json").open("w", encoding="utf-8") as fh:
        json.dump(labels, fh, indent=2)

    # --- Figure ------------------------------------------------------------------
    comp_sizes = np.bincount(comp_labels, minlength=n_components)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.bar(
        np.arange(1, n_components + 1),
        np.sort(comp_sizes)[::-1],
        color="steelblue",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Fitting-photo component (rank)")
    ax.set_ylabel("Photos in component (log scale)")
    ax.set_title(f"Photo graph: {n_components} component(s), "
                 f"largest covers {comp_sizes.max()}/{len(photo_nodes)} photos")

    ax = axes[1]
    opp = pd.Series(np.diff(photo_adj.indptr), index=photo_nodes)
    resp = photo_responses.reindex(photo_nodes)
    ax.scatter(resp, opp, s=8, alpha=0.35, color="darkorange")
    ax.set_xlabel("Fitting responses incident to photo")
    ax.set_ylabel("Unique fitting opponents")
    ax.set_title("Support per photo (fitting partition)")
    fig.tight_layout()
    fig.savefig(figures_dir / "graph_support.pdf")
    plt.close(fig)

    # --- Console summary ----------------------------------------------------------
    print("Stage 4 graphs and eligibility frozen.")
    print(f"  photo graph: {len(photo_nodes)} photos, {len(photo_edges)} unique "
          f"edges, {n_components} component(s), largest "
          f"{comp_sizes.max()} ({comp_sizes.max() / len(photo_nodes):.1%})")
    photo_scc = next(r for r in summary_rows
                     if r["graph_level"] == "photo"
                     and r["metric"] == "directed_n_scc")
    photo_scc_max = next(r for r in summary_rows
                         if r["graph_level"] == "photo"
                         and r["metric"] == "directed_largest_scc_nodes")
    print(f"  cat graph: {len(cat_nodes)} cats, {len(cat_edges)} unique edges")
    print(f"  directed photo graph: {photo_scc['value']} SCCs, largest "
          f"{photo_scc_max['value']} photos")
    print(f"  strict validation: {val_rows} responses, {val_pairs} pairs -> "
          f"{labels['strict_validation']['label']}")
    print(f"  strict test: {test_rows} responses, {test_pairs} pairs -> "
          f"{labels['strict_test']['label']}")
    print("  exclusions (validation/test):")
    for _, r in coverage.iterrows():
        print(f"    {r['partition']}/{r['eligibility']}: {r['n_rows']} rows, "
              f"{r['n_pairs']} pairs")
    if not labels["g2_claim_allowed"]:
        print("  WARNING: strict eligibility is empty - no G2 predictive "
              "evaluation may be claimed.")
        return 1
    print("  GO: eligibility frozen without using held-out outcomes.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 4 graphs/eligibility.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config.yaml",
    )
    args = parser.parse_args(argv)
    return run_graphs(args.config)


if __name__ == "__main__":
    sys.exit(main())
