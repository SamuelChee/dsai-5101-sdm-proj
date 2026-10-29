"""Stage 5 - Validate the estimator and run simulations A/B/C.

Demonstrates the three-gap concepts under known truth (plan Stage 5). These
are properties of the specified artificial worlds, not of the cat dataset.

* A - Correct BT: three items, equal pair weights, pair probabilities
  (2/3, 2/3, 0.8). These probabilities are exactly Bradley-Terry
  representable with scores (ln 2, 0, -ln 2): logits ln 2, ln 2, ln 4.
  (The plan's informal "(2, 0, -2)" score note would imply sigma(2) = 0.881,
  inconsistent with both the stated probabilities and the 0.5911 oracle
  reference; the probabilities - confirmed by the reference values - define
  the simulation.) Responses per pair: 10, 50, 200; 100 repetitions;
  no-position BT fitted with lambda = 1e-6; evaluated by EXACT expected loss
  under the known probabilities, never a noisy simulated test set.
* B - Recorded context: two equally common groups with score vectors u and
  -u (u = (ln 2, 0, -ln 2)); equal numbers per pair per group; pooled BT vs
  separate group BT; exact expected risks.
* C - Cyclic probabilities: qAB = qBC = qCA = 0.8, equal weights; BT and a
  scalar feature scorer with one-hot item features (algebraically the same
  model - their agreement is itself checked); compared with the
  unrestricted pair oracle.

Mathematical reference values are computed exactly and stored SEPARATELY
from simulated fitted results (artifacts/simulation_references.json).

Outputs
-------
artifacts/simulation_results.parquet
artifacts/simulation_references.json
figures/simulations_ABC.pdf
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
import yaml

from src.bt import (
    expected_log_loss,
    fit_penalized_logistic,
)

LN2 = float(np.log(2.0))
U_VECTOR = np.array([LN2, 0.0, -LN2])  # probabilities 2/3, 2/3, 0.8
N_REPETITIONS = 100
SIM_A_LAMBDA = 1e-6
SIM_B_N_PER_PAIR_PER_GROUP = 50
SIM_C_N_PER_PAIR = 50


def binary_entropy(p: float) -> float:
    """Bernoulli entropy in nats, with 0 log 0 := 0."""
    if p in (0.0, 1.0):
        return 0.0
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))


def bt_design(pairs: list[tuple[int, int]], n_items: int) -> sp.csr_matrix:
    """Sparse BT incidence design: +1 at the displayed-A item, -1 at B."""
    ii = np.arange(len(pairs))
    a_idx = [p[0] for p in pairs]
    b_idx = [p[1] for p in pairs]
    return sp.coo_matrix(
        (
            np.concatenate([np.ones(len(pairs)), -np.ones(len(pairs))]),
            (
                np.concatenate([ii, ii]),
                np.concatenate([a_idx, b_idx]),
            ),
        ),
        shape=(len(pairs), n_items),
    ).tocsr()


def fit_bt_on_counts(
    pairs: list[tuple[int, int]],
    wins_first: np.ndarray,
    n_per_pair: np.ndarray,
    lam: float,
) -> "object":
    """Fit no-position BT on (possibly fractional) aggregated counts."""
    X = bt_design(pairs, max(max(p) for p in pairs) + 1)
    y = wins_first / n_per_pair
    return fit_penalized_logistic(X, y, lam, weights=n_per_pair)


def pair_probabilities(fit: "object", pairs: list[tuple[int, int]],
                       n_items: int) -> np.ndarray:
    return fit.predict_proba(bt_design(pairs, n_items))


# ---------------------------------------------------------------------------
# Simulation definitions
# ---------------------------------------------------------------------------

PAIRS_ABC = [(0, 1), (1, 2), (0, 2)]
P_A = np.array([2 / 3, 2 / 3, 0.8])  # P(first item chosen) per pair


def run_simulation_a(rng: np.random.Generator) -> list[dict]:
    records = []
    for n_per_pair in (10, 50, 200):
        for rep in range(N_REPETITIONS):
            wins = rng.binomial(n_per_pair, P_A)
            fit = fit_bt_on_counts(
                PAIRS_ABC,
                wins.astype(float),
                np.full(3, float(n_per_pair)),
                SIM_A_LAMBDA,
            )
            if not fit.success:
                records.append({
                    "simulation": "A", "n_per_pair": n_per_pair,
                    "repetition": rep, "model": "BT_fitted",
                    "expected_loss": np.nan, "fit_status": "FAILED",
                })
                continue
            p_hat = pair_probabilities(fit, PAIRS_ABC, 3)
            loss = float(expected_log_loss(P_A, p_hat).mean())
            records.append({
                "simulation": "A", "n_per_pair": n_per_pair,
                "repetition": rep, "model": "BT_fitted",
                "expected_loss": loss, "fit_status": "ok",
            })
    return records


def run_simulation_b(rng: np.random.Generator) -> list[dict]:
    # Per group g: P(pair e resolved for first item) = sigma(s_g . u_diff_e)
    # with s_0 = +1, s_1 = -1.
    p_group = np.stack(
        [1 / (1 + np.exp(-(U_VECTOR[a] - U_VECTOR[b]))) for a, b in PAIRS_ABC]
    )
    p_per_group = np.stack([p_group, 1.0 - p_group])  # group 1 has -u
    records = []
    n = SIM_B_N_PER_PAIR_PER_GROUP
    for rep in range(N_REPETITIONS):
        wins_by_group = [
            rng.binomial(n, p_per_group[g]) for g in range(2)
        ]
        # Pooled fit: aggregate both groups per pair.
        wins_pool = sum(wins_by_group).astype(float)
        fit_pool = fit_bt_on_counts(
            PAIRS_ABC, wins_pool, np.full(3, 2.0 * n), SIM_A_LAMBDA
        )
        # Contextual fit: separate group BT scores.
        group_fits = [
            fit_bt_on_counts(
                PAIRS_ABC, wins_by_group[g].astype(float),
                np.full(3, float(n)), SIM_A_LAMBDA,
            )
            for g in range(2)
        ]
        ok = fit_pool.success and all(f.success for f in group_fits)
        if not ok:
            for model in ("pooled_BT", "contextual_BT"):
                records.append({
                    "simulation": "B",
                    "n_per_pair_per_group": n,
                    "repetition": rep, "model": model,
                    "expected_loss": np.nan, "fit_status": "FAILED",
                })
            continue
        p_pool = pair_probabilities(fit_pool, PAIRS_ABC, 3)
        # Exact expected pooled risk: truth is the equal mixture (p = 0.5).
        pooled_loss = float(
            expected_log_loss(np.full(3, 0.5), p_pool).mean()
        )
        contextual_loss = float(
            np.mean([
                expected_log_loss(
                    p_per_group[g], pair_probabilities(group_fits[g],
                                                       PAIRS_ABC, 3)
                ).mean()
                for g in range(2)
            ])
        )
        records.append({
            "simulation": "B", "n_per_pair_per_group": n,
            "repetition": rep, "model": "pooled_BT",
            "expected_loss": pooled_loss, "fit_status": "ok",
        })
        records.append({
            "simulation": "B", "n_per_pair_per_group": n,
            "repetition": rep, "model": "contextual_BT",
            "expected_loss": contextual_loss, "fit_status": "ok",
        })
    return records


def run_simulation_c(rng: np.random.Generator) -> list[dict]:
    # Cycle: 0 beats 1, 1 beats 2, 2 beats 0, each with probability 0.8.
    pairs_cyclic = [(0, 1), (1, 2), (2, 0)]
    p_true = np.full(3, 0.8)
    records = []
    n = SIM_C_N_PER_PAIR
    for rep in range(N_REPETITIONS):
        wins = rng.binomial(n, p_true)
        fit_bt = fit_bt_on_counts(
            pairs_cyclic, wins.astype(float), np.full(3, float(n)),
            SIM_A_LAMBDA,
        )
        # Scalar feature scorer with one-hot item features: score = w . x
        # with x one-hot is algebraically the BT design above. Fit through
        # the same estimator on the one-hot representation to demonstrate
        # the equivalence explicitly.
        one_hot_rows = []
        for a, b in pairs_cyclic:
            row_a = np.zeros(3)
            row_a[a] = 1.0
            row_b = np.zeros(3)
            row_b[b] = 1.0
            one_hot_rows.append((row_a, row_b))
        X_onehot = sp.csr_matrix(
            np.array([ra - rb for ra, rb in one_hot_rows])
        )
        fit_scorer = fit_penalized_logistic(
            X_onehot, wins / n, SIM_A_LAMBDA, weights=np.full(3, float(n))
        )
        if not (fit_bt.success and fit_scorer.success):
            for model in ("BT", "one_hot_scorer"):
                records.append({
                    "simulation": "C", "n_per_pair": n, "repetition": rep,
                    "model": model, "expected_loss": np.nan,
                    "fit_status": "FAILED",
                })
            continue
        p_bt = pair_probabilities(fit_bt, pairs_cyclic, 3)
        p_scorer = fit_scorer.predict_proba(X_onehot)
        # The two parameterizations must agree (same model class).
        agree = float(np.max(np.abs(p_bt - p_scorer)))
        records.append({
            "simulation": "C", "n_per_pair": n, "repetition": rep,
            "model": "BT",
            "expected_loss": float(expected_log_loss(p_true, p_bt).mean()),
            "fit_status": "ok", "param_agreement": agree,
        })
        records.append({
            "simulation": "C", "n_per_pair": n, "repetition": rep,
            "model": "one_hot_scorer",
            "expected_loss": float(
                expected_log_loss(p_true, p_scorer).mean()
            ),
            "fit_status": "ok", "param_agreement": agree,
        })
    return records


def reference_values() -> dict:
    """Exact mathematical references - stored separately from fitted runs."""
    a_oracle = float(np.mean([binary_entropy(p) for p in P_A]))
    return {
        "label": "Mathematical properties of the specified simulations - "
                 "not dataset findings",
        "A": {
            "oracle_expected_loss": a_oracle,
            "note": "mean of h(2/3), h(2/3), h(0.8); plan reference 0.5911",
        },
        "B": {
            "pooled_oracle": LN2,
            "contextual_oracle": a_oracle,
            "note": "equal mixture gives p=0.5 for every pair (h=ln 2); "
                    "plan references 0.6931 and 0.5911",
        },
        "C": {
            "pair_oracle": binary_entropy(0.8),
            "best_bt": LN2,
            "restriction_gap": LN2 - binary_entropy(0.8),
            "note": "symmetric cycle forces equal BT scores under equal "
                    "weights; plan references 0.5004, 0.6931, gap 0.1927",
        },
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure(results: pd.DataFrame, refs: dict, figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # Panel A: fitted expected loss vs responses per pair.
    ax = axes[0]
    a = results[results["simulation"] == "A"].dropna(subset=["expected_loss"])
    positions = [10, 50, 200]
    data = [a.loc[a["n_per_pair"] == n, "expected_loss"].to_numpy()
            for n in positions]
    bp = ax.boxplot(data, positions=[0, 1, 2], widths=0.5, showfliers=False,
                    medianprops=dict(color="#c62828"))
    ax.axhline(refs["A"]["oracle_expected_loss"], ls="--", color="#2e7d32",
               label=f"oracle reference {refs['A']['oracle_expected_loss']:.4f}")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([str(p) for p in positions])
    ax.set_xlabel("Responses per pair")
    ax.set_ylabel("Exact expected log loss")
    ax.set_title("A - BT is exactly correct")
    ax.legend(fontsize=8)

    # Panel B: pooled vs contextual.
    ax = axes[1]
    b = results[results["simulation"] == "B"].dropna(subset=["expected_loss"])
    means = b.groupby("model")["expected_loss"].mean()
    sems = b.groupby("model")["expected_loss"].sem()
    x = np.arange(2)
    ax.bar(x - 0.2, [refs["B"]["pooled_oracle"],
                     refs["B"]["contextual_oracle"]],
           width=0.4, color="#90a4ae",
           label="exact oracle (mathematical)")
    ax.bar(x + 0.2, [means["pooled_BT"], means["contextual_BT"]],
           width=0.4, color="#1565c0",
           yerr=[sems["pooled_BT"], sems["contextual_BT"]],
           label=f"fitted mean (n={N_REPETITIONS} sims)")
    ax.set_xticks(x)
    ax.set_xticklabels(["pooled BT", "contextual BT"])
    ax.set_ylabel("Exact expected log loss")
    ax.set_title("B - recorded opposite-score groups")
    ax.legend(fontsize=8)

    # Panel C: cycle - restriction gap.
    ax = axes[2]
    c = results[results["simulation"] == "C"].dropna(subset=["expected_loss"])
    cmeans = c.groupby("model")["expected_loss"].mean()
    labels = ["pair oracle\n(reference)", "best BT\n(reference)",
              "fitted BT", "fitted one-hot\nscorer"]
    values = [refs["C"]["pair_oracle"], refs["C"]["best_bt"],
              cmeans["BT"], cmeans["one_hot_scorer"]]
    colors = ["#2e7d32", "#90a4ae", "#1565c0", "#5e35b1"]
    ax.bar(np.arange(4), values, color=colors)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Exact expected log loss")
    ax.set_title("C - 0.8 cycle: restriction gap "
                 f"{refs['C']['restriction_gap']:.4f}")

    fig.suptitle(
        "Known-truth simulations (mathematical properties of the specified "
        "designs - not dataset findings)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figures_dir / "simulations_ABC.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run_simulations(config_path: Path) -> int:
    root = config_path.resolve().parent
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    artifacts_dir = root / cfg["paths"]["artifacts_dir"]
    figures_dir = root / cfg["paths"].get("figures_dir", "figures")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(cfg["seeds"]["simulation"]))

    records = (
        run_simulation_a(rng) + run_simulation_b(rng) + run_simulation_c(rng)
    )
    results = pd.DataFrame(records)
    refs = reference_values()

    results.to_parquet(artifacts_dir / "simulation_results.parquet",
                       index=False)
    with (artifacts_dir / "simulation_references.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(refs, fh, indent=2)
    make_figure(results, refs, figures_dir)

    n_failed = int((results["fit_status"] == "FAILED").sum())
    ok = results.dropna(subset=["expected_loss"])
    print("Stage 5 simulations complete.")
    print(f"  runs: {len(results)} ({n_failed} FAILED fits - kept visible)")
    for sim, group in ok.groupby("simulation"):
        summary = group.groupby("model")["expected_loss"].agg(
            ["mean", "min", "max"]
        )
        for model, row in summary.iterrows():
            print(f"  {sim}/{model}: mean={row['mean']:.4f} "
                  f"[{row['min']:.4f}, {row['max']:.4f}]")
    print(f"  references: A oracle {refs['A']['oracle_expected_loss']:.4f}; "
          f"B pooled {refs['B']['pooled_oracle']:.4f} / contextual "
          f"{refs['B']['contextual_oracle']:.4f}; C pair oracle "
          f"{refs['C']['pair_oracle']:.4f} / best BT {refs['C']['best_bt']:.4f}"
          f" (gap {refs['C']['restriction_gap']:.4f})")
    if n_failed:
        print("  NOTE: failed fits present - inspect before proceeding.")
        return 1
    print("  Next gate: pytest tests/test_bt_objective.py "
          "tests/test_recovery.py must pass.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 5 simulations A/B/C.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config.yaml",
    )
    args = parser.parse_args(argv)
    return run_simulations(args.config)


if __name__ == "__main__":
    sys.exit(main())
