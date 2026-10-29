"""Stage 9 - Development learning curves and controls.

Learning curves (plan Stage 9):
* models M1P, M2P, M3P, M4P (all available);
* nested 25% / 50% / 75% / 100% fitting-PAIR subsets under the two fixed
  seeds (nested prefixes of one shuffle per seed);
* the common validation subset contains strict-validation rows eligible
  under the SMALLEST fitting graph of BOTH seeds (photo graph: both photos
  present and in the same component);
* preprocessing (feature scaler, context encoding) is refit and penalties
  retuned over the frozen grid at every size;
* training and validation losses are saved; validation loss is plotted
  against the actual fitting-response count.

Coverage fallback: if the common validation subset has fewer than 200
responses or 50 pairs, the design is repeated once with 50% / 75% / 100%;
if still insufficient, learning curves are omitted and the coverage failure
is reported.

Controls (debugging checks, not leakage proofs):
* recheck the orientation identity and row/count objective equivalence;
* 50 repetitions of M2P and M4P at their frozen selected settings with
  independent Bernoulli(0.5) labels (control seed), compared with the 0.5
  baseline (ln 2); mean apparent improvement above 0.01 is flagged.

The internal test partition is never loaded.

Outputs
-------
tables/learning_curves.csv
figures/learning_curves.pdf
tables/control_results.csv
artifacts/learning_curve_summary.json
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

from src.bt import (
    fit_penalized_logistic,
    penalized_logistic_objective,
)
from src.identity_models import (
    RIDGE_GRID,
    incidence_design,
    row_log_loss,
    select_lambda,
)
from src.split import load_partition_rows
from src.visual_models import (
    STD_FLOOR,
    build_context_encoding,
    m4p_design,
)

FRACTIONS_FULL = (0.25, 0.50, 0.75, 1.00)
FRACTIONS_FALLBACK = (0.50, 0.75, 1.00)
MIN_COMMON_RESPONSES = 200
MIN_COMMON_PAIRS = 50
CONTROL_REPETITIONS = 50
CONTROL_FLAG_THRESHOLD = 0.01
LN2 = float(np.log(2.0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def subset_photo_graph(pairs: pd.DataFrame) -> tuple[set, dict]:
    """Photo graph over subset pairs: node set and component per photo."""
    nodes = np.sort(
        pd.unique(pd.concat([pairs["i_image"], pairs["j_image"]]))
    )
    index = {p: k for k, p in enumerate(nodes)}
    edges = {
        (min(index[a], index[b]), max(index[a], index[b]))
        for a, b in zip(pairs["i_image"], pairs["j_image"])
    }
    if edges:
        ii, jj = map(np.array, zip(*edges))
        adj = sp.coo_matrix(
            (np.ones(len(ii)), (ii, jj)), shape=(len(nodes), len(nodes))
        ).tocsr()
        adj = adj + adj.T
    else:
        adj = sp.csr_matrix((len(nodes), len(nodes)))
    _, labels = cs.connected_components(adj, directed=False)
    return set(nodes), dict(zip(nodes, labels))


def eligible_rows(
    rows: pd.DataFrame, nodes: set, component_of: dict
) -> pd.Series:
    both_in = rows["a_image"].isin(nodes) & rows["b_image"].isin(nodes)
    same_comp = [
        component_of.get(a, -1) == component_of.get(b, -2)
        for a, b in zip(rows["a_image"], rows["b_image"])
    ]
    return both_in & pd.Series(same_comp, index=rows.index)


def fit_model_at_size(
    model: str,
    sub: pd.DataFrame,
    common_val: pd.DataFrame,
    features: pd.DataFrame,
) -> dict:
    """Refit preprocessing and tune lambda at one subset size."""
    y_sub = sub["y_displayed"].astype(int).to_numpy(dtype=float)
    y_val = common_val["y_displayed"].astype(int).to_numpy(dtype=float)

    if model in ("M1P", "M2P"):
        left = "a_cat" if model == "M1P" else "a_image"
        right = "b_cat" if model == "M1P" else "b_image"
        items = np.sort(pd.unique(pd.concat([sub[left], sub[right]])))
        index = {it: k for k, it in enumerate(items)}
        # Common-validation rows reference photos present in the smallest
        # graph, hence present in every (nested) subset; unseen cats cannot
        # occur for the same reason.
        X_sub = incidence_design(sub[left], sub[right], index)
        X_val = incidence_design(
            common_val[left], common_val[right], index
        )
        n_params = X_sub.shape[1]
        coef_kind = "incidence"
    else:
        # Refit the scaler on unique subset photos.
        photos = sorted(set(sub["a_image"]) | set(sub["b_image"]))
        fit_feats = features.loc[[p for p in photos if p in features.index]]
        means = fit_feats.mean()
        stds = fit_feats.std(ddof=0)
        kept = stds[stds >= STD_FLOOR].index.tolist()
        z = (features[kept] - means[kept]) / stds[kept]
        dz_sub = (
            z.loc[sub["a_image"].to_numpy()].to_numpy()
            - z.loc[sub["b_image"].to_numpy()].to_numpy()
        )
        dz_val = (
            z.loc[common_val["a_image"].to_numpy()].to_numpy()
            - z.loc[common_val["b_image"].to_numpy()].to_numpy()
        )
        if model == "M3P":
            X_sub, X_val = dz_sub, dz_val
        else:
            encoding = build_context_encoding(sub)
            if not encoding["m4p_supported"]:
                return {"model": model, "status": "unsupported"}
            igroups = encoding["interaction_groups"]
            pet_sub = sub["pet_care"].where(
                sub["pet_care"].str.strip() != "", "unknown"
            )
            pet_val = common_val["pet_care"].where(
                common_val["pet_care"].str.strip() != "", "unknown"
            )
            X_sub = m4p_design(dz_sub, pet_sub, igroups)
            X_val = m4p_design(dz_val, pet_val, igroups)
        n_params = X_sub.shape[1]
        coef_kind = "features"

    grid_rows: list[dict] = []
    fits = {}
    for lam in RIDGE_GRID:
        fit = fit_penalized_logistic(X_sub, y_sub, lam, with_position=True)
        if fit.success:
            p_val = fit.predict_proba(X_val)
            val_loss = row_log_loss(y_val, p_val)
        else:
            val_loss = np.nan
        grid_rows.append({"lambda": lam, "fit_success": fit.success,
                          "validation_log_loss": val_loss})
        fits[lam] = fit
    lam_star = select_lambda(grid_rows)
    if lam_star is None:
        return {"model": model, "status": "all_fits_failed"}
    best = fits[lam_star]
    return {
        "model": model,
        "status": "ok",
        "lambda": lam_star,
        "train_loss": row_log_loss(y_sub, best.predict_proba(X_sub)),
        "validation_loss": row_log_loss(y_val, best.predict_proba(X_val)),
        "n_params": n_params,
        "coef_kind": coef_kind,
    }


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def recheck_orientation(rows: pd.DataFrame) -> bool:
    """y_canonical == y_displayed if s_t == 1 else 1 - y_displayed."""
    y = rows["y_displayed"].astype(int).to_numpy()
    s = rows["s_t"].to_numpy()
    return bool(
        (rows["y_canonical"].astype(int).to_numpy()
         == np.where(s == 1, y, 1 - y)).all()
    )


def recheck_count_row_equivalence(fitting: pd.DataFrame) -> bool:
    """Row-level objective equals count-level objective on real design."""
    rng = np.random.default_rng(0)
    items = np.sort(
        pd.unique(pd.concat([fitting["a_image"], fitting["b_image"]]))
    )
    index = {it: k for k, it in enumerate(items)}
    X_rows = incidence_design(fitting["a_image"], fitting["b_image"], index)
    y_rows = fitting["y_displayed"].astype(int).to_numpy(dtype=float)

    grouped = (
        fitting.groupby(["a_image", "b_image"])["y_displayed"]
        .agg(["size", "sum"])
        .reset_index()
    )
    X_counts = incidence_design(grouped["a_image"], grouped["b_image"], index)
    weights = grouped["size"].to_numpy(dtype=float)
    y_counts = grouped["sum"].to_numpy(dtype=float) / weights

    theta = rng.normal(size=X_rows.shape[1] + 1)
    obj_rows, _ = penalized_logistic_objective(
        theta, X_rows, y_rows, None, 1e-4, True
    )
    obj_counts, _ = penalized_logistic_objective(
        theta, X_counts, y_counts, weights, 1e-4, True
    )
    return bool(np.isclose(obj_rows, obj_counts, rtol=1e-10, atol=1e-12))


def run_controls(
    fitting: pd.DataFrame,
    strict_val: pd.DataFrame,
    features: pd.DataFrame,
    scaler: dict,
    encoding: dict,
    frozen_lambdas: dict,
    control_seed: int,
) -> pd.DataFrame:
    """50 Bernoulli(0.5)-label repetitions of M2P and M4P at frozen settings."""
    kept = list(scaler["means"].keys())
    z = (features[kept] - pd.Series(scaler["means"])) / pd.Series(
        scaler["stds"]
    )
    igroups = encoding["interaction_groups"]

    # Frozen designs (frozen preprocessing, per the selected settings).
    photos = np.sort(
        pd.unique(pd.concat([fitting["a_image"], fitting["b_image"]]))
    )
    index = {p: k for k, p in enumerate(photos)}
    designs = {
        "M2P": (
            incidence_design(fitting["a_image"], fitting["b_image"], index),
            incidence_design(
                strict_val["a_image"], strict_val["b_image"], index
            ),
        ),
    }
    if encoding["m4p_supported"]:
        dz_fit = (
            z.loc[fitting["a_image"].to_numpy()].to_numpy()
            - z.loc[fitting["b_image"].to_numpy()].to_numpy()
        )
        dz_val = (
            z.loc[strict_val["a_image"].to_numpy()].to_numpy()
            - z.loc[strict_val["b_image"].to_numpy()].to_numpy()
        )
        pet_fit = fitting["pet_care"].where(
            fitting["pet_care"].str.strip() != "", "unknown"
        )
        pet_val = strict_val["pet_care"].where(
            strict_val["pet_care"].str.strip() != "", "unknown"
        )
        designs["M4P"] = (
            m4p_design(dz_fit, pet_fit, igroups),
            m4p_design(dz_val, pet_val, igroups),
        )

    rng = np.random.default_rng(control_seed)
    records = []
    n_fit, n_val = len(fitting), len(strict_val)
    for rep in range(CONTROL_REPETITIONS):
        y_fit = rng.binomial(1, 0.5, n_fit).astype(float)
        y_val = rng.binomial(1, 0.5, n_val).astype(float)
        for model, (X_fit, X_val) in designs.items():
            fit = fit_penalized_logistic(
                X_fit, y_fit, frozen_lambdas[model], with_position=True
            )
            val_loss = (
                row_log_loss(y_val, fit.predict_proba(X_val))
                if fit.success else np.nan
            )
            records.append(
                {
                    "control": "synthetic_bernoulli_0.5_labels",
                    "model": model,
                    "repetition": rep,
                    "lambda": frozen_lambdas[model],
                    "fit_success": fit.success,
                    "baseline_loss": LN2,
                    "validation_loss": val_loss,
                    "apparent_improvement": LN2 - val_loss,
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run_learning_curves(config_path: Path) -> int:
    root = config_path.resolve().parent
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    artifacts_dir = root / cfg["paths"]["artifacts_dir"]
    tables_dir = root / cfg["paths"]["tables_dir"]
    figures_dir = root / cfg["paths"].get("figures_dir", "figures")
    models_dir = root / cfg["paths"].get("models_dir", "models")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fitting = load_partition_rows(config_path, "fitting")
    validation = load_partition_rows(config_path, "validation")
    manifest = pd.read_parquet(artifacts_dir / "split_manifest.parquet")
    eligible_ids = set(
        manifest.loc[
            (manifest["partition"] == "validation")
            & manifest["eligible_strict_g2"],
            "row_id",
        ]
    )
    strict_val = validation[validation["row_id"].isin(eligible_ids)].copy()
    features = pd.read_parquet(artifacts_dir / "image_features.parquet")
    with (artifacts_dir / "feature_scaler.json").open(
        "r", encoding="utf-8"
    ) as fh:
        scaler = json.load(fh)
    with (artifacts_dir / "context_encoding.json").open(
        "r", encoding="utf-8"
    ) as fh:
        encoding = json.load(fh)
    frozen_lambdas = {}
    for model in ("M2P", "M4P"):
        with (models_dir / model / "model.json").open(
            "r", encoding="utf-8"
        ) as fh:
            frozen_lambdas[model] = json.load(fh)["lambda_selected"]

    seeds = [int(s) for s in cfg["seeds"]["learning_curves"]]
    pairs = (
        fitting.groupby("pair_id", as_index=False)
        .agg(n_responses=("row_id", "size"))
    )
    pair_ids = pairs["pair_id"].to_numpy()

    # --- Nested subsets and common validation set (with fallback) --------------
    records: list[dict] = []
    coverage_note = None
    for fractions, attempt in ((FRACTIONS_FULL, "primary"),
                               (FRACTIONS_FALLBACK, "fallback")):
        # Smallest-graph eligibility per seed, then intersect across seeds.
        common_ids: set | None = None
        for seed in seeds:
            rng = np.random.default_rng(seed)
            shuffled = pair_ids[rng.permutation(len(pair_ids))]
            k_min = int(round(min(fractions) * len(pair_ids)))
            small_pairs = fitting[fitting["pair_id"].isin(shuffled[:k_min])]
            nodes, component_of = subset_photo_graph(small_pairs)
            eligible = eligible_rows(strict_val, nodes, component_of)
            ids = set(strict_val.loc[eligible, "row_id"])
            common_ids = ids if common_ids is None else common_ids & ids
        common_val = strict_val[
            strict_val["row_id"].isin(common_ids)
        ].copy()
        n_common_rows = len(common_val)
        n_common_pairs = common_val["pair_id"].nunique()
        if (n_common_rows >= MIN_COMMON_RESPONSES
                and n_common_pairs >= MIN_COMMON_PAIRS):
            break
        coverage_note = (
            f"{attempt} design: common validation subset too small "
            f"({n_common_rows} responses, {n_common_pairs} pairs)"
        )
        records = []
    else:
        coverage_note = (
            "coverage failure: learning curves omitted (common validation "
            "set insufficient under both designs)"
        )

    if records == [] and coverage_note and "omitted" in coverage_note:
        summary = {"status": "omitted", "reason": coverage_note}
        with (artifacts_dir / "learning_curve_summary.json").open(
            "w", encoding="utf-8"
        ) as fh:
            json.dump(summary, fh, indent=2)
        print(f"Stage 9: {coverage_note}")
        return 1

    fractions = FRACTIONS_FALLBACK if coverage_note else FRACTIONS_FULL
    for seed in seeds:
        rng = np.random.default_rng(seed)
        shuffled = pair_ids[rng.permutation(len(pair_ids))]
        for frac in fractions:
            k = int(round(frac * len(pair_ids)))
            sub = fitting[fitting["pair_id"].isin(shuffled[:k])]
            for model in ("M1P", "M2P", "M3P", "M4P"):
                result = fit_model_at_size(
                    model, sub, common_val, features
                )
                records.append(
                    {
                        "seed": seed,
                        "fraction": frac,
                        "model": model,
                        "status": result.get("status", "ok"),
                        "lambda_selected": result.get("lambda"),
                        "n_fitting_pairs": int(sub["pair_id"].nunique()),
                        "n_fitting_responses": len(sub),
                        "train_loss": result.get("train_loss"),
                        "validation_loss": result.get("validation_loss"),
                        "n_common_validation_rows": n_common_rows,
                        "n_common_validation_pairs": n_common_pairs,
                        "design": "fallback" if coverage_note else "primary",
                    }
                )

    curves = pd.DataFrame(records)
    curves.to_csv(tables_dir / "learning_curves.csv", index=False)

    # --- Figure ------------------------------------------------------------------
    ok = curves[curves["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = {"M1P": "#1565c0", "M2P": "#2e7d32", "M3P": "#ef6c00",
              "M4P": "#5e35b1"}
    for model, group in ok.groupby("model"):
        for seed, sub in group.groupby("seed"):
            sub = sub.sort_values("n_fitting_responses")
            ax.plot(sub["n_fitting_responses"], sub["validation_loss"],
                    color=colors.get(model), alpha=0.3, lw=1)
        mean_curve = group.groupby("n_fitting_responses")[
            "validation_loss"].mean().sort_index()
        ax.plot(mean_curve.index, mean_curve.values,
                color=colors.get(model), lw=2, marker="o", ms=4,
                label=model)
    ax.set_xlabel("Fitting responses (actual count)")
    ax.set_ylabel("Validation log loss (common eligible subset)")
    ax.set_title(
        f"Learning curves - {len(seeds)} repetitions, common validation "
        f"subset: {n_common_rows:,} responses, {n_common_pairs:,} pairs"
        + (" [FALLBACK design]" if coverage_note else "")
    )
    ax.legend()
    ax.text(0.01, 0.01,
            "A plateau over the tested range is not an estimated oracle "
            "floor.", transform=ax.transAxes, fontsize=8, color="#b71c1c")
    fig.tight_layout()
    fig.savefig(figures_dir / "learning_curves.pdf")
    plt.close(fig)

    # --- Controls -----------------------------------------------------------------
    orientation_ok = recheck_orientation(
        pd.concat([fitting, strict_val])
    )
    count_row_ok = recheck_count_row_equivalence(fitting)
    controls = run_controls(
        fitting, strict_val, features, scaler, encoding, frozen_lambdas,
        int(cfg["seeds"]["synthetic_label_control"]),
    )
    controls.to_csv(tables_dir / "control_results.csv", index=False)

    control_flags = {}
    for model, group in controls.groupby("model"):
        mean_improvement = float(group["apparent_improvement"].mean())
        control_flags[model] = {
            "mean_apparent_improvement": mean_improvement,
            "max_apparent_improvement": float(
                group["apparent_improvement"].max()
            ),
            "flagged": mean_improvement > CONTROL_FLAG_THRESHOLD,
        }

    summary = {
        "status": "ok",
        "design": "fallback" if coverage_note else "primary",
        "coverage_note": coverage_note,
        "fractions": list(fractions),
        "seeds": seeds,
        "common_validation": {"n_responses": n_common_rows,
                              "n_pairs": n_common_pairs},
        "rechecks": {
            "orientation_identity": orientation_ok,
            "count_row_objective_equivalence": count_row_ok,
        },
        "controls": control_flags,
        "controls_note": "debugging checks, not formal proof that leakage "
                         "is absent",
    }
    with (artifacts_dir / "learning_curve_summary.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(summary, fh, indent=2)

    # --- Console --------------------------------------------------------------------
    print("Stage 9 learning curves and controls complete.")
    print(f"  design: {summary['design']}; common validation subset: "
          f"{n_common_rows:,} responses, {n_common_pairs:,} pairs")
    if coverage_note:
        print(f"  note: {coverage_note}")
    for model in ("M1P", "M2P", "M3P", "M4P"):
        sub = ok[ok["model"] == model]
        if sub.empty:
            print(f"  {model}: no successful fits (see table)")
            continue
        full = sub[np.isclose(sub["fraction"], max(fractions))]
        small = sub[np.isclose(sub["fraction"], min(fractions))]
        print(f"  {model}: validation loss "
              f"{small['validation_loss'].mean():.4f} -> "
              f"{full['validation_loss'].mean():.4f} "
              f"({min(fractions):.0%} -> {max(fractions):.0%})")
    print(f"  rechecks: orientation={orientation_ok}, "
          f"count/row equivalence={count_row_ok}")
    for model, flag in control_flags.items():
        mark = "FLAGGED" if flag["flagged"] else "ok"
        print(f"  control {model}: mean apparent improvement "
              f"{flag['mean_apparent_improvement']:+.5f} ({mark})")
    if not (orientation_ok and count_row_ok):
        print("  STOP: orientation or count/row recheck failed.")
        return 1
    if any(f["flagged"] for f in control_flags.values()):
        print("  WARNING: control improvement above 0.01 - investigate "
              "before Stage 10.")
        return 1
    print("  GO: no unresolved unexpected control performance or "
          "orientation failure.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 9 curves/controls.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config.yaml",
    )
    args = parser.parse_args(argv)
    return run_learning_curves(args.config)


if __name__ == "__main__":
    sys.exit(main())
