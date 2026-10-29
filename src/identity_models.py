"""Stage 6 - Fit M0, M1, M1P, M2, and M2P.

Identity-based baselines and the matched photo-detail comparison (plan
Stage 6):

* sparse incidence designs over cats (M1/M1P) and photos (M2/M2P);
* every penalty in the frozen grid is fitted;
* penalties are selected on STRICT-G2 validation log loss; if two validation
  losses differ by less than 1e-6 the larger penalty is chosen;
* validation predictions and fitting diagnostics are saved per model;
* within-component scores and ranks are produced (component score means are
  reported for the numerical-zero check);
* validation contrasts M1P - M2P and M2 - M2P are computed (improvement =
  reference loss minus candidate loss).

The internal test partition is never loaded: rows are obtained through
``src.split.load_partition_rows``, which structurally refuses the test
partition for development stages.

Outputs
-------
models/M0 ... models/M2P     model.json, scores.csv, validation_predictions.parquet
tables/validation_identity_models.csv
tables/rankings.csv
figures/winrate_vs_score.pdf
artifacts/identity_model_summary.json
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
    expected_log_loss,  # noqa: F401  (re-exported for later stages)
    fit_penalized_logistic,
    m0_probability,
)
from src.split import load_partition_rows

RIDGE_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
SELECTION_TIE_TOL = 1e-6

MODEL_SPECS = {
    "M1": {"level": "cat", "with_position": False},
    "M1P": {"level": "cat", "with_position": True},
    "M2": {"level": "photo", "with_position": False},
    "M2P": {"level": "photo", "with_position": True},
}


def incidence_design(
    a_items: pd.Series, b_items: pd.Series, index: dict[str, int]
) -> sp.csr_matrix:
    """Sparse incidence design: +1 at item A, -1 at item B."""
    n = len(a_items)
    ii = np.arange(n)
    aa = a_items.map(index).to_numpy()
    bb = b_items.map(index).to_numpy()
    if np.isnan(aa.astype(float)).any() or np.isnan(bb.astype(float)).any():
        raise KeyError("row references an item absent from the design index")
    return sp.coo_matrix(
        (
            np.concatenate([np.ones(n), -np.ones(n)]),
            (np.concatenate([ii, ii]), np.concatenate([aa, bb])),
        ),
        shape=(n, len(index)),
    ).tocsr()


def row_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Response-weighted log loss (natural log), one weight per response."""
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def select_lambda(grid_rows: list[dict]) -> float | None:
    """argmin validation loss; ties within 1e-6 resolve to the larger penalty."""
    ok = [r for r in grid_rows if r["fit_success"]
          and np.isfinite(r["validation_log_loss"])]
    if not ok:
        return None
    best = min(r["validation_log_loss"] for r in ok)
    tied = [r["lambda"] for r in ok
            if r["validation_log_loss"] < best + SELECTION_TIE_TOL]
    return max(tied)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run_identity_models(config_path: Path) -> int:
    root = config_path.resolve().parent
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    artifacts_dir = root / cfg["paths"]["artifacts_dir"]
    tables_dir = root / cfg["paths"]["tables_dir"]
    figures_dir = root / cfg["paths"].get("figures_dir", "figures")
    models_dir = root / cfg["paths"].get("models_dir", "models")
    for d in (tables_dir, figures_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Fitting and strict-validation rows only; the test partition is refused
    # by load_partition_rows for this development stage.
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

    components = pd.read_parquet(artifacts_dir / "photo_components.parquet")
    component_of = dict(
        zip(components["photo"], components["fitting_component"])
    )

    y_fit = fitting["y_displayed"].astype(int).to_numpy(dtype=float)
    y_val = strict_val["y_displayed"].astype(int).to_numpy(dtype=float)

    # --- Support annotations from fitting data -------------------------------
    photo_responses = pd.concat(
        [fitting["a_image"], fitting["b_image"]]
    ).value_counts()
    photo_opponents = (
        pd.concat([
            fitting[["a_image", "b_image"]].rename(
                columns={"a_image": "item", "b_image": "opp"}),
            fitting[["b_image", "a_image"]].rename(
                columns={"b_image": "item", "a_image": "opp"}),
        ]).groupby("item")["opp"].nunique()
    )
    photo_wins = pd.concat([
        fitting.loc[fitting["y_displayed"] == 1, "a_image"],
        fitting.loc[fitting["y_displayed"] == 0, "b_image"],
    ]).value_counts()
    cat_responses = pd.concat(
        [fitting["a_cat"], fitting["b_cat"]]
    ).value_counts()
    cat_wins = pd.concat([
        fitting.loc[fitting["y_displayed"] == 1, "a_cat"],
        fitting.loc[fitting["y_displayed"] == 0, "b_cat"],
    ]).value_counts()

    # --- M0 --------------------------------------------------------------------
    n_fit = len(fitting)
    n_a = int(y_fit.sum())
    p0 = m0_probability(n_a, n_fit)
    m0_val_loss = row_log_loss(y_val, np.full(len(strict_val), p0))
    m0_dir = models_dir / "M0"
    m0_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"row_id": strict_val["row_id"], "p_a": p0}
    ).to_parquet(m0_dir / "validation_predictions.parquet", index=False)
    with (m0_dir / "model.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": "M0",
                "rule": "p_a = (n_A + 0.5) / (N + 1)",
                "p_a": p0,
                "n_fitting": n_fit,
                "n_a_fitting": n_a,
                "validation_log_loss": m0_val_loss,
                "n_strict_validation_rows": len(strict_val),
            },
            fh,
            indent=2,
        )

    # --- M1/M1P/M2/M2P over the frozen grid -------------------------------------
    grid_records: list[dict] = []
    fitted: dict[str, dict] = {}  # model -> selected fit bundle
    rankings_frames: list[pd.DataFrame] = []

    for model, spec in MODEL_SPECS.items():
        level = spec["level"]
        left_fit = fitting["a_cat"] if level == "cat" else fitting["a_image"]
        right_fit = fitting["b_cat"] if level == "cat" else fitting["b_image"]
        items = np.sort(pd.unique(pd.concat([left_fit, right_fit])))
        index = {item: k for k, item in enumerate(items)}
        X_fit = incidence_design(left_fit, right_fit, index)
        left_val = strict_val["a_cat"] if level == "cat" else strict_val["a_image"]
        right_val = strict_val["b_cat"] if level == "cat" else strict_val["b_image"]
        missing_val = (
            set(left_val) | set(right_val)
        ) - set(index)
        X_val = incidence_design(left_val, right_val, index)

        model_rows: list[dict] = []
        for lam in RIDGE_GRID:
            fit = fit_penalized_logistic(
                X_fit, y_fit, lam, with_position=spec["with_position"]
            )
            if fit.success:
                p_val = fit.predict_proba(X_val)
                finite = bool(np.all(np.isfinite(p_val)))
                val_loss = row_log_loss(y_val, p_val) if finite else np.nan
            else:
                p_val, finite, val_loss = None, False, np.nan
            record = {
                "model": model,
                "lambda": lam,
                "fit_success": bool(fit.success and finite),
                "validation_log_loss": val_loss,
                "objective": fit.objective,
                "max_abs_grad": fit.max_abs_grad,
                "n_iterations": fit.n_iterations,
                "retried": fit.retried,
                "beta": fit.beta,
                "solver_message": fit.message,
                "selected": False,
            }
            model_rows.append(record)
            grid_records.append(record)

        lam_star = select_lambda(model_rows)
        if lam_star is not None:
            for r in grid_records:
                if r["model"] == model and r["lambda"] == lam_star:
                    r["selected"] = True
            best_fit = fit_penalized_logistic(
                X_fit, y_fit, lam_star, with_position=spec["with_position"]
            )
            fitted[model] = {
                "fit": best_fit, "items": items, "index": index,
                "lam": lam_star, "X_val": X_val,
            }

    # --- Save selected models, predictions, rankings ------------------------------
    summary: dict = {"M0": {"p_a": p0, "validation_log_loss": m0_val_loss}}
    for model, bundle in fitted.items():
        fit, items, lam = bundle["fit"], bundle["items"], bundle["lam"]
        model_dir = models_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)

        p_val = fit.predict_proba(bundle["X_val"])
        val_loss = row_log_loss(y_val, p_val)
        pd.DataFrame({"row_id": strict_val["row_id"], "p_a": p_val}).to_parquet(
            model_dir / "validation_predictions.parquet", index=False
        )

        scores = pd.DataFrame({"item": items, "score": fit.theta})
        if model.startswith("M2"):
            scores["fitting_component"] = scores["item"].map(component_of)
            scores["n_fitting_responses"] = (
                scores["item"].map(photo_responses).fillna(0).astype(int)
            )
            scores["n_fitting_unique_opponents"] = (
                scores["item"].map(photo_opponents).fillna(0).astype(int)
            )
            # Component score means: reported for the numerical-zero check.
            comp_means = scores.groupby("fitting_component")["score"].mean()
            comp_mean_max_abs = float(comp_means.abs().max())
        else:
            scores["fitting_component"] = 0
            scores["n_fitting_responses"] = (
                scores["item"].map(cat_responses).fillna(0).astype(int)
            )
            scores["n_fitting_unique_opponents"] = np.nan
            comp_mean_max_abs = float(scores["score"].mean())
        scores["rank_within_component"] = scores.groupby(
            "fitting_component"
        )["score"].rank(ascending=False, method="min").astype(int)
        scores.insert(0, "model", model)
        scores.to_csv(model_dir / "scores.csv", index=False)
        rankings_frames.append(scores)

        with (model_dir / "model.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "model": model,
                    "lambda_selected": lam,
                    "selection_rule": "strict-G2 validation log loss; ties "
                                      "< 1e-6 resolve to the larger penalty",
                    "with_position": MODEL_SPECS[model]["with_position"],
                    "beta": fit.beta,
                    "validation_log_loss": val_loss,
                    "fit_diagnostics": {
                        "success": fit.success,
                        "objective": fit.objective,
                        "max_abs_grad": fit.max_abs_grad,
                        "n_iterations": fit.n_iterations,
                        "retried": fit.retried,
                        "message": fit.message,
                    },
                    "n_items": len(items),
                    "component_score_mean_max_abs": comp_mean_max_abs,
                    "n_fitting_rows": n_fit,
                    "n_strict_validation_rows": len(strict_val),
                },
                fh,
                indent=2,
            )
        summary[model] = {
            "lambda": lam,
            "validation_log_loss": val_loss,
            "beta": fit.beta,
            "component_score_mean_max_abs": comp_mean_max_abs,
        }

    pd.DataFrame(grid_records).to_csv(
        tables_dir / "validation_identity_models.csv", index=False
    )
    if rankings_frames:
        pd.concat(rankings_frames).to_csv(
            tables_dir / "rankings.csv", index=False
        )

    # --- Frozen validation contrasts (improvement = reference - candidate) --------
    contrasts = {}
    for name, ref, cand in (
        ("photo_detail_M1P_minus_M2P", "M1P", "M2P"),
        ("position_M2_minus_M2P", "M2", "M2P"),
    ):
        if ref in fitted and cand in fitted:
            contrasts[name] = {
                "reference": ref,
                "candidate": cand,
                "improvement": summary[ref]["validation_log_loss"]
                - summary[cand]["validation_log_loss"],
                "note": "validation-set development contrast; the primary "
                        "contrast is computed once on the internal test",
            }
    summary["validation_contrasts"] = contrasts
    with (artifacts_dir / "identity_model_summary.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(summary, fh, indent=2)

    # --- Figure: raw win rate vs fitted score --------------------------------------
    panels = [
        ("M1P", "cat", cat_wins, cat_responses, None),
        ("M2P", "photo", photo_wins, photo_responses, photo_opponents),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (model, level, wins, responses, opponents) in zip(axes, panels):
        if model not in fitted:
            ax.text(0.5, 0.5, f"{model} fit failed", ha="center",
                    transform=ax.transAxes)
            continue
        bundle = fitted[model]
        scores = pd.Series(bundle["fit"].theta, index=bundle["items"])
        df = pd.DataFrame({
            "score": scores,
            "win_rate": (wins / responses).reindex(scores.index),
            "n_responses": responses.reindex(scores.index),
        })
        if opponents is not None:
            df["n_opponents"] = opponents.reindex(scores.index)
            sc = ax.scatter(
                df["score"], df["win_rate"],
                s=np.clip(df["n_responses"], 5, None) * 1.5,
                c=df["n_opponents"], cmap="viridis", alpha=0.6,
            )
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("unique fitting opponents")
        else:
            ax.scatter(df["score"], df["win_rate"],
                       s=np.clip(df["n_responses"], 5, None) * 1.5,
                       alpha=0.6, color="#1565c0")
        ax.axhline(0.5, color="#9e9e9e", lw=0.8, ls=":")
        ax.axvline(0.0, color="#9e9e9e", lw=0.8, ls=":")
        ax.set_xlabel(f"Fitted {level} score ({model}, fitting data)")
        ax.set_ylabel("Raw win rate (fitting data)")
        ax.set_title(f"{model}: win rate vs score "
                     f"(point size = responses)")
    fig.tight_layout()
    fig.savefig(figures_dir / "winrate_vs_score.pdf")
    plt.close(fig)

    # --- Stop/go ----------------------------------------------------------------------
    m2p_ok = "M2P" in fitted and fitted["M2P"]["fit"].success
    m0_ok = np.isfinite(m0_val_loss)
    print("Stage 6 identity models complete.")
    print(f"  M0: p_a={p0:.4f}, validation log loss={m0_val_loss:.4f}")
    for model in MODEL_SPECS:
        if model in summary:
            s = summary[model]
            print(f"  {model}: lambda={s['lambda']:.0e}, "
                  f"validation log loss={s['validation_log_loss']:.4f}, "
                  f"beta={s['beta']}, "
                  f"max|component mean|={s['component_score_mean_max_abs']:.2e}")
        else:
            print(f"  {model}: ALL GRID FITS FAILED (kept visible)")
    for name, c in contrasts.items():
        print(f"  validation contrast {name}: {c['improvement']:+.5f}")
    if not (m0_ok and m2p_ok):
        print("  STOP: M0 and M2P must both predict all strict-validation "
              "rows.")
        return 1
    print("  GO: M0 and M2P produce valid predictions on all "
          "strict-validation rows.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 6 identity models.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config.yaml",
    )
    args = parser.parse_args(argv)
    return run_identity_models(args.config)


if __name__ == "__main__":
    sys.exit(main())
