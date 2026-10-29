"""Stage 10 - Freeze models and evaluate the internal test.

Produces the primary comparative results (plan Stage 10).

Protocol order:
1. A signed-off model freeze record (SHA-256 of every model artifact, the
   frozen contrast list, and the config) is written BEFORE any internal-test
   outcome is loaded. Models are fitting-only; nothing is refit on
   validation.
2. Only then are strict-test rows loaded (the sole ``allow_test=True`` call
   in the pipeline) and predictions computed from the frozen artifacts.

Metrics per model: response-weighted log loss (natural log, primary),
Brier score, accuracy (tie -> A iff p >= 0.5), within-/between-cat losses,
pair-weighted loss (equal weight per unordered pair).

Paired pair-bootstrap over test pairs (all responses retained when a pair is
sampled), 5,000 replicates with the evaluation bootstrap seed: point
estimates, 95% pointwise intervals, and 98.75% intervals for the four frozen
contrasts (the Bonferroni-adjusted level for a four-contrast family).
Positive reference-minus-candidate loss means candidate improvement;
intervals containing zero are reported as inconclusive. All intervals are
approximate and conditional on the fitted models; they do not account for
unidentified respondents, all shared-image dependence, or training
variability.

Nothing in this module changes models, bins, subgroups, or thresholds in
response to test results.

Outputs
-------
artifacts/model_freeze.json
artifacts/test_predictions.parquet
tables/main_results.csv
tables/model_contrasts.csv
tables/pair_weighted_results.csv
tables/subgroup_results.csv
figures/calibration.pdf
figures/test_probability_distribution.pdf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.special import expit

from src.split import load_partition_rows

MODELS = ["M0", "M1", "M1P", "M2", "M2P", "M3P", "M4P"]
CONTRASTS = [
    ("photo_detail", "M1P", "M2P"),
    ("position", "M2", "M2P"),
    ("visual_sharing", "M2P", "M3P"),
    ("recorded_context", "M3P", "M4P"),
]
BOOTSTRAP_REPLICATES = 5_000
N_CALIBRATION_BINS = 10
LOW_SUPPORT_ROWS = 30
LOW_SUPPORT_PAIRS = 10
SUBGROUP_MIN_ROWS = 100
SUBGROUP_MIN_PAIRS = 30
INTERVAL_NOTE = (
    "Approximate paired pair-bootstrap intervals, conditional on the fitted "
    "models; they do not account for unidentified respondents, all "
    "shared-image dependence, or training variability."
)


# ---------------------------------------------------------------------------
# Freeze record (written BEFORE test outcomes are loaded)
# ---------------------------------------------------------------------------


def write_freeze_record(
    root: Path, config_path: Path, models_dir: Path, artifacts_dir: Path
) -> dict:
    entries = {}
    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for file in sorted(model_dir.rglob("*")):
            if file.is_file():
                entries[str(file.relative_to(root))] = hashlib.sha256(
                    file.read_bytes()
                ).hexdigest()
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    record = {
        "signed_off_at_utc": datetime.now(timezone.utc).isoformat(),
        "statement": "All models, penalties, preprocessing, and contrasts "
                     "are frozen before loading internal-test outcomes. "
                     "Fitting-only models; no refit on validation. Any bug "
                     "correction triggers a versioned rerun and disclosure.",
        "config_sha256": config_hash,
        "model_artifact_sha256": entries,
        "frozen_contrasts": [
            {"name": n, "reference": r, "candidate": c,
             "improvement": "reference loss minus candidate loss"}
            for n, r, c in CONTRASTS
        ],
    }
    with (artifacts_dir / "model_freeze.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(record, fh, indent=2)
    return record


# ---------------------------------------------------------------------------
# Frozen predictions
# ---------------------------------------------------------------------------


def load_frozen_predictors(
    root: Path, models_dir: Path, artifacts_dir: Path
) -> dict:
    """Rebuild prediction functions from frozen model artifacts only."""
    predictors: dict = {}

    with (models_dir / "M0" / "model.json").open("r", encoding="utf-8") as fh:
        m0 = json.load(fh)
    predictors["M0"] = lambda rows: np.full(len(rows), m0["p_a"])

    for name in ("M1", "M1P", "M2", "M2P"):
        with (models_dir / name / "model.json").open(
            "r", encoding="utf-8"
        ) as fh:
            meta = json.load(fh)
        scores = pd.read_csv(models_dir / name / "scores.csv")
        score_of = dict(zip(scores["item"], scores["score"]))
        beta = meta["beta"] or 0.0
        left = "a_cat" if name.startswith("M1") else "a_image"
        right = "b_cat" if name.startswith("M1") else "b_image"

        def make(score_of=score_of, beta=beta, left=left, right=right):
            def predict(rows: pd.DataFrame) -> np.ndarray:
                eta = (rows[left].map(score_of).to_numpy()
                       - rows[right].map(score_of).to_numpy() + beta)
                if not np.isfinite(eta).all():
                    raise ValueError(f"{name}: missing score for a test item")
                return expit(eta)
            return predict
        predictors[name] = make()

    features = pd.read_parquet(
        artifacts_dir / "image_features.parquet"
    )
    with (artifacts_dir / "feature_scaler.json").open(
        "r", encoding="utf-8"
    ) as fh:
        scaler = json.load(fh)
    with (artifacts_dir / "context_encoding.json").open(
        "r", encoding="utf-8"
    ) as fh:
        encoding = json.load(fh)
    kept = list(scaler["means"].keys())
    z = (features[kept] - pd.Series(scaler["means"])) / pd.Series(
        scaler["stds"]
    )

    with (models_dir / "M3P" / "model.json").open("r", encoding="utf-8") as fh:
        m3p_meta = json.load(fh)
    w3 = pd.read_csv(models_dir / "M3P" / "coefficients.csv")["value"].to_numpy()
    beta3 = m3p_meta["beta"] or 0.0

    def predict_m3p(rows: pd.DataFrame) -> np.ndarray:
        dz = (z.loc[rows["a_image"].to_numpy()].to_numpy()
              - z.loc[rows["b_image"].to_numpy()].to_numpy())
        return expit(dz @ w3 + beta3)
    predictors["M3P"] = predict_m3p

    if encoding["m4p_supported"]:
        with (models_dir / "M4P" / "model.json").open(
            "r", encoding="utf-8"
        ) as fh:
            m4p_meta = json.load(fh)
        coefs = pd.read_csv(models_dir / "M4P" / "coefficients.csv")
        n_feat = len(kept)
        w4 = coefs["value"].to_numpy()[:n_feat]
        v_of = {}
        for k, group in enumerate(encoding["interaction_groups"]):
            block = coefs["value"].to_numpy()[
                n_feat * (k + 1): n_feat * (k + 2)
            ]
            v_of[group] = block
        beta4 = m4p_meta["beta"] or 0.0
        zero_v = np.zeros(n_feat)

        def predict_m4p(rows: pd.DataFrame) -> np.ndarray:
            dz = (z.loc[rows["a_image"].to_numpy()].to_numpy()
                  - z.loc[rows["b_image"].to_numpy()].to_numpy())
            pet = rows["pet_care"].where(
                rows["pet_care"].str.strip() != "", "unknown"
            )
            v = np.stack([v_of.get(g, zero_v) for g in pet])
            return expit(((w4 + v) * dz).sum(axis=1) + beta4)
        predictors["M4P"] = predict_m4p

    return predictors


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def log_loss_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def metric_block(y: np.ndarray, p: np.ndarray) -> dict:
    ll = log_loss_rows(y, p)
    return {
        "log_loss": float(ll.mean()),
        "brier": float(((p - y) ** 2).mean()),
        "accuracy": float(((p >= 0.5).astype(int) == y).mean()),
    }


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run_evaluate_test(config_path: Path) -> int:
    root = config_path.resolve().parent
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    artifacts_dir = root / cfg["paths"]["artifacts_dir"]
    tables_dir = root / cfg["paths"]["tables_dir"]
    figures_dir = root / cfg["paths"].get("figures_dir", "figures")
    models_dir = root / cfg["paths"].get("models_dir", "models")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Freeze record BEFORE loading internal-test outcomes.
    freeze = write_freeze_record(root, config_path, models_dir, artifacts_dir)
    print(f"  model freeze record signed: {len(freeze['model_artifact_sha256'])}"
          " artifacts hashed")

    # 2. Now load the strict internal-test rows.
    test = load_partition_rows(config_path, "test", allow_test=True)
    manifest = pd.read_parquet(artifacts_dir / "split_manifest.parquet")
    eligible_ids = set(
        manifest.loc[
            (manifest["partition"] == "test") & manifest["eligible_strict_g2"],
            "row_id",
        ]
    )
    strict_test = test[test["row_id"].isin(eligible_ids)].copy().reset_index(
        drop=True
    )

    predictors = load_frozen_predictors(root, models_dir, artifacts_dir)
    y = strict_test["y_displayed"].astype(int).to_numpy(dtype=float)

    predictions = pd.DataFrame({"row_id": strict_test["row_id"]})
    for model in MODELS:
        if model not in predictors:
            print(f"  {model}: not available (unsupported/failed) - excluded")
            continue
        predictions[model] = predictors[model](strict_test)
    available = [m for m in MODELS if m in predictions.columns]

    out = strict_test[
        ["row_id", "pair_id", "comparison_type", "pet_care", "y_displayed"]
    ].copy()
    for model in available:
        out[f"p_{model}"] = predictions[model].to_numpy()
        out[f"loss_{model}"] = log_loss_rows(y, predictions[model].to_numpy())
    out.to_parquet(artifacts_dir / "test_predictions.parquet", index=False)

    # --- Primary table ------------------------------------------------------------
    main_rows = []
    for model in available:
        p = predictions[model].to_numpy()
        block = metric_block(y, p)
        for ctype in ("within_cat", "between_cats"):
            mask = (strict_test["comparison_type"] == ctype).to_numpy()
            block[f"log_loss_{ctype}"] = float(
                log_loss_rows(y[mask], p[mask]).mean()
            )
        block["model"] = model
        block["n_rows"] = len(y)
        main_rows.append(block)
    main = pd.DataFrame(main_rows)[
        ["model", "log_loss", "brier", "accuracy",
         "log_loss_within_cat", "log_loss_between_cats", "n_rows"]
    ]
    main.to_csv(tables_dir / "main_results.csv", index=False)

    # --- Pair-weighted robustness ----------------------------------------------------
    pair_rows = []
    grouped = out.groupby("pair_id")
    for model in available:
        per_pair = grouped[f"loss_{model}"].mean()
        pair_rows.append({
            "model": model,
            "pair_weighted_log_loss": float(per_pair.mean()),
            "n_pairs": int(per_pair.size),
        })
    pd.DataFrame(pair_rows).to_csv(
        tables_dir / "pair_weighted_results.csv", index=False
    )

    # --- Context subgroups -------------------------------------------------------------
    subgroup_rows = []
    for group, sub in out.groupby(out["pet_care"].where(
        out["pet_care"].str.strip() != "", "unknown"
    )):
        n_rows_g = len(sub)
        n_pairs_g = sub["pair_id"].nunique()
        reportable = (n_rows_g >= SUBGROUP_MIN_ROWS
                      and n_pairs_g >= SUBGROUP_MIN_PAIRS)
        row = {
            "pet_care": group,
            "n_rows": n_rows_g,
            "n_pairs": n_pairs_g,
            "reportable": reportable,
        }
        y_g = sub["y_displayed"].astype(int).to_numpy(dtype=float)
        for model in available:
            row[f"log_loss_{model}"] = (
                float(log_loss_rows(y_g, sub[f"p_{model}"].to_numpy()).mean())
                if reportable else np.nan
            )
        subgroup_rows.append(row)
    subgroups = pd.DataFrame(subgroup_rows)
    subgroups.to_csv(tables_dir / "subgroup_results.csv", index=False)

    # --- Reconciliation (stop/go) --------------------------------------------------------
    for model in available:
        reconciled = float(out[f"loss_{model}"].mean())
        stated = float(main.loc[main["model"] == model, "log_loss"].iloc[0])
        assert abs(reconciled - stated) < 1e-12, f"{model} loss mismatch"
        weighted = sum(
            (r["n_rows"] / len(y)) * r[f"log_loss_{model}"]
            for r in subgroup_rows
            if r["reportable"] and not np.isnan(r[f"log_loss_{model}"])
        )
        covered = sum(r["n_rows"] for r in subgroup_rows if r["reportable"])
        if covered == len(y):
            assert abs(weighted - stated) < 1e-9, (
                f"{model}: weighted subgroup losses do not reproduce total"
            )

    # --- Paired pair-bootstrap ---------------------------------------------------------
    pairs = out["pair_id"].unique()
    n_pairs = len(pairs)
    pair_index = {pid: k for k, pid in enumerate(pairs)}
    pair_of_row = out["pair_id"].map(pair_index).to_numpy()
    n_resp_pair = np.bincount(pair_of_row, minlength=n_pairs).astype(float)
    loss_matrix = np.column_stack([
        np.bincount(pair_of_row, weights=out[f"loss_{model}"],
                    minlength=n_pairs)
        for model in available
    ])
    model_idx = {m: k for k, m in enumerate(available)}

    rng = np.random.default_rng(int(cfg["seeds"]["bootstrap"]))
    contrast_boots = np.full((BOOTSTRAP_REPLICATES, len(CONTRASTS)), np.nan)
    for b in range(BOOTSTRAP_REPLICATES):
        w = np.bincount(
            rng.integers(0, n_pairs, size=n_pairs), minlength=n_pairs
        ).astype(float)
        denom = w @ n_resp_pair
        losses_b = (w @ loss_matrix) / denom
        for c, (_, ref, cand) in enumerate(CONTRASTS):
            if ref in model_idx and cand in model_idx:
                contrast_boots[b, c] = (
                    losses_b[model_idx[ref]] - losses_b[model_idx[cand]]
                )

    contrast_rows = []
    for c, (name, ref, cand) in enumerate(CONTRASTS):
        if ref not in model_idx or cand not in model_idx:
            contrast_rows.append({
                "contrast": name, "reference": ref, "candidate": cand,
                "status": "unavailable",
            })
            continue
        point = float(
            main.loc[main["model"] == ref, "log_loss"].iloc[0]
            - main.loc[main["model"] == cand, "log_loss"].iloc[0]
        )
        boots = contrast_boots[:, c]
        lo95, hi95 = np.percentile(boots, [2.5, 97.5])
        lo98, hi98 = np.percentile(boots, [0.625, 99.375])
        if lo98 > 0:
            verdict = "improvement"
        elif hi98 < 0:
            verdict = "degradation"
        else:
            verdict = "inconclusive"
        contrast_rows.append({
            "contrast": name,
            "reference": ref,
            "candidate": cand,
            "status": "ok",
            "point_estimate": point,
            "ci95_lo": lo95, "ci95_hi": hi95,
            "ci9875_lo": lo98, "ci9875_hi": hi98,
            "verdict_9875": verdict,
            "note": INTERVAL_NOTE,
        })
    contrasts_df = pd.DataFrame(contrast_rows)
    contrasts_df.to_csv(tables_dir / "model_contrasts.csv", index=False)

    # --- Figures ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    edges = np.linspace(0, 1, N_CALIBRATION_BINS + 1)
    for ax, model in zip(axes.ravel(), available):
        p = predictions[model].to_numpy()
        bins = np.clip(np.digitize(p, edges) - 1, 0, N_CALIBRATION_BINS - 1)
        xs, ys, labels = [], [], []
        for b in range(N_CALIBRATION_BINS):
            mask = bins == b
            if not mask.any():
                continue
            n_rows_b = int(mask.sum())
            n_pairs_b = strict_test.loc[mask, "pair_id"].nunique()
            xs.append(float(p[mask].mean()))
            ys.append(float(y[mask].mean()))
            low = (n_rows_b < LOW_SUPPORT_ROWS
                   or n_pairs_b < LOW_SUPPORT_PAIRS)
            labels.append((low, n_rows_b, n_pairs_b))
        xs, ys = np.array(xs), np.array(ys)
        colors = ["#c62828" if l[0] else "#1565c0" for l in labels]
        ax.plot([0, 1], [0, 1], color="#9e9e9e", lw=0.8, ls="--")
        ax.scatter(xs, ys, c=colors, s=28, zorder=3)
        ax.plot(xs, ys, color="#1565c0", lw=0.8, alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("mean predicted P(A)", fontsize=8)
        ax.set_ylabel("observed A rate", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(available):]:
        ax.axis("off")
    fig.suptitle(
        "Calibration on strict internal test - 10 equal-width bins; red = "
        f"low support (<{LOW_SUPPORT_ROWS} rows or <{LOW_SUPPORT_PAIRS} "
        "pairs in bin)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figures_dir / "calibration.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    for ax, model in zip(axes.ravel(), available):
        ax.hist(predictions[model].to_numpy(), bins=edges, color="#1565c0",
                alpha=0.8)
        ax.set_title(model, fontsize=10)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("predicted P(A)", fontsize=8)
        ax.set_ylabel("rows", fontsize=8)
    for ax in axes.ravel()[len(available):]:
        ax.axis("off")
    fig.suptitle("Predicted probability distribution - strict internal test",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figures_dir / "test_probability_distribution.pdf")
    plt.close(fig)

    # --- Console summary ---------------------------------------------------------------------
    print("Stage 10 internal-test evaluation complete.")
    print(f"  strict test rows: {len(y)}, pairs: {n_pairs}")
    print(main.to_string(index=False,
                         float_format=lambda v: f"{v:.4f}"))
    for _, r in contrasts_df.iterrows():
        if r["status"] == "ok":
            print(f"  {r['contrast']} ({r['reference']} - {r['candidate']}): "
                  f"{r['point_estimate']:+.5f} "
                  f"98.75% CI [{r['ci9875_lo']:+.5f}, {r['ci9875_hi']:+.5f}] "
                  f"-> {r['verdict_9875']}")
    print("  GO: metrics reconcile with row-level losses.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 10 evaluation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config.yaml",
    )
    args = parser.parse_args(argv)
    return run_evaluate_test(args.config)


if __name__ == "__main__":
    sys.exit(main())
