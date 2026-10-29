"""Stage 11 - Optional reserved G3 experiment (new-photo prediction).

This stage is gated on the Stage 0 flag ``flags.enable_g3``. The reservation
rule (hash-sort photo IDs with the fixed split seed; ~5% G3 test, ~5% G3
validation) is applied in Stage 3 BEFORE any G2 partitioning - see
``src.split.reserve_g3_photos``.

Current run: the flag is OFF, so no photo pools were reserved and every
response is in the G2 pool. Per the plan's Stage 11 risks/fallbacks -
"If not enabled at Stage 0, do not add a supposedly untouched G3 test
later" - this stage records the not-run status and does nothing else.
Retroactively reserving photos now would contaminate the frozen split and
the already-completed primary evaluation.

When the flag IS enabled before splitting, the full experiment runs:
* fit visual models on core fitting responses (rows with no reserved photo);
* tune G3 penalties on the reserved G3 validation rows;
* evaluate once on the reserved G3 test rows;
* compare M0, known-cat M1P, M3P, and M4P (no learned photo-ID scores are
  invented for unseen images);
* bootstrap by held-out photo, retaining its eligible comparison rows.
Decision rule: proceed only if G3 validation and test each contain at least
200 responses, 50 pairs, and 20 held-out photos; otherwise report
insufficient coverage.

Outputs (skip path)
-------------------
artifacts/g3_record.json

Outputs (enabled path)
----------------------
artifacts/g3_manifest.parquet
artifacts/g3_record.json
tables/g3_results.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit

from src.bt import fit_penalized_logistic, m0_probability
from src.evaluate_test import log_loss_rows
from src.identity_models import RIDGE_GRID, incidence_design, select_lambda
from src.split import reserve_g3_photos

MIN_G3_RESPONSES = 200
MIN_G3_PAIRS = 50
MIN_G3_HELDOUT_PHOTOS = 20
BOOTSTRAP_REPLICATES = 5_000

SKIP_REASON = (
    "flags.enable_g3 is false (Stage 0 default); no photo pools were "
    "reserved before the G2 split in Stage 3. Per the plan (Stage 11 "
    "risks/fallbacks), no supposedly untouched G3 test may be added later; "
    "G3 capability of the visual models is noted in the report as a "
    "capability, not a demonstrated result."
)


# ---------------------------------------------------------------------------
# Enabled path (runs only when Stage 3 reserved photo pools)
# ---------------------------------------------------------------------------


def run_g3_enabled(root: Path, cfg: dict) -> int:
    artifacts_dir = root / cfg["paths"]["artifacts_dir"]
    tables_dir = root / cfg["paths"]["tables_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)

    with (artifacts_dir / "split_config.json").open("r", encoding="utf-8") as fh:
        split_cfg = json.load(fh)
    reservation = split_cfg.get("g3_reservation", {})
    if not reservation.get("enabled"):
        raise RuntimeError(
            "enable_g3 is true but Stage 3 recorded no G3 reservation; "
            "re-run the pipeline from Stage 3 with the flag set."
        )

    seed = int(cfg["seeds"]["split"])
    rows = pd.read_parquet(artifacts_dir / "rows_canonical.parquet")
    train = rows[rows["source"] == "train"].copy()
    all_photos = sorted(set(train["i_image"]) | set(train["j_image"]))
    pools = reserve_g3_photos(all_photos, seed)
    g3_test_photos = set(pools["g3_test_photos"])
    g3_val_photos = set(pools["g3_validation_photos"])
    reserved = g3_test_photos | g3_val_photos

    core = train[
        ~train["i_image"].isin(reserved) & ~train["j_image"].isin(reserved)
    ].copy()  # responses involving reserved photos are excluded from G2 work

    # "Core fitting" = core rows in the fitting partition of this run's split.
    manifest = pd.read_parquet(artifacts_dir / "split_manifest.parquet")
    fitting_ids = set(
        manifest.loc[manifest["partition"] == "fitting", "row_id"]
    )
    core_fit = core[core["row_id"].isin(fitting_ids)].copy()
    core_fitting_photos = set(
        pd.unique(pd.concat([core_fit["a_image"], core_fit["b_image"]]))
    )
    core_fitting_cats = set(
        pd.unique(pd.concat([core_fit["a_cat"], core_fit["b_cat"]]))
    )

    def g3_rows(pool: set) -> pd.DataFrame:
        """Rows with exactly one reserved photo from ``pool`` and one
        core-fitting photo; the held-out photo's cat must be represented in
        core fitting (known-cat claim). Reserved-vs-reserved comparisons are
        quarantined by the exactly-one-reserved-photo rule."""
        out_rows = []
        for _, r in train.iterrows():
            a_res, b_res = r["a_image"] in reserved, r["b_image"] in reserved
            if a_res == b_res:
                continue  # quarantine reserved-vs-reserved and core rows
            held = r["a_image"] if a_res else r["b_image"]
            other = r["b_image"] if a_res else r["a_image"]
            if held not in pool or other not in core_fitting_photos:
                continue
            held_cat = r["a_cat"] if a_res else r["b_cat"]
            if held_cat not in core_fitting_cats:
                continue
            out_rows.append(r)
        return pd.DataFrame(out_rows)

    g3_val = g3_rows(g3_val_photos)
    g3_test = g3_rows(g3_test_photos)

    manifest = pd.concat(
        [g3_val.assign(g3_partition="validation"),
         g3_test.assign(g3_partition="test")]
    )
    manifest.to_parquet(artifacts_dir / "g3_manifest.parquet", index=False)

    coverage = {
        part: {
            "n_responses": int(len(df)),
            "n_pairs": int(df["pair_id"].nunique()),
            "n_heldout_photos": int(
                len((set(df["a_image"]) | set(df["b_image"])) & pool)
            ) if len(df) else 0,
        }
        for part, df, pool in (
            ("validation", g3_val, g3_val_photos),
            ("test", g3_test, g3_test_photos),
        )
    }
    sufficient = all(
        c["n_responses"] >= MIN_G3_RESPONSES
        and c["n_pairs"] >= MIN_G3_PAIRS
        and c["n_heldout_photos"] >= MIN_G3_HELDOUT_PHOTOS
        for c in coverage.values()
    )
    if not sufficient:
        record = {"stage": 11, "status": "insufficient_coverage",
                  "coverage": coverage,
                  "thresholds": {"responses": MIN_G3_RESPONSES,
                                 "pairs": MIN_G3_PAIRS,
                                 "heldout_photos": MIN_G3_HELDOUT_PHOTOS}}
        with (artifacts_dir / "g3_record.json").open("w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        print(f"Stage 11: insufficient G3 coverage: {coverage}")
        return 1

    # --- Fit on core fitting responses; tune on G3 validation; evaluate once.
    features = pd.read_parquet(artifacts_dir / "image_features.parquet")
    with (artifacts_dir / "feature_scaler.json").open("r", encoding="utf-8") as fh:
        scaler = json.load(fh)
    with (artifacts_dir / "context_encoding.json").open("r", encoding="utf-8") as fh:
        encoding = json.load(fh)
    kept = list(scaler["means"].keys())
    z = (features[kept] - pd.Series(scaler["means"])) / pd.Series(scaler["stds"])

    y_core = core_fit["y_displayed"].astype(int).to_numpy(dtype=float)
    p0 = m0_probability(int(y_core.sum()), len(core_fit))

    def visual_design(df: pd.DataFrame, contextual: bool) -> np.ndarray:
        dz = (z.loc[df["a_image"].to_numpy()].to_numpy()
              - z.loc[df["b_image"].to_numpy()].to_numpy())
        if not contextual:
            return dz
        igroups = encoding["interaction_groups"]
        pet = df["pet_care"].where(df["pet_care"].str.strip() != "", "unknown")
        blocks = [dz]
        for g in igroups:
            blocks.append(dz * (pet.to_numpy() == g)[:, None])
        return np.hstack(blocks)

    def cat_design(df: pd.DataFrame, index: dict) -> "object":
        return incidence_design(df["a_cat"], df["b_cat"], index)

    cats = np.sort(pd.unique(pd.concat([core_fit["a_cat"], core_fit["b_cat"]])))
    cat_index = {c: k for k, c in enumerate(cats)}

    results = []
    boot_records: dict = {}

    def bootstrap_by_photo(df: pd.DataFrame, p: np.ndarray) -> tuple:
        """Bootstrap CIs over held-out photos, retaining each photo's rows."""
        yv = df["y_displayed"].astype(int).to_numpy(dtype=float)
        losses = log_loss_rows(yv, p)
        held_photo = np.where(
            df["a_image"].isin(reserved), df["a_image"], df["b_image"]
        )
        photos_u, inv = np.unique(held_photo, return_inverse=True)
        rng = np.random.default_rng(int(cfg["seeds"]["bootstrap"]))
        boots = []
        per_photo_loss = np.bincount(inv, weights=losses)
        per_photo_n = np.bincount(inv).astype(float)
        for _ in range(BOOTSTRAP_REPLICATES):
            w = np.bincount(rng.integers(0, len(photos_u), len(photos_u)),
                            minlength=len(photos_u)).astype(float)
            boots.append(float(w @ per_photo_loss / (w @ per_photo_n)))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return float(losses.mean()), float(lo), float(hi)

    # M0
    p_m0 = np.full(len(g3_test), p0)
    mean, lo, hi = bootstrap_by_photo(g3_test, p_m0)
    results.append({"model": "M0", "log_loss": mean, "ci95_lo": lo,
                    "ci95_hi": hi})

    # Known-cat M1P: tune lambda on G3 validation, evaluate once on G3 test.
    y_g3v = g3_val["y_displayed"].astype(int).to_numpy(dtype=float)
    grid = []
    for lam in RIDGE_GRID:
        fit = fit_penalized_logistic(
            cat_design(core_fit, cat_index), y_core, lam, with_position=True
        )
        p = expit(cat_design(g3_val, cat_index) @ fit.theta + fit.beta)
        grid.append({"lambda": lam, "fit_success": fit.success,
                     "validation_log_loss": float(
                         log_loss_rows(y_g3v, p).mean())})
    lam_star = select_lambda(grid)
    fit = fit_penalized_logistic(
        cat_design(core_fit, cat_index), y_core, lam_star, with_position=True
    )
    p_m1p = expit(cat_design(g3_test, cat_index) @ fit.theta + fit.beta)
    mean, lo, hi = bootstrap_by_photo(g3_test, p_m1p)
    results.append({"model": "M1P_known_cat", "lambda": lam_star,
                    "log_loss": mean, "ci95_lo": lo, "ci95_hi": hi})

    # M3P / M4P (visual models support unseen photos through features).
    for model, contextual in (("M3P", False), ("M4P", True)):
        if contextual and not encoding["m4p_supported"]:
            results.append({"model": model, "status": "unsupported"})
            continue
        grid = []
        for lam in RIDGE_GRID:
            fit = fit_penalized_logistic(
                visual_design(core_fit, contextual), y_core, lam,
                with_position=True
            )
            p = expit(visual_design(g3_val, contextual) @ fit.theta
                      + fit.beta)
            grid.append({"lambda": lam, "fit_success": fit.success,
                         "validation_log_loss": float(
                             log_loss_rows(y_g3v, p).mean())})
        lam_star = select_lambda(grid)
        fit = fit_penalized_logistic(
            visual_design(core_fit, contextual), y_core, lam_star,
            with_position=True
        )
        p = expit(visual_design(g3_test, contextual) @ fit.theta + fit.beta)
        mean, lo, hi = bootstrap_by_photo(g3_test, p)
        results.append({"model": model, "lambda": lam_star,
                        "log_loss": mean, "ci95_lo": lo, "ci95_hi": hi})

    pd.DataFrame(results).to_csv(tables_dir / "g3_results.csv", index=False)
    record = {"stage": 11, "status": "completed", "coverage": coverage,
              "note": "Separate G3 claim; no reserved G3 outcomes were used "
                      "in G2 development."}
    with (artifacts_dir / "g3_record.json").open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    print(f"Stage 11 G3 complete: {pd.DataFrame(results).to_string(index=False)}")
    return 0


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run_g3(config_path: Path) -> int:
    root = config_path.resolve().parent
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    artifacts_dir = root / cfg["paths"]["artifacts_dir"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not bool(cfg["flags"]["enable_g3"]):
        with (artifacts_dir / "split_config.json").open(
            "r", encoding="utf-8"
        ) as fh:
            reservation = json.load(fh).get("g3_reservation", {})
        record = {
            "stage": 11,
            "status": "not_run",
            "reason": SKIP_REASON,
            "stage3_reservation_record": reservation,
        }
        with (artifacts_dir / "g3_record.json").open(
            "w", encoding="utf-8"
        ) as fh:
            json.dump(record, fh, indent=2)
        print("Stage 11 G3: NOT RUN (optional experiment).")
        print(f"  {SKIP_REASON}")
        print("  No G3 claim will appear in the report; visual-model "
              "transfer capability is described as capability only.")
        return 0

    return run_g3_enabled(root, cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 11 optional G3.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config.yaml",
    )
    args = parser.parse_args(argv)
    return run_g3(args.config)


if __name__ == "__main__":
    sys.exit(main())
