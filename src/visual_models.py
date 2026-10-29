"""Stage 8 - Extract features and fit M3P/M4P.

Shared visual scorer and its pet-care interaction extension (plan Stage 8).

Image processing (per image):
  EXIF orientation -> RGB -> retain original width/height -> resize to
  256x256 bilinear -> scale channels to [0, 1].

Eight features (fixed; no feature search):
  1-3. mean R / G / B
  4.   std of luminance (0.2126 R + 0.7152 G + 0.0722 B)
  5.   mean HSV saturation
  6.   log(1 + Var(4-neighbor Laplacian of luminance, interior pixels))
  7.   original width / original height
  8.   log(original width * original height)

Learned preprocessing:
  means/stds are fitted on UNIQUE FITTING PHOTOS only; features with fitting
  std below 1e-12 are dropped; the frozen transformation is then applied to
  validation/test images unchanged.

Context encoding (pet_care):
  documented categories plus "unknown"; a category is interaction-eligible
  with >= 100 fitting responses and >= 30 distinct fitting pairs; the
  largest eligible group is the reference (ties lexicographic); other
  eligible groups receive feature-difference interactions; unsupported or
  unseen categories receive a zero interaction deviation; fewer than two
  eligible groups => M4P recorded as unsupported and omitted.

Models (displayed coordinates, position term included):
  M3P:  logit = w . (z_A - z_B) + beta
  M4P:  logit = (w + v_g) . (z_A - z_B) + beta,   v_reference = 0

Outputs
-------
artifacts/image_features.parquet     raw (unstandardized) features per image
artifacts/feature_scaler.json        frozen means/stds/dropped features
artifacts/context_encoding.json      eligibility, reference group, support
models/M3P, models/M4P               model.json, coefficients.csv,
                                     validation_predictions.parquet
tables/validation_visual_models.csv
artifacts/visual_model_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import rgb_to_hsv
from PIL import Image, ImageOps

from src.bt import fit_penalized_logistic
from src.identity_models import RIDGE_GRID, row_log_loss, select_lambda
from src.split import load_partition_rows

RESIZE = (256, 256)
STD_FLOOR = 1e-12
MIN_GROUP_RESPONSES = 100
MIN_GROUP_PAIRS = 30

FEATURE_NAMES = [
    "mean_red",
    "mean_green",
    "mean_blue",
    "std_luminance",
    "mean_saturation",
    "log_laplacian_var",
    "aspect_ratio",
    "log_area",
]

PET_CARE_CATEGORIES = [
    "never",
    "occasionally",
    "past_regular",
    "current_regular",
    "unknown",
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(path: Path) -> dict:
    """Eight frozen features for one image (plan Stage 8 specification)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)  # apply orientation metadata
        im = im.convert("RGB")
        orig_w, orig_h = im.size
        im = im.resize(RESIZE, Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float64) / 255.0

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    saturation = rgb_to_hsv(arr)[..., 1]

    lum = luminance
    lap = (
        lum[:-2, 1:-1] + lum[2:, 1:-1] + lum[1:-1, :-2] + lum[1:-1, 2:]
        - 4.0 * lum[1:-1, 1:-1]
    )  # 4-neighbor discrete Laplacian on interior pixels

    return {
        "mean_red": float(r.mean()),
        "mean_green": float(g.mean()),
        "mean_blue": float(b.mean()),
        "std_luminance": float(luminance.std()),
        "mean_saturation": float(saturation.mean()),
        "log_laplacian_var": float(np.log1p(lap.var())),
        "aspect_ratio": float(orig_w / orig_h),
        "log_area": float(np.log(orig_w * orig_h)),
    }


# ---------------------------------------------------------------------------
# Context encoding
# ---------------------------------------------------------------------------


def build_context_encoding(fitting: pd.DataFrame) -> dict:
    """Eligibility and reference-group selection from FITTING data only."""
    pet_care = fitting["pet_care"].where(
        fitting["pet_care"].str.strip() != "", "unknown"
    )
    stats = (
        fitting.assign(pet_care_filled=pet_care)
        .groupby("pet_care_filled")
        .agg(n_responses=("row_id", "size"),
             n_pairs=("pair_id", "nunique"))
    )
    eligibility = {}
    for cat in PET_CARE_CATEGORIES:
        if cat in stats.index:
            n_resp = int(stats.loc[cat, "n_responses"])
            n_pairs = int(stats.loc[cat, "n_pairs"])
        else:
            n_resp, n_pairs = 0, 0
        eligibility[cat] = {
            "n_fitting_responses": n_resp,
            "n_fitting_pairs": n_pairs,
            "eligible": n_resp >= MIN_GROUP_RESPONSES
            and n_pairs >= MIN_GROUP_PAIRS,
        }
    eligible = sorted(c for c, e in eligibility.items() if e["eligible"])
    reference = None
    if eligible:
        # Largest eligible group; ties broken lexicographically.
        reference = min(
            eligible,
            key=lambda c: (-eligibility[c]["n_fitting_responses"], c),
        )
    m4p_supported = len(eligible) >= 2
    return {
        "context_variable": "pet_care",
        "categories": PET_CARE_CATEGORIES,
        "missing_encoded_as": "unknown",
        "eligibility_thresholds": {
            "min_responses": MIN_GROUP_RESPONSES,
            "min_pairs": MIN_GROUP_PAIRS,
        },
        "per_category": eligibility,
        "eligible_groups": eligible,
        "reference_group": reference,
        "interaction_groups": [c for c in eligible if c != reference],
        "m4p_supported": m4p_supported,
        "unsupported_rule": "fewer than two eligible groups => M4P omitted",
    }


# ---------------------------------------------------------------------------
# Design matrices
# ---------------------------------------------------------------------------


def feature_differences(
    rows: pd.DataFrame, z: pd.DataFrame
) -> np.ndarray:
    """Standardized z_A - z_B per row (columns follow z.columns)."""
    za = z.loc[rows["a_image"].to_numpy()].to_numpy()
    zb = z.loc[rows["b_image"].to_numpy()].to_numpy()
    return za - zb


def m4p_design(
    dz: np.ndarray, groups: pd.Series, interaction_groups: list[str]
) -> np.ndarray:
    """[dz | dz * 1{group == g_k} for each non-reference eligible group]."""
    blocks = [dz]
    g = groups.to_numpy()
    for group in interaction_groups:
        mask = (g == group).astype(float)[:, None]
        blocks.append(dz * mask)
    return np.hstack(blocks)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run_visual_models(config_path: Path) -> int:
    root = config_path.resolve().parent
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    artifacts_dir = root / cfg["paths"]["artifacts_dir"]
    tables_dir = root / cfg["paths"]["tables_dir"]
    models_dir = root / cfg["paths"].get("models_dir", "models")
    images_dir = root / cfg["paths"]["images_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.read_parquet(artifacts_dir / "rows_canonical.parquet")
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

    # --- Feature extraction for every referenced image ------------------------
    referenced = sorted(set(rows["a_image"]) | set(rows["b_image"]))
    records = []
    failures = []
    for rel in referenced:
        try:
            feats = extract_features(images_dir / rel)
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append({"image": rel, "error": str(exc)})
            continue
        feats["image"] = rel
        records.append(feats)
    features = pd.DataFrame(records).set_index("image")[FEATURE_NAMES]
    features.to_parquet(artifacts_dir / "image_features.parquet")

    finite_mask = np.isfinite(features.to_numpy()).all(axis=1)
    if not finite_mask.all():
        bad = features.index[~finite_mask].tolist()
        failures.extend({"image": im, "error": "non-finite feature"}
                        for im in bad)
        features = features[finite_mask]

    # --- Frozen scaler from unique fitting photos ------------------------------
    fitting_photos = sorted(
        set(fitting["a_image"]) | set(fitting["b_image"])
    )
    fit_feats = features.loc[
        [p for p in fitting_photos if p in features.index]
    ]
    means = fit_feats.mean()
    stds = fit_feats.std(ddof=0)
    kept = stds[stds >= STD_FLOOR].index.tolist()
    dropped = stds[stds < STD_FLOOR].index.tolist()
    scaler = {
        "fitted_on": "unique fitting photos",
        "n_fitting_photos": len(fit_feats),
        "means": means[kept].to_dict(),
        "stds": stds[kept].to_dict(),
        "dropped_features": dropped,
        "std_floor": STD_FLOOR,
    }
    with (artifacts_dir / "feature_scaler.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(scaler, fh, indent=2)

    z = (features[kept] - means[kept]) / stds[kept]
    assert np.isfinite(z.to_numpy()).all(), "non-finite standardized features"

    # --- Context encoding -------------------------------------------------------
    encoding = build_context_encoding(fitting)
    with (artifacts_dir / "context_encoding.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(encoding, fh, indent=2)

    # --- Image availability on strict validation rows ---------------------------
    avail = strict_val["a_image"].isin(z.index) & strict_val["b_image"].isin(
        z.index
    )
    availability = float(avail.mean())
    if availability == 1.0:
        eval_rows = strict_val
        subset_label = "full strict-G2 validation subset"
    elif availability >= 0.95:
        eval_rows = strict_val[avail].copy()
        subset_label = ("common-image comparison on the complete-image "
                        "subset (clearly labeled)")
    else:
        print(f"  image availability {availability:.1%} < 95% - visual "
              "models dropped from the core presentation.")
        return 1

    y_fit = fitting["y_displayed"].astype(int).to_numpy(dtype=float)
    y_val = eval_rows["y_displayed"].astype(int).to_numpy(dtype=float)
    dz_fit = feature_differences(fitting, z)
    dz_val = feature_differences(eval_rows, z)

    pet_fit = fitting["pet_care"].where(
        fitting["pet_care"].str.strip() != "", "unknown"
    )
    pet_val = eval_rows["pet_care"].where(
        eval_rows["pet_care"].str.strip() != "", "unknown"
    )

    designs = {"M3P": (dz_fit, dz_val)}
    coef_names = {"M3P": [f"w:{c}" for c in kept]}
    if encoding["m4p_supported"]:
        igroups = encoding["interaction_groups"]
        designs["M4P"] = (
            m4p_design(dz_fit, pet_fit, igroups),
            m4p_design(dz_val, pet_val, igroups),
        )
        coef_names["M4P"] = (
            [f"w:{c}" for c in kept]
            + [f"v[{g}]:{c}" for g in igroups for c in kept]
        )
    else:
        print("  M4P recorded as UNSUPPORTED (fewer than two eligible "
              "groups); interaction fit omitted.")

    # --- Fit both models over the frozen grid -------------------------------------
    grid_records: list[dict] = []
    fitted: dict[str, dict] = {}
    for model, (X_fit, X_val) in designs.items():
        model_rows: list[dict] = []
        for lam in RIDGE_GRID:
            fit = fit_penalized_logistic(
                X_fit, y_fit, lam, with_position=True
            )
            if fit.success:
                p_val = fit.predict_proba(X_val)
                finite = bool(np.all(np.isfinite(p_val)))
                val_loss = row_log_loss(y_val, p_val) if finite else np.nan
            else:
                p_val, finite, val_loss = None, False, np.nan
            record = {
                "model": model, "lambda": lam,
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
                X_fit, y_fit, lam_star, with_position=True
            )
            fitted[model] = {"fit": best_fit, "lam": lam_star,
                             "X_val": X_val}

    # --- Save models ------------------------------------------------------------------
    summary: dict = {
        "evaluation_rows": subset_label,
        "image_availability_strict_validation": availability,
        "n_evaluation_rows": len(eval_rows),
        "n_features_kept": len(kept),
        "dropped_features": dropped,
        "m4p_supported": encoding["m4p_supported"],
    }
    for model, bundle in fitted.items():
        fit = bundle["fit"]
        model_dir = models_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        p_val = fit.predict_proba(bundle["X_val"])
        val_loss = row_log_loss(y_val, p_val)
        pd.DataFrame({"row_id": eval_rows["row_id"], "p_a": p_val}).to_parquet(
            model_dir / "validation_predictions.parquet", index=False
        )
        pd.DataFrame(
            {"coefficient": coef_names[model], "value": fit.theta}
        ).to_csv(model_dir / "coefficients.csv", index=False)
        with (model_dir / "model.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "model": model,
                    "lambda_selected": bundle["lam"],
                    "selection_rule": "strict-G2 validation log loss; ties "
                                      "< 1e-6 resolve to the larger penalty",
                    "with_position": True,
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
                    "n_features": len(kept),
                    "evaluation_rows": subset_label,
                    "n_fitting_rows": len(fitting),
                    "n_evaluation_rows": len(eval_rows),
                },
                fh,
                indent=2,
            )
        summary[model] = {"lambda": bundle["lam"],
                          "validation_log_loss": val_loss,
                          "beta": fit.beta}

    pd.DataFrame(grid_records).to_csv(
        tables_dir / "validation_visual_models.csv", index=False
    )

    # --- Validation contrasts (development evidence) ----------------------------------
    with (artifacts_dir / "identity_model_summary.json").open(
        "r", encoding="utf-8"
    ) as fh:
        identity_summary = json.load(fh)
    m2p_val = identity_summary["M2P"]["validation_log_loss"]
    contrasts: dict = {}
    if "M3P" in fitted:
        contrasts["visual_sharing_M2P_minus_M3P"] = {
            "reference": "M2P", "candidate": "M3P",
            "improvement": m2p_val - summary["M3P"]["validation_log_loss"],
        }
    if "M3P" in fitted and "M4P" in fitted:
        contrasts["recorded_context_M3P_minus_M4P"] = {
            "reference": "M3P", "candidate": "M4P",
            "improvement": summary["M3P"]["validation_log_loss"]
            - summary["M4P"]["validation_log_loss"],
        }
    summary["validation_contrasts"] = contrasts
    with (artifacts_dir / "visual_model_summary.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(summary, fh, indent=2)

    # --- Console summary ------------------------------------------------------------------
    print("Stage 8 visual/context models complete.")
    print(f"  images: {len(features)}/{len(referenced)} with finite features "
          f"({len(failures)} failures); availability on strict validation "
          f"{availability:.1%} -> {subset_label}")
    print(f"  features kept: {len(kept)} (dropped: {dropped})")
    print(f"  pet_care eligible groups: {encoding['eligible_groups']}; "
          f"reference: {encoding['reference_group']}")
    for model in ("M3P", "M4P"):
        if model in summary and isinstance(summary.get(model), dict):
            s = summary[model]
            print(f"  {model}: lambda={s['lambda']:.0e}, validation log "
                  f"loss={s['validation_log_loss']:.4f}, beta={s['beta']:.4f}")
        else:
            print(f"  {model}: not fitted (unsupported or failed)")
    for name, c in contrasts.items():
        print(f"  validation contrast {name}: {c['improvement']:+.5f}")
    if failures:
        print(f"  image failures reported: {failures[:3]}")
    if "M3P" not in fitted:
        print("  STOP: M3P did not produce valid matched-row predictions.")
        return 1
    print("  GO: finite features, frozen encodings, matched-row predictions.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 8 visual models.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config.yaml",
    )
    args = parser.parse_args(argv)
    return run_visual_models(args.config)


if __name__ == "__main__":
    sys.exit(main())
