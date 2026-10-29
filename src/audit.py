"""Stage 1 - Audit and validate the raw files.

Implements the Stage 1 operations from the project plan:

* preserve raw files unchanged and record SHA-256 hashes;
* check supplied row counts and required columns;
* verify row-ID uniqueness within each file;
* verify labeled choices are A/B and official test labels are blank;
* check exact comparison-type values;
* verify ``a_cat == b_cat`` iff ``comparison_type == "within_cat"``;
* verify each image path's folder agrees with its cat ID;
* detect self-comparisons of the same image (quarantine + report);
* count distinct cats, photos, unordered pairs, responses per pair/photo;
* tabulate context values and missingness;
* check image existence/readability and exact image hashes;
* classify official test rows by train-pair and train-image overlap.

Outputs
-------
artifacts/audit.json
artifacts/image_hashes.csv
artifacts/cleaning_log.csv
tables/data_audit.csv
tables/official_test_coverage.csv

Only structural information is used for audit summaries; no preference
summaries are produced here.

Exit status is non-zero if any check marked as a stop condition fails
(invalid labels, conflicting IDs, or ambiguous identity mappings).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:  # pragma: no cover - Pillow is in the project venv
    _HAS_PIL = False

# ---------------------------------------------------------------------------
# Fixed audit vocabulary (README.md field documentation)
# ---------------------------------------------------------------------------

CONTEXT_COLUMNS = [
    "gender",
    "pet_history",
    "pet_care",
    "pet_contact",
    "future_pet",
    "mbti_ie",
]

EXPECTED_VALUES = {
    "comparison_type": {"within_cat", "between_cats"},
    "pet_history": {"current", "past", "never"},
    "pet_care": {"never", "occasionally", "past_regular", "current_regular"},
    "pet_contact": {"never", "rarely", "sometimes", "often", "daily"},
    "future_pet": {"yes", "no", "unsure"},
    "mbti_ie": {"I", "E", "unknown"},
    # gender is free-form self-report: tabulated, not validated.
}

TRAIN_COLUMNS = [
    "train_row_id",
    "a_cat",
    "b_cat",
    "a_image",
    "b_image",
    "choice",
    "comparison_type",
] + CONTEXT_COLUMNS

TEST_COLUMNS = [
    "test_row_id",
    "a_cat",
    "b_cat",
    "a_image",
    "b_image",
    "choice",
    "prob_a",
    "comparison_type",
] + CONTEXT_COLUMNS

# Checks whose failure blocks all later stages (plan Stage 1 decision rules).
STOP_CHECKS = {
    "train_choice_labels",
    "test_labels_blank",
    "comparison_type_values",
    "cat_identity_vs_comparison_type",
    "image_folder_matches_cat",
    "train_row_id_uniqueness",
    "test_row_id_uniqueness",
    "train_required_columns",
    "test_required_columns",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> pd.DataFrame:
    """Load a CSV with every column as raw text; blanks stay empty strings."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def canonical_pair(a_image: str, b_image: str) -> str:
    """Display-order-independent pair key (lexicographic on image paths)."""
    lo, hi = sorted((a_image, b_image))
    return f"{lo}|{hi}"


class AuditRecorder:
    """Collects audit checks and cleaning actions, and writes the outputs."""

    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.cleaning_actions: list[dict] = []

    def check(
        self,
        check_id: str,
        scope: str,
        status: str,
        detail: str,
        n_affected: int = 0,
    ) -> None:
        self.checks.append(
            {
                "check_id": check_id,
                "scope": scope,
                "status": status,  # PASS / FAIL / WARN / INFO
                "n_affected": int(n_affected),
                "detail": detail,
            }
        )

    def action(
        self,
        action: str,
        scope: str,
        row_id: str,
        detail: str,
    ) -> None:
        self.cleaning_actions.append(
            {
                "action": action,
                "scope": scope,
                "row_id": row_id,
                "detail": detail,
            }
        )

    @property
    def failed_stop_checks(self) -> list[dict]:
        return [
            c
            for c in self.checks
            if c["status"] == "FAIL" and c["check_id"] in STOP_CHECKS
        ]


# ---------------------------------------------------------------------------
# Individual audit blocks
# ---------------------------------------------------------------------------


def audit_file_hashes(
    rec: AuditRecorder, root: Path, cfg: dict, audit: dict
) -> None:
    paths_cfg = cfg["paths"]
    hashes = {}
    for name in ("train_csv", "test_csv"):
        path = root / paths_cfg[name]
        hashes[name] = {
            "path": paths_cfg[name],
            "sha256": sha256_file(path),
            "n_bytes": path.stat().st_size,
        }
    audit["raw_file_hashes"] = hashes
    rec.check(
        "raw_file_hashes",
        "train.csv/test.csv",
        "INFO",
        "SHA-256 recorded; raw files left unchanged.",
    )


def audit_structure(
    rec: AuditRecorder, train: pd.DataFrame, test: pd.DataFrame, cfg: dict, audit: dict
) -> None:
    expected = cfg["expected"]

    for name, df, want_cols, want_rows, id_col in (
        ("train", train, TRAIN_COLUMNS, expected["train_rows"], "train_row_id"),
        ("test", test, TEST_COLUMNS, expected["test_rows"], "test_row_id"),
    ):
        got_cols = list(df.columns)
        if got_cols == want_cols:
            rec.check(
                f"{name}_required_columns", name, "PASS",
                f"All {len(want_cols)} required columns present in order.",
            )
        else:
            missing = sorted(set(want_cols) - set(got_cols))
            extra = sorted(set(got_cols) - set(want_cols))
            rec.check(
                f"{name}_required_columns", name, "FAIL",
                f"Column mismatch. missing={missing} extra={extra}",
                n_affected=len(missing) + len(extra),
            )

        n_rows = len(df)
        status = "PASS" if n_rows == want_rows else "WARN"
        rec.check(
            f"{name}_row_count", name, status,
            f"{n_rows} rows (README states {want_rows}); "
            + ("matches." if n_rows == want_rows else "MISMATCH - resolve before modeling."),
            n_affected=abs(n_rows - want_rows),
        )

        n_dup = int(df[id_col].duplicated().sum())
        rec.check(
            f"{name}_row_id_uniqueness", name,
            "PASS" if n_dup == 0 else "FAIL",
            f"{n_dup} duplicated {id_col} values.",
            n_affected=n_dup,
        )


def audit_labels_and_identity(
    rec: AuditRecorder, train: pd.DataFrame, test: pd.DataFrame
) -> None:
    # Labeled choices must be exactly A/B.
    bad_choice = sorted(set(train["choice"]) - {"A", "B"})
    rec.check(
        "train_choice_labels", "train",
        "PASS" if not bad_choice else "FAIL",
        "All train choices in {A, B}." if not bad_choice
        else f"Invalid train choice values: {bad_choice}",
        n_affected=int((~train["choice"].isin(["A", "B"])).sum()),
    )

    # Official test labels must be blank.
    blank_choice = int((test["choice"].str.strip() == "").sum())
    blank_prob = int((test["prob_a"].str.strip() == "").sum())
    ok = blank_choice == len(test) and blank_prob == len(test)
    rec.check(
        "test_labels_blank", "test",
        "PASS" if ok else "FAIL",
        f"test choice blank in {blank_choice}/{len(test)} rows; "
        f"prob_a blank in {blank_prob}/{len(test)} rows.",
        n_affected=len(test) - min(blank_choice, blank_prob),
    )

    # Exact comparison-type values.
    for name, df in (("train", train), ("test", test)):
        bad = sorted(set(df["comparison_type"]) - EXPECTED_VALUES["comparison_type"])
        rec.check(
            "comparison_type_values" if name == "train"
            else "comparison_type_values_test",
            name,
            "PASS" if not bad else "FAIL",
            f"comparison_type values exactly {sorted(EXPECTED_VALUES['comparison_type'])}."
            if not bad else f"Unexpected comparison_type values: {bad}",
            n_affected=int((~df["comparison_type"].isin(
                EXPECTED_VALUES["comparison_type"])).sum()),
        )

    # a_cat == b_cat iff comparison_type == "within_cat".
    for name, df, id_col in (
        ("train", train, "train_row_id"), ("test", test, "test_row_id")
    ):
        same_cat = df["a_cat"] == df["b_cat"]
        within = df["comparison_type"] == "within_cat"
        mismatch = df[same_cat != within]
        rec.check(
            "cat_identity_vs_comparison_type" if name == "train"
            else "cat_identity_vs_comparison_type_test",
            name,
            "PASS" if mismatch.empty else "FAIL",
            "a_cat == b_cat iff comparison_type == 'within_cat'."
            if mismatch.empty
            else f"{len(mismatch)} rows violate the identity rule, e.g. "
                 f"{mismatch[id_col].head(5).tolist()}",
            n_affected=len(mismatch),
        )

    # Image path folder agrees with the cat ID on both sides.
    for name, df in (("train", train), ("test", test)):
        for side in ("a", "b"):
            folder = df[f"{side}_image"].str.split("/").str[0]
            n_bad = int((folder != df[f"{side}_cat"]).sum())
            rec.check(
                "image_folder_matches_cat" if (name, side) == ("train", "a")
                else f"image_folder_matches_cat_{name}_{side}",
                f"{name}.{side}_image",
                "PASS" if n_bad == 0 else "FAIL",
                f"{n_bad} rows where the image folder != {side}_cat.",
                n_affected=n_bad,
            )


def audit_self_comparisons_and_duplicates(
    rec: AuditRecorder, train: pd.DataFrame, test: pd.DataFrame
) -> None:
    for name, df, id_col in (
        ("train", train, "train_row_id"), ("test", test, "test_row_id")
    ):
        self_cmp = df[df["a_image"] == df["b_image"]]
        rec.check(
            "self_comparisons" if name == "train" else "self_comparisons_test",
            name,
            "PASS" if self_cmp.empty else "WARN",
            "No same-image self-comparisons." if self_cmp.empty
            else f"{len(self_cmp)} self-comparisons quarantined and reported.",
            n_affected=len(self_cmp),
        )
        for _, row in self_cmp.iterrows():
            rec.action(
                "quarantine", name, row[id_col],
                f"Self-comparison of image {row['a_image']} on side A and B.",
            )

    # Identical-looking response rows are retained (plan Stage 1 decision rule).
    dup_mask = train.duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    n_dup_groups = int(train[dup_mask].groupby(list(train.columns)).ngroups) if n_dup_rows else 0
    rec.check(
        "identical_response_rows", "train", "INFO",
        f"{n_dup_rows} rows in {n_dup_groups} identical-value groups; retained "
        "(they may be distinct annotators giving the same answer).",
        n_affected=n_dup_rows,
    )
    if n_dup_rows:
        rec.action(
            "retain", "train", "(multiple)",
            f"{n_dup_rows} identical-looking response rows in {n_dup_groups} "
            "groups retained per Stage 1 decision rule.",
        )


def audit_units(rec: AuditRecorder, train: pd.DataFrame, test: pd.DataFrame, audit: dict) -> None:
    train_pairs = train.apply(lambda r: canonical_pair(r["a_image"], r["b_image"]), axis=1)
    test_pairs = test.apply(lambda r: canonical_pair(r["a_image"], r["b_image"]), axis=1)

    train_images = set(train["a_image"]) | set(train["b_image"])
    test_images = set(test["a_image"]) | set(test["b_image"])
    train_cats = set(train["a_cat"]) | set(train["b_cat"])
    test_cats = set(test["a_cat"]) | set(test["b_cat"])

    per_pair = train_pairs.value_counts()
    per_photo = pd.concat([train["a_image"], train["b_image"]]).value_counts()

    units = {
        "train": {
            "n_rows": len(train),
            "n_distinct_cats": len(train_cats),
            "n_distinct_photos": len(train_images),
            "n_unordered_pairs": int(train_pairs.nunique()),
            "responses_per_pair": _describe(per_pair),
            "responses_per_photo": _describe(per_photo),
        },
        "test": {
            "n_rows": len(test),
            "n_distinct_cats": len(test_cats),
            "n_distinct_photos": len(test_images),
            "n_unordered_pairs": int(test_pairs.nunique()),
        },
        "overlap": {
            "cats_in_both": len(train_cats & test_cats),
            "photos_in_both": len(train_images & test_images),
            "pairs_in_both": len(set(train_pairs) & set(test_pairs)),
        },
    }
    audit["units"] = units
    rec.check(
        "unit_counts", "train/test", "INFO",
        f"train: {len(train_cats)} cats, {len(train_images)} photos, "
        f"{train_pairs.nunique()} unordered pairs; "
        f"test: {len(test_cats)} cats, {len(test_images)} photos, "
        f"{test_pairs.nunique()} unordered pairs; "
        f"overlap: {units['overlap']['photos_in_both']} photos, "
        f"{units['overlap']['pairs_in_both']} pairs.",
    )


def _describe(series: pd.Series) -> dict:
    return {
        "min": int(series.min()),
        "median": float(series.median()),
        "mean": float(series.mean()),
        "max": int(series.max()),
        "count_bands": {
            "1": int((series == 1).sum()),
            "2-4": int(((series >= 2) & (series <= 4)).sum()),
            "5-9": int(((series >= 5) & (series <= 9)).sum()),
            "10+": int((series >= 10).sum()),
        },
    }


def audit_context(rec: AuditRecorder, train: pd.DataFrame, test: pd.DataFrame, audit: dict) -> None:
    tabulation: dict = {}
    for name, df in (("train", train), ("test", test)):
        tabulation[name] = {}
        for col in CONTEXT_COLUMNS:
            values = df[col].str.strip()
            n_missing = int((values == "").sum())
            counts = Counter(values[values != ""])
            unexpected = sorted(set(counts) - EXPECTED_VALUES.get(col, set(counts)))
            status = "PASS"
            if unexpected:
                status = "WARN"
            elif n_missing:
                status = "WARN"
            rec.check(
                f"context_{col}_{name}", f"{name}.{col}", status,
                f"missing={n_missing}; values={dict(sorted(counts.items()))}"
                + (f"; UNEXPECTED values: {unexpected}" if unexpected else ""),
                n_affected=n_missing,
            )
            tabulation[name][col] = {
                "n_missing": n_missing,
                "values": dict(sorted(counts.items())),
                "unexpected_values": unexpected,
            }
            if n_missing:
                rec.action(
                    "encode_unknown", f"{name}.{col}", "(column)",
                    f"{n_missing} missing values to be encoded as 'unknown' "
                    "per Stage 1 decision rule.",
                )
    audit["context_tabulation"] = tabulation


def audit_images(
    rec: AuditRecorder,
    root: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: dict,
    audit: dict,
) -> None:
    images_dir = root / cfg["paths"]["images_dir"]
    referenced = sorted(
        (set(train["a_image"]) | set(train["b_image"]))
        | (set(test["a_image"]) | set(test["b_image"]))
    )
    on_disk = {
        str(p.relative_to(images_dir)).replace("\\", "/")
        for p in images_dir.rglob("*")
        if p.is_file()
    }

    referenced_set = set(referenced)
    missing = sorted(referenced_set - on_disk)
    unreferenced = sorted(on_disk - referenced_set)

    rec.check(
        "image_existence", "images",
        "PASS" if not missing else "WARN",
        f"{len(referenced) - len(missing)}/{len(referenced)} referenced images exist; "
        f"{len(on_disk)} files on disk; {len(unreferenced)} unreferenced files.",
        n_affected=len(missing),
    )
    for rel in missing:
        rec.action(
            "report_missing_image", "images", rel,
            "Referenced image not found on disk; Stage 8 image fallback applies.",
        )

    rec.check(
        "image_count_vs_readme", "images",
        "PASS" if len(on_disk) == cfg["expected"]["n_images"] else "WARN",
        f"{len(on_disk)} image files on disk "
        f"(README states {cfg['expected']['n_images']}).",
        n_affected=abs(len(on_disk) - cfg["expected"]["n_images"]),
    )

    # Hash every existing referenced image and verify readability.
    hash_rows = []
    unreadable = []
    for rel in referenced:
        path = images_dir / rel
        if not path.is_file():
            continue
        digest = sha256_file(path)
        readable = True
        if _HAS_PIL:
            try:
                with Image.open(path) as im:
                    im.verify()
            except Exception:  # noqa: BLE001 - any decode failure counts
                readable = False
        if not readable:
            unreadable.append(rel)
        hash_rows.append({"image": rel, "sha256": digest, "readable": readable})

    hashes_df = pd.DataFrame(hash_rows)
    rec.check(
        "image_readability", "images",
        "PASS" if not unreadable else "WARN",
        f"{len(hashes_df) - len(unreadable)}/{len(hashes_df)} existing referenced "
        "images readable by Pillow." + ("" if _HAS_PIL else " Pillow unavailable; skipped."),
        n_affected=len(unreadable),
    )
    for rel in unreadable:
        rec.action(
            "report_corrupt_image", "images", rel,
            "Image exists but failed Pillow decode; Stage 8 image fallback applies.",
        )

    dup_hashes = (
        hashes_df.groupby("sha256")["image"]
        .apply(sorted)
        .loc[lambda s: s.str.len() > 1]
    )
    rec.check(
        "image_duplicate_content", "images", "INFO",
        f"{len(dup_hashes)} SHA-256 values shared by more than one image path.",
        n_affected=int(sum(len(v) for v in dup_hashes)),
    )

    audit["images"] = {
        "n_referenced": len(referenced),
        "n_on_disk": len(on_disk),
        "n_missing": len(missing),
        "missing": missing,
        "n_unreferenced_on_disk": len(unreferenced),
        "n_unreadable": len(unreadable),
        "unreadable": unreadable,
        "duplicate_hash_groups": {k: v for k, v in dup_hashes.items()},
    }

    out_dir = root / cfg["paths"]["artifacts_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes_df.to_csv(out_dir / "image_hashes.csv", index=False)


def audit_official_test_coverage(
    rec: AuditRecorder,
    root: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    train_pairs = {
        canonical_pair(a, b) for a, b in zip(train["a_image"], train["b_image"])
    }
    train_images = set(train["a_image"]) | set(train["b_image"])

    rows = []
    for _, r in test.iterrows():
        pair = canonical_pair(r["a_image"], r["b_image"])
        a_in = r["a_image"] in train_images
        b_in = r["b_image"] in train_images
        pair_in = pair in train_pairs
        if pair_in:
            category = "pair_seen_in_train"
        elif a_in and b_in:
            category = "both_images_seen_new_pair"
        elif a_in or b_in:
            category = "one_image_seen"
        else:
            category = "no_images_seen"
        rows.append(
            {
                "test_row_id": r["test_row_id"],
                "comparison_type": r["comparison_type"],
                "pair_in_train": pair_in,
                "a_image_in_train": a_in,
                "b_image_in_train": b_in,
                "coverage_category": category,
            }
        )

    coverage = pd.DataFrame(rows)
    counts = coverage["coverage_category"].value_counts().to_dict()
    rec.check(
        "official_test_coverage", "test", "INFO",
        f"Official test rows by train overlap: {counts}.",
    )

    out_dir = root / cfg["paths"]["tables_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(out_dir / "official_test_coverage.csv", index=False)
    return coverage


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_audit(config_path: Path) -> int:
    root = config_path.resolve().parent
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    train = load_rows(root / cfg["paths"]["train_csv"])
    test = load_rows(root / cfg["paths"]["test_csv"])

    rec = AuditRecorder()
    audit: dict = {"stage": 1, "config": cfg}

    audit_file_hashes(rec, root, cfg, audit)
    audit_structure(rec, train, test, cfg, audit)
    audit_labels_and_identity(rec, train, test)
    audit_self_comparisons_and_duplicates(rec, train, test)
    audit_units(rec, train, test, audit)
    audit_context(rec, train, test, audit)
    audit_images(rec, root, train, test, cfg, audit)
    audit_official_test_coverage(rec, root, train, test, cfg)

    stop_failures = rec.failed_stop_checks
    audit["stop_go"] = {
        "stop_condition_failures": [c["check_id"] for c in stop_failures],
        "proceed": not stop_failures,
        "rule": "No unresolved label or identity errors.",
    }
    audit["checks"] = rec.checks

    artifacts_dir = root / cfg["paths"]["artifacts_dir"]
    tables_dir = root / cfg["paths"]["tables_dir"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    with (artifacts_dir / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    pd.DataFrame(rec.checks).to_csv(tables_dir / "data_audit.csv", index=False)
    pd.DataFrame(
        rec.cleaning_actions,
        columns=["action", "scope", "row_id", "detail"],
    ).to_csv(artifacts_dir / "cleaning_log.csv", index=False)

    # Console summary.
    counts = Counter(c["status"] for c in rec.checks)
    print("Stage 1 audit complete.")
    print(f"  checks: {dict(counts)}")
    print(f"  cleaning actions: {len(rec.cleaning_actions)}")
    print(f"  artifacts/audit.json, artifacts/image_hashes.csv, "
          f"artifacts/cleaning_log.csv")
    print(f"  tables/data_audit.csv, tables/official_test_coverage.csv")
    if stop_failures:
        print("  STOP: unresolved label/identity errors:")
        for c in stop_failures:
            print(f"    - {c['check_id']}: {c['detail']}")
        return 1
    print("  GO: no unresolved label or identity errors.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 1 raw-file audit.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config.yaml",
        help="Path to config.yaml (defaults to the project root copy).",
    )
    args = parser.parse_args(argv)
    return run_audit(args.config)


if __name__ == "__main__":
    sys.exit(main())
