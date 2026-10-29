"""Stage 3 - Reserve experiments and freeze the primary split.

Defines exactly what the primary evaluation predicts (plan Stage 3):

* G3 reservation runs first *if* ``flags.enable_g3`` is true (it is false by
  default, so the reservation is recorded as skipped and every canonical row
  stays in the G2 pool). The Stage 11 reservation rule is implemented here so
  that enabling G3 later does not change this module's control flow.
* Rows are grouped by unordered ``pair_id``.
* Pairs are stratified by comparison type x response-count band
  (1 / 2-4 / 5-9 / 10+). A stratum with fewer than 20 pairs is merged into
  its comparison-type stratum.
* Within each stratum, pairs are shuffled once with the fixed seed and
  assigned approximately 70/15/15 by largest-remainder integer allocation.
* All pair and row assignments are saved. Seeds are never searched.

Outputs
-------
artifacts/split_manifest.parquet   row-level assignments (pair attributes on every row)
artifacts/split_pairs.parquet      pair-level assignments
artifacts/split_config.json        frozen configuration, strata summary, assertions

Decision rule: no pair may occur in more than one G2 partition (asserted).
Stop/go: disjoint-pair assertions pass and internal-test labels are
inaccessible to development scripts (enforced structurally by
:func:`load_partition_rows`, which refuses the test partition unless the
caller passes ``allow_test=True``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PARTITIONS = ("fitting", "validation", "test")

BAND_LABELS = ("1", "2-4", "5-9", "10+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def count_band(n_responses: int) -> str:
    """Frozen response-count bands: 1; 2-4; 5-9; 10+."""
    if n_responses <= 1:
        return "1"
    if n_responses <= 4:
        return "2-4"
    if n_responses <= 9:
        return "5-9"
    return "10+"


def largest_remainder_allocation(
    n: int, fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)
) -> tuple[int, int, int]:
    """Largest-remainder integer allocation of n pairs across partitions.

    Ties on the fractional remainder are broken deterministically in favour
    of the earlier partition (fitting, then validation, then test).
    """
    raw = [f * n for f in fractions]
    base = [int(np.floor(x)) for x in raw]
    remaining = n - sum(base)
    order = sorted(range(len(raw)), key=lambda k: (-(raw[k] - base[k]), k))
    for k in order[:remaining]:
        base[k] += 1
    return tuple(base)  # type: ignore[return-value]


def load_partition_rows(
    config_path: Path,
    partition: str,
    allow_test: bool = False,
) -> pd.DataFrame:
    """Load canonical rows for one partition.

    Structural guard for the Stage 3 stop/go rule: development scripts may
    load only ``fitting`` and ``validation`` outcomes. The internal test
    partition raises unless the caller explicitly passes ``allow_test=True``
    (reserved for the Stage 10 evaluation entry point).
    """
    if partition not in PARTITIONS:
        raise ValueError(f"Unknown partition: {partition!r}")
    if partition == "test" and not allow_test:
        raise PermissionError(
            "Internal-test outcomes are inaccessible to development scripts "
            "(plan Stage 3 stop/go). Pass allow_test=True only from the "
            "frozen Stage 10 evaluation."
        )
    root = config_path.resolve().parent
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    manifest = pd.read_parquet(root / cfg["paths"]["artifacts_dir"]
                               / "split_manifest.parquet")
    rows = pd.read_parquet(root / cfg["paths"]["artifacts_dir"]
                           / "rows_canonical.parquet")
    keys = manifest.loc[manifest["partition"] == partition, "row_id"]
    return rows[rows["row_id"].isin(set(keys))].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Optional G3 reservation (Stage 11 rule, applied before any G2 partitioning)
# ---------------------------------------------------------------------------


def reserve_g3_photos(photos: list[str], seed: int) -> dict:
    """Hash-sort photo IDs with the fixed seed; reserve ~5% test, ~5% validation."""
    keyed = sorted(
        photos,
        key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).hexdigest(),
    )
    n = len(keyed)
    n_each = int(round(0.05 * n))
    return {
        "g3_test_photos": sorted(keyed[:n_each]),
        "g3_validation_photos": sorted(keyed[n_each: 2 * n_each]),
        "g2_pool_photos": sorted(keyed[2 * n_each:]),
    }


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run_split(config_path: Path) -> int:
    root = config_path.resolve().parent
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    seed = int(cfg["seeds"]["split"])
    fractions = tuple(float(f) for f in cfg["split"]["fractions"])
    merge_threshold = int(cfg["split"]["merge_threshold"])
    enable_g3 = bool(cfg["flags"]["enable_g3"])

    rows = pd.read_parquet(
        root / cfg["paths"]["artifacts_dir"] / "rows_canonical.parquet"
    )
    train = rows[rows["source"] == "train"].copy()

    # --- Optional G3 reservation, before any G2 partitioning ----------------
    all_photos = sorted(set(train["i_image"]) | set(train["j_image"]))
    if enable_g3:
        g3 = reserve_g3_photos(all_photos, seed)
        reserved = set(g3["g3_test_photos"]) | set(g3["g3_validation_photos"])
        g2_pool = train[
            ~train["i_image"].isin(reserved) & ~train["j_image"].isin(reserved)
        ].copy()
        g3_record = {
            "enabled": True,
            "rule": "hash-sort photo IDs with the split seed; ~5% test, ~5% validation",
            "n_g3_test_photos": len(g3["g3_test_photos"]),
            "n_g3_validation_photos": len(g3["g3_validation_photos"]),
            "n_g2_pool_photos": len(g3["g2_pool_photos"]),
            "n_rows_excluded_from_g2": int(len(train) - len(g2_pool)),
            "g3_test_photos": g3["g3_test_photos"],
            "g3_validation_photos": g3["g3_validation_photos"],
        }
    else:
        g2_pool = train
        g3_record = {
            "enabled": False,
            "skipped": True,
            "reason": "flags.enable_g3 is false (Stage 0 default); the full "
                      "labeled pool is used for G2 and no G3 test may be "
                      "added later (plan Stage 11 risks/fallbacks).",
        }

    # --- Pair-level table ---------------------------------------------------
    pair = (
        g2_pool.groupby("pair_id")
        .agg(
            n_responses=("row_id", "size"),
            comparison_types=("comparison_type", lambda s: sorted(set(s))),
        )
        .reset_index()
    )
    pair["comparison_type_unique"] = pair["comparison_types"].str.len() == 1
    pair["comparison_type"] = pair["comparison_types"].str[0]
    pair["count_band"] = pair["n_responses"].map(count_band)

    # --- Strata with the <20-pair merge rule --------------------------------
    pair["stratum"] = pair["comparison_type"] + "|" + pair["count_band"]
    strata_sizes = pair["stratum"].value_counts()
    small_strata = set(strata_sizes[strata_sizes < merge_threshold].index)
    merge_note: dict[str, str] = {}
    for stratum in sorted(small_strata):
        ctype = stratum.split("|")[0]
        pair.loc[pair["stratum"] == stratum, "stratum"] = ctype + "|merged"
        merge_note[stratum] = (
            f"merged into {ctype}|merged (<{merge_threshold} pairs)"
        )

    # --- Shuffle once per stratum and allocate ------------------------------
    rng = np.random.default_rng(seed)
    assignments: dict[str, str] = {}
    strata_summary = []
    for stratum in sorted(pair["stratum"].unique()):
        sub = pair[pair["stratum"] == stratum]
        pair_ids = sub["pair_id"].to_numpy()
        shuffled = pair_ids[rng.permutation(len(pair_ids))]  # one shuffle
        n_fit, n_val, n_test = largest_remainder_allocation(
            len(shuffled), fractions
        )
        for pid, part in zip(shuffled[:n_fit], [PARTITIONS[0]] * n_fit):
            assignments[pid] = part
        for pid, part in zip(
            shuffled[n_fit: n_fit + n_val], [PARTITIONS[1]] * n_val
        ):
            assignments[pid] = part
        for pid, part in zip(shuffled[n_fit + n_val:], [PARTITIONS[2]] * n_test):
            assignments[pid] = part
        ctype, band = stratum.split("|")
        strata_summary.append(
            {
                "stratum": stratum,
                "comparison_type": ctype,
                "count_band": band,
                "n_pairs": int(len(shuffled)),
                "n_pairs_fitting": int(n_fit),
                "n_pairs_validation": int(n_val),
                "n_pairs_test": int(n_test),
                "n_rows": int(sub["n_responses"].sum()),
                "n_rows_fitting": int(
                    sub.loc[sub["pair_id"].isin(shuffled[:n_fit]),
                            "n_responses"].sum()
                ),
                "n_rows_validation": int(
                    sub.loc[sub["pair_id"].isin(shuffled[n_fit: n_fit + n_val]),
                            "n_responses"].sum()
                ),
                "n_rows_test": int(
                    sub.loc[sub["pair_id"].isin(shuffled[n_fit + n_val:]),
                            "n_responses"].sum()
                ),
            }
        )

    pair["partition"] = pair["pair_id"].map(assignments)

    # --- Row-level manifest ---------------------------------------------------
    manifest = g2_pool[["row_id", "pair_id"]].merge(
        pair[["pair_id", "comparison_type", "n_responses", "count_band",
              "stratum", "partition"]],
        on="pair_id",
        how="left",
        validate="many_to_one",
    )

    # --- Stop/go assertions ---------------------------------------------------
    parts = pair.groupby("partition")["pair_id"].apply(set)
    disjoint = (
        parts.get(PARTITIONS[0], set()).isdisjoint(parts.get(PARTITIONS[1], set()))
        and parts.get(PARTITIONS[0], set()).isdisjoint(parts.get(PARTITIONS[2], set()))
        and parts.get(PARTITIONS[1], set()).isdisjoint(parts.get(PARTITIONS[2], set()))
    )
    assertions = {
        "disjoint_pairs": bool(disjoint),
        "every_pair_assigned_exactly_once": bool(
            pair["partition"].isin(PARTITIONS).all()
            and pair["pair_id"].is_unique
        ),
        "every_g2_row_assigned": bool(
            manifest["partition"].isin(PARTITIONS).all()
            and len(manifest) == len(g2_pool)
        ),
        "pair_comparison_type_unique": bool(
            pair["comparison_type_unique"].all()
        ),
        "internal_test_labels_inaccessible": (
            "structural: src.split.load_partition_rows refuses the test "
            "partition unless allow_test=True (Stage 10 only)"
        ),
    }
    ok = all(v is True for k, v in assertions.items()
             if k != "internal_test_labels_inaccessible")

    # --- Outputs --------------------------------------------------------------
    artifacts_dir = root / cfg["paths"]["artifacts_dir"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(artifacts_dir / "split_manifest.parquet", index=False)
    pair.drop(columns=["comparison_types", "comparison_type_unique"]).to_parquet(
        artifacts_dir / "split_pairs.parquet", index=False
    )

    totals = (
        manifest.groupby("partition")
        .agg(n_rows=("row_id", "size"), n_pairs=("pair_id", "nunique"))
        .reindex(PARTITIONS)
        .to_dict(orient="index")
    )
    config_out = {
        "stage": 3,
        "seed": seed,
        "seed_search": "none - the fixed seed is used exactly once per stratum",
        "fractions": dict(zip(PARTITIONS, fractions)),
        "allocation": "largest-remainder integer allocation; remainder ties "
                      "favour fitting, then validation, then test",
        "count_bands": list(BAND_LABELS),
        "merge_threshold_pairs": merge_threshold,
        "merged_strata": merge_note,
        "g3_reservation": g3_record,
        "strata": strata_summary,
        "totals": totals,
        "assertions": assertions,
        "stop_go": {"proceed": ok},
    }
    with (artifacts_dir / "split_config.json").open("w", encoding="utf-8") as fh:
        json.dump(config_out, fh, indent=2)

    print("Stage 3 split frozen.")
    print(f"  G3 reservation: {'enabled' if enable_g3 else 'skipped (flag off)'}")
    print(f"  G2 pool: {len(g2_pool)} rows, {pair['pair_id'].nunique()} pairs, "
          f"{len(strata_summary)} strata")
    for part in PARTITIONS:
        t = totals[part]
        print(f"  {part}: {t['n_pairs']} pairs, {t['n_rows']} rows "
              f"({t['n_rows'] / len(g2_pool):.1%} of rows)")
    if merge_note:
        print(f"  merged strata: {merge_note}")
    print(f"  assertions: {assertions}")
    if not ok:
        print("  STOP: a split assertion failed.")
        return 1
    print("  GO: disjoint-pair assertions pass.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 3 primary split.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config.yaml",
    )
    args = parser.parse_args(argv)
    return run_split(args.config)


if __name__ == "__main__":
    sys.exit(main())
