"""Stage 2 - Canonicalize pairs and test transformations.

Makes pair identity independent of display order (plan Stage 2):

* ``pair_id``: permanent unordered-pair identifier, ``"<i_image>|<j_image>"``
  where ``i_image`` is the lexicographically smaller image path.
* ``s_t``: +1 if canonical image ``i`` was displayed as A, else -1.
* ``y_displayed``: 1{choice == A} in displayed coordinates.
* ``y_canonical``: ``y_displayed`` if ``s_t == +1`` else ``1 - y_displayed``.

Displayed and canonical outcomes are stored separately. Displayed-coordinate
fitting remains the sole implementation path; canonical coordinates are for
grouping and summaries only.

Output: ``artifacts/rows_canonical.parquet`` (train and test rows; the test
outcome columns are null because official labels are withheld).

The transformation tests live in ``tests/test_orientation.py``; any failed
test blocks all model fitting (plan Stage 2 decision rule).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CONTEXT_COLUMNS = [
    "gender",
    "pet_history",
    "pet_care",
    "pet_contact",
    "future_pet",
    "mbti_ie",
]

BASE_COLUMNS = [
    "a_cat",
    "b_cat",
    "a_image",
    "b_image",
    "comparison_type",
] + CONTEXT_COLUMNS


# ---------------------------------------------------------------------------
# Row-level transformation helpers (also exercised by tests/test_orientation.py)
# ---------------------------------------------------------------------------


def canonical_endpoints(a_image: str, b_image: str) -> tuple[str, str, int]:
    """Return (i_image, j_image, s_t) with i_image < j_image lexicographically.

    s_t is +1 if the canonical image i was displayed as A, otherwise -1.
    """
    if a_image <= b_image:
        return a_image, b_image, 1
    return b_image, a_image, -1


def canonical_outcome(y_displayed: int, s_t: int) -> int:
    """Convert a displayed-coordinate outcome to canonical coordinates."""
    return y_displayed if s_t == 1 else 1 - y_displayed


def pair_id_of(i_image: str, j_image: str) -> str:
    """Permanent unordered-pair identifier from canonical endpoints."""
    return f"{i_image}|{j_image}"


# ---------------------------------------------------------------------------
# Frame canonicalization
# ---------------------------------------------------------------------------


def load_raw(root: Path, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(
        root / cfg["paths"]["train_csv"], dtype=str, keep_default_na=False
    )
    test = pd.read_csv(
        root / cfg["paths"]["test_csv"], dtype=str, keep_default_na=False
    )
    return train, test


def canonicalize_frame(df: pd.DataFrame, source: str, id_col: str) -> pd.DataFrame:
    """Add canonical-pair columns to validated raw rows.

    Displayed-coordinate fields (a_*, b_*, y_displayed) are preserved;
    canonical fields (i_image, j_image, s_t, y_canonical) are added.
    """
    out = df.copy()
    out.insert(0, "source", source)
    out = out.rename(columns={id_col: "row_id"})

    endpoints = [
        canonical_endpoints(a, b) for a, b in zip(out["a_image"], out["b_image"])
    ]
    out["i_image"] = [e[0] for e in endpoints]
    out["j_image"] = [e[1] for e in endpoints]
    out["s_t"] = np.array([e[2] for e in endpoints], dtype=np.int8)
    out["pair_id"] = [
        pair_id_of(e[0], e[1]) for e in endpoints
    ]

    if source == "train":
        y = (out["choice"] == "A").astype(np.int8)
        out["y_displayed"] = y
        out["y_canonical"] = [
            canonical_outcome(int(yv), int(sv))
            for yv, sv in zip(out["y_displayed"], out["s_t"])
        ]
    else:
        # Official test labels are withheld; keep outcomes null.
        out["y_displayed"] = pd.Series(pd.NA, index=out.index, dtype="Int8")
        out["y_canonical"] = pd.Series(pd.NA, index=out.index, dtype="Int8")

    out["y_displayed"] = out["y_displayed"].astype("Int8")
    out["y_canonical"] = out["y_canonical"].astype("Int8")

    columns = [
        "source",
        "row_id",
        "pair_id",
        "i_image",
        "j_image",
        "s_t",
        "y_displayed",
        "y_canonical",
        "choice",
        "prob_a",
        *BASE_COLUMNS,
    ]
    return out[[c for c in columns if c in out.columns]]


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run_canonicalize(config_path: Path) -> int:
    root = config_path.resolve().parent
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    train, test = load_raw(root, cfg)
    canon = pd.concat(
        [
            canonicalize_frame(train, "train", "train_row_id"),
            canonicalize_frame(test, "test", "test_row_id"),
        ],
        ignore_index=True,
    )

    artifacts_dir = root / cfg["paths"]["artifacts_dir"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "rows_canonical.parquet"
    canon.to_parquet(out_path, index=False)

    train_part = canon[canon["source"] == "train"]
    print("Stage 2 canonicalization complete.")
    print(f"  rows: {len(canon)} ({len(train_part)} train, "
          f"{len(canon) - len(train_part)} test)")
    print(f"  unordered pairs: train={train_part['pair_id'].nunique()}, "
          f"test={canon[canon['source'] == 'test']['pair_id'].nunique()}")
    print(f"  s_t=+1 share (train): {(train_part['s_t'] == 1).mean():.4f}")
    print(f"  wrote {out_path.relative_to(root)}")
    print("  Next gate: pytest tests/test_orientation.py must pass before "
          "any model fitting.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2 pair canonicalization.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config.yaml",
    )
    args = parser.parse_args(argv)
    return run_canonicalize(args.config)


if __name__ == "__main__":
    sys.exit(main())
