"""Stage 2 orientation tests (plan Stage 2, decision rule: any failure blocks
all model fitting).

Verifies the four required transformations:

1. Reversing A/B and reversing the choice preserves the canonical outcome.
2. No-position predictions complement when endpoints are swapped.
3. Displayed- and canonical-coordinate position-model predictions agree
   after conversion.
4. Context attached to a row is unchanged under coordinate relabeling.

Plus real-data invariants on the validated raw CSVs (recomputed in memory;
the raw files are never modified).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.special import expit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.canonicalize import (  # noqa: E402
    CONTEXT_COLUMNS,
    canonical_endpoints,
    canonical_outcome,
    canonicalize_frame,
    load_raw,
    pair_id_of,
)

RNG = np.random.default_rng(20261028)


def _flip(choice: str) -> str:
    return {"A": "B", "B": "A"}[choice]


# ---------------------------------------------------------------------------
# Test 1 - reversal invariance of the canonical outcome
# ---------------------------------------------------------------------------


def test_reversal_preserves_canonical_outcome() -> None:
    cases = [
        ("cat_000002/cat_000002_0001.jpg", "cat_000002/cat_000002_0003.jpg", "A"),
        ("cat_000002/cat_000002_0003.jpg", "cat_000002/cat_000002_0001.jpg", "B"),
        ("cat_000751/cat_000751_0005.jpg", "cat_005017/cat_005017_0001.jpg", "A"),
        ("cat_005017/cat_005017_0001.jpg", "cat_000751/cat_000751_0005.jpg", "B"),
    ]
    for a_image, b_image, choice in cases:
        y = 1 if choice == "A" else 0
        i1, j1, s1 = canonical_endpoints(a_image, b_image)
        i2, j2, s2 = canonical_endpoints(b_image, a_image)
        assert (i1, j1) == (i2, j2)
        assert s2 == -s1
        y_c1 = canonical_outcome(y, s1)
        y_c2 = canonical_outcome(1 - y, s2)  # reversed display, reversed choice
        assert y_c1 == y_c2


def test_reversal_invariance_fuzz() -> None:
    for _ in range(2000):
        cat_a, cat_b = RNG.choice(["cat_000002", "cat_000751"], size=2)
        a_image = f"{cat_a}/{cat_a}_{RNG.integers(1, 9):04d}.jpg"
        b_image = f"{cat_b}/{cat_b}_{RNG.integers(1, 9):04d}.jpg"
        if a_image == b_image:
            continue
        y = int(RNG.integers(0, 2))
        i1, j1, s1 = canonical_endpoints(a_image, b_image)
        i2, j2, s2 = canonical_endpoints(b_image, a_image)
        assert (i1, j1) == (i2, j2)
        assert s2 == -s1
        assert canonical_outcome(y, s1) == canonical_outcome(1 - y, s2)


# ---------------------------------------------------------------------------
# Test 2 - no-position predictions complement under endpoint swap
# ---------------------------------------------------------------------------


def test_no_position_predictions_complement() -> None:
    scores = {f"img_{k}": float(v) for k, v in enumerate(RNG.normal(size=50))}
    images = list(scores)
    for _ in range(1000):
        a, b = RNG.choice(images, size=2, replace=False)
        p_ab = expit(scores[a] - scores[b])
        p_ba = expit(scores[b] - scores[a])
        assert p_ab + p_ba == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Test 3 - displayed vs canonical position-model predictions agree
# ---------------------------------------------------------------------------


def test_displayed_and_canonical_position_agree() -> None:
    # P_displayed(A) = sigma(u_A - u_B + beta)
    # P_canonical(i chosen) = sigma(u_i - u_j + s_t * beta)
    # Conversion: displayed A is canonical i when s_t=+1, canonical j when
    # s_t=-1, so p_canonical == p_displayed if s_t=+1 else 1 - p_displayed.
    scores = {f"img_{k}": float(v) for k, v in enumerate(RNG.normal(size=50))}
    images = list(scores)
    for _ in range(1000):
        a, b = RNG.choice(images, size=2, replace=False)
        beta = float(RNG.normal())
        i_img, j_img, s_t = canonical_endpoints(a, b)
        p_displayed = expit(scores[a] - scores[b] + beta)
        p_canonical = expit(scores[i_img] - scores[j_img] + s_t * beta)
        p_converted = p_displayed if s_t == 1 else 1.0 - p_displayed
        assert p_canonical == pytest.approx(p_converted, abs=1e-12)


# ---------------------------------------------------------------------------
# Test 4 - context is unchanged under coordinate relabeling
# ---------------------------------------------------------------------------


def _relabeled_row(row: dict) -> dict:
    """The same response with A/B display swapped and the choice flipped."""
    swapped = dict(row)
    for col in ("a_cat", "b_cat", "a_image", "b_image"):
        swapped[col] = row[col.replace("a_", "b_") if col.startswith("a_")
                        else col.replace("b_", "a_")]
    swapped["choice"] = _flip(row["choice"])
    return swapped


def test_context_invariant_under_relabeling() -> None:
    base = {
        "a_cat": "cat_000751",
        "b_cat": "cat_005017",
        "a_image": "cat_000751/cat_000751_0005.jpg",
        "b_image": "cat_005017/cat_005017_0001.jpg",
        "choice": "A",
        "gender": "woman",
        "pet_history": "current",
        "pet_care": "occasionally",
        "pet_contact": "rarely",
        "future_pet": "unsure",
        "mbti_ie": "I",
    }
    swapped = _relabeled_row(base)
    for col in CONTEXT_COLUMNS:
        assert swapped[col] == base[col]
    # And the canonical identity/outcome still match after relabeling.
    i1, j1, s1 = canonical_endpoints(base["a_image"], base["b_image"])
    i2, j2, s2 = canonical_endpoints(swapped["a_image"], swapped["b_image"])
    assert (i1, j1) == (i2, j2) and s2 == -s1
    assert canonical_outcome(1, s1) == canonical_outcome(0, s2)


# ---------------------------------------------------------------------------
# Real-data invariants (raw CSVs canonicalized in memory)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def canonical_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    train, test = load_raw(PROJECT_ROOT, cfg)
    return (
        canonicalize_frame(train, "train", "train_row_id"),
        canonicalize_frame(test, "test", "test_row_id"),
    )


def test_real_canonical_fields_consistent(
    canonical_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, _ = canonical_frames
    # s_t in {+1, -1}; pair_id matches canonical endpoints.
    assert set(train["s_t"].unique()) == {1, -1}
    assert (train["pair_id"] == train["i_image"] + "|" + train["j_image"]).all()
    assert (train["i_image"] < train["j_image"]).all()
    # y_displayed / y_canonical consistency with the raw choice.
    y = (train["choice"] == "A").astype(int).to_numpy()
    s = train["s_t"].to_numpy()
    assert (train["y_displayed"].to_numpy() == y).all()
    assert (
        train["y_canonical"].to_numpy() == np.where(s == 1, y, 1 - y)
    ).all()
    # pair_id is display-order independent: same unordered images, same id.
    assert train["pair_id"].nunique() == len(
        {pair_id_of(*canonical_endpoints(a, b)[:2])
         for a, b in zip(train["a_image"], train["b_image"])}
    )


def test_real_test_outcomes_null(
    canonical_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    _, test = canonical_frames
    assert test["y_displayed"].isna().all()
    assert test["y_canonical"].isna().all()
    assert (test["choice"].str.strip() == "").all()
