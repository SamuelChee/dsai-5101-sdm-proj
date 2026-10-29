"""Stage 5 recovery tests (plan Stage 5 recovery checks, part 2).

Fit fractional expected counts under simulation A with a very small penalty;
require the maximum probability error below 1e-3. Also check the fitted
expected loss approaches the exact oracle reference from above.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.bt import expected_log_loss  # noqa: E402
from src.simulations import (  # noqa: E402
    PAIRS_ABC,
    P_A,
    bt_design,
    fit_bt_on_counts,
    reference_values,
)


def test_fractional_count_recovery() -> None:
    """Fitting exact expected counts must recover the true probabilities."""
    n_per_pair = 200.0
    expected_wins = n_per_pair * P_A  # fractional expected counts
    fit = fit_bt_on_counts(
        PAIRS_ABC,
        expected_wins,
        np.full(3, n_per_pair),
        lam=1e-6,  # very small penalty
    )
    assert fit.success, f"fit failed: {fit.message}"
    p_hat = fit.predict_proba(bt_design(PAIRS_ABC, 3))
    max_err = float(np.max(np.abs(p_hat - P_A)))
    assert max_err < 1e-3, f"max probability error {max_err:.2e} >= 1e-3"


def test_recovered_expected_loss_matches_oracle() -> None:
    """With exact counts, fitted expected loss is within 1e-3 of the oracle."""
    n_per_pair = 200.0
    fit = fit_bt_on_counts(
        PAIRS_ABC, n_per_pair * P_A, np.full(3, n_per_pair), lam=1e-6
    )
    assert fit.success
    p_hat = fit.predict_proba(bt_design(PAIRS_ABC, 3))
    fitted_expected = float(expected_log_loss(P_A, p_hat).mean())
    oracle = reference_values()["A"]["oracle_expected_loss"]
    assert 0.0 <= fitted_expected - oracle < 1e-3


def test_cycle_best_bt_matches_reference() -> None:
    """Under the symmetric 0.8 cycle, fitted BT sits at the ln 2 reference."""
    pairs_cyclic = [(0, 1), (1, 2), (2, 0)]
    p_true = np.full(3, 0.8)
    n_per_pair = 200.0
    fit = fit_bt_on_counts(
        pairs_cyclic, n_per_pair * p_true, np.full(3, n_per_pair), lam=1e-6
    )
    assert fit.success
    p_hat = fit.predict_proba(bt_design(pairs_cyclic, 3))
    refs = reference_values()["C"]
    fitted_expected = float(expected_log_loss(p_true, p_hat).mean())
    assert fitted_expected - refs["best_bt"] < 1e-3
    # And the restriction gap is real: the pair oracle is strictly better.
    assert refs["best_bt"] - refs["pair_oracle"] > 0.19
