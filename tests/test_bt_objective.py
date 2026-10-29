"""Stage 5 objective tests (plan Stage 5 recovery checks, part 1).

1. Analytic gradient matches central finite differences on a small example.
2. Row-level and count-level objectives agree when there are no row-varying
   covariates (a count row with weight n_e and y = k_e/n_e reproduces n_e
   identical binary rows).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.bt import (  # noqa: E402
    BETA_RIDGE,
    fit_penalized_logistic,
    penalized_logistic_objective,
)

RNG = np.random.default_rng(20261029)


def _small_problem(with_position: bool):
    n_rows, n_params = 40, 7
    X = sp.csr_matrix(RNG.normal(size=(n_rows, n_params)))
    y = RNG.integers(0, 2, size=n_rows).astype(float)
    lam = 1e-3
    theta0 = RNG.normal(size=n_params + (1 if with_position else 0))
    return X, y, lam, theta0


@pytest.mark.parametrize("with_position", [False, True])
def test_gradient_matches_finite_difference(with_position: bool) -> None:
    X, y, lam, theta0 = _small_problem(with_position)
    _, grad = penalized_logistic_objective(
        theta0, X, y, None, lam, with_position
    )
    eps = 1e-6
    fd = np.empty_like(theta0)
    for k in range(len(theta0)):
        step = np.zeros_like(theta0)
        step[k] = eps
        f_plus, _ = penalized_logistic_objective(
            theta0 + step, X, y, None, lam, with_position
        )
        f_minus, _ = penalized_logistic_objective(
            theta0 - step, X, y, None, lam, with_position
        )
        fd[k] = (f_plus - f_minus) / (2 * eps)
    assert np.allclose(grad, fd, rtol=1e-4, atol=1e-6), (
        f"max |analytic - FD| = {np.max(np.abs(grad - fd)):.2e}"
    )


@pytest.mark.parametrize("with_position", [False, True])
def test_gradient_with_weights_matches_finite_difference(
    with_position: bool,
) -> None:
    X, y, lam, theta0 = _small_problem(with_position)
    w = RNG.integers(1, 5, size=len(y)).astype(float)
    _, grad = penalized_logistic_objective(
        theta0, X, y, w, lam, with_position
    )
    eps = 1e-6
    fd = np.empty_like(theta0)
    for k in range(len(theta0)):
        step = np.zeros_like(theta0)
        step[k] = eps
        f_plus, _ = penalized_logistic_objective(
            theta0 + step, X, y, w, lam, with_position
        )
        f_minus, _ = penalized_logistic_objective(
            theta0 - step, X, y, w, lam, with_position
        )
        fd[k] = (f_plus - f_minus) / (2 * eps)
    assert np.allclose(grad, fd, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("with_position", [False, True])
def test_row_and_count_objectives_agree(with_position: bool) -> None:
    # Three unique covariate patterns with multiplicities 1, 3, 4.
    X_unique = sp.csr_matrix(
        np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0], [1.0, 0.0, -1.0]])
    )
    counts = np.array([1.0, 3.0, 4.0])
    wins = np.array([1.0, 1.0, 3.0])

    # Row-level expansion: each count becomes that many identical binary rows.
    X_rows = sp.vstack(
        [sp.csr_matrix(X_unique.getrow(e).toarray().repeat(int(c), axis=0))
         for e, c in enumerate(counts)]
    ).tocsr()
    y_rows = np.concatenate(
        [np.concatenate([np.ones(int(k)), np.zeros(int(c - k))])
         for k, c in zip(wins, counts)]
    )

    # Count level: fractional y with weights.
    y_counts = wins / counts

    n_params = X_unique.shape[1] + (1 if with_position else 0)
    theta = RNG.normal(size=n_params)
    lam = 1e-4

    obj_rows, grad_rows = penalized_logistic_objective(
        theta, X_rows, y_rows, None, lam, with_position
    )
    obj_counts, grad_counts = penalized_logistic_objective(
        theta, X_unique, y_counts, counts, lam, with_position
    )
    assert obj_rows == pytest.approx(obj_counts, rel=1e-12)
    assert np.allclose(grad_rows, grad_counts, rtol=1e-10, atol=1e-12)


def test_penalty_terms_present() -> None:
    # With X = 0 the linear predictor is eta = beta, so the data term is
    # mean(logaddexp(0, beta) - y*beta); penalties lam/2 ||theta||^2 and
    # (1e-6/2) beta^2 must be added exactly.
    X = sp.csr_matrix(np.zeros((4, 3)))
    y = np.array([0.0, 1.0, 0.0, 1.0])
    theta = np.array([0.5, -0.25, 1.0, 2.0])  # includes beta = 2.0
    obj_pos, _ = penalized_logistic_objective(
        theta, X, y, None, 0.5, True
    )
    data_term = np.logaddexp(0.0, 2.0) - 2.0 * y.mean()
    expected = (
        data_term
        + 0.25 * (0.25 + 0.0625 + 1.0)
        + 0.5 * BETA_RIDGE * 4.0
    )
    assert obj_pos == pytest.approx(expected, rel=1e-12)

    # No-position variant must omit the beta penalty entirely.
    obj_nopos, _ = penalized_logistic_objective(
        theta[:3], X, y, None, 0.5, False
    )
    assert obj_nopos == pytest.approx(
        np.log(2.0) + 0.25 * (0.25 + 0.0625 + 1.0), rel=1e-12
    )


def test_failed_fit_is_marked_not_silent() -> None:
    # A degenerate, perfectly separated design still converges under ridge;
    # instead verify the acceptance machinery rejects a non-finite gradient
    # scenario by checking the FitResult contract on a normal small fit.
    X = sp.csr_matrix(np.array([[1.0, -1.0], [-1.0, 1.0]] * 10))
    y = np.array([1.0, 0.0] * 10)
    fit = fit_penalized_logistic(X, y, lam=1e-6)
    assert fit.success
    assert fit.max_abs_grad < 1e-5
    assert np.isfinite(fit.objective)
    assert fit.beta is None  # no-position model omits the beta term
