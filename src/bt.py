"""Common penalized logistic estimator for the Bradley-Terry model family.

Implements the plan's common estimation specification (Section 3, Estimation).
All nonconstant models minimize, in displayed coordinates,

    (1/N) sum_t [ log(1 + exp(eta_t)) - y_t eta_t ]
        + (lam / 2) ||theta||_2^2
        + (1e-6 / 2) beta^2,

where eta_t = x_t . theta (+ beta when a position term is included), y_t =
1{choice_t = A}, and theta holds item scores or visual/context coefficients.
The beta penalty term is omitted for no-position models.

Conventions (fixed by the plan):
* numerically stable ``logaddexp(0, eta)`` loss;
* analytic gradient, L-BFGS-B, start at zero;
* initial maximum 5,000 iterations;
* acceptance requires a finite objective, finite parameters/predictions, and
  maximum absolute gradient below 1e-5;
* one retry with 10,000 iterations and tighter function/gradient tolerances;
* if acceptance still fails the fit is marked failed (``success=False``) and
  must remain visibly failed - never silently reported.

Count-level fitting is supported through ``weights``: a row with weight n_e
and fractional y_e = k_e / n_e reproduces the row-level objective of n_e
identical binary rows, with N = sum_t w_t in both cases (plan Stage 5
recovery checks rely on this equivalence).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit

BETA_RIDGE = 1e-6
MAX_ABS_GRAD_TOL = 1e-5
INITIAL_MAXITER = 5_000
RETRY_MAXITER = 10_000


@dataclass
class FitResult:
    """Outcome of one penalized logistic fit."""

    theta: np.ndarray
    beta: float | None
    success: bool
    message: str
    n_iterations: int
    objective: float
    max_abs_grad: float
    lam: float
    with_position: bool
    retried: bool = False

    def predict_eta(self, X: sp.spmatrix | np.ndarray) -> np.ndarray:
        eta = np.asarray(X @ self.theta, dtype=float).ravel()
        if self.with_position:
            eta = eta + float(self.beta)
        return eta

    def predict_proba(self, X: sp.spmatrix | np.ndarray) -> np.ndarray:
        return expit(self.predict_eta(X))


def penalized_logistic_objective(
    theta_full: np.ndarray,
    X: sp.spmatrix | np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None,
    lam: float,
    with_position: bool,
) -> tuple[float, np.ndarray]:
    """Objective value and analytic gradient at ``theta_full``.

    ``theta_full`` is ``theta`` concatenated with ``beta`` when
    ``with_position`` is true. ``weights`` defaults to 1 per row; the data
    term is normalized by N = sum of weights.
    """
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    n_total = w.sum()

    if with_position:
        theta, beta = theta_full[:-1], float(theta_full[-1])
    else:
        theta, beta = theta_full, 0.0

    eta = np.asarray(X @ theta, dtype=float).ravel() + beta
    loss = np.logaddexp(0.0, eta) - y * eta  # stable log(1+exp(eta)) - y*eta
    objective = float((w * loss).sum() / n_total)
    objective += 0.5 * lam * float(theta @ theta)
    if with_position:
        objective += 0.5 * BETA_RIDGE * beta * beta

    residual = w * (expit(eta) - y) / n_total
    grad_theta = np.asarray(X.T @ residual, dtype=float).ravel() + lam * theta
    if with_position:
        grad_beta = float(residual.sum()) + BETA_RIDGE * beta
        gradient = np.concatenate([grad_theta, [grad_beta]])
    else:
        gradient = grad_theta
    return objective, gradient


def fit_penalized_logistic(
    X: sp.spmatrix | np.ndarray,
    y: np.ndarray,
    lam: float,
    with_position: bool = False,
    weights: np.ndarray | None = None,
) -> FitResult:
    """Fit the common objective with L-BFGS-B from a zero start.

    Applies the plan's acceptance rule and single retry; a fit that still
    fails is returned with ``success=False`` and must be reported as failed.
    """
    y = np.asarray(y, dtype=float)
    n_params = X.shape[1] + (1 if with_position else 0)
    x0 = np.zeros(n_params)

    def objective(theta_full: np.ndarray) -> tuple[float, np.ndarray]:
        return penalized_logistic_objective(
            theta_full, X, y, weights, lam, with_position
        )

    retried = False
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": INITIAL_MAXITER},
    )
    accepted, final_obj, max_grad = _acceptance(result, objective)
    if not accepted:
        retried = True
        result = minimize(
            objective,
            result.x,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": RETRY_MAXITER,
                "ftol": 1e-15,
                "gtol": 1e-9,
            },
        )
        accepted, final_obj, max_grad = _acceptance(result, objective)

    if with_position:
        theta, beta = result.x[:-1], float(result.x[-1])
    else:
        theta, beta = result.x, None

    message = str(result.message)
    if not accepted:
        message = f"FIT FAILED acceptance checks: {message}"

    return FitResult(
        theta=np.asarray(theta, dtype=float),
        beta=beta,
        success=bool(accepted),
        message=message,
        n_iterations=int(result.nit),
        objective=float(final_obj),
        max_abs_grad=float(max_grad),
        lam=float(lam),
        with_position=with_position,
        retried=retried,
    )


def _acceptance(result, objective) -> tuple[bool, float, float]:
    """Finite objective/parameters/predictions and max|grad| < 1e-5."""
    final_obj, grad = objective(result.x)
    finite = np.isfinite(final_obj) and np.all(np.isfinite(result.x))
    max_grad = float(np.max(np.abs(grad))) if grad.size else 0.0
    return bool(finite and max_grad < MAX_ABS_GRAD_TOL), final_obj, max_grad


def m0_probability(n_a: int, n: int) -> float:
    """M0 baseline: smoothed constant A probability, (n_A + 0.5) / (N + 1)."""
    return (n_a + 0.5) / (n + 1.0)


def expected_log_loss(p_true: np.ndarray, p_hat: np.ndarray) -> np.ndarray:
    """Exact expected log loss of predictions under known probabilities."""
    p_true = np.asarray(p_true, dtype=float)
    p_hat = np.clip(np.asarray(p_hat, dtype=float), 1e-15, 1 - 1e-15)
    return -(p_true * np.log(p_hat) + (1 - p_true) * np.log(1 - p_hat))
