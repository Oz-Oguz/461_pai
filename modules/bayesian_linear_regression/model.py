"""Bayesian Linear Regression — exact posterior and predictive computations.

Model:
    y = Φ(x) w + ε
    w  ~ N(0, α⁻¹ I)          prior over weights
    ε  ~ N(0, β⁻¹)            observation noise

Posterior (Bayes' theorem, Gaussian conjugacy):
    w | X, y ~ N(μ_w, Σ_w)
    Σ_w = (β Φᵀ Φ + α I)⁻¹
    μ_w = β Σ_w Φᵀ y

Predictive distribution:
    p(y* | x*, X, y) = N(μ_w · φ(x*),  φ(x*)ᵀ Σ_w φ(x*) + β⁻¹)
"""

from __future__ import annotations

import numpy as np


# ── Feature maps ─────────────────────────────────────────────────────

def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """Return Vandermonde design matrix Φ ∈ ℝ^{N×(d+1)}."""
    return np.column_stack([x ** i for i in range(degree + 1)])


def rbf_features(x: np.ndarray, centers: np.ndarray, length_scale: float) -> np.ndarray:
    """Return RBF basis feature matrix Φ ∈ ℝ^{N×K}."""
    return np.exp(-0.5 * ((x[:, None] - centers[None, :]) / length_scale) ** 2)


# ── Core computation ──────────────────────────────────────────────────

def fit(
    x_data: list[float],
    y_data: list[float],
    prior_variance: float = 1.0,     # α⁻¹  (σ_w²)
    noise_variance: float = 0.3,     # β⁻¹  (σ_n²)
    degree: int = 1,
    x_min: float = -3.0,
    x_max: float = 3.0,
    n_grid: int = 200,
    basis: str = "polynomial",       # "polynomial" or "rbf"
) -> dict:
    """Compute BLR posterior and predictive distribution.

    Returns a dict with everything the API / frontend needs.
    """
    alpha = 1.0 / prior_variance   # prior precision
    beta = 1.0 / noise_variance    # noise precision

    x_grid = np.linspace(x_min, x_max, n_grid)
    rbf_centers = np.linspace(x_min, x_max, 7)
    rbf_ls = (x_max - x_min) / 8
    if basis == "rbf":
        Phi_grid = rbf_features(x_grid, rbf_centers, rbf_ls)
        d = 7
    else:
        Phi_grid = polynomial_features(x_grid, degree)
        d = degree + 1

    n_data = len(x_data)
    N_SAMPLES = 10

    if n_data == 0:
        # ── Prior predictive ──────────────────────────────────────────
        # p(y*|x*) = N(0,  φ(x*)ᵀ (α⁻¹ I) φ(x*) + β⁻¹)
        prior_cov = prior_variance * np.eye(d)
        pred_mean = np.zeros(n_grid)
        pred_var = prior_variance * np.sum(Phi_grid ** 2, axis=1) + noise_variance
        pred_std = np.sqrt(pred_var)
        samples = np.random.multivariate_normal(np.zeros(d), prior_cov, size=N_SAMPLES)
        return {
            "n_data": 0,
            "x_grid": x_grid.tolist(),
            "pred_mean": pred_mean.tolist(),
            "pred_std": pred_std.tolist(),
            "pred_std2": (pred_std * 2).tolist(),
            "posterior_mean": None,
            "posterior_var_diag": None,
            "posterior_cov": prior_cov.tolist(),
            "posterior_samples": samples.tolist(),
            "log_marginal_likelihood": None,
            "is_prior": True,
            "x_data": [],
            "y_data": [],
        }

    x = np.array(x_data)
    y = np.array(y_data)
    Phi = rbf_features(x, rbf_centers, rbf_ls) if basis == "rbf" else polynomial_features(x, degree)

    # ── Posterior ─────────────────────────────────────────────────────
    A = beta * Phi.T @ Phi + alpha * np.eye(d)     # precision matrix d×d
    Sigma_w = np.linalg.inv(A)                     # posterior covariance
    mu_w = beta * Sigma_w @ Phi.T @ y              # posterior mean (d+1,)

    # ── Predictive distribution on grid ──────────────────────────────
    pred_mean = Phi_grid @ mu_w                    # (n_grid,)
    pred_var = np.array(
        [phi @ Sigma_w @ phi + noise_variance for phi in Phi_grid]
    )                                              # (n_grid,)
    pred_std = np.sqrt(np.maximum(pred_var, 0))

    # ── Posterior samples (function samples) ─────────────────────────
    samples = np.random.multivariate_normal(mu_w, Sigma_w, size=N_SAMPLES)

    # ── Log marginal likelihood (Bishop Eq. 3.86) ─────────────────────
    # ln p(y|α,β) = d/2·ln α + N/2·ln β − E(μ_w) − ½·ln|A| − N/2·ln 2π
    E_mn = 0.5 * beta * float(np.sum((y - Phi @ mu_w) ** 2)) + \
           0.5 * alpha * float(mu_w @ mu_w)
    _, logdet_A = np.linalg.slogdet(A)
    N = n_data
    log_ml = (d * np.log(alpha) + N * np.log(beta)) / 2.0 \
             - E_mn - 0.5 * float(logdet_A) - N / 2.0 * np.log(2 * np.pi)

    return {
        "n_data": n_data,
        "x_grid": x_grid.tolist(),
        "pred_mean": pred_mean.tolist(),
        "pred_std": pred_std.tolist(),
        "pred_std2": (pred_std * 2).tolist(),
        "posterior_mean": mu_w.tolist(),
        "posterior_var_diag": np.diag(Sigma_w).tolist(),
        "posterior_cov": Sigma_w.tolist(),
        "posterior_samples": samples.tolist(),
        "log_marginal_likelihood": float(log_ml),
        "is_prior": False,
        "x_data": x_data,
        "y_data": y_data,
    }


# ── Step-by-step solver ───────────────────────────────────────────────

def solver_steps(
    x_data: list[float],
    y_data: list[float],
    prior_variance: float,
    noise_variance: float,
    degree: int,
) -> list[dict]:
    """Generate step-by-step LaTeX derivation of the BLR posterior update."""
    alpha = 1.0 / prior_variance
    beta  = 1.0 / noise_variance

    steps = [
        {
            "title": "1. Model Definition",
            "text": "We place a Gaussian prior on the weights and model observation noise:",
            "latex": (
                r"\mathbf{y} = \boldsymbol{\Phi}\mathbf{w} + \boldsymbol{\varepsilon}"
                r"\qquad"
                r"\mathbf{w} \sim \mathcal{N}(\mathbf{0},\, \alpha^{-1}\mathbf{I})"
                r"\qquad"
                r"\varepsilon \sim \mathcal{N}(0,\, \beta^{-1})"
            ),
        },
        {
            "title": "2. Likelihood",
            "text": "The likelihood of the data given weights is:",
            "latex": (
                r"p(\mathbf{y} \mid \mathbf{w}) = \mathcal{N}\!\left("
                r"\mathbf{y};\;\boldsymbol{\Phi}\mathbf{w},\;\beta^{-1}\mathbf{I}\right)"
            ),
        },
        {
            "title": "3. Posterior (Gaussian Conjugacy)",
            "text": (
                "Because prior and likelihood are both Gaussian, the posterior is also "
                "Gaussian. Completing the square gives the posterior precision and mean:"
            ),
            "latex": (
                r"\boldsymbol{\Sigma}_w = \left(\beta\,\boldsymbol{\Phi}^\top\boldsymbol{\Phi}"
                r"+ \alpha\,\mathbf{I}\right)^{-1}"
                r"\qquad"
                r"\boldsymbol{\mu}_w = \beta\,\boldsymbol{\Sigma}_w\,\boldsymbol{\Phi}^\top\mathbf{y}"
            ),
        },
    ]

    if x_data:
        x = np.array(x_data)
        y = np.array(y_data)
        Phi = polynomial_features(x, degree)
        d = degree + 1
        Sigma_w = np.linalg.inv(beta * Phi.T @ Phi + alpha * np.eye(d))
        mu_w = beta * Sigma_w @ Phi.T @ y

        # Format matrices compactly
        def fmt_vec(v: np.ndarray) -> str:
            entries = r"\\".join(f"{vi:.3f}" for vi in v)
            return rf"\begin{{pmatrix}} {entries} \end{{pmatrix}}"

        steps.append({
            "title": f"4. Numerical Result  (N={len(x_data)}, degree={degree})",
            "text": (
                f"With α = {alpha:.2f} (prior precision), β = {beta:.2f} (noise precision) "
                f"and N = {len(x_data)} observations:"
            ),
            "latex": (
                rf"\boldsymbol{{\mu}}_w = {fmt_vec(mu_w)}"
                rf"\qquad"
                rf"\text{{diag}}(\boldsymbol{{\Sigma}}_w) = {fmt_vec(np.diag(Sigma_w))}"
            ),
        })

    steps.append({
        "title": f"{'5' if x_data else '4'}. Predictive Distribution",
        "text": "For a new input x*, the predictive distribution integrates out the weights:",
        "latex": (
            r"p(y^* \mid x^*, \mathbf{X}, \mathbf{y}) = \mathcal{N}\!\left("
            r"\boldsymbol{\mu}_w^\top \boldsymbol{\phi}(x^*),\;"
            r"\boldsymbol{\phi}(x^*)^\top \boldsymbol{\Sigma}_w \boldsymbol{\phi}(x^*)"
            r"+ \beta^{-1}\right)"
        ),
    })

    return steps
