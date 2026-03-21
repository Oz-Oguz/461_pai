"""Gaussian Process Regression — exact posterior and predictive computations.

GP prior:
    f ~ GP(0, k(x, x'))

Posterior:
    f | X, y ~ GP(μ_*, Σ_*)
    μ_*(x*)  = K(x*, X) [K(X,X) + σ²_n I]⁻¹ y
    Σ_*(x*,x**) = K(x*,x**) − K(x*,X) [K(X,X) + σ²_n I]⁻¹ K(X,x**)

Log marginal likelihood (Rasmussen & Williams 2.30):
    log p(y|X,θ) = −½ yᵀ K_y⁻¹ y − ½ log|K_y| − N/2 log 2π
    where K_y = K(X,X) + σ²_n I
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.optimize import minimize


# ── Kernel functions ───────────────────────────────────────────────────

def rbf_kernel(X1: np.ndarray, X2: np.ndarray,
               length_scale: float, signal_variance: float) -> np.ndarray:
    """Squared Exponential: k(x,x') = σ²_f · exp(−||x−x'||² / 2ℓ²)."""
    X1 = np.asarray(X1).reshape(-1, 1)
    X2 = np.asarray(X2).reshape(-1, 1)
    sq_dists = (X1 - X2.T) ** 2
    return signal_variance * np.exp(-0.5 * sq_dists / length_scale ** 2)


def matern32_kernel(X1: np.ndarray, X2: np.ndarray,
                    length_scale: float, signal_variance: float) -> np.ndarray:
    """Matérn 3/2: k(x,x') = σ²_f · (1 + √3|r|/ℓ) · exp(−√3|r|/ℓ)."""
    X1 = np.asarray(X1).reshape(-1, 1)
    X2 = np.asarray(X2).reshape(-1, 1)
    r = np.abs(X1 - X2.T)
    sqrt3r_l = np.sqrt(3) * r / length_scale
    return signal_variance * (1.0 + sqrt3r_l) * np.exp(-sqrt3r_l)


def linear_kernel(X1: np.ndarray, X2: np.ndarray,
                  signal_variance: float, offset: float = 0.0) -> np.ndarray:
    """Linear: k(x,x') = σ²_f · (x − c)(x' − c)."""
    X1 = np.asarray(X1).reshape(-1, 1) - offset
    X2 = np.asarray(X2).reshape(-1, 1) - offset
    return signal_variance * (X1 @ X2.T)


def periodic_kernel(X1: np.ndarray, X2: np.ndarray,
                    length_scale: float, signal_variance: float,
                    period: float) -> np.ndarray:
    """Periodic: k(x,x') = σ²_f · exp(−2 sin²(π|x−x'|/p) / ℓ²)."""
    X1 = np.asarray(X1).reshape(-1, 1)
    X2 = np.asarray(X2).reshape(-1, 1)
    r = np.abs(X1 - X2.T)
    return signal_variance * np.exp(
        -2.0 * np.sin(np.pi * r / period) ** 2 / length_scale ** 2
    )


def _get_kernel_matrix(
    X1: np.ndarray, X2: np.ndarray,
    kernel: str, length_scale: float, signal_variance: float, period: float,
) -> np.ndarray:
    if kernel == "rbf":
        return rbf_kernel(X1, X2, length_scale, signal_variance)
    elif kernel == "matern32":
        return matern32_kernel(X1, X2, length_scale, signal_variance)
    elif kernel == "linear":
        return linear_kernel(X1, X2, signal_variance)
    elif kernel == "periodic":
        return periodic_kernel(X1, X2, length_scale, signal_variance, period)
    else:
        raise ValueError(f"Unknown kernel: {kernel!r}")


# ── Core computation ───────────────────────────────────────────────────

JITTER = 1e-6
N_SAMPLES = 10


def fit(
    x_data: list[float],
    y_data: list[float],
    kernel: str = "rbf",
    length_scale: float = 1.0,
    signal_variance: float = 1.0,
    noise_variance: float = 0.3,
    period: float = 3.14159,
    x_min: float = -5.0,
    x_max: float = 5.0,
    n_grid: int = 200,
    kernel_ref: float = 0.0,
) -> dict:
    """Compute GP posterior and predictive distribution.

    Returns a dict with everything the API / frontend needs.
    """
    x_grid = np.linspace(x_min, x_max, n_grid)

    # Kernel slice for visualization: k(kernel_ref, x_grid)
    kernel_slice = _get_kernel_matrix(
        [kernel_ref], x_grid, kernel, length_scale, signal_variance, period
    )[0].tolist()

    n_data = len(x_data)

    if n_data == 0:
        # ── Prior predictive ──────────────────────────────────────────
        K_prior = _get_kernel_matrix(
            x_grid, x_grid, kernel, length_scale, signal_variance, period
        ) + JITTER * np.eye(n_grid)
        pred_mean = np.zeros(n_grid)
        pred_std = np.sqrt(np.maximum(np.diag(K_prior), 0))

        # Function samples from prior via Cholesky
        try:
            L_prior = cholesky(K_prior, lower=True)
            samples = [L_prior @ np.random.randn(n_grid) for _ in range(N_SAMPLES)]
        except Exception:
            samples = [np.zeros(n_grid) for _ in range(N_SAMPLES)]

        return {
            "n_data": 0,
            "x_grid": x_grid.tolist(),
            "pred_mean": pred_mean.tolist(),
            "pred_std": pred_std.tolist(),
            "pred_std2": (pred_std * 2).tolist(),
            "posterior_samples": [s.tolist() for s in samples],
            "log_marginal_likelihood": None,
            "kernel_slice": kernel_slice,
            "is_prior": True,
            "x_data": [],
            "y_data": [],
        }

    x = np.asarray(x_data)
    y = np.asarray(y_data)

    # ── Kernel matrices ────────────────────────────────────────────────
    K_XX = _get_kernel_matrix(
        x, x, kernel, length_scale, signal_variance, period
    ) + (noise_variance + JITTER) * np.eye(n_data)  # N×N

    K_Xs = _get_kernel_matrix(
        x, x_grid, kernel, length_scale, signal_variance, period
    )  # N × n_grid

    k_ss_diag = _get_kernel_matrix(
        x_grid, x_grid, kernel, length_scale, signal_variance, period
    ).diagonal()  # n_grid  (only diagonal needed for predictive var)

    # ── Posterior via Cholesky ────────────────────────────────────────
    L_fac = cho_factor(K_XX)           # Cholesky of K_y
    alpha = cho_solve(L_fac, y)        # K_y⁻¹ y
    v = cho_solve(L_fac, K_Xs)         # K_y⁻¹ K(X, x*)   N × n_grid

    pred_mean = K_Xs.T @ alpha         # n_grid
    pred_var = k_ss_diag - np.sum(K_Xs * v, axis=0)
    pred_var = np.maximum(pred_var, 0)
    pred_std = np.sqrt(pred_var)

    # ── Posterior function samples (via posterior covariance Cholesky) ─
    K_grid = _get_kernel_matrix(
        x_grid, x_grid, kernel, length_scale, signal_variance, period
    )
    K_post = K_grid - K_Xs.T @ v + JITTER * np.eye(n_grid)
    # Ensure positive definiteness
    K_post = 0.5 * (K_post + K_post.T)
    try:
        L_post = cholesky(K_post, lower=True)
        samples = [pred_mean + L_post @ np.random.randn(n_grid) for _ in range(N_SAMPLES)]
    except Exception:
        samples = [pred_mean + pred_std * np.random.randn(n_grid) for _ in range(N_SAMPLES)]

    # ── Log marginal likelihood ───────────────────────────────────────
    # log p(y|X,θ) = −½yᵀα − Σ log L_ii − N/2 log 2π
    # Extract diagonal of Cholesky factor for log-det term
    L_mat = L_fac[0]  # upper or lower triangle depending on cho_factor
    log_diag = np.sum(np.log(np.abs(np.diag(L_mat))))
    log_ml = float(
        -0.5 * float(y @ alpha) - log_diag - 0.5 * n_data * np.log(2 * np.pi)
    )

    return {
        "n_data": n_data,
        "x_grid": x_grid.tolist(),
        "pred_mean": pred_mean.tolist(),
        "pred_std": pred_std.tolist(),
        "pred_std2": (pred_std * 2).tolist(),
        "posterior_samples": [s.tolist() for s in samples],
        "log_marginal_likelihood": log_ml,
        "kernel_slice": kernel_slice,
        "is_prior": False,
        "x_data": x_data,
        "y_data": y_data,
    }


# ── Hyperparameter optimisation ────────────────────────────────────────

def optimize_hyperparameters(
    x_data: list[float],
    y_data: list[float],
    kernel: str = "rbf",
    length_scale: float = 1.0,
    signal_variance: float = 1.0,
    noise_variance: float = 0.3,
    period: float = 3.14159,
) -> dict:
    """Maximise log marginal likelihood over (ℓ, σ²_f, σ²_n) via L-BFGS-B.

    Optimises in log-space for positivity and numerical stability.
    For the Linear kernel length_scale is unused but still optimised harmlessly.
    """
    x = np.asarray(x_data)
    y = np.asarray(y_data)

    def neg_log_ml(log_params: np.ndarray) -> float:
        ls, sv, nv = np.exp(log_params)
        try:
            result = fit(
                x_data, y_data,
                kernel=kernel,
                length_scale=float(ls),
                signal_variance=float(sv),
                noise_variance=float(nv),
                period=period,
            )
            lml = result["log_marginal_likelihood"]
            return -lml if lml is not None and np.isfinite(lml) else 1e8
        except Exception:
            return 1e8

    x0 = np.log([length_scale, signal_variance, noise_variance])
    # Log-space bounds: ℓ ∈ (e⁻³,e³), σ²_f ∈ (e⁻³,e³), σ²_n ∈ (e⁻⁶,e¹)
    bounds = [(-3.0, 3.0), (-3.0, 3.0), (-6.0, 1.0)]

    res = minimize(neg_log_ml, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 200, "ftol": 1e-9})

    opt_ls, opt_sv, opt_nv = np.exp(res.x)
    return {
        "length_scale": float(opt_ls),
        "signal_variance": float(opt_sv),
        "noise_variance": float(opt_nv),
        "success": bool(res.success),
    }


# ── Solver steps ───────────────────────────────────────────────────────

_KERNEL_LATEX: dict[str, str] = {
    "rbf": (
        r"k(x, x') = \sigma^2_f \exp\!\left(-\frac{(x-x')^2}{2\ell^2}\right)"
    ),
    "matern32": (
        r"k(x, x') = \sigma^2_f\!\left(1 + \frac{\sqrt{3}\,|x-x'|}{\ell}\right)"
        r"\exp\!\left(-\frac{\sqrt{3}\,|x-x'|}{\ell}\right)"
    ),
    "linear": (
        r"k(x, x') = \sigma^2_f \cdot x \cdot x'"
    ),
    "periodic": (
        r"k(x, x') = \sigma^2_f \exp\!\left(-\frac{2\sin^2\!\left(\pi|x-x'|/p\right)}{\ell^2}\right)"
    ),
}

_KERNEL_NAMES: dict[str, str] = {
    "rbf": "Squared Exponential (RBF)",
    "matern32": "Matérn 3/2",
    "linear": "Linear",
    "periodic": "Periodic",
}


def solver_steps(
    x_data: list[float],
    y_data: list[float],
    kernel: str = "rbf",
    length_scale: float = 1.0,
    signal_variance: float = 1.0,
    noise_variance: float = 0.3,
    period: float = 3.14159,
) -> list[dict]:
    """Return step-by-step LaTeX derivation of the GP posterior."""
    kname = _KERNEL_NAMES.get(kernel, kernel)
    N = len(x_data)

    steps = [
        {
            "title": "1. GP Prior",
            "text": (
                f"We place a zero-mean GP prior over functions, with {kname} kernel. "
                "The kernel encodes our prior beliefs about function smoothness."
            ),
            "latex": (
                r"f \sim \mathcal{GP}(0,\; k(x,x'))"
                r"\qquad "
                + _KERNEL_LATEX[kernel]
            ),
        },
        {
            "title": "2. Noisy Observations",
            "text": (
                "Observations are noisy evaluations of the latent function. "
                "The observation model adds independent Gaussian noise:"
            ),
            "latex": (
                r"y_i = f(x_i) + \varepsilon_i"
                r"\qquad \varepsilon_i \sim \mathcal{N}(0,\, \sigma^2_n)"
            ),
        },
        {
            "title": "3. Kernel Matrix",
            "text": (
                "Build the N×N kernel matrix of the training inputs, "
                "then add noise to the diagonal:"
            ),
            "latex": (
                r"\mathbf{K}_y = \mathbf{K}(\mathbf{X},\mathbf{X}) + \sigma^2_n \mathbf{I}"
                r"\quad\text{where}\quad"
                r"[\mathbf{K}]_{ij} = k(x_i, x_j)"
            ),
        },
        {
            "title": "4. Posterior Predictive",
            "text": (
                "Conditioning on the data gives a Gaussian posterior predictive "
                "at any new point x*:"
            ),
            "latex": (
                r"p(f^* \mid x^*, \mathbf{X}, \mathbf{y}) = \mathcal{N}\!\left(\mu^*,\; (\sigma^*)^2\right)"
                r"\\"
                r"\mu^* = \mathbf{k}_*^\top \mathbf{K}_y^{-1} \mathbf{y}"
                r"\qquad"
                r"(\sigma^*)^2 = k(x^*,x^*) - \mathbf{k}_*^\top \mathbf{K}_y^{-1} \mathbf{k}_*"
            ),
        },
    ]

    if x_data:
        x = np.asarray(x_data)
        y = np.asarray(y_data)
        K_XX = _get_kernel_matrix(
            x, x, kernel, length_scale, signal_variance, period
        ) + (noise_variance + JITTER) * np.eye(N)
        L_fac = cho_factor(K_XX)
        alpha = cho_solve(L_fac, y)
        L_mat = L_fac[0]
        log_diag = np.sum(np.log(np.abs(np.diag(L_mat))))
        log_ml = float(
            -0.5 * float(y @ alpha) - log_diag - 0.5 * N * np.log(2 * np.pi)
        )

        alpha_str = ", ".join(f"{a:.3f}" for a in alpha[:5])
        if N > 5:
            alpha_str += ", \\ldots"

        steps.append({
            "title": f"5. Numerical Result  (N={N}, kernel={kname})",
            "text": (
                f"With ℓ = {length_scale:.2f}, σ²_f = {signal_variance:.2f}, "
                f"σ²_n = {noise_variance:.2f} and N = {N} observations, "
                "the Cholesky solve gives weight vector α = K_y⁻¹y:"
            ),
            "latex": (
                rf"\boldsymbol{{\alpha}} = \mathbf{{K}}_y^{{-1}}\mathbf{{y}} "
                rf"= \begin{{pmatrix}} {alpha_str} \end{{pmatrix}}"
            ),
        })

    steps.append({
        "title": f"{'6' if x_data else '5'}. Log Marginal Likelihood",
        "text": (
            "The log marginal likelihood scores how well the kernel hyperparameters "
            "explain the data. Maximising it performs Bayesian model selection:"
        ),
        "latex": (
            r"\log p(\mathbf{y} \mid \mathbf{X},\theta) = "
            r"-\tfrac{1}{2}\mathbf{y}^\top\mathbf{K}_y^{-1}\mathbf{y}"
            r" - \tfrac{1}{2}\log|\mathbf{K}_y|"
            r" - \tfrac{N}{2}\log 2\pi"
            + (rf"\;=\; {log_ml:.3f}" if x_data else "")
        ),
    })

    return steps
