"""1-D Kalman Filter — predict / update cycle.

State-space model:
    x_t = x_{t-1} + q_t,   q_t ~ N(0, Q)   (random-walk dynamics)
    z_t = x_t    + r_t,   r_t ~ N(0, R)   (noisy measurement)

Predict step:
    x̂_{t|t-1} = x̂_{t-1|t-1}
    P_{t|t-1}  = P_{t-1|t-1} + Q

Update step:
    K_t = P_{t|t-1} / (P_{t|t-1} + R)          Kalman gain
    x̂_{t|t} = x̂_{t|t-1} + K_t (z_t - x̂_{t|t-1})
    P_{t|t}  = (1 - K_t) P_{t|t-1}
"""

from __future__ import annotations

import numpy as np


def simulate(
    n_steps: int = 30,
    process_noise_q: float = 0.5,
    measurement_noise_r: float = 2.0,
    x0_true: float = 0.0,
    seed: int = 42,
) -> dict:
    """Simulate a 1-D random-walk, generate measurements, run Kalman filter."""
    rng = np.random.default_rng(seed)

    # ── Generate ground truth ─────────────────────────────────────────
    true_states = np.zeros(n_steps)
    true_states[0] = x0_true
    for t in range(1, n_steps):
        true_states[t] = true_states[t - 1] + rng.normal(0, np.sqrt(process_noise_q))

    # ── Noisy measurements ────────────────────────────────────────────
    measurements = true_states + rng.normal(0, np.sqrt(measurement_noise_r), n_steps)

    # ── Kalman filter ─────────────────────────────────────────────────
    x_est = np.zeros(n_steps)
    p_est = np.zeros(n_steps)   # posterior variance
    x_pred_arr = np.zeros(n_steps)
    p_pred_arr = np.zeros(n_steps)
    gains = np.zeros(n_steps)

    # Initial conditions (uninformative)
    x_hat = 0.0
    P = 1.0

    for t in range(n_steps):
        # Predict
        x_hat_pred = x_hat
        P_pred = P + process_noise_q
        x_pred_arr[t] = x_hat_pred
        p_pred_arr[t] = P_pred

        # Update
        K = P_pred / (P_pred + measurement_noise_r)
        x_hat = x_hat_pred + K * (measurements[t] - x_hat_pred)
        P = (1 - K) * P_pred

        x_est[t] = x_hat
        p_est[t] = P
        gains[t] = K

    steps_detail = _build_step_details(
        true_states, measurements, x_pred_arr, p_pred_arr,
        x_est, p_est, gains, process_noise_q, measurement_noise_r,
    )

    return {
        "timesteps": list(range(n_steps)),
        "true_states": true_states.tolist(),
        "measurements": measurements.tolist(),
        "estimated_states": x_est.tolist(),
        "estimated_stds": np.sqrt(p_est).tolist(),
        "predicted_states": x_pred_arr.tolist(),
        "predicted_stds": np.sqrt(p_pred_arr).tolist(),
        "kalman_gains": gains.tolist(),
        "steps_detail": steps_detail,
    }


def _build_step_details(
    true_states, measurements, x_pred, p_pred, x_est, p_est, gains, Q, R
) -> list[dict]:
    """Build per-timestep detail for the step-through UI."""
    details = []
    for t in range(len(true_states)):
        details.append({
            "t": t,
            "true": round(float(true_states[t]), 3),
            "measurement": round(float(measurements[t]), 3),
            "x_predicted": round(float(x_pred[t]), 3),
            "p_predicted": round(float(p_pred[t]), 3),
            "kalman_gain": round(float(gains[t]), 3),
            "x_updated": round(float(x_est[t]), 3),
            "p_updated": round(float(p_est[t]), 3),
        })
    return details


def solver_steps(process_noise_q: float, measurement_noise_r: float) -> list[dict]:
    """Return the canonical Kalman filter equations as LaTeX derivation steps."""
    return [
        {
            "title": "1. State-Space Model",
            "text": "We model the hidden state as a random walk with noisy observations:",
            "latex": (
                r"x_t = x_{t-1} + q_t, \quad q_t \sim \mathcal{N}(0, Q)"
                r"\qquad "
                r"z_t = x_t + r_t, \quad r_t \sim \mathcal{N}(0, R)"
            ),
        },
        {
            "title": "2. Predict Step",
            "text": "Propagate the current belief forward one time step (no control input):",
            "latex": (
                r"\hat{x}_{t \mid t-1} = \hat{x}_{t-1 \mid t-1}"
                r"\qquad "
                r"P_{t \mid t-1} = P_{t-1 \mid t-1} + Q"
            ),
        },
        {
            "title": "3. Update Step",
            "text": "Fuse the new measurement z_t with the prediction using the Kalman gain:",
            "latex": (
                r"\begin{aligned}"
                r"K_t &= \frac{P_{t \mid t-1}}{P_{t \mid t-1} + R} && \text{(Kalman gain)} \\"
                r"\hat{x}_{t \mid t} &= \hat{x}_{t \mid t-1} + K_t \bigl(z_t - \hat{x}_{t \mid t-1}\bigr) && \text{(state update)} \\"
                r"P_{t \mid t} &= (1 - K_t)\, P_{t \mid t-1} && \text{(variance shrinks)}"
                r"\end{aligned}"
            ),
        },
        {
            "title": "4. Kalman Gain Intuition",
            "text": (
                f"With Q = {process_noise_q} and R = {measurement_noise_r}: "
                "when R → 0 (perfect sensor), K → 1 and we trust the measurement fully. "
                "When R → ∞ (terrible sensor), K → 0 and we ignore it."
            ),
            "latex": (
                r"K \to 1 \Leftrightarrow R \ll P \quad \text{(trust sensor)}"
                r"\qquad"
                r"K \to 0 \Leftrightarrow R \gg P \quad \text{(trust prediction)}"
            ),
        },
    ]
