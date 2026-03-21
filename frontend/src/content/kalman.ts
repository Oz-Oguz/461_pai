import type { ModuleContent } from './types';

export const kalmanContent: ModuleContent = {
  id: 'kalman',
  title: 'Kalman Filter',
  subtitle: 'State Estimation · Chapter 3',

  overview: {
    paragraphs: [
      'The Kalman Filter is an optimal recursive Bayesian estimator for linear systems with Gaussian noise. At each time step it alternates between two phases: a Predict step that propagates the current Gaussian belief forward through the dynamics model, and an Update step that fuses the new measurement to produce a sharper posterior. The result is the minimum mean-square-error (MMSE) state estimate under the Gaussian assumption.',
      'The key quantity is the Kalman gain K_t ∈ [0,1], which determines how much to trust the sensor versus the prediction. When measurement noise R is small relative to the predicted variance P, K → 1 and the filter snaps to the measurement. When process noise Q is small (the state barely changes), K decreases and the filter relies more on the prediction.',
      'The Kalman Filter is the linear-Gaussian special case of the Bayes filter, and the foundation for the Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF) used in nonlinear robotics systems. It is also the exact solution to the linear-Gaussian state-space model (Kalman Smoother = Kalman Filter + backward pass).',
    ],
    equations: [
      {
        label: 'Predict step',
        latex: '\\hat{x}_{t|t-1} = \\hat{x}_{t-1|t-1}, \\qquad P_{t|t-1} = P_{t-1|t-1} + Q',
        explanation: 'The state estimate is propagated forward (identity dynamics here). The variance P grows by Q — the process uncertainty.',
      },
      {
        label: 'Kalman gain',
        latex: 'K_t = \\frac{P_{t|t-1}}{P_{t|t-1} + R}',
        explanation: 'K ∈ [0,1] balances prediction vs. measurement. K = 1 means trust the sensor fully; K = 0 means ignore it.',
      },
      {
        label: 'Update step',
        latex: '\\hat{x}_{t|t} = \\hat{x}_{t|t-1} + K_t\\bigl(z_t - \\hat{x}_{t|t-1}\\bigr), \\qquad P_{t|t} = (1 - K_t)\\,P_{t|t-1}',
        explanation: 'The innovation (z_t − x̂) is the "surprise". Multiplied by K and added to the prediction. Variance always shrinks after a measurement.',
      },
    ],
    keyInsights: [
      'High Q, low R: the state wanders fast and the sensor is accurate — the filter tracks measurements tightly (K near 1).',
      'Low Q, high R: the state is stable but the sensor is noisy — the filter smooths over many measurements (K near 0).',
      'Watch the Kalman gain K_t converge quickly to a steady-state value — this is the Riccati equation in action.',
      'The uncertainty band (σ) shrinks after each update and grows during each predict step — observe the breathing pattern.',
      'Use the seed slider to try different trajectories. Notice that the filter "lags" sudden jumps — it needs several measurements to catch up.',
    ],
  },

  algorithm: {
    name: 'Kalman Filter (1-D Random Walk)',
    complexity: 'O(N) time · O(1) space',
    steps: [
      { kind: 'header', indent: 0, text: 'Algorithm: Kalman Filter' },
      { kind: 'io', indent: 0, text: 'Input: measurements z_{1:T}, process noise Q, measurement noise R' },
      { kind: 'io', indent: 0, text: 'Output: state estimates x̂_{t|t} and variances P_{t|t} for t = 1…T' },
      { kind: 'divider', indent: 0, text: '' },
      { kind: 'step', indent: 0, text: 'Initialise: x̂₀ = 0, P₀ = 1' },
      { kind: 'step', indent: 0, text: 'For t = 1 to T:' },
      { kind: 'comment', indent: 1, text: '── Predict ──' },
      { kind: 'step', indent: 1, math: true, text: '\\hat{x}_{t|t-1} \\leftarrow \\hat{x}_{t-1|t-1}' },
      { kind: 'step', indent: 1, math: true, text: 'P_{t|t-1} \\leftarrow P_{t-1|t-1} + Q' },
      { kind: 'comment', indent: 1, text: '── Update ──' },
      { kind: 'step', indent: 1, math: true, text: 'K_t \\leftarrow P_{t|t-1}\\, /\\, (P_{t|t-1} + R)' },
      { kind: 'step', indent: 1, math: true, text: '\\hat{x}_{t|t} \\leftarrow \\hat{x}_{t|t-1} + K_t\\,(z_t - \\hat{x}_{t|t-1})' },
      { kind: 'step', indent: 1, math: true, text: 'P_{t|t} \\leftarrow (1 - K_t)\\,P_{t|t-1}' },
      { kind: 'return', indent: 0, text: '{x̂_{t|t}, P_{t|t}} for all t' },
    ],
    note: 'For multi-dimensional systems, replace scalars with vectors/matrices: Σ → KΣK^T, K → ΣH^T(HΣH^T + R)^{-1}. The EKF extends this to nonlinear f(x) and h(x) via Jacobian linearisation.',
  },

  references: [
    {
      label: 'Probabilistic Artificial Intelligence — Krause & Hübotter',
      authors: 'A. Krause, J. Hübotter (ETH Zürich, 2025)',
      url: 'https://arxiv.org/abs/2502.05244',
      description: 'Chapter 3: Filtering. Derives the Kalman Filter as a special case of the Bayes filter under linear-Gaussian assumptions. Covers the multi-dimensional case and Kalman Smoother.',
      type: 'book',
    },
    {
      label: 'Probabilistic Robotics — Thrun, Burgard & Fox',
      authors: 'S. Thrun, W. Burgard, D. Fox (MIT Press, 2005)',
      url: 'https://mitpress.mit.edu/9780262201629/probabilistic-robotics/',
      description: 'Chapter 3: The Kalman Filter. Essential reading for robotics applications — covers EKF, UKF, and particle filters in the context of robot localisation and mapping.',
      type: 'book',
    },
    {
      label: "An Introduction to the Kalman Filter — Welch & Bishop",
      authors: 'G. Welch, G. Bishop (UNC Chapel Hill, 2006)',
      url: 'https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf',
      description: 'The classic 16-page tutorial. Clear, concise, and full of practical intuition. The best starting point for students encountering the Kalman Filter for the first time.',
      type: 'tutorial',
    },
    {
      label: 'Kalman Filter — Wikipedia',
      authors: 'Wikipedia contributors',
      url: 'https://en.wikipedia.org/wiki/Kalman_filter',
      description: 'Comprehensive article covering 1-D and multivariate derivations, history (Apollo program), extensions (EKF, UKF, particle filter), and applications.',
      type: 'wiki',
    },
  ],

  explore: [
    {
      title: 'The predict-update rhythm',
      instruction: 'Reset to t = 0. Press Step Forward one step at a time. After each step, read the Step Detail card: look at x̂(pred), z(meas), and x̂(post) for each step.',
      watch: 'x̂(post) always lies strictly between the prediction and the measurement — never at either extreme. This weighted average IS the Kalman update equation: the filter allocates credit between "what the model expected" and "what the sensor saw" according to their relative reliability. No other algorithm type shows this interpolation so directly.',
    },
    {
      title: 'Kalman gain as a trust dial',
      instruction: 'Set Q = 0.01 and R = 0.01. Step to t = 3 and note the Kalman gain K shown in the step detail. Now reset, set R = 10.00 (noisy sensor), re-run to t = 3, and read K again.',
      watch: 'With low R (trustworthy sensor) K approaches 1 — the estimate jumps almost entirely to the measurement. With high R, K drops toward 0 — the estimate barely moves from the prediction. K is not a free parameter you tune by hand; it is computed optimally from P and R at every step, automatically encoding the trust balance.',
    },
    {
      title: 'Process noise Q: the breathing rate of uncertainty',
      instruction: 'Set R = 2.00. Set Q = 0.01 and run to t = 10 — note the steady-state band width. Reset, set Q = 3.00, run again, compare.',
      watch: 'High Q inflates the predicted variance fast between steps (the band bulges during each predict), then each measurement contracts it. Low Q barely breathes. Q is your prior belief about how rapidly the world changes; getting it right calibrates whether the filter tracks a slow drift or a rapidly jumping process.',
    },
    {
      title: 'Steady-state Kalman gain (Riccati convergence)',
      instruction: 'Set Q = 0.5, R = 2.0. Press Play. Pause at t = 2, 5, 10, 20 and read K from the Step Detail card.',
      watch: 'K starts large (high initial uncertainty), then locks to a fixed value around t = 5. This is the Riccati equation reaching its fixed point — the steady-state optimal gain that can be computed algebraically without running the filter at all. Once a real system reaches steady state, you can precompute K and hardcode it in firmware.',
    },
    {
      title: 'Two extreme limits: sensor vs. model',
      instruction: 'Set Q = 5.00, R = 0.01. Press Play — observe how the estimate track measurements. Then swap to Q = 0.01, R = 9.00. Press Play.',
      watch: 'First run: sensor dominates — the estimate hugs every measurement tightly (K ≈ 1) regardless of model prediction. Second run: model dominates — the estimate barely reacts to measurements and drifts smoothly. Every real Kalman filter deployment sits somewhere between these extremes; calibrating Q and R is the engineer\'s primary task.',
    },
    {
      title: 'Robustness across random runs (seed slider)',
      instruction: 'Set Q = 0.5, R = 2.0. Run to completion and note the final MSE. Drag the Seed slider to 15, run again. Then try seed 80.',
      watch: 'Each seed produces a completely different state trajectory and measurement sequence, yet the Kalman gain convergence curve looks identical and the band breathing pattern repeats. Kalman optimality is distributional — it minimises E[error²] over all possible realisations, so the gain schedule is the same regardless of which specific trajectory occurred.',
    },
    {
      title: 'Calibration check: how often should points escape the band?',
      instruction: 'Set max time steps = 50, Q = 1.00, R = 3.00. Run and count how many measurement dots fall outside the ±1σ band. Then reduce R to 0.1 and repeat.',
      watch: 'For a correctly calibrated filter, roughly 32% of observations should fall outside ±1σ (since 68% should lie inside a one-standard-deviation Gaussian interval). With R = 0.1 the filter over-trusts the sensor and underestimates its own uncertainty — far more than 32% fall outside, signalling a miscalibrated R. This intuitive count is a simplified version of the Normalised Innovation Squared (NIS) consistency check used in real system validation.',
    },
  ],
};
