---
marp: true
theme: default
paginate: true
math: katex
style: |
  section {
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    font-size: 22px;
    color: #1e293b;
    padding: 30px 48px 36px 48px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
  }
  h1 { 
    font-size: 1.35em; 
    color: #1e3a5f; 
    border-bottom: 3px solid #3b82f6; 
    padding-bottom: 8px; 
    margin-bottom: 16px; 
    margin-top: 0;
    padding-top: 0;
  }
  h2 { font-size: 1.05em; color: #2563eb; margin-top: 0; margin-bottom: 8px; }
  ul { margin-top: 6px; }
  li { margin-bottom: 4px; }

  section.title {
    background: linear-gradient(135deg, #0f2a52 0%, #0369a1 100%);
    color: white;
    display: flex; flex-direction: column; justify-content: center;
  }
  section.title h1 { color: white; border-color: rgba(255,255,255,0.35); font-size: 1.8em; }
  section.title h2 { color: rgba(255,255,255,0.8); font-size: 1em; }
  section.title p  { color: rgba(255,255,255,0.65); font-size: 0.75em; }

  section.divider {
    background: #0c4a6e;
    color: white;
    display: flex; flex-direction: column; justify-content: center;
  }
  section.divider h1 { color: white; border-color: rgba(255,255,255,0.3); }
  section.divider p  { color: rgba(255,255,255,0.7); font-size: 0.85em; }

  .demo { background: #eff6ff; border-left: 4px solid #3b82f6; padding: 8px 12px; border-radius: 0 6px 6px 0; font-size: 0.78em; margin-top: 10px; }
  .demo strong { color: #1d4ed8; }
  .insight { background: #f0fdf4; border-left: 4px solid #22c55e; padding: 8px 12px; border-radius: 0 6px 6px 0; font-size: 0.78em; margin-top: 10px; }
  .insight strong { color: #15803d; }
  .warning { background: #fff7ed; border-left: 4px solid #f97316; padding: 8px 12px; border-radius: 0 6px 6px 0; font-size: 0.78em; margin-top: 10px; }
  .eq-box { background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px 18px; text-align: center; margin: 10px 0; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
  .col { }
  .small { font-size: 0.74em; color: #64748b; }
  table { font-size: 0.76em; }
  section::after { font-size: 0.62em; color: #94a3b8; }
  .algorithm {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 8px 12px;
    font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
    font-size: 0.64em;
    line-height: 1.4;
    margin: 8px 0;
  }
  .algorithm .kw { color: #2563eb; font-weight: 700; }
  .algorithm .cm { color: #94a3b8; font-style: italic; }
  .algorithm p { margin: 0; }
  .algorithm p:empty { display: none; }
---

<!-- _class: title -->

# Kalman Filter

## Probabilistic AI — Chapter 3

Özgür S. Öğüz · Bilkent University

---

<!-- _paginate: false -->

# Agenda

1. **State estimation problem** — what are we trying to do?
2. **Linear-Gaussian state-space model** — the setup
3. **Predict step** — propagating uncertainty forward
4. **Kalman gain** — the optimal trust allocation
5. **Update step** — fusing the measurement
6. **Steady-state behaviour** — Riccati convergence
7. **Calibrating Q and R** — engineering the filter
8. **Extensions** — EKF, UKF, and beyond

---

<!-- _class: divider -->

# Part 1
## The State Estimation Problem

---

# Hidden Markov Intuition

We observe noisy **measurements** $z_t$ of an underlying **hidden state** $x_t$:

$$x_t = f(x_{t-1}) + \text{process noise}$$
$$z_t = h(x_t) + \text{measurement noise}$$

We want $p(x_t \mid z_{1:t})$ — the **belief** over the current state given all past measurements.

**Examples:**
- Robot position (state) from noisy GPS (measurement)
- Temperature (state) from imprecise thermometer
- Stock price (state) from bid/ask spread (noisy proxy)
- Aircraft position (state) from radar (noisy, delayed)

> *The world changes. Our sensor lies. What do we actually believe?*

---

# The Bayes Filter

The general recursive belief update:

<div class="eq-box">

$$\underbrace{p(x_t \mid z_{1:t})}_{\text{posterior}} \propto \underbrace{p(z_t \mid x_t)}_{\text{measurement model}} \int \underbrace{p(x_t \mid x_{t-1})}_{\text{motion model}} \underbrace{p(x_{t-1} \mid z_{1:t-1})}_{\text{prior belief}} \, dx_{t-1}$$

</div>

Two steps at every time $t$:
1. **Predict** — propagate $p(x_{t-1} \mid z_{1:t-1})$ forward through the motion model
2. **Update** — condition on the new measurement $z_t$

The integral is generally **intractable** — but for linear-Gaussian systems it is exact.

---

<!-- _class: divider -->

# Part 2
## Linear-Gaussian State-Space Model

---

# The Linear-Gaussian Setup

Restrict to linear dynamics and Gaussian noise:

<div class="eq-box">

**Motion model:** $\quad x_t = x_{t-1} + w_t, \qquad w_t \sim \mathcal{N}(0, Q)$

**Measurement model:** $\quad z_t = x_t + v_t, \qquad v_t \sim \mathcal{N}(0, R)$

</div>

(1-D random walk — the simplest nontrivial case)

| Parameter | Meaning | Interpretation |
|---|---|---|
| $Q$ | Process noise variance | How much does the state jump per step? |
| $R$ | Measurement noise variance | How trustworthy is the sensor? |
| $x_0$ | Initial state | Our starting belief |
| $P_0$ | Initial variance | How uncertain are we at $t=0$? |

---

# Why Gaussians? — Closure

Under linear-Gaussian dynamics, if the belief at $t-1$ is Gaussian, the belief at $t$ is also Gaussian:

$$p(x_{t-1} \mid z_{1:t-1}) = \mathcal{N}(x_{t-1};\; \hat{x}_{t-1|t-1},\; P_{t-1|t-1})$$

$$\Downarrow \text{ predict + update}$$

$$p(x_t \mid z_{1:t}) = \mathcal{N}(x_t;\; \hat{x}_{t|t},\; P_{t|t})$$

The Gaussian family is **closed under linear transformations** and **Gaussian conditioning** — two operations that Predict and Update perform, respectively.

This is the special structure the Kalman Filter exploits.

---

<!-- _class: divider -->

# Part 3
## Predict Step

---

# Predict: Propagating Uncertainty

Starting from $p(x_{t-1} \mid z_{1:t-1}) = \mathcal{N}(\hat{x}_{t-1|t-1},\; P_{t-1|t-1})$:

<div class="eq-box">

$$\hat{x}_{t|t-1} = \hat{x}_{t-1|t-1}$$

$$P_{t|t-1} = P_{t-1|t-1} + Q$$

</div>

- The **mean** is propagated forward through the (identity) dynamics
- The **variance grows by Q** — the state could have moved; we don't know where

<div class="insight">
<strong>Interpretation:</strong> Between measurements, uncertainty only increases. Q is your prior belief about how fast the world changes. A slowly drifting temperature needs small Q; a quickly manoeuvring aircraft needs large Q.
</div>

---

<!-- _class: divider -->

# Part 4
## Kalman Gain — The Trust Dial

---

# How Much to Trust the Measurement?

After predicting, we receive measurement $z_t$. How do we fuse it?

The **innovation** (surprise) is: $\;\nu_t = z_t - \hat{x}_{t|t-1}$

We want to move the estimate by some fraction $K_t$ of the innovation. What is the optimal $K_t$?

**The Kalman gain** minimises the posterior variance (MMSE criterion):

<div class="eq-box">

$$K_t = \frac{P_{t|t-1}}{P_{t|t-1} + R}$$

</div>

$K_t \in [0, 1]$ always. It is **computed from the data** — not a tunable hyperparameter.

---

# Reading the Kalman Gain

$$K_t = \frac{P_{t|t-1}}{P_{t|t-1} + R}$$

<div class="two-col">
<div class="col insight">

**$K_t \to 1$ when $R \ll P$**

Sensor is accurate relative to model uncertainty → trust the measurement fully → estimate jumps to $z_t$

</div>
<div class="col warning">

**$K_t \to 0$ when $R \gg P$**

Sensor is noisy relative to model → ignore the measurement → estimate stays at prediction

</div>
</div>

<div class="demo">
<strong>PAI Tool — Explore Step 2:</strong> Set Q = 0.01, R = 0.01 → note K at t=3. Reset, set R = 10.00, re-run → read K again. With high R, K ≈ 0; the estimate barely reacts to the noisy sensor.
</div>

---

<!-- _class: divider -->

# Part 5
## Update Step

---

# Update: Fusing the Measurement

<div class="eq-box">

$$\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t \underbrace{(z_t - \hat{x}_{t|t-1})}_{\text{innovation } \nu_t}$$

$$P_{t|t} = (1 - K_t)\, P_{t|t-1}$$

</div>

**Key observations:**

- $\hat{x}_{t|t}$ is always strictly **between** the prediction and the measurement (weighted average)
- $P_{t|t} \leq P_{t|t-1}$ always — measurements can only **reduce** uncertainty
- If $K_t = 0$: no update at all. If $K_t = 1$: snap to measurement, $P_{t|t} = 0$

<div class="demo">
<strong>PAI Tool — Explore Step 1:</strong> Step forward one at a time. Read x̂(pred), z(meas), x̂(post) each step. The posterior always lies strictly between prediction and measurement — this weighted average IS the update equation.
</div>

---

# The Full Kalman Filter Algorithm

<div class="algorithm">

<span class="kw">Initialise:</span> $\hat{x}_0 = 0$, &ensp; $P_0 = 1$<br>
<span class="kw">for</span> $t = 1$ <span class="kw">to</span> $T$:<br>
&emsp; <span class="cm">── Predict ──────────────────────────────</span><br>
&emsp; $\hat{x}_{t|t-1} \;\leftarrow\; \hat{x}_{t-1|t-1}$ &ensp; <span class="cm">(propagate mean)</span><br>
&emsp; $P_{t|t-1} \;\leftarrow\; P_{t-1|t-1} + Q$ &ensp; <span class="cm">(grow variance)</span><br>
&emsp; <span class="cm">── Update ───────────────────────────────</span><br>
&emsp; $K_t \;\leftarrow\; P_{t|t-1} \;/\; (P_{t|t-1} + R)$ &ensp; <span class="cm">(gain)</span><br>
&emsp; $\hat{x}_{t|t} \;\leftarrow\; \hat{x}_{t|t-1} + K_t\,(z_t - \hat{x}_{t|t-1})$<br>
&emsp; $P_{t|t} \;\leftarrow\; (1 - K_t)\,P_{t|t-1}$<br>
<span class="kw">return</span> $\{\hat{x}_{t|t},\; P_{t|t}\}$ for all $t$

</div>

Complexity: **O(T)** time, **O(1)** space — update one step at a time, discard history.

---

<!-- _class: divider -->

# Part 6
## Steady-State Behaviour

---

# Riccati Convergence

The predicted variance $P_{t|t-1}$ converges to a **fixed point** regardless of the initial $P_0$:

Setting $P_{t|t-1} = P_{t-1|t-1} + Q$ and $P_{t|t} = (1-K_t)P_{t|t-1}$ at steady state:

<div class="eq-box">

$$P_\infty = \frac{Q}{2} + \sqrt{\frac{Q^2}{4} + QR}$$

$$K_\infty = \frac{P_\infty}{P_\infty + R}$$

</div>

Once steady-state is reached, $K$ is **constant** — you can precompute it and hardcode it in firmware.

<div class="demo">
<strong>PAI Tool — Explore Step 4:</strong> Set Q = 0.5, R = 2.0, press Play. Pause at t = 2, 5, 10, 20. K starts large and locks to a fixed value around t = 5 — Riccati fixed point reached.
</div>

---

<!-- _class: divider -->

# Part 7
## Calibrating Q and R

---

# The Two Extreme Limits

<div class="two-col">
<div class="col">

**High Q, Low R**
- State wanders fast
- Sensor is accurate
- Filter tracks measurements tightly
- $K \approx 1$ at every step

</div>
<div class="col">

**Low Q, High R**
- State is nearly stationary
- Sensor is noisy
- Filter smooths over many steps
- $K \approx 0$ at every step

</div>
</div>

<div class="demo">
<strong>PAI Tool — Explore Step 5:</strong> Set Q = 5.00, R = 0.01 → sensor dominates, estimate hugs every measurement. Then swap to Q = 0.01, R = 9.00 → model dominates, estimate barely reacts. Every deployment sits between these extremes.
</div>

---

# Calibration Check: NIS

How do we know if $Q$ and $R$ are correctly specified?

**Normalised Innovation Squared (NIS):**

$$\text{NIS}_t = \frac{(z_t - \hat{x}_{t|t-1})^2}{P_{t|t-1} + R}$$

For a correctly calibrated filter, $\text{NIS}_t \sim \chi^2(1)$, so the empirical mean should be $\approx 1$.

**Simple check:** roughly **32% of observations** should fall outside the $\pm 1\sigma$ band. More than 32% → filter is overconfident (underestimated $R$ or $Q$). Less than 32% → filter is underconfident.

<div class="demo">
<strong>PAI Tool — Explore Step 7:</strong> Run 50 steps with Q=1, R=3. Count dots outside ±1σ. Then reduce R to 0.1 — far more dots escape, signalling a miscalibrated filter.
</div>

---

# Robustness Across Realisations

**Kalman optimality is distributional:**

The gain schedule is derived from the *expected* squared error over all possible trajectories — so the same $K$ schedule is optimal regardless of which specific trajectory occurred.

<div class="demo">
<strong>PAI Tool — Explore Step 6:</strong> Set Q=0.5, R=2.0. Run to completion, note MSE. Drag Seed to 15, run again. Try seed 80. Different trajectories → same Kalman gain curve. The filter is optimally calibrated for the distribution, not for one realisation.
</div>

<div class="insight">
<strong>Implication:</strong> You evaluate a Kalman filter by its long-run statistical behaviour (NIS, MSE averaged over seeds), not by how well it tracked one particular run.
</div>

---

<!-- _class: divider -->

# Part 8
## Extensions

---

# Beyond 1-D: The Multivariate Kalman Filter

Replace scalars with matrices. For state $\mathbf{x}_t \in \mathbb{R}^n$, measurement $\mathbf{z}_t \in \mathbb{R}^m$:

<div class="eq-box">

**Motion model:** $\;\mathbf{x}_t = \mathbf{F}\mathbf{x}_{t-1} + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$

**Measurement model:** $\;\mathbf{z}_t = \mathbf{H}\mathbf{x}_t + \mathbf{v}_t, \quad \mathbf{v}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R})$

**Kalman gain:** $\;\mathbf{K}_t = \mathbf{P}_{t|t-1}\mathbf{H}^\top(\mathbf{H}\mathbf{P}_{t|t-1}\mathbf{H}^\top + \mathbf{R})^{-1}$

</div>

| Variant | Handles | Key idea |
|---|---|---|
| Kalman Filter | Linear + Gaussian | Exact, analytic |
| Extended KF (EKF) | Nonlinear, smooth | Jacobian linearisation |
| Unscented KF (UKF) | Nonlinear | Sigma-point propagation |
| Particle Filter | Arbitrary | Monte Carlo sampling |

---

# Summary

<div class="two-col">
<div class="col">

**The Predict-Update loop**
- Predict: $P$ grows by $Q$
- Update: $P$ shrinks by factor $(1-K)$
- The "breathing" uncertainty band

**Kalman gain $K_t$**
- Not tuned by hand — computed optimally
- $K = P/(P+R)$ balances model vs. sensor
- Converges to steady-state $K_\infty$

</div>
<div class="col insight">

**Key design decisions**
1. Choose $Q$: how fast does state move?
2. Choose $R$: how noisy is the sensor?
3. Validate with NIS / coverage check

**The payoff:** Minimum MSE estimate, exact under linear-Gaussian assumptions, $O(T)$ runtime.

</div>
</div>

---

# Key Equations at a Glance

<div class="eq-box">

**Predict:**
$$\hat{x}_{t|t-1} = \hat{x}_{t-1|t-1} \qquad P_{t|t-1} = P_{t-1|t-1} + Q$$

**Gain:**
$$K_t = \frac{P_{t|t-1}}{P_{t|t-1} + R}$$

**Update:**
$$\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t(z_t - \hat{x}_{t|t-1}) \qquad P_{t|t} = (1-K_t)P_{t|t-1}$$

**Steady-state:**
$$P_\infty = \frac{Q}{2} + \sqrt{\frac{Q^2}{4} + QR} \qquad K_\infty = \frac{P_\infty}{P_\infty + R}$$

</div>

---

# References

- **Krause & Hübotter** — *Probabilistic Artificial Intelligence* (2025), Ch. 3. Derives KF from the Bayes filter; covers multivariate case and Kalman Smoother.
- **Welch & Bishop** — *An Introduction to the Kalman Filter* (2006). Classic 16-page tutorial, best starting point.
- **Thrun, Burgard & Fox** — *Probabilistic Robotics* (2005), Ch. 3. EKF, UKF, particle filters in robotics.
- **Kalman (1960)** — *A New Approach to Linear Filtering and Prediction Problems.* The original paper.

<div class="insight">
<strong>Connection to BLR:</strong> The Kalman Filter is the online, sequential equivalent of BLR for time series. Each update step is a one-step BLR posterior update with the measurement $z_t$ as the single new observation.
</div>

---

<!-- _class: title -->

# Questions?

## PAI Interactive Tool

Explore tab → Module 3: Kalman Filter

*Seven interactive steps: predict-update rhythm, Kalman gain, Q/R calibration, Riccati convergence.*
