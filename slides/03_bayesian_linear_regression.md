---
marp: true
theme: default
paginate: true
math: katex
style: |
  /* ── Base ───────────────────────────────────────────── */
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
  h3 { font-size: 0.95em; color: #475569; font-weight: 600; }
  ul { margin-top: 6px; }
  li { margin-bottom: 4px; }
  code { background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; }

  /* ── Title slide ────────────────────────────────────── */
  section.title {
    background: linear-gradient(135deg, #0f2a52 0%, #1d4ed8 100%);
    color: white;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.title h1 { color: white; border-color: rgba(255,255,255,0.35); font-size: 1.8em; }
  section.title h2 { color: rgba(255,255,255,0.8); font-size: 1em; }
  section.title p  { color: rgba(255,255,255,0.65); font-size: 0.75em; }

  /* ── Section divider ─────────────────────────────────── */
  section.divider {
    background: #1e3a5f;
    color: white;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.divider h1 { color: white; border-color: rgba(255,255,255,0.3); }
  section.divider p  { color: rgba(255,255,255,0.7); font-size: 0.85em; }

  /* ── Callout boxes ───────────────────────────────────── */
  .demo {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    padding: 8px 12px;
    border-radius: 0 6px 6px 0;
    font-size: 0.78em;
    margin-top: 10px;
  }
  .demo strong { color: #1d4ed8; }
  .insight {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
    padding: 8px 12px;
    border-radius: 0 6px 6px 0;
    font-size: 0.78em;
    margin-top: 10px;
  }
  .insight strong { color: #15803d; }
  .warning {
    background: #fff7ed;
    border-left: 4px solid #f97316;
    padding: 8px 12px;
    border-radius: 0 6px 6px 0;
    font-size: 0.78em;
    margin-top: 10px;
  }
  .eq-box {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 12px 18px;
    text-align: center;
    margin: 10px 0;
  }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
  .three-col { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 18px; }
  .col { }
  .small { font-size: 0.74em; color: #64748b; }

  /* ── Footer override ─────────────────────────────────── */
  section::after { font-size: 0.62em; color: #94a3b8; }
---

<!-- _class: title -->

# Bayesian Linear Regression

## Probabilistic AI — Chapter 2

Özgür S. Öğüz · Bilkent University

---

<!-- _paginate: false -->

# Agenda

0. **The Gaussian distribution** — why it is everywhere in probabilistic ML
1. **From point estimates to distributions** — why go Bayesian?
2. **Model specification** — likelihood, prior over weights
3. **Posterior derivation** — Gaussian conjugacy, closed form
4. **Predictive distribution** — two sources of uncertainty
5. **Weight-space view** — the posterior ellipse
6. **Non-linear regression** — basis functions (polynomial, RBF)
7. **Model evidence** — automatic Occam's razor
8. **Summary & key equations**

---

# Recap: From BN to BLR

<div class="two-col">
<div class="col">

**Bayesian Networks (Module 1)**

- Discrete random variables
- CPTs as probability tables
- Bayes' theorem with **sums**:

$$p(\theta \mid D) = \frac{\sum_h p(D, h \mid \theta)\,p(\theta)}{\sum_\theta \sum_h p(D, h \mid \theta)\,p(\theta)}$$

- Inference: VE, enumeration

</div>
<div class="col">

**Bayesian Linear Regression (this module)**

- Continuous weight vector $\mathbf{w}$
- Gaussian prior + Gaussian likelihood
- Bayes' theorem with **integrals**:

$$p(\mathbf{w} \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \mathbf{w})\,p(\mathbf{w})}{\int p(\mathcal{D} \mid \mathbf{w})\,p(\mathbf{w})\,d\mathbf{w}}$$

- Inference: analytic (conjugacy)

</div>
</div>

<div class="insight">
<strong>The posterior update is the same mathematical operation — only the domain changes.</strong> Sums become integrals; tables become density functions; the logic of conditioning on evidence is identical.
</div>

---

<!-- _class: divider -->

# Part 0
## The Gaussian Distribution

---

# Why the Gaussian?

The normal distribution is the foundation of probabilistic ML — not by convention, but for three deep reasons:

<div class="three-col">
<div class="col insight">

**Maximum Entropy**
Given only a fixed mean and variance, the Gaussian is the distribution that makes the *fewest additional assumptions* — the most "spread out" shape consistent with the constraints.

</div>
<div class="col insight">

**Conjugacy**
Gaussian prior × Gaussian likelihood = Gaussian posterior. Closed-form update, no approximation. This is why BLR is exactly solvable.

</div>
<div class="col insight">

**Central Limit Theorem**
The sum of many independent random variables converges to a Gaussian, regardless of the individual distributions. Measurement noise is a sum of many small errors → Gaussian.

</div>
</div>

<div class="eq-box">

$$\mathcal{N}(x;\,\mu,\,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

</div>

68% of mass within $\mu\pm\sigma$ &ensp;·&ensp; 95% within $\mu\pm 2\sigma$ &ensp;·&ensp; 99.7% within $\mu\pm 3\sigma$

---

# Multivariate Gaussian

For a vector $\mathbf{x} \in \mathbb{R}^d$, the distribution generalises as:

<div class="eq-box">

$$\mathcal{N}(\mathbf{x};\,\boldsymbol{\mu},\,\boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\!\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

</div>

<div class="two-col">
<div class="col">

- $\boldsymbol{\mu} \in \mathbb{R}^d$ — mean vector
- $\boldsymbol{\Sigma} \in \mathbb{R}^{d\times d}$ — covariance matrix
- Diagonal $\boldsymbol{\Sigma}$: independent components
- Off-diagonal terms: correlations between variables
- Level sets are **ellipsoids** in $\mathbb{R}^d$

</div>
<div class="col demo">

**In BLR:** the weight posterior $p(\mathbf{w}\mid\mathcal{D}) = \mathcal{N}(\boldsymbol{\mu}_w, \boldsymbol{\Sigma}_w)$ is a $(d\!+\!1)$-dimensional Gaussian. For degree = 1 it is a **2D ellipse** over $(w_0, w_1)$ — visible in the Weight-Space panel of the PAI tool.

</div>
</div>

<div class="insight">
<strong>Key property — linear transform:</strong> if $\mathbf{x}\sim\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\Sigma})$ and $y = \mathbf{a}^\top\mathbf{x}$, then $y\sim\mathcal{N}(\mathbf{a}^\top\boldsymbol{\mu},\,\mathbf{a}^\top\boldsymbol{\Sigma}\mathbf{a})$. This is exactly how the predictive variance $\boldsymbol{\phi}(x^*)^\top\boldsymbol{\Sigma}_w\boldsymbol{\phi}(x^*)$ arises.
</div>

---

# Motivating Example: Sensor Calibration

A robot's joint-temperature sensor needs calibration before deployment:

$$y = w_1\, x + w_0 + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0,\,\sigma_n^2)$$

$x$ = reference temperature &ensp;·&ensp; $y$ = sensor reading &ensp;·&ensp; $w_0, w_1$ = unknown bias & scale

**Calibrate at 5 reference points** in the normal operating range, then ask:

- **What are $w_1$ and $w_0$?** → posterior over weights
- **How confident?** → uncertainty band (tight in-range, wide outside)
- **Reading at extreme temperatures never tested?** → extrapolation danger
- **Need degree 2, or is linear sufficient?** → model evidence answers automatically — no test set needed

<div class="demo">
<strong>PAI Tool:</strong> Load "Sensor Calibration". The uncertainty band is tight in [−1, 1] but <em>widens beyond the calibrated range</em> — the model tells you when it is operating outside its training knowledge.
</div>

---

<!-- _class: divider -->

# Part 1
## From Point Estimates to Distributions

---

# The Frequentist Limit

**Ordinary Least Squares** (OLS) gives a *single* answer:

$$\hat{\mathbf{w}}_{\text{OLS}} = \arg\min_\mathbf{w} \|\mathbf{y} - \boldsymbol{\Phi}\mathbf{w}\|^2$$

Problems with a single point estimate:

- **No uncertainty** — we don't know how confident to be
- **No regularisation** built in by default
- **Overfits** when $N \ll d$
- **Extrapolation is silent** — no warning when we leave the data domain

> *We want a model that says "I don't know" when it genuinely doesn't.*

---

# The Bayesian Approach

Instead of a single weight vector, maintain a **distribution over weights**:

<div class="eq-box">

$$p(\mathbf{w} \mid \mathcal{D}) \propto \underbrace{p(\mathbf{y} \mid \mathbf{X}, \mathbf{w})}_{\text{likelihood}} \cdot \underbrace{p(\mathbf{w})}_{\text{prior}}$$

</div>

- **Prior** $p(\mathbf{w})$ — what we believe before seeing data
- **Likelihood** $p(\mathbf{y} \mid \mathbf{X}, \mathbf{w})$ — how well weights explain the data
- **Posterior** $p(\mathbf{w} \mid \mathcal{D})$ — updated belief after seeing data

The posterior is a *full distribution*: it quantifies uncertainty, not just a best guess.

---

# Analogy: The New Employee

Every BLR concept maps to a familiar workplace story:

| BLR concept | Analogy |
|---|---|
| Prior $p(\mathbf{w})$ | Manager's initial belief: *"probably average, not extraordinary"* |
| Data points | Tasks the employee completes; each one updates the belief |
| Epistemic uncertainty | Week 1: wide range of possible skill levels |
| Aleatoric uncertainty | Even after knowing them well, some days they're just off |
| Posterior update | After 2 months: much tighter belief about their abilities |
| Extrapolation (new domain) | *"Great at frontend — but we've never seen them do ML"* → wide band |
| Prior as regularisation | One exceptional result doesn't immediately crown them a genius |
| Model evidence | Does skill follow a simple linear trend, or something more complex? |

<div class="insight">
<strong>The posterior after many tasks is sharp — a near-certain belief. After one task it is still wide. This is the Bayesian update equation, expressed in a language every student already understands.</strong>
</div>

---

<!-- _class: divider -->

# Part 2
## Model Specification

---

# Likelihood: Gaussian Noise Model

Assume observations are generated by a linear model plus i.i.d. noise:

$$y_i = \mathbf{w}^\top \boldsymbol{\phi}(x_i) + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma_n^2)$$

where $\boldsymbol{\phi}(x) \in \mathbb{R}^{d+1}$ is a **feature vector** (e.g., $[1, x, x^2, \ldots, x^d]^\top$).

The likelihood for $N$ observations is:

$$p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) = \mathcal{N}(\mathbf{y};\; \boldsymbol{\Phi}\mathbf{w},\; \sigma_n^2 \mathbf{I})$$

Parameters: $\beta = \sigma_n^{-2}$ (**noise precision**) — how much we trust each observation.

---

# Prior: Gaussian Over Weights

Place a zero-mean isotropic Gaussian prior over the weight vector:

$$p(\mathbf{w}) = \mathcal{N}(\mathbf{w};\; \mathbf{0},\; \sigma_w^2 \mathbf{I})$$

Parameters: $\alpha = \sigma_w^{-2}$ (**prior precision**) — how strongly we believe weights should be near zero.

<div class="two-col">
<div class="col">

**Small $\sigma_w^2$** → strong prior → weights are pulled toward 0 → simple (flat) functions

</div>
<div class="col">

**Large $\sigma_w^2$** → weak prior → weights can be anything → expressive functions

</div>
</div>

<div class="demo">
<strong>PAI Tool — Explore Step 1:</strong> Load "Clear (Prior)", enable Posterior samples. Drag σ²_w from 0.10 → 5.00. Watch the fan of prior functions open and close.
</div>

---

<!-- _class: divider -->

# Part 3
## Posterior Derivation

---

# Gaussian Conjugacy

Both the likelihood and prior are Gaussian → the posterior is **also Gaussian** (conjugacy).

No approximation needed. The posterior is computed in closed form:

<div class="eq-box">

$$\boldsymbol{\Sigma}_w = \bigl(\beta\,\boldsymbol{\Phi}^\top\boldsymbol{\Phi} + \alpha\mathbf{I}\bigr)^{-1}$$

$$\boldsymbol{\mu}_w = \beta\,\boldsymbol{\Sigma}_w\,\boldsymbol{\Phi}^\top\mathbf{y}$$

$$p(\mathbf{w} \mid \mathcal{D}) = \mathcal{N}(\mathbf{w};\; \boldsymbol{\mu}_w,\; \boldsymbol{\Sigma}_w)$$

</div>

- $\boldsymbol{\mu}_w$ = regularised least-squares solution (ridge regression!)
- $\boldsymbol{\Sigma}_w$ = remaining parameter uncertainty

---

# Algorithm: BLR Posterior Update

| Step | Operation | Cost |
|------|-----------|------|
| 1 | Build feature matrix $\boldsymbol{\Phi} \in \mathbb{R}^{N \times (d+1)}$ | $O(Nd)$ |
| 2 | Compute $\boldsymbol{\Sigma}_w = (\beta\boldsymbol{\Phi}^\top\boldsymbol{\Phi} + \alpha\mathbf{I})^{-1}$ | $O(Nd^2 + d^3)$ |
| 3 | Compute $\boldsymbol{\mu}_w = \beta\boldsymbol{\Sigma}_w\boldsymbol{\Phi}^\top\mathbf{y}$ | $O(Nd)$ |
| 4 | Predict: $\mu_{y^*} = \boldsymbol{\mu}_w^\top\boldsymbol{\phi}(x^*)$ | $O(d)$ |
| 5 | Predict: $\sigma^2_{y^*} = \boldsymbol{\phi}(x^*)^\top\boldsymbol{\Sigma}_w\boldsymbol{\phi}(x^*) + \beta^{-1}$ | $O(d^2)$ |

Total: **$O(Nd^2 + d^3)$** — matrix inversion is the bottleneck, not the data size.

<div class="insight">
<strong>Key insight:</strong> When $N \gg d$, the bottleneck is building Φ. When $d \gg N$, use the matrix inversion lemma (Woodbury identity) to flip the cost to O(N³).
</div>

---

# Connection to Ridge Regression

The posterior mean $\boldsymbol{\mu}_w$ is exactly the **ridge regression** solution:

$$\boldsymbol{\mu}_w = \arg\min_\mathbf{w} \left[ \|\mathbf{y} - \boldsymbol{\Phi}\mathbf{w}\|^2 + \frac{\alpha}{\beta}\|\mathbf{w}\|^2 \right]$$

The regularisation strength $\lambda = \alpha/\beta = \sigma_n^2/\sigma_w^2$ emerges naturally from the prior ratio.

<div class="demo">
<strong>PAI Tool — Explore Step 3:</strong> Load "Linear Trend" (Degree=1). Drag σ²_w slowly from 1.00 → 0.01. Watch the slope flatten toward 0 even though the data show a clear trend — this is L2 regularisation emerging from the prior.
</div>

<div class="warning">
Ridge regression is a <em>special case</em> of BLR — but BLR also gives you the full uncertainty, not just the MAP point.
</div>

---

<!-- _class: divider -->

# Part 4
## Predictive Distribution

---

# Marginalising Over Weights

To predict at a new input $x^*$, we integrate out the weights:

<div class="eq-box">

$$p(y^* \mid x^*, \mathcal{D}) = \int p(y^* \mid x^*, \mathbf{w})\, p(\mathbf{w} \mid \mathcal{D})\, d\mathbf{w}$$

</div>

For Gaussians this integral is **analytic**:

$$p(y^* \mid x^*, \mathcal{D}) = \mathcal{N}\!\left(y^*;\;\; \underbrace{\boldsymbol{\mu}_w^\top\boldsymbol{\phi}(x^*)}_{\text{posterior mean}},\;\; \underbrace{\boldsymbol{\phi}(x^*)^\top\boldsymbol{\Sigma}_w\boldsymbol{\phi}(x^*) + \beta^{-1}}_{\text{total variance}}\right)$$

---

# Two Sources of Uncertainty

The predictive variance decomposes as:

<div class="eq-box">

$$\sigma^2_{y^*} = \underbrace{\boldsymbol{\phi}(x^*)^\top\boldsymbol{\Sigma}_w\boldsymbol{\phi}(x^*)}_{\text{epistemic uncertainty}} + \underbrace{\beta^{-1} = \sigma_n^2}_{\text{aleatoric uncertainty}}$$

</div>

<div class="two-col">
<div class="col insight">

**Epistemic** (model uncertainty)
- Due to limited data
- Shrinks as $N \to \infty$
- Wide in unobserved regions
- Can be reduced

</div>
<div class="col warning">

**Aleatoric** (observation noise)
- Due to sensor noise
- Constant: $\sigma_n^2$ everywhere
- Cannot be reduced
- Encoded by $\beta^{-1}$

</div>
</div>

---

# Seeing Uncertainty in Action

<div class="demo">
<strong>PAI Tool — Explore Step 2 (Sequential Updating):</strong>

Load "Clear (Prior)", set Degree = 1. Click one point near (0, 1). Then click two more roughly collinear points.

- First click: band collapses **only at that x** — the prior dominates everywhere else
- Three collinear points: narrow corridor, but ends still **flare** (extrapolation uncertainty)
- The tightening is sharpest where φ(x) is large — the posterior update equation made visible
</div>

<div class="demo">
<strong>PAI Tool — Explore Step 4 (Noise variance):</strong>

Load "Nonlinear Data", Degree = 2. Drag σ²_n from 0.01 → 2.00 and back.

- Low σ²_n: mean threads through every point (near-noiseless sensor)
- High σ²_n: smooth curve — jitter attributed to noise, not signal
</div>

---

<!-- _class: divider -->

# Part 5
## Weight-Space View

---

# The Posterior Ellipse

In the weight space $(\mathbf{w} \in \mathbb{R}^{d+1})$, the posterior $\mathcal{N}(\boldsymbol{\mu}_w, \boldsymbol{\Sigma}_w)$ is an ellipsoid.

For Degree = 1 ($\mathbf{w} = [w_0, w_1]^\top$): a 2D ellipse showing the joint uncertainty over intercept and slope.

<div class="two-col">
<div class="col">

**Elongated ellipse**
- One direction poorly constrained
- Observations lie along a confounded axis
- Multiple $(w_0, w_1)$ pairs fit equally well

</div>
<div class="col">

**Compact ellipse**
- Dense data → small $\boldsymbol{\Sigma}_w$
- Slope and intercept well-identified
- Function-space band also narrow

</div>
</div>

<div class="demo">
<strong>PAI Tool — Explore Step 5:</strong> Load "Dense Observations", Degree=1, enable samples. Open Weight-Space panel. Drag σ²_n from 0.30 → 2.00 → 0.05. The ellipse and uncertainty band are two projections of the same Σ_w.
</div>

---

<!-- _class: divider -->

# Part 6
## Non-Linear Regression via Basis Functions

---

# The Feature Map Trick

BLR is linear in **weights**, but the feature map $\boldsymbol{\phi}(x)$ can be *anything*:

<div class="three-col">
<div class="col">

**Polynomial**
$$\boldsymbol{\phi}(x) = \begin{bmatrix}1 \\ x \\ x^2 \\ \vdots \\ x^d\end{bmatrix}$$

</div>
<div class="col">

**RBF (Gaussian)**
$$\phi_j(x) = \exp\!\left(-\frac{(x-c_j)^2}{2\ell^2}\right)$$

</div>
<div class="col">

**Fourier**
$$\boldsymbol{\phi}(x) = \begin{bmatrix}\cos(\omega_1 x) \\ \sin(\omega_1 x) \\ \vdots\end{bmatrix}$$

</div>
</div>

The posterior update equations **do not change** — only $\boldsymbol{\Phi}$ changes.

> *Bayesian Linear Regression is really "Bayesian regression in any feature space."*

---

# Polynomial Basis

$$\boldsymbol{\phi}(x) = [1,\; x,\; x^2,\; \ldots,\; x^d]^\top \quad \Rightarrow \quad d+1 \text{ parameters}$$

- Degree 1: line (2 parameters)
- Degree 3: cubic (4 parameters)
- Degree 7+: very flexible but **high uncertainty at edges** (Runge's phenomenon)

<div class="insight">
<strong>Observation:</strong> Increasing degree beyond what the data supports inflates the uncertainty band near the boundaries — a visible warning from the model that complexity is unjustified.
</div>

---

# RBF (Radial Basis Function) Basis

Place $J$ Gaussian bumps at centres $c_1, \ldots, c_J$ across the input domain:

$$\phi_j(x) = \exp\!\left(-\frac{(x - c_j)^2}{2\ell^2}\right), \qquad j = 1, \ldots, J$$

- Each basis function has **local support** — a bump around $c_j$
- $\sigma_w^2$ controls **bump amplitude**
- $\ell$ controls **bump width** (smoothness)

<div class="demo">
<strong>PAI Tool — Explore Step 7:</strong> Load "Nonlinear Data". Switch Basis from Polynomial → RBF. Add points along a sine wave. Then switch back to Polynomial Degree 3 and compare. Seven bumps capture arbitrary curvature without changing degree.
</div>

---

<!-- _class: divider -->

# Part 7
## Model Evidence & Occam's Razor

---

# The Model Evidence

Given data, which model complexity (degree) is best? Use the **marginal likelihood**:

$$p(\mathbf{y} \mid \mathbf{X}, \mathcal{M}) = \int p(\mathbf{y} \mid \mathbf{X}, \mathbf{w})\, p(\mathbf{w} \mid \mathcal{M})\, d\mathbf{w}$$

For BLR this is analytic:

<div class="eq-box">

$$\log p(\mathbf{y} \mid \mathbf{X}, d) = -\frac{1}{2}\bigl[\mathbf{y}^\top \mathbf{C}_y^{-1} \mathbf{y} + \log|\mathbf{C}_y| + N\log 2\pi\bigr]$$

</div>

where $\mathbf{C}_y = \sigma_n^2\mathbf{I} + \sigma_w^2\boldsymbol{\Phi}\boldsymbol{\Phi}^\top$.

**No train/test split needed. No cross-validation.**

---

# Automatic Occam's Razor

<div class="two-col">
<div class="col">

A **more complex model** must spread its probability mass over *more functions*.

If the data does not justify that complexity, the evidence **penalises** the complex model.

$$\underbrace{\text{Evidence}}_{\text{observed}} = \underbrace{\text{Data fit}}_{\text{want high}} - \underbrace{\text{Complexity penalty}}_{\text{model spread}}$$

</div>
<div class="col insight">

**Bayes is automatically parsimonious:**
- Degree 1 for linear data → highest evidence
- Degree 4 for cubic data → evidence drops (fits noise)
- No regularisation hyperparameter to tune by hand

</div>
</div>

<div class="demo">
<strong>PAI Tool — Explore Step 6:</strong> Load "Model Selection". Click Degree 1 → 2 → 3 → 4. Read log p(y|degree) each time. Degree 3 peaks (data was cubic). Degree 4 drops — a more complex model is penalised when the data don't justify it.
</div>

---

<!-- _class: divider -->

# Part 8
## Summary

---

# What BLR Gives You

| OLS / Ridge | Bayesian Linear Regression |
|---|---|
| Point estimate $\hat{\mathbf{w}}$ | Full distribution $p(\mathbf{w}\mid\mathcal{D})$ |
| No uncertainty | Epistemic + aleatoric |
| Regularisation by hand ($\lambda$) | Prior $\sigma_w^2$ with probabilistic meaning |
| Model selection via CV | Model evidence (analytic) |
| Silent extrapolation | Wide band = warning |
| Any basis | Any basis (same update eqs) |

**The Bayesian posterior collapses to the MAP (ridge) estimate when you take its mode — but carries far more information.**

---

# Key Equations at a Glance

<div class="eq-box">

**Posterior:**
$$\boldsymbol{\Sigma}_w = (\beta\boldsymbol{\Phi}^\top\boldsymbol{\Phi} + \alpha\mathbf{I})^{-1} \qquad \boldsymbol{\mu}_w = \beta\boldsymbol{\Sigma}_w\boldsymbol{\Phi}^\top\mathbf{y}$$

**Predictive:**
$$p(y^*\mid x^*,\mathcal{D}) = \mathcal{N}\!\left(\boldsymbol{\mu}_w^\top\boldsymbol{\phi}(x^*),\;\; \boldsymbol{\phi}(x^*)^\top\boldsymbol{\Sigma}_w\boldsymbol{\phi}(x^*) + \beta^{-1}\right)$$

**Model evidence:**
$$\log p(\mathbf{y}\mid d) = -\tfrac{1}{2}\!\left[\mathbf{y}^\top\mathbf{C}_y^{-1}\mathbf{y} + \log|\mathbf{C}_y| + N\log 2\pi\right]$$

</div>

<div class="small">

$\alpha = 1/\sigma_w^2$ (prior precision) · $\beta = 1/\sigma_n^2$ (noise precision) · $\boldsymbol{\Phi}_{ij} = \phi_j(x_i)$

</div>

---

# References

- **Krause & Hübotter** — *Probabilistic Artificial Intelligence* (2025), Ch. 2. Primary course text. → `arxiv.org/abs/2502.05244`
- **Bishop** — *Pattern Recognition and Machine Learning* (2006), §3.3. Canonical derivation with evidence approximation.
- **Murphy** — *Machine Learning: A Probabilistic Perspective* (2012), §7.6. Accessible treatment with Python examples.
- **Rasmussen & Williams** — *GP for ML* (2006), §2.1. Function-space view, connection to GPs.

<div class="insight">
<strong>Coming up — Module 3:</strong> The function-space view of BLR leads directly to <em>Gaussian Processes</em>: replace the finite basis φ(x) with an infinite-dimensional kernel k(x,x′) and the same posterior update equations still apply.
</div>

---

<!-- _class: title -->

# Questions?

## PAI Interactive Tool

`pai.local` · Explore tab → Module 2: Bayesian Linear Regression

*Eight interactive steps to build intuition before the next lecture.*
