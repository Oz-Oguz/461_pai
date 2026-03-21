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
    border-bottom: 3px solid #0891b2; 
    padding-bottom: 8px; 
    margin-bottom: 16px; 
    margin-top: 0;
    padding-top: 0;
  }
  h2 { font-size: 1.05em; color: #0891b2; margin-top: 0; margin-bottom: 8px; }
  ul { margin-top: 6px; }
  li { margin-bottom: 4px; }

  section.title {
    background: linear-gradient(135deg, #0c4a6e 0%, #0891b2 100%);
    color: white;
    display: flex; flex-direction: column; justify-content: center;
  }
  section.title h1 { color: white; border-color: rgba(255,255,255,0.35); font-size: 1.8em; }
  section.title h2 { color: rgba(255,255,255,0.8); font-size: 1em; }
  section.title p  { color: rgba(255,255,255,0.65); font-size: 0.75em; }

  section.divider {
    background: #083344;
    color: white;
    display: flex; flex-direction: column; justify-content: center;
  }
  section.divider h1 { color: white; border-color: rgba(255,255,255,0.3); }
  section.divider p  { color: rgba(255,255,255,0.7); font-size: 0.85em; }

  .demo { background: #ecfeff; border-left: 4px solid #0891b2; padding: 8px 12px; border-radius: 0 6px 6px 0; font-size: 0.78em; margin-top: 10px; }
  .demo strong { color: #0e7490; }
  .insight { background: #f0fdf4; border-left: 4px solid #22c55e; padding: 8px 12px; border-radius: 0 6px 6px 0; font-size: 0.78em; margin-top: 10px; }
  .insight strong { color: #15803d; }
  .warning { background: #fff7ed; border-left: 4px solid #f97316; padding: 8px 12px; border-radius: 0 6px 6px 0; font-size: 0.78em; margin-top: 10px; }
  .eq-box { background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px 18px; text-align: center; margin: 10px 0; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
  .three-col { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 18px; }
  .col { }
  .small { font-size: 0.74em; color: #64748b; }
  table { font-size: 0.76em; }
  section::after { font-size: 0.62em; color: #94a3b8; }
---

<!-- _class: title -->

# Gaussian Processes

## Probabilistic AI — Chapter 4

Özgür S. Öğüz · Bilkent University

---

<!-- _paginate: false -->

# Agenda

1. **From BLR to GPs** — the function-space limit
2. **GP Prior** — a distribution over functions
3. **Kernels** — the design choice
4. **GP Posterior** — closed-form update
5. **Two sources of uncertainty** — epistemic vs. aleatoric
6. **Hyperparameter optimisation** — Type-II MLE
7. **Kernel design** — RBF, Matérn, Linear, Periodic
8. **Scalability** — $O(N^3)$ and sparse approximations

---

<!-- _class: divider -->

# Part 1
## From BLR to Gaussian Processes

---

# Recall: BLR in Function Space

In BLR we placed a prior over weight vectors $\mathbf{w} \in \mathbb{R}^{d+1}$:

$$p(\mathbf{w}) = \mathcal{N}(\mathbf{0}, \sigma_w^2\mathbf{I})$$

The prior over **functions** induced by this weight prior is:

$$f(x) = \mathbf{w}^\top\boldsymbol{\phi}(x) \implies f \sim \mathcal{GP}(0,\; \sigma_w^2\,\boldsymbol{\phi}(x)^\top\boldsymbol{\phi}(x'))$$

The **kernel** $k(x, x') = \sigma_w^2\,\boldsymbol{\phi}(x)^\top\boldsymbol{\phi}(x')$ is the covariance between function values at any two inputs.

**The key insight:** instead of choosing a finite basis $\boldsymbol{\phi}(x)$ and implicitly specifying $k$, we can specify $k$ *directly* — and this corresponds to an *infinite-dimensional* basis.

---

# The Function-Space View

**BLR:** Prior over weights → implicit prior over functions

**GP:** Prior **directly** over functions

<div class="eq-box">

$$f \sim \mathcal{GP}(m(x),\; k(x, x'))$$

</div>

Meaning: for any finite collection of inputs $\{x_1, \ldots, x_n\}$:

$$\begin{bmatrix}f(x_1) \\ \vdots \\ f(x_n)\end{bmatrix} \sim \mathcal{N}\!\left(\begin{bmatrix}m(x_1) \\ \vdots \\ m(x_n)\end{bmatrix},\; \mathbf{K}\right), \quad K_{ij} = k(x_i, x_j)$$

We use $m(x) = 0$ (zero mean prior — the data will shift it).

---

<!-- _class: divider -->

# Part 2
## The GP Prior

---

# The Prior as a Distribution Over Functions

A GP prior samples entire **functions**, not weight vectors.

Each function is characterised by how correlated its values are at different inputs — this is exactly what the kernel $k(x, x')$ encodes.

<div class="demo">
<strong>PAI Tool — Explore Step 1:</strong> Load "Clear (Prior)", enable Posterior samples. Drag Length Scale ℓ from 1.0 → 0.10, then → 3.00.

- ℓ = 0.10: wildly wiggly — each point nearly independent of neighbours
- ℓ = 3.00: long gentle curves — distant points highly correlated

You are watching "how far does information travel?" — invisible in the formula, obvious in the samples.
</div>

---

# Kernel Matrix $\mathbf{K}$

Evaluate the kernel at all pairs of training inputs:

$$K_{ij} = k(x_i, x_j)$$

Properties of a valid kernel matrix:
- **Symmetric:** $K_{ij} = K_{ji}$
- **Positive semi-definite:** $\mathbf{v}^\top\mathbf{K}\mathbf{v} \geq 0$ for all $\mathbf{v}$

This guarantees the implied covariance structure is consistent with a Gaussian distribution.

**With noise:** $\mathbf{K}_y = \mathbf{K}(X, X) + \sigma_n^2\mathbf{I}$

The diagonal $\sigma_n^2\mathbf{I}$ ensures numerical stability and models observation noise.

---

<!-- _class: divider -->

# Part 3
## Kernels — The Design Choice

---

# Common Kernels

<div class="three-col">
<div class="col">

**RBF / Squared Exponential**

$$k(x,x') = \sigma_f^2 e^{-\frac{(x-x')^2}{2\ell^2}}$$

- Infinitely differentiable
- Very smooth functions
- Most widely used

</div>
<div class="col">

**Matérn 3/2**

$$k(x,x') = \sigma_f^2\!\left(1+\frac{\sqrt{3}r}{\ell}\right)e^{-\frac{\sqrt{3}r}{\ell}}$$

- Once differentiable
- Functions can have kinks
- More realistic for physical data

</div>
<div class="col">

**Periodic**

$$k(x,x') = \sigma_f^2 e^{-\frac{2\sin^2(\pi r/p)}{\ell^2}}$$

- Repeating patterns
- Period $p$ is explicit
- Extrapolates oscillations

</div>
</div>

**Linear kernel:** $k(x,x') = \sigma_f^2\,x\,x'$ → GP becomes exactly BLR.

---

# What the Kernel Encodes

The kernel $k(x_0, x)$ tells you: *"if I know $f(x_0)$, how much does that constrain $f(x)$?"*

| Kernel shape | Implication |
|---|---|
| Narrow bell (small ℓ) | Only nearby points are correlated → wiggly fits |
| Wide bell (large ℓ) | Distant points correlated → smooth global behaviour |
| Sharp peak (Matérn) | Correlation drops quickly → allows kinks |
| Oscillating | Periodic correlation → repeating structure |

<div class="demo">
<strong>PAI Tool — Explore Step 3 (Kernel Panel):</strong> Load "Clear (Prior)". Switch Kernel: RBF → Matérn 3/2 → Linear. Watch the Kernel Shape panel below the plot — you are watching k(0, x) live. The bell shape IS the prior's notion of similarity.
</div>

---

# Length Scale: Smoothness Control

<div class="demo">
<strong>PAI Tool — Explore Step 5:</strong> Load "Short Length Scale" (ℓ = 0.3). Look at the Kernel panel — narrow bell. Drag ℓ → 2.00, watching both plot and kernel panel simultaneously.

The bell widens exactly as the posterior mean smooths out. You can read the entire smoothness structure of the posterior from a single cross-section of the kernel.
</div>

<div class="insight">
<strong>Design principle:</strong> Before fitting a GP, ask: "What length scale makes physical sense for my problem?" A sensor measurement taken 1 second ago is highly correlated with now → large ℓ (relative to sampling rate). A stock price 1 year ago is weakly correlated → small ℓ.
</div>

---

<!-- _class: divider -->

# Part 4
## GP Posterior

---

# Conditioning a GP on Data

Given data $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ with $y_i = f(x_i) + \varepsilon_i$, $\varepsilon_i \sim \mathcal{N}(0, \sigma_n^2)$:

The posterior is also a GP:

<div class="eq-box">

$$f \mid \mathcal{D} \sim \mathcal{GP}\!\left(\mu_*(\cdot),\; k_*(\cdot, \cdot)\right)$$

$$\mu_*(x^*) = \mathbf{k}_*^\top \mathbf{K}_y^{-1} \mathbf{y}$$

$$\sigma^2_*(x^*) = k(x^*, x^*) - \mathbf{k}_*^\top \mathbf{K}_y^{-1} \mathbf{k}_*$$

</div>

where $\mathbf{k}_* = [k(x^*, x_1), \ldots, k(x^*, x_N)]^\top$ and $\mathbf{K}_y = \mathbf{K}(X,X) + \sigma_n^2\mathbf{I}$.

---

# Interpreting the Posterior

$$\mu_*(x^*) = \mathbf{k}_*^\top \underbrace{\mathbf{K}_y^{-1} \mathbf{y}}_{\boldsymbol{\alpha}}$$

- $\mu_*$ is a **weighted sum of kernel evaluations** — each training point $x_i$ contributes $\alpha_i \cdot k(x^*, x_i)$
- Points far from $x^*$ contribute little (their kernel value is small)
- Points near $x^*$ dominate — **local prediction**

$$\sigma^2_*(x^*) = \underbrace{k(x^*,x^*)}_{\text{prior variance}} - \underbrace{\mathbf{k}_*^\top\mathbf{K}_y^{-1}\mathbf{k}_*}_{\text{information gained}}$$

- Far from all data: $\mathbf{k}_* \approx \mathbf{0}$ → $\sigma^2_* \approx k(x^*,x^*)$ (recover prior variance)
- Near data: the second term is large → $\sigma^2_*$ is small

---

# GP Algorithm

| Step | Operation | Cost |
|------|-----------|------|
| 1 | Build $\mathbf{K}_y = K(X,X) + \sigma_n^2\mathbf{I}$ | $O(N^2)$ |
| 2 | Cholesky: $\mathbf{K}_y = \mathbf{L}\mathbf{L}^\top$ | $O(N^3)$ |
| 3 | Solve: $\boldsymbol{\alpha} = \mathbf{K}_y^{-1}\mathbf{y}$ (via backsubstitution) | $O(N^2)$ |
| 4 | Compute $\mathbf{k}_* = K(x^*, X)$ | $O(N)$ |
| 5 | Predict: $\mu_* = \mathbf{k}_*^\top\boldsymbol{\alpha}$ | $O(N)$ |
| 6 | Predict: $\sigma^2_* = k(x^*,x^*) - \mathbf{k}_*^\top(\mathbf{L}\mathbf{L}^\top)^{-1}\mathbf{k}_*$ | $O(N^2)$ |

Bottleneck: **Cholesky at $O(N^3)$**. Memory: $O(N^2)$ for $\mathbf{K}_y$.

---

# Seeing the Posterior Update

<div class="demo">
<strong>PAI Tool — Explore Step 2:</strong> Load "Posterior Update". Click one point on the right (x ≈ 3, y ≈ 1). Then click directly on an existing data point in the middle.

- First click: band collapses to near-zero at new point, widens everywhere else
- Clicking on existing point: almost nothing changes — GP already "knew" the value there
- Far from all data: band width identical to the prior

These three behaviours are the posterior variance formula made visible.
</div>

---

<!-- _class: divider -->

# Part 5
## Two Sources of Uncertainty

---

# Noise vs. Epistemic Uncertainty

The **predictive distribution** for a noisy observation $y^* = f(x^*) + \varepsilon$:

$$p(y^* \mid x^*, \mathcal{D}) = \mathcal{N}(\mu_*(x^*),\; \sigma^2_*(x^*) + \sigma_n^2)$$

<div class="two-col">
<div class="col insight">

**Epistemic** $\sigma^2_*(x^*)$
- Uncertainty about the function $f$
- Collapses near training data
- Recovers prior variance far away
- Can be reduced with more data

</div>
<div class="col warning">

**Aleatoric** $\sigma_n^2$
- Observation noise
- Constant everywhere
- Cannot be reduced
- Correctly modelled by not interpolating

</div>
</div>

<div class="demo">
<strong>PAI Tool — Explore Step 4:</strong> Load "Noisy Observations". Drag σ²_n from 0.01 → 1.50. Low: GP threads exactly through points. High: smooth regression curve — individual scatter attributed to noise, not signal. Both are correct for their respective σ²_n.
</div>

---

<!-- _class: divider -->

# Part 6
## Hyperparameter Optimisation

---

# The Log Marginal Likelihood

Hyperparameters $\theta = (\ell, \sigma_f^2, \sigma_n^2)$ are chosen by maximising:

<div class="eq-box">

$$\log p(\mathbf{y} \mid \mathbf{X}, \theta) = -\frac{1}{2}\mathbf{y}^\top\mathbf{K}_y^{-1}\mathbf{y} - \frac{1}{2}\log|\mathbf{K}_y| - \frac{N}{2}\log 2\pi$$

</div>

| Term | Role |
|---|---|
| $-\frac{1}{2}\mathbf{y}^\top\mathbf{K}_y^{-1}\mathbf{y}$ | Data fit — pushes $\theta$ toward interpolating the data |
| $-\frac{1}{2}\log\|\mathbf{K}_y\|$ | Complexity penalty — penalises over-parameterised models |
| $-\frac{N}{2}\log 2\pi$ | Constant |

The **automatic trade-off** between fit and complexity is built in — no cross-validation needed.

---

# Type-II Maximum Likelihood

**Training:**
$$\theta^* = \arg\max_\theta \log p(\mathbf{y} \mid \mathbf{X}, \theta)$$

Optimised via gradient ascent (analytic gradients available through the Cholesky factors).

This is called **Type-II MLE** (or *empirical Bayes*):
- We are not marginalising over $\theta$ (that would be fully Bayesian hyperparameter learning)
- We are finding the *most likely* $\theta$ under the marginal likelihood
- Practical and usually sufficient — the marginal likelihood is a much better objective than held-out log-likelihood

<div class="demo">
<strong>PAI Tool — Explore Step 7:</strong> Load "Noisy Observations". Set ℓ=0.30, σ²_f=0.50, σ²_n=1.50. Read log p(y|θ). Click "Optimise Hyperparams". Sliders jump to data-driven values; log-likelihood rises; fit visibly improves.
</div>

---

# Periodic Kernel: Structured Extrapolation

<div class="demo">
<strong>PAI Tool — Explore Step 6:</strong> Load "Periodic Pattern". Enable Posterior samples. Drag Period p from 3.14 → 1.50 → 6.00, watching the extrapolated region (x > 3).

The GP confidently extrapolates the oscillation far beyond the data range. No polynomial or RBF GP could do this without the right inductive bias — the periodic kernel encodes structural prior knowledge that propagates into unobserved regions.
</div>

<div class="insight">
<strong>Kernel engineering:</strong> The most powerful use of GPs in practice is encoding domain knowledge through kernel choice. Additive kernels (RBF + Periodic), product kernels, and deep kernels allow expressing complex prior beliefs that improve generalisation dramatically over generic models.
</div>

---

<!-- _class: divider -->

# Part 7
## Scalability

---

# The $O(N^3)$ Problem

The Cholesky decomposition of $\mathbf{K}_y \in \mathbb{R}^{N \times N}$ costs $O(N^3)$:

| $N$ | Approximate cost |
|---|---|
| 100 | Instant |
| 1,000 | ~1 second |
| 10,000 | ~15 minutes |
| 100,000 | ~weeks |

**Standard GPs are intractable for $N > 10,000$.**

---

# Sparse GP Approximations

**Inducing points:** Replace $N$ training points with $M \ll N$ inducing points $\mathbf{Z}$:

$$p(f \mid \mathcal{D}) \approx p(f \mid \mathbf{u})\,p(\mathbf{u} \mid \mathcal{D})$$

where $\mathbf{u} = f(\mathbf{Z})$ are function values at inducing points.

| Method | Cost | Notes |
|---|---|---|
| Exact GP | $O(N^3)$ | Exact |
| FITC / SGPR | $O(NM^2)$ | Inducing points, deterministic |
| SVGP | $O(NM^2)$ per batch | Stochastic, scales to $N = 10^6$ |
| Deep GP | $O(NM^2 L)$ | $L$ layers, handles non-stationarity |

In all cases: choose $\mathbf{Z}$ and $M$ to balance accuracy vs. cost.

---

# Summary: BLR vs. GP

| | Bayesian Linear Regression | Gaussian Process |
|---|---|---|
| Prior over | Weight vectors $\mathbf{w}$ | Functions $f(\cdot)$ |
| Basis | Explicit $\boldsymbol{\phi}(x)$, finite $d$ | Implicit, infinite-dimensional |
| Expressiveness | Limited to $d$ basis functions | Infinite-dimensional (nonparametric) |
| Hyperparams | $\sigma_w^2, \sigma_n^2$ | $\ell, \sigma_f^2, \sigma_n^2$ (via MLE) |
| Cost | $O(Nd^2 + d^3)$ | $O(N^3)$ |
| Connection | GP with $k(x,x') = \sigma_w^2\boldsymbol{\phi}(x)^\top\boldsymbol{\phi}(x')$ | Generalises BLR |

> *GPs are BLR in an infinite-dimensional feature space, with the kernel encoding the implicit basis.*

---

# Key Equations at a Glance

<div class="eq-box">

**GP Prior:** $f \sim \mathcal{GP}(0, k(x,x'))$

**Posterior mean:** $\mu_*(x^*) = \mathbf{k}_*^\top\mathbf{K}_y^{-1}\mathbf{y}$

**Posterior variance:** $\sigma^2_*(x^*) = k(x^*,x^*) - \mathbf{k}_*^\top\mathbf{K}_y^{-1}\mathbf{k}_*$

**Log marginal likelihood:** $\log p(\mathbf{y}\mid\mathbf{X},\theta) = -\tfrac{1}{2}\mathbf{y}^\top\mathbf{K}_y^{-1}\mathbf{y} - \tfrac{1}{2}\log|\mathbf{K}_y| - \tfrac{N}{2}\log 2\pi$

</div>

<div class="small">
$\mathbf{K}_y = K(X,X) + \sigma_n^2\mathbf{I}$ · $\mathbf{k}_* = [k(x^*,x_1),\ldots,k(x^*,x_N)]^\top$ · $\theta = (\ell, \sigma_f^2, \sigma_n^2)$
</div>

---

# References

- **Rasmussen & Williams** — *Gaussian Processes for Machine Learning* (2006). The canonical GP textbook. Full text free online. `gaussianprocess.org/gpml`
- **Krause & Hübotter** — *Probabilistic AI* (2025), Ch. 4. GP regression, kernel design, marginal likelihood optimisation.
- **Görtler, Kehlbeck & Deussen** — *A Visual Exploration of Gaussian Processes* (Distill, 2019). Beautiful interactive tutorial. `distill.pub/2019/visual-exploration-gaussian-processes`
- **Murphy** — *Probabilistic Machine Learning: Advanced Topics* (2023), Ch. 18. Sparse GPs, deep kernels.

<div class="insight">
<strong>The grand arc:</strong> BN (discrete graphical models) → BLR (Gaussian, parametric) → Kalman Filter (Gaussian, sequential) → GP (Gaussian, nonparametric). Each module is the same probabilistic reasoning machinery applied in a different structural setting.
</div>

---

<!-- _class: title -->

# Questions?

## PAI Interactive Tool

Explore tab → Module 4: Gaussian Processes

*Seven steps: prior over functions, posterior collapse, kernel design, noise interpolation, length scale, periodic extrapolation, Type-II MLE.*
