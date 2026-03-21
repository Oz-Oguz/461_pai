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
    color: #0f766e; 
    border-bottom: 3px solid #14b8a6; 
    padding-bottom: 8px; 
    margin-bottom: 16px; 
    margin-top: 0;
    padding-top: 0;
  }
  h2 { font-size: 1.05em; color: #0d9488; margin-top: 0; margin-bottom: 8px; }
  ul { margin-top: 6px; }
  li { margin-bottom: 4px; }

  section.title {
    background: linear-gradient(135deg, #042f2e 0%, #0d9488 100%);
    color: white;
    display: flex; flex-direction: column; justify-content: center;
  }
  section.title h1 { color: white; border-color: rgba(255,255,255,0.35); font-size: 1.8em; }
  section.title h2 { color: rgba(255,255,255,0.82); font-size: 1em; }
  section.title p  { color: rgba(255,255,255,0.68); font-size: 0.75em; }

  section.divider {
    background: #115e59;
    color: white;
    display: flex; flex-direction: column; justify-content: center;
  }
  section.divider h1 { color: white; border-color: rgba(255,255,255,0.3); }
  section.divider p  { color: rgba(255,255,255,0.75); font-size: 0.85em; }

  .demo { background: #f0fdfa; border-left: 4px solid #14b8a6; padding: 8px 12px; border-radius: 0 6px 6px 0; font-size: 0.78em; margin-top: 10px; }
  .demo strong { color: #0f766e; }
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
  .algorithm .kw { color: #0d9488; font-weight: 700; }
  .algorithm .cm { color: #94a3b8; font-style: italic; }
  .algorithm p { margin: 0; }
  .algorithm p:empty { display: none; }
---

<!-- _class: title -->

# Hidden Markov Models

## Probabilistic AI - Chapter 2 (Companion Deck)

Ozgur S. Oguz - Bilkent University

---

<!-- _paginate: false -->

# Agenda

1. **From Markov chains to HMMs** - what is hidden vs. observed?
2. **Three inference questions** - filtering, smoothing, decoding
3. **Forward algorithm** - recursive Bayesian filtering
4. **Viterbi algorithm** - MAP state sequence via dynamic programming
5. **Forward-Backward** - using future evidence for smoothing
6. **Learning parameters** - supervised counts and Baum-Welch (EM)
7. **Robot localization** - grid-world Bayes filtering
8. **Connections and references** - DBNs, particle filters, modern sequence models

---

<!-- _class: divider -->

# Part 1
## Markov Assumptions and HMM Structure

---

# Why HMMs?

Many temporal systems have:
- A **latent state** we care about but cannot observe directly
- A noisy **observation** that gives partial evidence about that state

Examples:
- Weather (hidden) from mood or umbrella observations
- Robot position (hidden) from wall sensors
- POS tags (hidden) from words in a sentence
- Health condition (hidden) from medical test outcomes

<div class="insight">
<strong>Core idea:</strong> HMMs separate how the world evolves (transition model) from how sensors/language reveal it (emission model).
</div>

---

# Markov Chain Refresher

A first-order Markov chain assumes the **Markov property**:

$$P(X_t \mid X_{1:t-1}) = P(X_t \mid X_{t-1})$$

This models systems where the next state depends only on the current state, not the full history.

**Classic applications:**
- Random walks (1D: −4, −3, ..., +4; step left/right with fixed probability)
- Language models (n-grams: predict next word from previous words)
- Web browsing (PageRank: stationary distribution over pages)

---

In an HMM, we add observations $E_t$ such that:

$$P(E_t \mid X_{1:t}, E_{1:t-1}) = P(E_t \mid X_t)$$

<div class="eq-box">

Joint HMM factorization:
 and CPTs

**Dynamic Bayesian Network view:**

$$X_{t-1} \rightarrow X_t \rightarrow X_{t+1}, \qquad X_t \rightarrow E_t$$

**Parameters (stored as matrices or tables):**

| Component | Symbol | Meaning |
|---|---|---|
| Initial distribution | $\pi$ | $P(X_1)$ |
| Transition matrix | $A$ | $A_{ij}=P(X_t=j\mid X_{t-1}=i)$ |
| Emission matrix | $B$ | $B_{jk}=P(E_t=k\mid X_t=j)$ |

**Example: Weather HMM** with two states $\{S(\text{un}), R(\text{ain})\}$:
<!-- |---|---|---| -->

| $X_{t-1}$ | $P(X_t=\text{Sun} \mid X_{t-1})$ | $P(X_t=\text{Rain} \mid X_{t-1})$ |

--- 

# Stationary Distributions

For a fixed transition matrix, long sequences converge to a **stationary distribution** $\pi_\infty$ satisfying: 
$$\footnotesize \pi_\infty = \pi_\infty^T A$$

**Example**: Weather Markov chain with transition matrix 
$$\footnotesize A = \begin{pmatrix} 0.9 & 0.1 \\ 0.3 & 0.7 \end{pmatrix}$$

Solving $\pi_\infty = \pi_\infty A$ with $\pi_\infty[0] + \pi_\infty[1]=1$:

$$\footnotesize 0.9p + 0.3(1-p) = p \implies p = 0.75$$

So $\pi_\infty = [0.75, 0.25]$ regardless of starting distribution—stationary probability of Sun is 75%.

<!-- | Sun | 0.9 | 0.1 |
| Rain | 0.3 | 0.7 | -->

Initial: $P(X_1) = [0.5, 0.5]$ (start equally likely in either state)
<!-- - Vertical edges: **sensor / observation model** -->

| Component | Symbol | Meaning |
|---|---|---|
| Initial distribution | $\pi$ | $P(X_1)$ |
| Transition matrix | $A$ | $A_{ij}=P(X_t=j\mid X_{t-1}=i)$ |
| Emission matrix | $B$ | $B_{jk}=P(E_t=k\mid X_t=j)$ |

---

<!-- _class: divider -->

# Part 2
## The Three Inference Tasks

---

# Filtering, Smoothing, Decoding

Given evidence $e_{1:T}$, common tasks are:

1. **Filtering** (online):
$$P(X_t \mid e_{1:t})$$

2. **Smoothing** (offline):
$$P(X_t \mid e_{1:T})$$

3. **Decoding** (single best path):
$$\arg\max_{x_{1:T}} P(x_{1:T} \mid e_{1:T})$$

<div class="warning">
Filtering gives a distribution now; Viterbi gives one best path overall. They answer different questions and can disagree.
</div>

---

# Companion Mapping to the PAI App

Your HMM app already covers the three core pedagogical examples:

- **Example 1: Weather-Mood** -> Filtering via Forward
- **Example 2: Viterbi Decoding** -> MAP sequence
- **Example 3: Robot Localization** -> Bayes filter on a 2D discrete grid

<div class="demo">
<strong>In class flow:</strong> run Example 1 to build intuition, Example 2 to contrast sum vs max, then Example 3 to make belief dynamics spatial and tangible.
</div>

---

<!-- _class: divider -->

# Part 3
## Forward Algorithm (Filtering)

---

# Forward Recursion

Define forward messages:

$$\alpha_t(i) = P(e_{1:t}, X_t=i)$$

Recurrence:

<div class="eq-box">

$$\alpha_1(i)=\pi_i\,B_i(e_1)$$

$$\alpha_t(j)=B_j(e_t)\sum_i \alpha_{t-1}(i)A_{ij}$$

</div>

Normalized belief:

$$P(X_t=j\mid e_{1:t}) = \frac{\alpha_t(j)}{\sum_k \alpha_t(k)}$$

---

# Forward as Predict-Update

Equivalent Bayes filter view:

1. **Predict**
$$\bar{bel}_t(x_t)=\sum_{x_{t-1}}P(x_t\mid x_{t-1})\,bel_{t-1}(x_{t-1})$$

2. **Update**
$$bel_t(x_t)=\eta\,P(e_t\mid x_t)\,\bar{bel}_t(x_t)$$

3. **Normalize** with $\eta^{-1}=\sum_{x_t}P(e_t\mid x_t)\bar{bel}_t(x_t)$

<div class="insight">
<strong>Computational win:</strong> O($T N^2$) time instead of O($N^T$) enumeration over state trajectories.
</div>

---

# Forward Algorithm (Pseudo-code)

<div class="algorithm">

<span class="kw">input</span>: transition $A$, emission $B$, initial $\pi$, observations $e_{1:T}$

<span cla Algorithm: State Trellis

Viterbi finds the most likely path through a **state trellis**:

```
     t=0      t=1      t=2      t=3
      Sun      Sun      Sun      Sun
       |  \   / |  \   / |  \   / |
       |   \ /  |   \ /  |   \ /  |
      Rain    Rain    Rain    Rain
```

Each node is a state at time $t$; each edge has weight $P(e_t|X_t) \cdot A_{i,j}$.

**Viterbi DP** computes best (highest-weight) path:

<div class="eq-box">

$$\delta_1(j)=\pi_j B_j(e_1)$$

$$\delta_t(j)=B_j(e_t)\max_i\left[\delta_{t-1}(i)A_{ij}\right]$$

$$\psi_t(j)=\arg\max_i\left[\delta_{t-1}(i)A_{ij}\right]$$

</div>

Backtrack: start at $x_**sum**-product | **max**-product |
| Question answered | Sum over all paths | Single best path |
| Output | Distribution $P(X_t \mid e_{1:t})$ | Sequence $\arg\max_{x_{1:T}} P(x_{1:T} \mid e_{1:T})$ |
| Uncertainty? | Yes, retains alternatives | No, commits to one path |
| Use case | Tracking, filtering | Speech recognition, segmentation |

**Key insight**: Forward sums across all paths; Viterbi picks just the max path. They can give different marginal state estimates!

<div class="demo">
<strong>PAI app Example 2 scenario:</strong> Compare Forward belief at time $t$ with the Viterbi path's marginal state at $t$—they differ when multiple high-probability paths exist
# Viterbi: Max Over Paths, Not Sum

Viterbi dynamic program:

$$\delta_t(j)=\max_{x_{1:t-1}} P(x_{1:t-1},X_t=j,e_{1:t})$$

Recurrence:

<div class="eq-box">

$$\delta_1(j)=\pi_j B_j(e_1)$$

$$\delta_t(j)=B_j(e_t)\max_i\left[\delta_{t-1}(i)A_{ij}\right]$$

$$\psi_t(j)=\arg\max_i\left[\delta_{t-1}(i)A_{ij}\right]$$

</div>

Backtrack from $x_T^*=\arg\max_j\delta_T(j)$ using $\psi_t$.

---

# Forward vs. Viterbi at a Glance

| Property | Forward | Viterbi |
|---|---|---|
| Semiring operation | sum-product | max-product |
| Output | full posterior over states | one best sequence |
| Uncertainty retained? | Yes | No |
| Typical use | tracking, monitoring | segmentation, decoding |
Online Belief Updates: Two Step View

Breaking down the forward algorithm into separate conceptual steps:

**Predict** (time step):
$$\bar{bel}_t(x_t) = \sum_{x_{t-1}} P(x_t \mid x_{t-1}) \, bel_{t-1}(x_{t-1})$$

**Update** (observe evidence):
$$bel_t(x_t) = \eta \, P(e_t \mid x_t) \, \bar{bel}_t(x_t)$$

The forward algorithm does both at once without explicit normalization until the end (for efficiency).

<div class="insight">
<strong>Intuition:</strong> Predict spreads belief across possible next states (entropy often increases). Observe concentrates it where the evidence points (entropy decreases).
</div>

---

# Smoothing and Future Evidence

When does smoothing help most?

- Ambiguous local evidence at time $t$
- Strong, disambiguating evidence later at $t+k$

Example intuition:
- Early observation looks consistent with both Sun and Rain
- Later observations strongly indicate persistent Rain
- Smoothing shifts earlier posterior significantly toward Rain

<div class="insight">
<strong>Teaching pattern:</strong> show filtering and smoothing side-by-side on one sequence to make "future evidence" concrete. Popular exam question: which states' beliefs change most between filtering and smoothing?
# Why Smoothing?

Filtering uses past evidence only. But once the sequence ends, future observations can revise earlier beliefs.

Backward message:

$$\beta_t(i)=P(e_{t+1:T}\mid X_t=i)$$

Recurrence:

$$\beta_t(i)=\sum_j A_{ij}B_j(e_{t+1})\beta_{t+1}(j), \quad \beta_T(i)=1$$

Smoothed posterior:

<div class="eq-box">

$$P(X_t=i\mid e_{1:T}) \propto \alpha_t(i)\beta_t(i)$$

</div>

---

# Typical Exam and Project Question

When does smoothing help most?

- Ambiguous local evidence at time $t$
- Strong, disambiguating evidence later at $t+k$

Example intuition:
- Early mood looks like both Sun and Rain
- Later sequence strongly indicates persistent Rain
- Smoothing shifts earlier posterior toward Rain

<div class="insight">
<strong>Teaching pattern:</strong> show filtering and smoothing side-by-side on one sequence to make "future evidence" concrete.
</div>

---

<!-- _class: divider -->

# Part 6
## Learning HMM Parameters

---

# Parameter Learning Modes

1. **Supervised** (states observed):
- Count transitions and emissions
- Apply Laplace/Dirichlet smoothing

2. **Unsupervised** (states hidden):
- Use **Baum-Welch** = EM for HMMs

EM cycle:
- E-step: compute expected transition/emission counts via forward-backward
- M-step: re-estimate $A, B, \pi$ from those expected counts

<div class="warning">
EM improves likelihood monotonically but may converge to local optima; initialization matters.
</div>

---

# Baum-Welch Quantities

Expected occupancy and transition statistics:

$$\gamma_t(i)=P(X_t=i\mid e_{1:T})$$

$$\xi_t(i,j)=P(X_t=i, X_{t+1}=j\mid e_{1:T})$$

Updates:

<div class="eq-box">

$$\hat{A}_{ij}=\frac{\sum_{t=1}^{T-1}\xi_t(i,j)}{\sum_{t=1}^{T-1}\gamma_t(i)}$$

$$\hat{B}_{j}(k)=\frac{\sum_{t: e_t=k}\gamma_t(j)}{\sum_{t=1}^{T}\gamma_t(j)}$$

</div>

---

<!-- _class: divider -->

# Part 7
## Robot Localization and Bayes Filtering

---

# From 1D HMM to 2D Grid Localization

State becomes cell index $(r,c)$ on a map.

- Transition model: action command + motion noise
- Emission model: sensor signature likelihood at each cell
- Belief: categorical distribution over all free cells

Core loop is unchanged:

$$bel_t \leftarrow \text{normalize}\left( P(e_t\mid x_t) \odot \sum_{x_{t-1}}P(x_t\mid x_{t-1})bel_{t-1} \right)$$

---

# Entropy as Uncertainty Meter

For discrete belief $bel_t(x)$:

$$H(bel_t)=-\sum_x bel_t(x)\log_2 bel_t(x)$$

Interpretation in the app:
- **Move** step tends to diffuse mass -> higher entropy
- **Sense** step tends to concentrate mass -> lower entropy

<div class="demo">
<strong>PAI app tie-in (Example 3):</strong> press movement arrows repeatedly, then Sense; watch heatmap sharpen and entropy drop.
</div>

---

<!-- _class: divider -->

# Part 8
## How Top Courses Teach HMMs

---

# Common HMM Pedagogy Pattern

Across major courses, the recurring structure is:

1. Markov chains and temporal factorization
2. HMM inference tasks (filtering/smoothing/decoding)
3. Forward and Viterbi dynamic programming
4. Particle filtering or DBN extensions
5. Parameter learning (EM/Baum-Welch), often in advanced tracks

Observed in:
- Berkeley CS188 (HMM -> Forward/Viterbi -> Particle Filtering)
- MIT OCW 6.867 (Markov Models -> HMM -> HMM continuation)
- Stanford CS228 notes (latent variable inference + EM framing)

---

# Suggested Teaching Sequence for Your Course

Week module arc (90-120 min):

- **Lecture 1**: HMM model + Forward + in-class weather demo
- **Lecture 2**: Viterbi + smoothing + robot localization exercise
- **Optional advanced session**: Baum-Welch and numerical stability (log-space)

Homework ideas:
- Implement Forward and Viterbi from scratch
- Compare filtered vs smoothed marginals on same sequence
- Tune transition/emission for localization map and report entropy trends

---

# Numerical Stability Note (Practical)

Long sequences underflow in probability space.

Two standard fixes:
- **Scaling factors** each timestep
- **Log-space** recursion with log-sum-exp

<div class="algorithm">

<span class="kw">log-forward</span>: $\ell_t(j)=\log B_j(e_t)+\operatorname{LSE}_i\left(\ell_{t-1}(i)+\log A_{ij}\right)$

<sKey Takeaways

1. **HMMs = Markov chains + observations**: separate state evolution from sensor models
2. **Forward algorithm**: online, tracks full distribution over states  
3. **Viterbi algorithm**: offline, finds best single path through state trellis
4. **Smoothing**: uses future evidence to refine past state beliefs
5. **EM/Baum-Welch**: learn models from unlabeled sequences
6. **Practical**: use log-space arithmetic for long sequences to avoid underflow

---

# Questions?

## PAI Interactive Tool

Module: **Hidden Markov Models**

✓ Forward algorithm (Weather-Mood filtering)  
✓ Viterbi decoding (best path with arrows)  
✓ Robot localization (2D grid Bayes filter with entropy)

---

# Summary

<div class="two-col">
<div class="col">

**Conceptual takeaways**
- HMM = latent dynamics + noisy observations
- Forward: online belief tracking
- Viterbi: best-path decoding
- Smoothing: revise past using future

</div>
<div class="col">

**Engineering takeaways**
- Dynamic programming makes inference tractable
- Normalize carefully and use log-space for long sequences
- Match algorithm to question: distribution vs single path

</div>
</div>

---

# References

Core textbooks:
- Krause and Hubotter, *Probabilistic Artificial Intelligence* (HMM and sequential inference chapters)
- Murphy, *Probabilistic Machine Learning: An Introduction* (state-space models, HMM inference and EM)
- Murphy, *Probabilistic Machine Learning: Advanced Topics* (structured latent-variable extensions)

Course resources used for topic alignment:
- Berkeley CS188 schedule/textbook: HMMs, Forward, Viterbi, particle filtering
- MIT OCW 6.867 lecture notes: Markov models and HMM sequence
- Stanford CS228 notes: latent variable inference and EM framing

Classic papers/books:
- Rabiner (1989), "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition"
- Bishop (2006), *Pattern Recognition and Machine Learning* (Ch. 13)

---

<!-- _class: title -->

# Questions?

## PAI Interactive Tool

Module: Hidden Markov Models

Forward, Viterbi, and Robot Localization demos
