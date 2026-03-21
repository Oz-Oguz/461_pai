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
    border-bottom: 3px solid #7c3aed; 
    padding-bottom: 8px; 
    margin-bottom: 16px; 
    margin-top: 0;
    padding-top: 0;
  }
  h2 { font-size: 1.05em; color: #7c3aed; margin-top: 0; margin-bottom: 8px; }
  ul { margin-top: 6px; }
  li { margin-bottom: 4px; }

  section.title {
    background: linear-gradient(135deg, #2e1065 0%, #7c3aed 100%);
    color: white;
    display: flex; flex-direction: column; justify-content: center;
  }
  section.title h1 { color: white; border-color: rgba(255,255,255,0.35); font-size: 1.8em; }
  section.title h2 { color: rgba(255,255,255,0.8); font-size: 1em; }
  section.title p  { color: rgba(255,255,255,0.65); font-size: 0.75em; }

  section.divider {
    background: #3b0764;
    color: white;
    display: flex; flex-direction: column; justify-content: center;
  }
  section.divider h1 { color: white; border-color: rgba(255,255,255,0.3); }
  section.divider p  { color: rgba(255,255,255,0.7); font-size: 0.85em; }

  .demo { background: #faf5ff; border-left: 4px solid #7c3aed; padding: 8px 12px; border-radius: 0 6px 6px 0; font-size: 0.78em; margin-top: 10px; }
  .demo strong { color: #6d28d9; }
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
  .algorithm .kw  { color: #7c3aed; font-weight: 700; }
  .algorithm .cm  { color: #94a3b8; font-style: italic; }
  .algorithm .ind  { display: inline-block; padding-left: 1.5em; vertical-align: top; }
  .algorithm .ind2 { display: inline-block; padding-left: 3.0em; vertical-align: top; }
  .algorithm p { margin: 0 !important; padding: 0 !important; }
  .algorithm p:empty { display: none !important; }
---

<!-- _class: title -->

# Bayesian Networks

## Probabilistic AI — Chapter 1

Özgür S. Öğüz · Bilkent University

---

<!-- _paginate: false -->

# Agenda

1. **Foundations of probability** — frequentist vs. Bayesian, sum/product rules
2. **Why graphical models?** — the curse of dimensionality
3. **DAG structure** — nodes, edges, conditional independence
4. **Joint factorisation & CPTs** — including conjugate priors
5. **D-separation** — independence from structure, Reichenbach
6. **Inference by Enumeration** — exact, exponential
7. **Variable Elimination** — exact, efficient, tensor view
8. **Reasoning patterns** — causal, diagnostic, explaining away
9. **Plate notation** — repeated variables, bridge to supervised learning
10. **Summary** — the road from discrete BN to continuous BLR

---

<!-- _class: divider -->

# Part 0
## Foundations of Probability

---

# Frequentist vs. Bayesian

<div class="two-col">
<div class="col">

**Frequentist interpretation**

Probability = limiting frequency of an event in repeated experiments.

- $P(\text{heads}) = 0.5$ means: in infinitely many flips, half are heads
- Only meaningful for *repeatable* experiments
- Parameters are fixed but unknown constants

</div>
<div class="col">

**Bayesian interpretation**

Probability = degree of belief (rational agent's uncertainty).

- $P(\text{rain tomorrow}) = 0.3$ is a statement about *our knowledge*
- Applies to one-off events and hypotheses
- Parameters have distributions — we update beliefs with data

</div>
</div>

<div class="insight">
<strong>Key insight (Cox, 1946):</strong> If you want a consistent system for quantifying uncertainty that reduces to Boolean logic in the certain case, you <em>must</em> use probability theory. "Bayesian probability is the direct mathematical generalisation of Aristotelian logic." — Jaynes
</div>

---

# Probability as Generalised Logic

Probability theory rests on two rules and one powerful consequence:

<div class="eq-box">

**Sum rule (marginalisation):** $\quad p(x) = \sum_y p(x, y)$

**Product rule (chain rule):** $\quad p(x, y) = p(x \mid y)\,p(y)$

</div>

Every result in this course — Bayes' theorem, marginal likelihood, variable elimination, the Kalman gain — follows from applying these two rules.

<div class="insight">
<strong>Machine learning as mechanised Bayesian logic:</strong> Given a model (likelihood + prior), a computer applies the sum and product rules to compute posteriors. ML = automating probabilistic inference on Turing machines.
</div>

---

# The Sum and Product Rules → Bayes' Theorem

Combining the product rule in both directions:

$$p(x, y) = p(y \mid x)\,p(x) = p(x \mid y)\,p(y)$$

Rearranging yields **Bayes' theorem** — one step, no magic:

<div class="eq-box">

$$p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta)\,p(\theta)}{p(\mathcal{D})} \qquad \text{where } p(\mathcal{D}) = \sum_\theta p(\mathcal{D} \mid \theta)\,p(\theta)$$

</div>

| Symbol | Name | Role |
|---|---|---|
| $p(\theta)$ | Prior | What we believe before data |
| $p(\mathcal{D} \mid \theta)$ | Likelihood | How probable is the data given $\theta$? |
| $p(\theta \mid \mathcal{D})$ | Posterior | Updated belief after seeing data |
| $p(\mathcal{D})$ | Evidence | Normalising constant (model comparison) |

---

<!-- _class: divider -->

# Part 1
## Why Graphical Models?

---

# The Curse of Dimensionality

For $n$ binary random variables, the full joint distribution has $2^n - 1$ parameters.

| $n$ | Parameters in full joint |
|-----|-------------------------|
| 5 | 31 |
| 10 | 1,023 |
| 20 | $\approx 10^6$ |
| 30 | $\approx 10^9$ |

**Problem:** Neither storing, estimating, nor computing with $2^n$ numbers is feasible.

**Key observation:** Most variables are *conditionally independent* of most others given a small set of parents. Graphical models exploit this structure.

---

# The Solution: Structured Factorisation

Instead of one big table, represent the joint as a product of *small local tables*:

<div class="eq-box">

$$p(x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n} p\bigl(x_i \mid \text{parents}(x_i)\bigr)$$

</div>

Each factor $p(x_i \mid \text{parents}(x_i))$ involves only $x_i$ and its (few) parents.

**Example:** $n = 10$ variables, each with at most 2 parents → at most $4 \times 10 = 40$ parameters (vs. 1,023 for the full joint).

---

<!-- _class: divider -->

# Part 2
## Bayesian Network Structure

---

# Definition: Bayesian Network

A **Bayesian Network** is a pair $(G, \Theta)$ where:

- $G = (V, E)$ is a **Directed Acyclic Graph (DAG)**
  - Nodes $V$: random variables
  - Edges $E$: direct probabilistic dependencies ($X \to Y$ means $X$ is a parent of $Y$)
- $\Theta$: a set of **CPTs** (Conditional Probability Tables), one per node

**The Markov condition:** Each variable is conditionally independent of its non-descendants given its parents.

This is both a *representational* and an *independence* assumption.

---

# Reading the Graph

<div class="two-col">
<div class="col">

**Three structural patterns:**

1. **Chain** $A \to B \to C$
   - $A \perp C \mid B$ (B blocks)

2. **Fork** $A \leftarrow B \rightarrow C$
   - $A \perp C \mid B$ (B blocks)

3. **V-structure / Collider** $A \to C \leftarrow B$
   - $A \perp B$ marginally
   - $A \not\perp B \mid C$ (C activates!)

</div>
<div class="col warning">

**The collider is counterintuitive:**

Two independent causes become dependent once we observe their common effect.

Observing the effect "activates" the path — explaining away.

</div>
</div>

---

<!-- _class: divider -->

# Part 3
## Joint Factorisation & CPTs

---

# The Factorisation Property

For any BN with topological ordering $x_1, \ldots, x_n$:

<div class="eq-box">

$$p(x_1, \ldots, x_n) = \prod_{i=1}^{n} p(x_i \mid \mathbf{x}_{\text{pa}(i)})$$

</div>

This is not an approximation — it is exact for any distribution that satisfies the Markov condition w.r.t. the DAG.

**Parameter count:** $\sum_{i} k_i \cdot \prod_{j \in \text{pa}(i)} k_j$ where $k_i$ = number of states of variable $i$.

For the Robot Battery example (Level 1): 3 nodes, all binary → just **5 parameters** (vs. $2^3 - 1 = 7$ for the full joint).

---

# Conditional Probability Tables (CPTs)

Each node stores $p(X_i \mid \mathbf{x}_{\text{pa}(i)})$ for all parent combinations:

| Battery | P(WarningLight = On) |
|---------|---------------------|
| Full    | 0.05 |
| Low     | 0.90 |

**Root nodes** (no parents) store a simple prior: $p(\text{Battery} = \text{Full}) = 0.80$.

**Interpretation:** CPT entries encode *domain knowledge* — how probable is each outcome given what we know about its direct causes?

<div class="demo">
<strong>PAI Tool — Explore Step 1:</strong> Load Level 1. Click VoltageSensor → set High. Watch P(Battery=Full) jump from 80% to ~95%. This is Bayes' theorem: the sensor CPT determines how informative the observation is.
</div>

---

# From Discrete Tables to Continuous Priors

So far, CPT entries are fixed numbers. But what if we want to **learn** them from data?

**Beta-Binomial conjugacy** — the simplest case. For a binary variable with unknown parameter $\theta$:

<div class="eq-box">

**Prior:** $\theta \sim \text{Beta}(\alpha, \beta)$ &emsp; **Likelihood:** $x \sim \text{Bernoulli}(\theta)$ &emsp; **Posterior:** $\theta \mid x \sim \text{Beta}(\alpha + x,\; \beta + 1 - x)$

</div>

<div class="two-col">
<div class="col">

**Interpretation:** $\alpha$ and $\beta$ are **pseudo-counts** — imaginary data you pretend you've seen before collecting real observations.

- $\text{Beta}(1,1)$ = uniform (no prior opinion)
- $\text{Beta}(10,2)$ = strong belief $\theta \approx 0.83$

</div>
<div class="col insight">

**Generalisation:** For a variable with $K$ states, replace Beta with **Dirichlet**:

$$\boldsymbol{\theta} \sim \text{Dir}(\alpha_1, \ldots, \alpha_K)$$

Each observation increments the corresponding pseudo-count. The posterior is always another Dirichlet — **conjugacy**.

</div>
</div>

---

<!-- _class: divider -->

# Part 4
## D-Separation

---

# D-Separation: Independence from Structure

$X$ and $Y$ are **d-separated** by set $Z$ (written $X \perp\!\!\!\perp_d Y \mid Z$) if every path between them is **blocked** by $Z$.

A path is blocked by $Z$ if it contains:

| Path type | Blocking condition |
|---|---|
| Chain $A \to B \to C$ | $B \in Z$ |
| Fork $A \leftarrow B \rightarrow C$ | $B \in Z$ |
| Collider $A \to B \leftarrow C$ | $B \notin Z$ **and** no descendant of $B$ in $Z$ |

**If d-separated:** $p(X, Y \mid Z) = p(X \mid Z)\,p(Y \mid Z)$ — they are truly independent.

**If not d-separated:** They may (or may not) be dependent — the graph is silent on the magnitude.

---

# D-Separation Examples

**Chain:** $\text{Rain} \to \text{WetGrass} \to \text{Slippery}$

- $\text{Rain} \perp\!\!\!\perp \text{Slippery} \mid \text{WetGrass}$ ✓ (WetGrass blocks)
- $\text{Rain} \not\perp\!\!\!\perp \text{Slippery}$ (path unblocked without conditioning)

**Fork:** $\text{Season} \to \text{Rain}$, $\text{Season} \to \text{Sprinkler}$

- $\text{Rain} \perp\!\!\!\perp \text{Sprinkler} \mid \text{Season}$ ✓ (Season blocks)
- $\text{Rain} \not\perp\!\!\!\perp \text{Sprinkler}$ (correlated through Season)

**Collider:** $\text{Rain} \to \text{WetGrass} \leftarrow \text{Sprinkler}$

- $\text{Rain} \perp\!\!\!\perp \text{Sprinkler}$ ✓ (unobserved collider blocks)
- $\text{Rain} \not\perp\!\!\!\perp \text{Sprinkler} \mid \text{WetGrass}$ ← **explaining away!**

---

# Reichenbach's Common Cause Principle

> *"If two random variables $X$ and $Y$ are statistically dependent, then either $X$ causes $Y$, $Y$ causes $X$, or there exists a common cause $Z$ that causes both."*
> — Hans Reichenbach (1956)

This principle **formalises the intuition behind d-separation:**

- Statistical dependence implies a **causal path** (direct or through a common ancestor)
- If we block all such paths by conditioning on the right variables, dependence vanishes
- D-separation is the graphical algorithm for checking this

<div class="two-col">
<div class="col insight">

**From correlation to causation:** Reichenbach tells us that observed correlations are not arbitrary — they must have a structural explanation in the DAG.

</div>
<div class="col warning">

**The catch:** Reichenbach does not tell us the *direction* of causation. Observational data alone cannot distinguish $X \to Y$ from $Y \to X$ (unless colliders break the symmetry). This is the domain of **causal discovery**.

</div>
</div>

---

# The Collider in Action

<div class="demo">
<strong>PAI Tool — Explore Step 5 (D-separation):</strong>

Load "d-sep ③ V-Structure: Blocked". Drag Object prior 0.4 → 0.9. Watch Weather node → **no change** (collider Camera is unobserved, blocks the path).

Load "d-sep ④ V-Structure: Active". Same drag → Weather **reacts**. Observing Camera opens the path between Object and Weather.

This is the collider effect, impossible to see from a static formula.
</div>

<div class="insight">
<strong>Practical consequence:</strong> When you control for a collider (common effect) in a regression, you can introduce spurious correlations between its causes that did not exist in the raw data. This is a major source of errors in causal inference.
</div>

---

<!-- _class: divider -->

# Part 5
## Exact Inference

---

# Inference by Enumeration

**Goal:** Compute $P(Q \mid E = e)$ for query $Q$ given evidence $E$.

**Strategy:** Use the joint and marginalise:

$$P(Q \mid E = e) = \frac{P(Q, E=e)}{P(E=e)} = \frac{\sum_{\mathbf{h}} P(Q, \mathbf{h}, E=e)}{\sum_{q,\mathbf{h}} P(Q=q, \mathbf{h}, E=e)}$$

**Algorithm:**
1. Generate the full truth table (all $2^n$ states)
2. Filter to states consistent with evidence $E = e$
3. Sum joint weights for each value of $Q$
4. Normalise

**Complexity:** $O(2^n)$ — feasible only for small networks (≤ 20 variables or so).

---

# Variable Elimination

**Key insight:** The sum over all assignments can be *factored* using the BN structure.

Rather than enumerating all $2^n$ states, process one variable at a time:

<div class="algorithm">

<span class="kw">for each</span> variable $Z_j$ in elimination order $\pi$:<br>
<span class="ind">1. Collect all factors (CPTs) mentioning $Z_j$</span><br>
<span class="ind">2. Multiply them into a single product factor $\psi$</span><br>
<span class="ind">3. Sum out $Z_j$: $\psi'(\mathbf{x}_{\setminus Z_j}) = \sum_{Z_j} \psi(\mathbf{x})$</span>

</div>

The resulting factor $\psi'$ no longer mentions $Z_j$ — we eliminated it.

**Complexity:** $O(n \cdot 2^w)$ where $w$ = **treewidth** (max scope size of any intermediate factor).

For trees: $w = 1$ → linear! For general DAGs: minimising $w$ is NP-hard.

---

# VE: A Simple Example

Network: $A \to B \to C$. Query: $P(C)$. Eliminate $A$, then $B$.

$$P(C) = \sum_B \sum_A P(A)\,P(B\mid A)\,P(C\mid B)$$

**Step 1 — Eliminate $A$:**
$$\psi_B(B) = \sum_A P(A)\,P(B\mid A) \qquad \leftarrow \text{this is just } P(B)$$

**Step 2 — Eliminate $B$:**
$$P(C) = \sum_B \psi_B(B)\,P(C\mid B)$$

At each step we only touch **local factors** — we never build the full $2^3$ joint table.

<div class="demo">
<strong>PAI Tool — Explore Step 6 (Level 3):</strong> Load "Rough Terrain". Hit "Solve Step-by-Step". Watch each factor being multiplied and marginalised across the 6-node Robot Mission network.
</div>

---

# Inference as Tensor Operations

Reframe factors as **multi-dimensional arrays** (tensors). VE becomes a sequence of tensor contractions:

<div class="two-col">
<div class="col">

**Factor = tensor**
- $P(A)$ → vector of length $|A|$
- $P(B \mid A)$ → matrix of shape $|A| \times |B|$
- Joint product = outer product of tensors

</div>
<div class="col">

**Marginalisation = contraction**
- Summing out $A$ = contracting the $A$-axis
- VE = a sequence of `einsum` operations
- Elimination order = contraction order

</div>
</div>

<div class="algorithm">

<span class="cm"># Variable Elimination as einsum (3-node chain A→B→C)</span><br><br><br>
<span class="kw">import</span> numpy <span class="kw">as</span> np<br><br><br>
psi_B = np.einsum('a,ab->b', P_A, P_B_given_A) &ensp; <span class="cm"># eliminate A</span><br><br><br>
P_C &ensp;= np.einsum('b,bc->c', psi_B, P_C_given_B) <span class="cm"># eliminate B</span>

</div>

The treewidth $w$ determines the size of the largest intermediate tensor: $O(2^w)$ entries.

---

<!-- _class: divider -->

# Part 6
## Reasoning Patterns

---

# Three Types of Reasoning

<div class="three-col">
<div class="col">

**Causal**
Top → Down
(predictive)

Set a cause, predict effects.

*"If it rains, how likely is the grass wet?"*

</div>
<div class="col">

**Diagnostic**
Bottom → Up
(abductive)

Observe an effect, infer cause.

*"Grass is wet — was it rain or sprinkler?"*

</div>
<div class="col">

**Inter-causal**
Across
(explaining away)

One cause reduces another's probability.

*"Sprinkler is on — rain less likely."*

</div>
</div>

A single BN model supports all three — inference is just conditioned probability.

<div class="demo">
<strong>PAI Tool — Explore Step 2:</strong> Set WarningLight=On (diagnostic). Note P(Battery=Low) rises. Reset, set Battery=Low (causal). Compare how WarningLight probability shifts in each direction.
</div>

---

# Explaining Away in Detail

**Setup:** Two causes $C_1, C_2$ of a common effect $E$. Marginally: $C_1 \perp C_2$.

**After observing $E$:** $C_1 \not\perp C_2 \mid E$ — knowing one cause reduces the probability of the other.

**Intuition:** If $E$ is observed, the total probability of causes must explain $E$. Knowing that $C_1$ occurred already "accounts for" $E$, leaving less need for $C_2$.

<div class="demo">
<strong>PAI Tool — Explore Step 4:</strong> Level 2 "Explaining Away": Camera=None, LiDAR=Detected, Weather=Fog → P(Pedestrian) is high (fog explains camera failure). Switch Weather→Clear: camera failure can't be blamed on fog → P(Pedestrian) drops sharply. Same evidence, different inference — caused by attributing blame to one explanation.
</div>

---

# Multi-Hop Cascades

<div class="demo">
<strong>PAI Tool — Explore Step 6 (Level 3):</strong>

Load "Rough Terrain" → note P(Mission=Abort).
Add Battery=Low.

Rough Terrain activates Sensors→Localization→Mission.
Battery=Low activates an independent path via Navigation.
**Two paths compound nonlinearly** — Mission Abort jumps more than either cause alone.

This is how a robot safety monitor reasons about concurrent failures.
</div>

<div class="demo">
<strong>PAI Tool — Explore Step 7:</strong>

Load "Mission Aborted". Observe upward shifts in P(Terrain=Rough) and P(Battery=Low). Then set Navigation=Success — P(Localization=Drifted) spikes via explaining away. Six nodes, backward inference, automatic.
</div>

---

<!-- _class: divider -->

# Part 7
## From Discrete BN to Continuous BLR

---

# Plate Notation

When a BN has $N$ repeated i.i.d. variables, we use **plate notation** instead of drawing $N$ identical copies:

<div class="two-col">
<div class="col">

**Without plate:**

$\theta \to x_1$, $\theta \to x_2$, $\ldots$, $\theta \to x_N$

(Impractical for large $N$)

</div>
<div class="col">

**With plate:**

A single box labelled $N$ containing $x_i$, with $\theta$ outside pointing in.

The plate says: "repeat this subgraph $N$ times, all sharing the same $\theta$."

</div>
</div>

<div class="insight">
<strong>This is supervised learning as a BN:</strong> A latent parameter node $\theta$ (or $\mathbf{w}$) connects to $N$ observed data nodes inside a plate. The prior on $\theta$ is the root CPT; the likelihood is the edge $\theta \to x_i$. Inference = computing the posterior $p(\theta \mid x_{1:N})$.
</div>

---

# From Discrete to Continuous: The Road to BLR

<div class="two-col">
<div class="col">

**Bayesian Networks (Module 1)**

- Variables: **discrete** (binary, categorical)
- Parameters: **CPTs** (tables)
- Inference: **sums** ($\sum$)
- Algorithms: VE, enumeration

</div>
<div class="col">

**Bayesian Linear Regression (Module 2)**

- Variables: **continuous** (Gaussian)
- Parameters: **weight vector** $\mathbf{w}$
- Inference: **integrals** ($\int$)
- Algorithms: analytic Gaussian posterior

</div>
</div>

<div class="eq-box">

**Same Bayes' theorem, same factorisation, same posterior update:**

$$\underbrace{p(\theta \mid \mathcal{D})}_{\text{posterior}} = \frac{\overbrace{p(\mathcal{D} \mid \theta)}^{\text{likelihood}} \;\cdot\; \overbrace{p(\theta)}^{\text{prior}}}{\underbrace{p(\mathcal{D})}_{\text{evidence}}}$$

The math changes from **sums to integrals** — the logic is identical.

</div>

---

<!-- _class: title -->

# Part 8: Approximate Inference via Sampling

---

# Why Approximate Inference?

**Exact methods hit a wall for large networks:**

| Method | Cost | Bottleneck |
|---|---|---|
| Enumeration | $O(2^n)$ | Exponential in variables |
| Variable Elimination | $O(n \cdot 2^w)$ | Exponential in treewidth |
| Real-world nets | 20–100+ nodes | Often NP-hard |

**Sampling offers a practical alternative:**

<div class="two-col">
<div class="col">

- Cost: $O(N \cdot n)$ — linear in samples and nodes
- Memory: $O(n)$ — just one sample at a time
- Scales to networks with thousands of variables
- Converges to true posterior as $N \to \infty$

</div>
<div class="col">

<div class="insight">

**Key idea:** Draw $N$ samples from a sampling distribution, count frequencies, normalize.

As $N \to \infty$, the estimate converges to the true distribution $P$.

</div>

</div>
</div>

---

# Sampling a Categorical Distribution

**Step 1:** Draw $u \sim \text{Uniform}[0, 1)$

**Step 2:** Map $u$ to a state via cumulative probabilities

<div class="eq-box">

$$\text{Sample } C \sim P(C): \quad u \mapsto \begin{cases} \text{red} & 0 \le u < 0.6 \\ \text{green} & 0.6 \le u < 0.7 \\ \text{blue} & 0.7 \le u < 1 \end{cases}$$

</div>

This is the **inverse CDF** (quantile) method. In Python: `numpy.random.choice(states, p=probs)`.

<div class="algorithm">

<span class="cm"># Sample from categorical P(C) = {red: 0.6, green: 0.1, blue: 0.3}</span><br>
<span class="kw">import</span> numpy <span class="kw">as</span> np<br>
probs = np.array([0.6, 0.1, 0.3])<br>
sample = np.random.choice([<span class="cm">'red'</span>, <span class="cm">'green'</span>, <span class="cm">'blue'</span>], p=probs)

</div>

<div class="insight">

This generalizes directly to sampling any node $X_i$ from its CPT row $P(X_i \mid \text{parents}(X_i))$.

</div>

---

# Prior Sampling

Sample a **complete assignment** by walking down the DAG in topological order:

<div class="algorithm">

<span class="kw">for</span> $i = 1, 2, \ldots, n$ (topological order):<br>
<span class="ind">Sample $X_i \sim P(X_i \mid \text{parents}(X_i))$</span><br>
<span class="kw">return</span> $(X_1, X_2, \ldots, X_n)$

</div>

**Correctness:** Each sample is drawn from the BN's joint:

$$S_\text{PS}(x_1,\ldots,x_n) = \prod_i P(x_i \mid \text{parents}(X_i)) = P(x_1,\ldots,x_n)$$

**Marginal estimate** from $N$ samples: $\hat{P}(X{=}s) = \frac{\#\{x_i = s\}}{N} \xrightarrow{N \to \infty} P(X{=}s)$

<div class="warning">

**Limitation:** Prior sampling ignores any evidence $\mathbf{e}$. It estimates the **prior** marginals $P(Q)$, not the posterior $P(Q \mid \mathbf{e})$.

</div>

---

# Rejection Sampling

**Condition on evidence by discarding inconsistent samples:**

<div class="algorithm">

<span class="kw">for</span> $i = 1, 2, \ldots, n$ (topological order):<br>
<span class="ind">Sample $X_i \sim P(X_i \mid \text{parents}(X_i))$</span><br>
<span class="ind"><span class="kw">if</span> $X_i$ contradicts evidence $\mathbf{e}$: <span class="kw">reject</span> this sample, start over</span><br>
<span class="kw">return</span> $(X_1, \ldots, X_n)$ &ensp;<span class="cm"># accepted only if consistent with e</span>

</div>

**Consistent** — in the limit, accepted samples come from $P(\mathbf{x} \mid \mathbf{e})$.

<div class="warning">

**Drawback: waste.** Acceptance rate $\approx P(\mathbf{e})$. For rare evidence (e.g. $P(\mathbf{e}) = 0.01$), only 1 in 100 samples is kept — the other 99 are thrown away. Efficiency collapses as evidence gets more specific.

</div>

<div class="demo">
<strong>PAI Tool — Sampling:</strong> Load <em>Weather</em> model → set evidence <em>Rain=False, WetGrass=True</em> → run Rejection Sampling with N=500. Watch the acceptance rate badge — it will be very low (rare combination). Compare with Likelihood Weighting.
</div>

---

# Likelihood Weighting

**Fix evidence, sample the rest, weight each sample by how well it predicts the evidence:**

<div class="algorithm">

$w \leftarrow 1.0$<br>
<span class="kw">for</span> $i = 1, 2, \ldots, n$ (topological order):<br>
<span class="ind"><span class="kw">if</span> $X_i$ is an evidence variable:</span><br>
<span class="ind2">$x_i \leftarrow$ observed value &ensp;;&ensp; $w \mathrel{*}= P(x_i \mid \text{parents}(X_i))$</span><br>
<span class="ind"><span class="kw">else</span>: sample $x_i \sim P(X_i \mid \text{parents}(X_i))$</span><br>
<span class="kw">return</span> $(x_1,\ldots,x_n),\, w$

</div>

**Weighted marginal:**

$$\hat{P}(X{=}s \mid \mathbf{e}) = \frac{\sum_i w_i \cdot \mathbf{1}[x_i^{(i)} = s]}{\sum_i w_i}$$

**All $N$ samples are used** — no rejection waste. But upstream variables are sampled from the *prior*, unaware of downstream evidence → weights can collapse for leaf-node evidence.

---

# Gibbs Sampling (MCMC)

**Wander through the joint space, one variable at a time:**

<div class="algorithm">

Initialise all non-evidence variables randomly; fix $\mathbf{e}$<br>
<span class="kw">repeat</span> $T$ times:<br>
<span class="ind">Pick a non-evidence variable $X_i$ uniformly at random</span><br>
<span class="ind">Resample $X_i \sim P(X_i \mid \text{Markov blanket}(X_i))$</span><br>
<span class="kw">return</span> collected samples (after burn-in)

</div>

**Markov blanket** of $X_i$ = parents + children + co-parents (children's other parents):

$$P(X_i \mid \text{MB}(X_i)) \;\propto\; P(X_i \mid \text{pa}(X_i)) \prod_{Y_j \in \text{children}(X_i)} P(Y_j \mid \text{pa}(Y_j))$$

<div class="insight">

**Why this beats LW for leaf evidence:** every variable conditions on *all* evidence through repeated resampling. Downstream evidence propagates back to upstream variables through the Markov chain.

</div>

<div class="warning">

**Burn-in:** the chain starts from a *random* state — early samples reflect that initialisation, not the target posterior. We discard the first $B$ samples (the burn-in period) to let the chain "forget" where it started. Rule of thumb: $B \approx 10\text{–}20\%$ of total steps. The PAI tool lets you set $B = 0$ to see the effect directly.

</div>

---

# Algorithm Comparison

| Algorithm | Estimates | All samples used | Handles leaf evidence | Correlated samples |
|---|---|---|---|---|
| Prior Sampling | $P(Q)$ | ✓ | ✗ (no evidence) | ✗ |
| Rejection Sampling | $P(Q \mid \mathbf{e})$ | ✗ (rejects) | ✓ (but slow) | ✗ |
| Likelihood Weighting | $P(Q \mid \mathbf{e})$ | ✓ | Partial (weights collapse) | ✗ |
| Gibbs Sampling | $P(Q \mid \mathbf{e})$ | ✓ | ✓ | ✓ (burn-in needed) |

**Rule of thumb:** LW for upstream evidence; Gibbs for downstream/leaf evidence.

<div class="demo">
<strong>PAI Tool — Sampling tab:</strong> Load <em>Weather</em> model → <em>Wet Grass Observed</em> scenario → compare all four algorithms with N=500. Watch the convergence chart: LW and Gibbs converge fastest; rejection sampling shows low acceptance rate; prior sampling ignores the evidence entirely.
</div>

---

# Summary

| Property | Bayesian Network |
|---|---|
| Representation | DAG + CPTs |
| Joint distribution | $\prod_i p(x_i \mid \mathbf{x}_{\text{pa}(i)})$ |
| Independence | D-separation from graph structure |
| Exact inference | VE: $O(n \cdot 2^w)$; Enumeration: $O(2^n)$ |
| Supports | Causal + diagnostic + inter-causal reasoning |

**Key equations:**

<div class="eq-box">

$$p(x_1,\ldots,x_n) = \prod_{i} p(x_i \mid \mathbf{x}_{\text{pa}(i)}) \qquad P(H\mid E) = \frac{P(E\mid H)\,P(H)}{P(E)}$$

</div>

---

# References

- **Krause & Hübotter** — *Probabilistic Artificial Intelligence* (2025), §1.1.8. Primary course text.
- **Koller & Friedman** — *Probabilistic Graphical Models* (2009). The definitive reference. Chapters 3–10 cover BNs, d-separation, VE, and approximate inference.
- **Russell & Norvig** — *AI: A Modern Approach* (4th ed., 2020), Ch. 12–13. Accessible introduction.
- **Pearl (1988)** — *Probabilistic Reasoning in Intelligent Systems.* Original BN paper — d-separation, belief propagation.
- **Hennig** — *Probabilistic Machine Learning* (Tübingen, 2023), Lectures 1–3. Probability as generalised logic, array-centric inference.
- **Jaynes** — *Probability Theory: The Logic of Science* (2003). Cox's theorem, maximum entropy.

<div class="insight">
<strong>Coming up — Module 2:</strong> Bayesian Networks encode structure as a DAG over discrete variables. The next module extends Bayesian reasoning to <em>continuous</em> variables with Gaussian posteriors — Bayesian Linear Regression.
</div>

---

<!-- _class: title -->

# Questions?

## PAI Interactive Tool

Explore tab → Module 1: Bayesian Networks

*Seven guided steps: prior-posterior update, causal/diagnostic reasoning, explaining away, d-separation, VE trace.*
