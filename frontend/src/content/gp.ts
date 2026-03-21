import type { ModuleContent } from './types';

export const gpContent: ModuleContent = {
  id: 'gp',
  title: 'Gaussian Processes',
  subtitle: 'Non-Parametric ML · Krause Ch. 4',

  overview: {
    paragraphs: [
      'A Gaussian Process (GP) defines a prior directly over functions. Instead of placing a prior over a finite weight vector (as in BLR), a GP specifies a joint Gaussian distribution over function values at any finite collection of inputs — allowing infinitely flexible function representations.',
      'The kernel k(x, x\') is the heart of a GP. It encodes prior beliefs: how correlated should f(x) and f(x\') be? The RBF kernel produces infinitely smooth functions; Matérn 3/2 produces once-differentiable ones; Linear recovers Bayesian linear regression; Periodic extrapolates oscillatory patterns indefinitely.',
      'After observing data, the GP posterior is still a GP — with a closed-form mean and covariance. The posterior mean threads through the observations; the posterior variance narrows near the data and widens in unseen regions. This gives principled, calibrated uncertainty quantification.',
    ],
    equations: [
      {
        label: 'GP Prior',
        latex:
          'f \\sim \\mathcal{GP}\\!\\left(0,\\; k(x, x\'\\right))',
        explanation:
          'The prior places a Gaussian distribution over every finite set of function evaluations. The mean is zero; the covariance is the kernel.',
      },
      {
        label: 'Posterior Mean',
        latex:
          '\\mu_*(x^*) = \\mathbf{k}_*^\\top \\mathbf{K}_y^{-1} \\mathbf{y}',
        explanation:
          'k* = K(x*, X). The posterior mean is a weighted sum of observations — weights determined by how correlated x* is with each training point.',
      },
      {
        label: 'Posterior Variance',
        latex:
          '\\sigma^2_*(x^*) = k(x^*,x^*) - \\mathbf{k}_*^\\top \\mathbf{K}_y^{-1} \\mathbf{k}_*',
        explanation:
          'Prior variance minus the information gained from data. Far from observations the second term ≈ 0 and we recover the prior variance.',
      },
      {
        label: 'Log Marginal Likelihood',
        latex:
          '\\log p(\\mathbf{y}\\mid\\mathbf{X},\\theta) = -\\tfrac{1}{2}\\mathbf{y}^\\top\\mathbf{K}_y^{-1}\\mathbf{y} - \\tfrac{1}{2}\\log|\\mathbf{K}_y| - \\tfrac{N}{2}\\log 2\\pi',
        explanation:
          'Maximising this w.r.t. θ = (ℓ, σ²_f, σ²_n) automatically balances data fit (first term) against model complexity (second term).',
      },
    ],
    keyInsights: [
      'With no data: the shaded band is the prior — each sample is a plausible function consistent with the kernel.',
      'Add data points → the posterior mean snaps to fit them; uncertainty collapses near observations.',
      'Length scale ℓ controls smoothness: short ℓ = wiggly, long ℓ = smooth. Try dragging the slider.',
      'Noise variance σ²_n controls how tightly the GP interpolates: high σ²_n = smooth fit (trusts the kernel more than individual points).',
      'The Kernel panel below the plot shows k(x_ref, x) — the correlation profile. Each kernel has a characteristic shape.',
      'Click "Optimise Hyperparams" to find ℓ, σ²_f, σ²_n that maximise log p(y|θ). Watch the sliders jump to the data-driven values.',
    ],
  },

  algorithm: {
    name: 'GP Posterior Prediction',
    complexity: 'O(N³) — dominated by Cholesky decomposition of K_y',
    steps: [
      { kind: 'header', indent: 0, text: 'Algorithm: GP Regression' },
      { kind: 'io', indent: 0, text: 'Input: training set 𝒟 = {(xᵢ, yᵢ)}, kernel k(·,·), hyperparams θ' },
      { kind: 'io', indent: 0, text: 'Output: predictive distribution p(f* | x*, 𝒟, θ)' },
      { kind: 'divider', indent: 0, text: '' },
      { kind: 'step', indent: 0, text: '1. Build noisy kernel matrix:' },
      { kind: 'step', indent: 1, math: true, text: '\\mathbf{K}_y = \\mathbf{K}(\\mathbf{X},\\mathbf{X}) + \\sigma^2_n \\mathbf{I}' },
      { kind: 'step', indent: 0, text: '2. Cholesky decomposition (numerical stability):' },
      { kind: 'step', indent: 1, math: true, text: '\\mathbf{K}_y = \\mathbf{L}\\mathbf{L}^\\top' },
      { kind: 'step', indent: 0, text: '3. Solve for weights via back-substitution:' },
      { kind: 'step', indent: 1, math: true, text: '\\boldsymbol{\\alpha} = \\mathbf{K}_y^{-1}\\mathbf{y} \\quad (\\text{via } \\mathbf{L}\\mathbf{L}^\\top\\boldsymbol{\\alpha} = \\mathbf{y})' },
      { kind: 'step', indent: 0, text: '4. Compute cross-kernel vector k* = K(x*, X)' },
      { kind: 'step', indent: 0, text: '5. Predictive mean and variance:' },
      { kind: 'step', indent: 1, math: true, text: '\\mu^* = \\mathbf{k}_*^\\top\\boldsymbol{\\alpha}' },
      { kind: 'step', indent: 1, math: true, text: '\\sigma^{*2} = k(x^*,x^*) - \\mathbf{k}_*^\\top(\\mathbf{L}\\mathbf{L}^\\top)^{-1}\\mathbf{k}_*' },
      { kind: 'return', indent: 0, text: '𝒩(μ*, σ*²)' },
    ],
    note:
      'The O(N³) cost comes from step 2. For large N use sparse GPs or inducing-point approximations (SGPR, SVGP). scipy.linalg.cho_factor/cho_solve is used here for numerical robustness.',
  },

  references: [
    {
      label: 'Gaussian Processes for Machine Learning — Rasmussen & Williams',
      authors: 'C.E. Rasmussen, C.K.I. Williams (MIT Press, 2006)',
      url: 'http://gaussianprocess.org/gpml/',
      description: 'The canonical GP textbook. Chapters 2–5 cover regression, classification, model selection, and approximations.',
      type: 'book',
    },
    {
      label: 'Probabilistic AI — Krause & Hübotter',
      authors: 'A. Krause, J. Hübotter (ETH Zürich, 2025)',
      url: 'https://las.inf.ethz.ch/pai',
      description: 'Chapter 4: Gaussian Processes. Covers GP regression, kernel design, marginal likelihood optimisation.',
      type: 'book',
    },
    {
      label: 'A Visual Exploration of Gaussian Processes — distill.pub',
      authors: 'J. Görtler, R. Kehlbeck, O. Deussen (2019)',
      url: 'https://distill.pub/2019/visual-exploration-gaussian-processes/',
      description: 'Beautiful interactive introduction to GPs with live visualisations of kernels and posteriors.',
      type: 'tutorial',
    },
  ],

  explore: [
    {
      title: 'The prior as a distribution over functions',
      instruction: 'Load \'Clear (Prior)\' — no data points. Enable \'Posterior samples\' in the Visualisation section. Drag the Length Scale ℓ slider from 1.0 down to 0.10, then slowly up to 3.00.',
      watch: 'At ℓ = 0.10 the sampled functions are wildly wiggly — each point is nearly independent of its neighbours. At ℓ = 3.00 the samples are long gentle curves. ℓ is your prior assumption about "how far does information travel?" — a question that is invisible in the posterior formula but becomes obvious the moment you see the prior samples change shape. No data is needed to see this; it is purely the kernel\'s definition of correlation.',
    },
    {
      title: 'Posterior update: uncertainty collapses at data',
      instruction: 'Load \'Posterior Update\'. Click one new point on the right side of the plot (around x = 3, y = 1). Then click directly on top of an existing data point in the middle.',
      watch: 'After the first click: the uncertainty band collapses to near-zero at your new point and widens everywhere else. After clicking on an existing point: almost nothing changes — the GP already "knew" the function value there and a second observation adds no information. Far from all data the band width is identical to the prior. These three behaviours are the posterior variance formula made visible.',
    },
    {
      title: 'The kernel panel is the design choice',
      instruction: 'Load \'Clear (Prior)\'. Switch the Kernel selector through RBF → Matérn 3/2 → Linear, watching the Kernel Shape panel below the main plot each time.',
      watch: 'RBF: smooth Gaussian bell — functions are infinitely differentiable. Matérn 3/2: a sharper-peaked bell — functions are once-differentiable and can have kinks. Linear: a straight line through the origin — the GP becomes exactly Bayesian Linear Regression. The kernel panel IS showing you the correlation k(0, x) between x=0 and every other point. Choosing a kernel means choosing what kind of world you believe you live in before seeing any data.',
    },
    {
      title: 'Noise variance: interpolation vs. regression',
      instruction: 'Load \'Noisy Observations\'. Drag σ²_n from 0.01 to 1.50 and back.',
      watch: 'At σ²_n = 0.01 the posterior mean threads through each data point exactly — the GP treats every observation as a perfect measurement. At σ²_n = 1.50 the mean becomes a smooth curve that ignores individual scatter. This is not a failure: setting σ²_n = 1.50 is the correct Bayesian thing to do when your sensor truly is that noisy. The GP is not averaging away information; it is correctly down-weighting unreliable observations.',
    },
    {
      title: 'Length scale and kernel panel in sync',
      instruction: 'Load \'Short Length Scale\' (ℓ = 0.3). Look at the Kernel panel — note the narrow bell. Now slowly drag ℓ up to 2.00, watching both the main plot and the kernel panel simultaneously.',
      watch: 'The bell in the kernel panel widens exactly as the posterior mean smooths out. A narrow bell means "only my immediate neighbours correlate with me" → wiggly, local fits. A wide bell means "points far away are still correlated with me" → smooth, global behaviour. You can read the entire smoothness structure of the posterior from a single cross-section of the kernel — the panel makes that cross-section live.',
    },
    {
      title: 'Periodic kernel: structured extrapolation',
      instruction: 'Load \'Periodic Pattern\'. Enable \'Posterior samples\'. Drag the Period p slider from 3.14 down to 1.50, then up to 6.00, watching the extrapolated region (x > 3) carefully.',
      watch: 'The GP confidently extrapolates the oscillation far beyond the data range, and the extrapolated period matches p exactly. No polynomial or RBF GP could do this: without the right inductive bias (a periodic kernel), extrapolation reverts to the prior mean. This is the central lesson of kernel design: the right kernel encodes structural prior knowledge that propagates into regions where data is absent.',
    },
    {
      title: 'Optimise Hyperparams: automatic kernel learning',
      instruction: 'Load \'Noisy Observations\'. Manually set ℓ = 0.30, σ²_f = 0.50, σ²_n = 1.50 (deliberately poor values). Read the log p(y|θ) from the Hyperparameter Summary. Now click \'Optimise Hyperparams\'. Compare the before/after log-likelihood and the slider positions.',
      watch: 'The sliders jump to values that maximise log marginal likelihood. The evidence rises and the fit visibly improves. Notice that the optimiser found a balance you would not have guessed by hand: it simultaneously increased data fit (first term of log p) and penalised model complexity (second term). This is Type-II MLE — the standard way GPs learn kernel hyperparameters. It is the GP equivalent of cross-validation, but analytic and computed in one shot.',
    },
  ],
};
