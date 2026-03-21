import type { ModuleContent } from './types';

export const blrContent: ModuleContent = {
  id: 'blr',
  title: 'Bayesian Linear Regression',
  subtitle: 'Probabilistic ML · Chapter 2',

  overview: {
    paragraphs: [
      'Imagine calibrating a robot sensor: you take 5 readings at known reference values, then ask "what will it read at a temperature you never tested?" A point-estimate fit gives you a number but no warning. Bayesian Linear Regression (BLR) gives you a number and a band — narrow where you calibrated, wide where you extrapolate. This is the essence of BLR: place a Gaussian prior over the weight vector w, observe data, and compute a full posterior distribution p(w | X, y) rather than a single best-guess line.',
      'The key distinction is between two sources of uncertainty: epistemic uncertainty (uncertainty about the weights, which shrinks as more data arrives) and aleatoric uncertainty (irreducible observation noise σ²_n). The predictive distribution integrates over all plausible weight vectors, yielding a mean prediction plus a data-dependent uncertainty band.',
      'Because both the prior and the likelihood are Gaussian, the posterior is also Gaussian — a closed-form result known as Gaussian conjugacy. This means BLR is exact (no approximation needed) and computationally efficient. Try adding points along a line: watch the uncertainty collapse around data and remain wide in unobserved regions.',
    ],
    equations: [
      {
        label: 'Posterior over weights (Gaussian conjugacy)',
        latex: '\\boldsymbol{\\Sigma}_w = \\bigl(\\beta\\,\\boldsymbol{\\Phi}^\\top\\boldsymbol{\\Phi} + \\alpha\\mathbf{I}\\bigr)^{-1}, \\quad \\boldsymbol{\\mu}_w = \\beta\\,\\boldsymbol{\\Sigma}_w\\,\\boldsymbol{\\Phi}^\\top\\mathbf{y}',
        explanation: 'α = 1/σ²_w (prior precision), β = 1/σ²_n (noise precision). The posterior mean μ_w is the regularised least-squares solution; Σ_w captures remaining parameter uncertainty.',
      },
      {
        label: 'Predictive distribution',
        latex: 'p(y^* \\mid x^*, \\mathcal{D}) = \\mathcal{N}\\!\\left(y^*;\\; \\boldsymbol{\\mu}_w^\\top\\boldsymbol{\\phi}(x^*),\\; \\boldsymbol{\\phi}(x^*)^\\top\\boldsymbol{\\Sigma}_w\\boldsymbol{\\phi}(x^*) + \\beta^{-1}\\right)',
        explanation: 'The variance has two terms: φ(x*)ᵀΣ_wφ(x*) is the epistemic uncertainty (shrinks with data) and β⁻¹ = σ²_n is the irreducible aleatoric noise.',
      },
      {
        label: 'Law of Total Probability (marginalising weights)',
        latex: 'p(y^* \\mid x^*, \\mathcal{D}) = \\int p(y^* \\mid x^*, \\mathbf{w})\\, p(\\mathbf{w} \\mid \\mathcal{D})\\, d\\mathbf{w}',
        explanation: 'The predictive distribution is obtained by marginalising (integrating out) the weights under their posterior. For Gaussians this integral is analytic.',
      },
    ],
    keyInsights: [
      'With no data, the shaded band shows the prior predictive — broad Gaussian uncertainty everywhere.',
      'Add points along a line: the band collapses near observations but stays wide far from them. The Gaussian posterior contracts where data constrains it.',
      'Add noisy points: even at observed locations, some uncertainty remains — that is the irreducible aleatoric Gaussian noise σ²_n.',
      'Open the Weight-Space panel (degree = 1): the posterior is a 2D Gaussian ellipse over (w₀, w₁). Use the axis sliders to fix the range, then add data and watch the ellipse shrink.',
      'Increase degree to 3 or 4, then add few points: the uncertainty grows sharply at the edges — the multivariate Gaussian in weight space is poorly constrained.',
      'Increase prior variance σ²_w to allow more extreme weights; decrease it to enforce a "simple" model near zero. This is the Gaussian prior\'s width at work.',
    ],
  },

  algorithm: {
    name: 'Bayesian Linear Regression — Posterior Update',
    complexity: 'O(N·d² + d³)',
    steps: [
      { kind: 'header', indent: 0, text: 'Algorithm: BLR Posterior Update' },
      { kind: 'io', indent: 0, text: 'Input: data 𝒟 = {(xᵢ, yᵢ)}, prior α = 1/σ²_w, noise β = 1/σ²_n, degree d' },
      { kind: 'io', indent: 0, text: 'Output: posterior p(w | 𝒟) = 𝒩(μ_w, Σ_w)' },
      { kind: 'divider', indent: 0, text: '' },
      { kind: 'step',    indent: 0, text: '1. Build feature matrix Φ ∈ ℝ^{N×(d+1)}:' },
      { kind: 'step',    indent: 1, math: true, text: '\\Phi_{ij} = x_i^{j-1}, \\quad j = 1,\\ldots,d+1' },
      { kind: 'step',    indent: 0, text: '2. Compute posterior covariance:' },
      { kind: 'step',    indent: 1, math: true, text: '\\boldsymbol{\\Sigma}_w = \\bigl(\\beta\\boldsymbol{\\Phi}^\\top\\boldsymbol{\\Phi} + \\alpha\\mathbf{I}\\bigr)^{-1}' },
      { kind: 'step',    indent: 0, text: '3. Compute posterior mean:' },
      { kind: 'step',    indent: 1, math: true, text: '\\boldsymbol{\\mu}_w = \\beta\\,\\boldsymbol{\\Sigma}_w\\,\\boldsymbol{\\Phi}^\\top\\mathbf{y}' },
      { kind: 'step',    indent: 0, text: '4. For prediction at x*:' },
      { kind: 'step',    indent: 1, math: true, text: '\\mu_{y^*} = \\boldsymbol{\\mu}_w^\\top\\boldsymbol{\\phi}(x^*)' },
      { kind: 'step',    indent: 1, math: true, text: '\\sigma^2_{y^*} = \\boldsymbol{\\phi}(x^*)^\\top\\boldsymbol{\\Sigma}_w\\boldsymbol{\\phi}(x^*) + \\beta^{-1}' },
      { kind: 'return',  indent: 0, text: '𝒩(μ_w, Σ_w) and predictive 𝒩(μ_{y*}, σ²_{y*})' },
    ],
    note: 'Step 2 involves a (d+1)×(d+1) matrix inversion — O(d³). For high-dimensional or large-N settings, use the matrix inversion lemma (Woodbury identity) or sparse Gaussian Processes.',
  },

  references: [
    {
      label: 'Probabilistic Artificial Intelligence — Krause & Hübotter',
      authors: 'A. Krause, J. Hübotter (ETH Zürich, 2025)',
      url: 'https://arxiv.org/abs/2502.05244',
      description: 'Chapter 2: Bayesian Linear Regression. Covers weight-space view, aleatoric vs. epistemic uncertainty, non-linear regression via basis functions, and the function-space view leading to GPs.',
      type: 'book',
    },
    {
      label: 'Pattern Recognition and Machine Learning — Bishop',
      authors: 'C. M. Bishop (Springer, 2006)',
      url: 'https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/',
      description: 'Chapter 3.3: Bayesian Linear Regression. The canonical treatment with full derivations, evidence approximation, and model comparison via marginal likelihood.',
      type: 'book',
    },
    {
      label: 'Machine Learning: A Probabilistic Perspective — Murphy',
      authors: 'K. P. Murphy (MIT Press, 2012)',
      url: 'https://probml.github.io/pml-book/',
      description: 'Chapter 7.6: Bayesian Linear Regression. Accessible treatment with Python code examples in the companion notebooks (PML2).',
      type: 'book',
    },
    {
      label: 'Bayesian Linear Regression — Wikipedia',
      authors: 'Wikipedia contributors',
      url: 'https://en.wikipedia.org/wiki/Bayesian_linear_regression',
      description: 'Concise reference covering the conjugate prior derivation, predictive distribution, and connection to ridge regression.',
      type: 'wiki',
    },
  ],

  explore: [
    {
      title: 'Start here: calibration uncertainty',
      instruction: 'Load "Sensor Calibration". This models a robot sensor calibrated at 5 reference points in x ∈ [−1, 1] (e.g., 10°C–30°C of a joint-temperature probe). Look at the shaded uncertainty band across the full x range.',
      watch: 'The band is narrow inside [−1, 1] — the model is confident where it was trained. Outside that range the band widens noticeably: the model is telling you "I was not calibrated here — trust my reading less." This is BLR\'s core promise: silence is replaced by quantified uncertainty. Now drag σ²_n from 0.10 up to 0.50: the entire band lifts uniformly — noisier sensors leave more uncertainty everywhere, independent of whether a point was observed.',
    },
    {
      title: 'Gaussian in weight space: watch the ellipse shrink',
      instruction: 'Load "Sensor Calibration", set Degree = 1. Open the Weight-Space panel. Fix both axis sliders at [−4, 4] (they start there by default). Now manually click 3–5 additional data points near the existing calibration line.',
      watch: 'With fixed axes you can see the 1σ/2σ/3σ ellipse visibly contracting as each point arrives — the posterior Gaussian is tightening. If you had not fixed the axes, the view would auto-rescale and hide this shrinkage. The ellipse shape also rotates as correlated information arrives: slope and intercept become jointly constrained. This is the multivariate Gaussian update, visible directly.',
    },
    {
      title: 'The Gaussian prior: width controls model complexity',
      instruction: 'Load the default \'Clear (Prior)\' state. Enable \'Posterior samples\' in the Visualisation panel. Drag the σ²_w slider from 0.10 all the way to 5.00, then back down.',
      watch: 'At low σ²_w the sampled lines are nearly horizontal — the Gaussian prior N(0, σ²_w I) is tight around zero, so weights must be small. At high σ²_w the lines fan out wildly — the prior is almost flat, allowing any function. You are watching the Gaussian prior\'s standard deviation (its "spread") directly control the complexity of functions the model is willing to consider, before seeing a single data point.',
    },
    {
      title: 'Sequential updating: one click at a time',
      instruction: 'In \'Clear (Prior)\', set Degree = 1. Click one point near (0, 1). Watch the band. Then click two more points roughly collinear with the first. Watch after each click.',
      watch: 'The first click collapses the band only at that x — everywhere else the prior still dominates. Three collinear points collapse the band to a narrow corridor around the implied line, but notice the ends still flare. This is the posterior update equation live: each observation tightens Σ_w, and the tightening is sharpest where φ(x) is large.',
    },
    {
      title: 'Prior variance as regularisation strength',
      instruction: 'Load \'Linear Trend\' (Degree = 1). Note the fitted line. Now drag σ²_w slowly from 1.00 down to 0.01 while watching both the main plot and the Weight-Space panel.',
      watch: 'As σ²_w shrinks, the posterior mean is pulled toward zero — the slope flattens even though the data clearly show a trend. This is L2 regularisation / ridge regression, emerging naturally from the Bayesian prior. The weight-space ellipse also collapses toward the origin. No penalty term was added by hand; it fell out of the prior automatically.',
    },
    {
      title: 'Noise variance: interpolation vs. smoothing',
      instruction: 'Load \'Nonlinear Data\', Degree = 2. Drag σ²_n from 0.01 up to 2.00 and back.',
      watch: 'At low σ²_n the posterior mean threads through every point exactly (the model believes observations are nearly noise-free). At high σ²_n the curve smooths over the scatter — individual jitter is attributed to noise rather than signal. σ²_n is your explicit encoding of sensor precision. Cranking it up is not a weakness; it is the correct thing to do when your data really is noisy.',
    },
    {
      title: 'The weight-space posterior ellipse',
      instruction: 'Load \'Dense Observations\', Degree = 1, enable \'Posterior samples\'. Open the Weight-Space panel. Drag σ²_n from 0.30 → 2.00 → 0.05 while watching the ellipse.',
      watch: 'Low σ²_n: the ellipse collapses to a near-dot — both slope and intercept are tightly constrained. High σ²_n: the ellipse elongates, showing the trade-off between slope and intercept that remains uncertain. The function-space uncertainty band and the weight-space ellipse are two projections of the same posterior Σ_w. Watching them move together makes the connection concrete.',
    },
    {
      title: 'Model evidence automatically selects complexity',
      instruction: 'Load \'Model Selection\'. Look at the Model Evidence panel at the bottom. Click Degree 1, then 2, then 3, then 4, reading the log p(y | degree) value each time.',
      watch: 'Degree 3 peaks — the data were generated by a cubic. Degree 4 wiggles to fit noise and the evidence drops despite a lower training error. Occam\'s razor emerges automatically from Bayesian integration: a more complex model must spread its probability mass over more function shapes, and is penalised when the data do not justify that complexity. No separate test set or cross-validation is needed.',
    },
    {
      title: 'RBF basis: nonlinear regression without changing degree',
      instruction: 'Load \'Nonlinear Data\'. Switch the Basis toggle from Polynomial to RBF. Add several points along a rough sine wave. Then switch back to Polynomial at Degree 3 and compare the fits.',
      watch: 'Seven Gaussian bumps placed evenly across the x-axis capture arbitrary curved shapes — the fit improves without touching the polynomial degree. σ²_w now controls bump amplitude rather than polynomial coefficient size. The key lesson: Bayesian Linear Regression is really \'Bayesian regression in any feature space\'. Once you change the basis function φ(x), the same posterior update equations apply unchanged — the Bayesian machinery is indifferent to the feature map.',
    },
  ],
};
