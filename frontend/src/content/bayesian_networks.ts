import type { ModuleContent } from './types';

export const bayesianNetworksContent: ModuleContent = {
  id: 'bayesian-networks',
  title: 'Bayesian Networks',
  subtitle: 'Probabilistic Graphical Models · Chapter 1',

  overview: {
    paragraphs: [
      'A Bayesian Network (BN) is a directed acyclic graph (DAG) in which nodes represent random variables and directed edges encode direct probabilistic dependencies. The joint distribution factorises as a product of local CPTs — one per node given its parents — dramatically reducing the number of parameters needed.',
      'Inference means computing posterior probabilities of hidden variables given observed evidence. The "explaining away" phenomenon (inter-causal reasoning) is one of the most counter-intuitive effects: observing one cause of an effect reduces the posterior probability of other causes, even though the causes are marginally independent.',
      'Two exact algorithms are shown here. Inference by Enumeration (Levels 1–2) materialises the full joint and is easy to understand. Variable Elimination (Level 3) exploits conditional independence to work only with local factors — it multiplies and marginalises factors one variable at a time, avoiding the exponential joint table.',
    ],
    equations: [
      {
        label: 'Joint factorisation',
        latex: 'p(x_1, \\ldots, x_n) = \\prod_{i=1}^{n} p\\bigl(x_i \\mid \\text{parents}(x_i)\\bigr)',
        explanation: 'The global joint distribution equals the product of local CPTs. This is the defining property of a Bayesian Network.',
      },
      {
        label: "Bayes' theorem (posterior update)",
        latex: 'P(H \\mid E) = \\frac{P(E \\mid H)\\, P(H)}{P(E)} = \\frac{P(E \\mid H)\\, P(H)}{\\sum_{h} P(E \\mid h)\\, P(h)}',
        explanation: 'Given evidence E, we reweight each hypothesis H by its likelihood P(E|H) and normalise.',
      },
      {
        label: 'D-separation (conditional independence)',
        latex: 'X \\perp Y \\mid Z \\iff Z \\text{ d-separates } X \\text{ and } Y \\text{ in the DAG}',
        explanation: 'D-separation is the graphical criterion that tells us which variables become independent given which observations.',
      },
    ],
    keyInsights: [
      'Change a prior slider and watch how the marginals of child nodes shift — this is causal (top-down) reasoning.',
      'Click evidence on a child node and observe how root nodes update — this is diagnostic (bottom-up) reasoning.',
      'In Sensor Fusion (L2): set Camera = None, then add Weather = Fog. Fog "explains away" the camera failure, raising the Pedestrian probability back up.',
      'In Robot Mission (L3): observe the cascade from Terrain → Sensors → Localization → Mission. Hit "Solve Step-by-Step" to see Variable Elimination trace each factor being multiplied and marginalised.',
      'Select Mission = Abort in L3 and work backwards — BN diagnosis reveals which root causes (Terrain, Battery) are most likely responsible for failure.',
      'The shift indicators (±%) show exactly how much each observation moved each belief from its prior.',
    ],
  },

  algorithm: {
    name: 'Exact Inference — Two Algorithms',
    complexity: 'BF: O(2ⁿ) · VE: O(n·2ʷ)',
    steps: [
      // ── Inference by Enumeration ───────────────────────────────────
      { kind: 'header', indent: 0, text: 'Algorithm 1 — Inference by Enumeration  (Levels 1 & 2)' },
      { kind: 'io', indent: 0, text: 'Input: DAG, CPTs, priors, evidence E = {(Xᵢ, eᵢ)}' },
      { kind: 'io', indent: 0, text: 'Output: Posterior marginals P(Xⱼ | E) for all j' },
      { kind: 'divider', indent: 0, text: '' },
      { kind: 'step',    indent: 0, text: '1. Generate truth table — Cartesian product of all states' },
      { kind: 'step',    indent: 0, text: '2. Filter states consistent with evidence E' },
      { kind: 'step',    indent: 0, text: '3. For each consistent state s, compute joint weight:' },
      { kind: 'step',    indent: 1, math: true, text: 'w(s) = \\prod_{\\text{roots}} P(s_i) \\cdot \\prod_{\\text{children}} P(s_j \\mid s_{\\text{pa}(j)})' },
      { kind: 'step',    indent: 0, text: '4. Normalise: α = Σ w(s) = P(E)' },
      { kind: 'step',    indent: 1, math: true, text: 'P(X_j = v \\mid E) = \\tfrac{1}{\\alpha}\\sum_{s:\\, s_j = v} w(s)' },
      { kind: 'return',  indent: 0, text: 'marginals for all nodes' },

      // ── Variable Elimination ───────────────────────────────────────
      { kind: 'divider', indent: 0, text: '' },
      { kind: 'header', indent: 0, text: 'Algorithm 2 — Variable Elimination  (Level 3)' },
      { kind: 'comment', indent: 0, text: 'Works with local factors; never builds the full joint.' },
      { kind: 'io', indent: 0, text: 'Input: query Q, evidence E, elimination order π' },
      { kind: 'io', indent: 0, text: 'Output: P(Q | E)' },
      { kind: 'divider', indent: 0, text: '' },
      { kind: 'step',    indent: 0, text: '1. Initialise factors: φᵢ ← CPT of node i' },
      { kind: 'step',    indent: 0, text: '2. Restrict each factor containing eᵢ ∈ E:' },
      { kind: 'step',    indent: 1, math: true, text: '\\phi_i(\\mathbf{x}) \\to \\phi_i(\\mathbf{x}_{-e})\\big|_{e = \\text{observed}}' },
      { kind: 'step',    indent: 0, text: '3. For each variable Zⱼ in elimination order π:' },
      { kind: 'step',    indent: 1, text: 'a. Collect all φ containing Zⱼ' },
      { kind: 'step',    indent: 1, text: 'b. Multiply them into a product factor ψ' },
      { kind: 'step',    indent: 1, text: 'c. Sum out Zⱼ from ψ → new factor ψ\'' },
      { kind: 'step',    indent: 1, math: true, text: "\\psi'(\\mathbf{x}_{-Z_j}) = \\sum_{Z_j} \\psi(\\mathbf{x})" },
      { kind: 'step',    indent: 0, text: '4. Multiply remaining factors → unnormalised P̃(Q | E)' },
      { kind: 'step',    indent: 0, text: '5. Normalise: divide by Z = Σ_q P̃(Q=q | E)' },
      { kind: 'return',  indent: 0, text: 'P(Q | E)' },
    ],
    note: 'VE complexity is O(n · 2ʷ) where w is the treewidth — the maximum scope size of any intermediate factor. Choosing the elimination order to minimise w is NP-hard in general but easy for polytrees.',
  },

  references: [
    {
      label: 'Probabilistic Artificial Intelligence — Krause & Hübotter',
      authors: 'A. Krause, J. Hübotter (ETH Zürich, 2025)',
      url: 'https://arxiv.org/abs/2502.05244',
      description: 'Section 1.1.8: Directed Graphical Models. The primary reference for this course — covers BN factorisation, d-separation, and inference.',
      type: 'book',
    },
    {
      label: 'Probabilistic Graphical Models — Koller & Friedman',
      authors: 'D. Koller, N. Friedman (MIT Press, 2009)',
      url: 'https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/',
      description: 'The definitive textbook on PGMs. Chapters 3–10 cover BNs, d-separation, exact and approximate inference in depth.',
      type: 'book',
    },
    {
      label: 'Artificial Intelligence: A Modern Approach — Russell & Norvig',
      authors: 'S. Russell, P. Norvig (4th ed., 2020)',
      url: 'https://aima.cs.berkeley.edu/',
      description: 'Chapters 12–13 provide an accessible introduction to BNs, CPTs, and exact inference algorithms for students new to the topic.',
      type: 'book',
    },
    {
      label: 'Bayesian Networks — Wikipedia',
      authors: 'Wikipedia contributors',
      url: 'https://en.wikipedia.org/wiki/Bayesian_network',
      description: 'Good starting point covering structure, inference, and learning with worked examples and visualisations.',
      type: 'wiki',
    },
  ],

  explore: [
    {
      title: 'Prior to posterior: the one-node update',
      instruction: 'Load the Level 1: Robot Battery model (the default). In the graph, click the VoltageSensor node to open its detail panel, then click "High" to set it as observed evidence.',
      watch: 'P(Battery = Full) jumps from roughly 80% to ~95%, and a green shift badge appears. This is Bayes\' theorem in action: a noisy sensor (P(High|Full)=0.9 vs P(High|Low)=0.2) is still informative enough to cut the probability of being wrong by 75% — from 20% down to 5%. You cannot read that speed of belief revision from a static formula — you have to see the bar move.',
    },
    {
      title: 'Causal vs. diagnostic reasoning',
      instruction: 'In Level 1, set WarningLight = On (diagnostic: observing an effect). Note how P(Battery = Low) rises. Then click Reset Network and instead set Battery = Low (causal: asserting a cause). Compare how WarningLight probability shifts in each direction.',
      watch: 'The same CPT drives inference in both directions simultaneously. Observing an effect raises the probability of its causes; asserting a cause predicts its effects. Bayesian Networks are a single model that supports both causal and diagnostic reasoning — unlike a lookup table or decision tree which only works one way.',
    },
    {
      title: 'Conflicting evidence: sensors disagree',
      instruction: 'In Level 1, load the "Conflict: Light On + Voltage High" scenario from the sidebar. Observe the Battery posterior. Now open the VoltageSensor node\'s CPT panel and slide P(High | Battery=Low) from 0.20 up toward 0.50.',
      watch: 'As the sensor\'s false-positive rate rises, the conflicting evidence loses its force — the Battery posterior drifts back toward the prior. The network does not simply AND evidence together: it weights each sensor by its reliability encoded in the CPT. An unreliable sensor contributes less, exactly proportional to how uninformative it is.',
    },
    {
      title: 'Explaining away in Sensor Fusion',
      instruction: 'Switch to Level 2: Sensor Fusion. Load the "Explaining Away" scenario (Camera=None, LiDAR=Detected, Weather=Fog). Note P(Object=Pedestrian). Now toggle Weather from Fog to Clear while keeping Camera and LiDAR evidence fixed.',
      watch: 'With Fog, Camera=None is naturally explained by poor visibility — so P(Pedestrian) stays high to match LiDAR. Switch to Clear weather: Camera failure can no longer be blamed on fog, so the network must re-attribute it to the object\'s absence. P(Pedestrian) drops sharply. One cause absorbs blame from another — with no new information about the object itself.',
    },
    {
      title: 'D-separation: the collider effect',
      instruction: 'In Level 2, load "d-sep ③ V-Structure: Blocked". Drag the Object prior slider (left sidebar) from 0.4 to 0.9 and watch the Weather node\'s posterior. Then load "d-sep ④ V-Structure: Active (Collider Observed)" and repeat the same drag.',
      watch: 'In ③ with Camera unobserved, dragging Object prior leaves Weather unchanged — two independent causes cannot influence each other through an unobserved common effect. In ④ with Camera observed, Weather reacts to Object prior. Observing the collider opens a path that did not exist before. This counter-intuitive activation is the heart of d-separation, and it is nearly impossible to grasp from a formula alone.',
    },
    {
      title: 'Multi-hop cascade in Mission Planning',
      instruction: 'Switch to Level 3: Robot Mission Planning. Load "Rough Terrain" and note P(Mission=Abort). Then additionally click Battery node and set Battery=Low. Compare the final Mission Abort probability to the Rough Terrain case alone.',
      watch: 'Rough Terrain activates the Sensors→Localization→Mission chain. Adding Battery=Low activates a completely independent path through Navigation→Mission. The two paths compound: Mission Abort probability jumps more than either cause alone would suggest. This is how a robot\'s safety monitor must reason about multiple concurrent failures — the nonlinear combination is only visible in the full joint distribution.',
    },
    {
      title: 'Backward inference from Mission = Abort',
      instruction: 'In Level 3, load the "Mission Aborted" scenario (Mission=Abort). Note how P(Terrain=Rough) and P(Battery=Low) shift upward. Now open the Navigation node\'s detail panel and also set Navigation=Success.',
      watch: 'Setting Navigation=Success while Mission=Abort remains observed activates explaining away between Navigation and Localization: since navigation succeeded, the abort must be explained by localization drift, so P(Localization=Drifted) spikes. This is Variable Elimination running backwards through six nodes simultaneously — the network is reasoning about the most plausible combination of causes for a given outcome.',
    },
    {
      title: 'Prior Sampling: no evidence, no conditioning',
      instruction: 'Load any model. Open the Sampling panel (click the "Sample" tab on the right edge). Select the "Prior" method, set N=500, and click Run.',
      watch: 'The convergence chart shows estimated P(X=state) bouncing around early then settling near the prior marginals. Since no evidence is set, all methods should converge to the same values. Increase N to 2000 and re-run — the curve becomes noticeably smoother. This is the Law of Large Numbers at work: the estimator is unbiased and variance shrinks as 1/N.',
    },
    {
      title: 'Rejection Sampling: the cost of rare evidence',
      instruction: 'Load the Level 1: Robot Battery model. Set WarningLight = On and VoltageSensor = Low (a fairly rare combination). Open Sampling, choose Rejection, N=500, and run. Note the acceptance rate badge.',
      watch: 'With both sensors pointing to a low battery, only a small fraction of prior samples are consistent with the evidence — the acceptance badge turns amber or red. Run again with WarningLight = On only: acceptance rate improves. This illustrates why rejection sampling breaks down for unlikely evidence: most computation is wasted on discarded samples.',
    },
    {
      title: 'Likelihood Weighting: all samples used',
      instruction: 'Keep the same rare evidence (WarningLight=On, VoltageSensor=Low). Switch to Likelihood Weighting and run N=500. Compare the ESS (Effective Sample Size) badge to N.',
      watch: 'All 500 samples are used — no rejections. But ESS < N: when evidence is unlikely, most weight concentrates on a few "good" samples. Low ESS means the effective information is much less than the raw count. Compare the convergence chart to Rejection: LW often converges faster because it never wastes samples, but ESS exposes when the weighting scheme is degenerate.',
    },
    {
      title: 'Gibbs Sampling: MCMC convergence and burn-in',
      instruction: 'Load "Sampling: Student Grades" and set evidence Letter=Strong (scenario ①). Switch to Gibbs, N=500, Burn-in=0. Run and observe the Hard-working convergence chart. Then slide Burn-in to 100 and re-run.',
      watch: 'With Burn-in=0, the chain starts from a random state — early samples reflect that initialisation rather than the posterior, so the curve can start far from the true value before drifting in. With Burn-in=100 those first 100 steps are silently discarded before averaging begins, giving a much cleaner estimate from the start. Gibbs is the only method here that propagates leaf evidence (Letter) upstream through the Markov chain.',
    },

    // ── Robot Sensor Fusion (sampling-focused) ──────────────────────────
    {
      title: '[Robot] Why Likelihood Weighting struggles with leaf evidence',
      instruction: 'Switch to "Sampling: Robot Sensor Fusion". Load scenario ② (AccelReading=Spike, SpeedSensor=Stall — both leaf nodes observed). Run Likelihood Weighting with N=500. Note the ESS badge. Then switch to scenario ① (Terrain=Rough — a root node) and re-run LW with the same N.',
      watch: 'Leaf evidence forces LW to assign low weights to most samples, because the sampled root values rarely produce the observed leaf readings. ESS drops sharply — sometimes below 50 even with 500 raw samples. With upstream evidence (scenario ①), LW is perfectly efficient: root evidence is "clamped" directly, so all weights are equal and ESS ≈ N. This is the fundamental asymmetry: LW is a top-down sampler, so it pays a heavy price for bottom-up evidence.',
    },
    {
      title: '[Robot] Gibbs handles leaf evidence gracefully',
      instruction: 'Keep scenario ② (leaf evidence). Switch to Gibbs, N=500, Burn-in=100, and run. Compare the convergence of P(Terrain) and P(BatteryAge) against the LW run from the previous step.',
      watch: 'Gibbs converges to sensible marginals for the hidden root nodes even though both sensors are leaves. Each Gibbs step resamples one variable conditioned on its Markov blanket — blanket conditioning automatically propagates the leaf observations upstream. LW cannot do this: it samples top-down and then reweights, so rare leaf outcomes crush most weights. Gibbs pays its cost in autocorrelation (slow chain mixing), not in wasted samples.',
    },
    {
      title: '[Robot] Collider at WheelSlip: explaining away via sampling',
      instruction: 'Load scenario ③ (WheelSlip=True — the collider is observed). Run Gibbs N=500. Note P(Terrain=Rough) and P(BatteryAge=Old). Then switch to scenario ④ (both AccelReading=Spike AND WheelSlip=True) and re-run.',
      watch: 'With WheelSlip alone (scenario ③), both Terrain and BatteryAge rise — observing the collider activates the v-structure, creating a dependency between the two root causes. Adding AccelReading=Spike (scenario ④) shifts blame: acceleration spikes are more consistent with rough terrain, so P(Terrain=Rough) rises further while P(BatteryAge=Old) is pulled back down. This is explaining away emerging from a sample-based method — no symbolic manipulation, just the Markov chain exploring the posterior.',
    },

    // ── Student Grades ──────────────────────────────────────────────────
    {
      title: '[Student] Hand-trace Gibbs on a 5-node network',
      instruction: 'Switch to "Sampling: Student Grades". Load scenario ① (Letter=Strong — a leaf). Open the Sampling panel, choose Gibbs, set N=20 and Burn-in=0. Run and inspect the trajectory table (one row per sample).',
      watch: 'With only 20 samples you can trace each Gibbs step by hand. The chain resamples each of the 5 variables in turn (Difficulty, Hard-working, Grade, SAT, Letter — but Letter is fixed as evidence). Watch how Grade flips: it is a collider between Difficulty and Hard-working, so its state feeds both upstream nodes in the next step. With so few samples the chain has not mixed yet — this is exactly what burn-in is for. Re-run with Burn-in=50 and compare.',
    },
    {
      title: '[Student] Explaining away through Grade',
      instruction: 'In "Sampling: Student Grades", load scenario ③ (Grade=A, no other evidence). Note P(Difficulty=Easy) and P(HardWorking=High) from Gibbs. Then add SAT=High to the evidence (manually click the SAT node) and re-run.',
      watch: 'Grade=A raises both P(Easy course) and P(Hard-working=High) — two competing explanations for a top grade. Adding SAT=High (an independent signal of diligence) tips the balance: if the student is clearly hard-working, the good grade needs less help from an easy course. Watch P(Difficulty=Easy) drop. This is explaining away through a collider: two causes become anti-correlated once their shared effect is observed, and the additional SAT evidence reinforces one explanation at the expense of the other.',
    },
    {
      title: '[Student] Rejection sampling failure on leaf evidence',
      instruction: 'Load scenario ③ in "Sampling: Student Grades" (Letter=Strong, Grade=A, SAT=High — three leaves observed). Switch to Rejection Sampling, set N=500, and run. Note the acceptance count in the metadata.',
      watch: 'Three simultaneous leaf observations create a very narrow acceptance region. The prior is unlikely to jointly produce a strong letter, an A grade, and a high SAT score in the same sample — so most draws are rejected. The acceptance rate badge should be well below 10%. This is rejection sampling\'s Achilles heel: evidence probability P(E) is the acceptance rate, and when E is a specific combination of leaves, P(E) can be tiny. Gibbs with the same evidence runs comfortably in the same time.',
    },

    // ── Ising Grid / Image Denoising ────────────────────────────────────
    {
      title: '[Ising] Why only Gibbs works on a 32-node grid',
      instruction: 'Switch to "Sampling: Ising Denoising (4×4)". Load scenario ① (noisy cross — 16 observation nodes observed). Try Rejection Sampling with N=100 and note how long it takes and what the acceptance rate is.',
      watch: 'With 16 binary observations, the probability of a random prior sample matching all 16 observed pixels is at most (0.85)¹⁶ ≈ 7% even in the best case — and the noisy scenario is far worse. Rejection sampling is completely intractable on image-sized models. Likelihood Weighting avoids rejections but ESS collapses to near 1 because one or two samples carry almost all the weight. Gibbs is the only method that works: it conditions each hidden pixel on its observed neighbour, propagating denoising one pixel at a time along the Markov chain.',
    },
    {
      title: '[Ising] Watch Gibbs denoise the noisy cross',
      instruction: 'In "Sampling: Ising Denoising", keep scenario ① (noisy cross). Run Gibbs with N=200, Burn-in=50. Look at the marginals panel for the hidden pixel nodes (P00–P33): P(Black) should be high for pixels in the cross arms and low for the background.',
      watch: 'The Ising coupling (φ=0.85) strongly encourages neighbouring pixels to share the same colour, while the observation noise (η=0.15) allows occasional flips. Gibbs finds the denoised image by repeatedly sampling each hidden pixel conditioned on its neighbours and its noisy observation — a Markov blanket of at most 5 nodes. After burn-in, the chain\'s average marginals reconstruct the underlying clean cross. This is exactly how early probabilistic image restoration worked, long before deep learning.',
    },
    {
      title: '[Ising] Burn-in effect on image reconstruction',
      instruction: 'In "Sampling: Ising Denoising", run Gibbs with N=300 and Burn-in=0. Note a few pixel marginals. Re-run with Burn-in=100, then Burn-in=200. Compare P(Black) for a centre pixel across the three runs.',
      watch: 'With Burn-in=0, the chain starts from a random pixel assignment unrelated to the observations — early samples contaminate the averages. The centre pixel (part of the cross arm) may initially show P(Black)≈0.5 before drifting toward the true posterior. Increasing burn-in discards this transient. Notice that the chain mixes slowly on the grid because a pixel\'s state is highly correlated with its neighbours — this is typical of Ising models near the ferromagnetic coupling regime, and it is why Gibbs on grids needs relatively long burn-in compared to a tree-structured BN.',
    },

    // ── Medical Diagnosis (15-node comparison network) ──────────────────────
    {
      title: '[Medical] Why rejection sampling fails with symptom evidence',
      instruction: 'Load "Sampling: Medical Diagnosis (15 nodes)". Go scenario ② (fever, cough, SOB, fatigue). Run Rejection Sampling with N=500 and note the acceptance rate and time.',
      watch: 'The acceptance rate will be very low (~1-5%) because these specific symptom combinations are relatively rare in the prior (P(Fever AND Cough AND SOB AND Fatigue) is tiny). Rejection sampling discards hundreds of samples to get 500 accepted ones. Meanwhile, try Prior Sampling — it runs instantly since it ignores evidence completely. This is the key failure mode of rejection sampling: high-dimensional evidence makes the acceptance rate exponentially small.',
    },
    {
      title: '[Medical] When Gibbs struggles: multimodal posteriors',
      instruction: 'Load "Sampling: Medical Diagnosis (15 nodes)". Go to scenario ④ (fever + positive antigen test, ambiguous Flu vs COVID). Switch to Compare Mode: Prior, Rejection, Likelihood Weighting, Gibbs. Run with N=1000, Runs=5, Burn-in=200. Look at P(Flu) and P(COVID) convergence.',
      watch: 'Rejection and LW converge consistently: P(Flu)≈0.62±0.01. But Gibbs shows WIDE CI bands: individual runs give P(Flu) anywhere from 0.2 to 1.0 because the posterior is multimodal (Flu and COVID are nearly equally plausible). Gibbs gets trapped in local modes: once the chain commits to \"Flu dominant\", the symptoms reinforce that choice, making it hard to flip to \"COVID dominant\". This is mode-locking, a classic MCMC failure when the posterior has separated modes. Use rejection/LW here; Gibbs excels only when evidence strongly singles out one mode.',
    },
    {
      title: '[Medical] Why Gibbs excels on this network',
      instruction: 'Load "Sampling: Medical Diagnosis (15 nodes)". Go scenario ③ (confirmed COVID + chest infiltrates + fever + SOB). Run Gibbs with N=500, Burn-in=200, Runs=5. Look at P(COVID), P(Severity), and the ±1σ bands.',
      watch: 'Gibbs rapidly converges to P(COVID)≈0.998 and shows the tightest CI bands across multiple runs. The chain needs sufficient burn-in (~200 steps) for complex networks to mix properly across the disease subspace. Once burnt-in, Gibbs outperforms rejection sampling (acceptance ≈1.5%) and likelihood weighting (ESS ≈50): it explores the posterior efficiently through local Markov blanket moves, finding that COVID is nearly certain given strong evidence (PCR+, X-ray infiltrates, SOB). The tight ±1σ bands indicate consistent posterior estimates across independent runs.',
    },
  ],
};
