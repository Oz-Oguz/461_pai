"""Approximate inference via sampling for Bayesian Networks.

Four algorithms:
  - prior_sampling:        samples from the joint P(x1,...,xn)
  - rejection_sampling:    conditions on evidence by discarding inconsistent samples
  - likelihood_weighting:  fixes evidence, weights each sample by P(evidence|parents)
  - gibbs_sampling:        MCMC — resamples one variable at a time from its Markov blanket
"""

from __future__ import annotations

import numpy as np

from shared.types import BayesianNetworkModel, CPT


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lookup_prob(node_id: str, val: str, sample: dict[str, str],
                 node_parents: list[str],
                 priors: dict[str, dict[str, float]],
                 cpts: dict[str, CPT]) -> float:
    """Return P(node_id = val | parents sampled in `sample`)."""
    if not node_parents:  # root node
        return priors[node_id].get(val, 0.0)
    parent_vals = [sample[p] for p in node_parents]
    key = ",".join(parent_vals)
    cpt = cpts[node_id]
    row = cpt.table.get(key)
    if row is None and len(parent_vals) == 1:
        row = cpt.table.get(parent_vals[0])
    return (row or {}).get(val, 0.0)


def _sample_node(node_id: str, sample: dict[str, str],
                 node_parents: list[str], node_states: list[str],
                 priors: dict[str, dict[str, float]],
                 cpts: dict[str, CPT],
                 rng: np.random.Generator) -> str:
    """Draw a state for node_id given current sample (parent values already set)."""
    probs = np.array([
        _lookup_prob(node_id, s, sample, node_parents, priors, cpts)
        for s in node_states
    ], dtype=float)
    total = probs.sum()
    if total <= 0:
        return rng.choice(node_states)
    probs /= total
    return node_states[rng.choice(len(node_states), p=probs)]


def _downsample_trajectory(
    node_curves: dict[str, dict[str, list[float]]],
    all_steps: list[int],
    max_pts: int = 500,
) -> tuple[list[int], dict[str, dict[str, list[float]]]]:
    """Reduce trajectory to at most max_pts data points."""
    n = len(all_steps)
    if n <= max_pts:
        return all_steps, node_curves
    stride = n // max_pts
    idxs = list(range(0, n, stride))
    steps_out = [all_steps[i] for i in idxs]
    curves_out: dict[str, dict[str, list[float]]] = {}
    for node_id, state_curves in node_curves.items():
        curves_out[node_id] = {
            state: [vals[i] for i in idxs]
            for state, vals in state_curves.items()
        }
    return steps_out, curves_out


def _build_children_map(model: BayesianNetworkModel) -> dict[str, list[str]]:
    """Map each node to its list of child node IDs."""
    children: dict[str, list[str]] = {n.id: [] for n in model.nodes}
    for node in model.nodes:
        for parent_id in node.parents:
            children[parent_id].append(node.id)
    return children


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _prior_sampling_single_run(
    model: BayesianNetworkModel,
    n_samples: int,
    priors: dict[str, dict[str, float]],
    cpts: dict[str, CPT],
    rng: np.random.Generator,
) -> tuple[dict, dict, dict]:
    """Single run of prior sampling. Returns (all_steps, node_curves, marginals)."""
    node_info = [(n.id, n.parents, n.states) for n in model.nodes]
    counts: dict[str, dict[str, float]] = {
        n.id: {s: 0.0 for s in n.states} for n in model.nodes
    }
    node_curves: dict[str, dict[str, list[float]]] = {
        n.id: {s: [] for s in n.states} for n in model.nodes
    }
    all_steps: list[int] = []

    for i in range(n_samples):
        sample: dict[str, str] = {}
        for node_id, parents, states in node_info:
            sample[node_id] = _sample_node(node_id, sample, parents, states, priors, cpts, rng)
        for node_id, _, states in node_info:
            counts[node_id][sample[node_id]] += 1.0
        n_done = i + 1
        all_steps.append(n_done)
        for node_id, _, states in node_info:
            for s in states:
                node_curves[node_id][s].append(counts[node_id][s] / n_done)

    marginals = {
        nid: {s: counts[nid][s] / n_samples for s in counts[nid]}
        for nid in counts
    }
    return all_steps, node_curves, marginals


def prior_sampling(
    model: BayesianNetworkModel,
    n_samples: int,
    priors: dict[str, dict[str, float]] | None = None,
    cpts: dict[str, CPT] | None = None,
    seed: int | None = None,
    n_runs: int = 1,
) -> dict:
    """
    Prior sampling: draw N complete samples from the joint P(x1,...,xn).
    Iterates nodes in topological order (as listed in model.nodes).
    """
    priors = priors if priors is not None else model.priors
    cpts = cpts if cpts is not None else model.cpts

    base_seed = seed if seed is not None else 0
    all_marginals = []
    all_curves_list = []
    steps_out = []

    for r in range(n_runs):
        rng = np.random.default_rng(base_seed + r)
        all_steps, node_curves, marginals = _prior_sampling_single_run(model, n_samples, priors, cpts, rng)
        s_out, c_out = _downsample_trajectory(node_curves, all_steps)
        if r == 0:
            steps_out = s_out
        all_marginals.append(marginals)
        all_curves_list.append(c_out)

    # Aggregate results
    if n_runs == 1:
        marginals = all_marginals[0]
        curves_out = all_curves_list[0]
        ci_bands = None
    else:
        # Mean marginals across runs
        marginals = {}
        for nid in all_marginals[0]:
            marginals[nid] = {}
            for s in all_marginals[0][nid]:
                vals = [run[nid][s] for run in all_marginals]
                marginals[nid][s] = np.mean(vals)

        # Mean curves and CI bands
        curves_out = all_curves_list[0]  # use trajectory structure from first run
        ci_bands = {}
        for nid in all_curves_list[0]:
            ci_bands[nid] = {}
            for s in all_curves_list[0][nid]:
                curves = [run[nid][s] for run in all_curves_list]
                mean = np.mean(curves, axis=0)
                std = np.std(curves, axis=0)
                ci_bands[nid][s] = {
                    "low": np.clip(mean - std, 0, 1).tolist(),
                    "high": np.clip(mean + std, 0, 1).tolist(),
                }
                curves_out[nid][s] = mean.tolist()

    return {
        "method": "prior",
        "marginals": marginals,
        "trajectory": {"steps": steps_out, "curves": curves_out, **({"ci_bands": ci_bands} if ci_bands else {})},
        "metadata": {"n_samples": n_samples,
                     "n_accepted": None, "acceptance_rate": None,
                     "effective_samples": None, "n_burn": None,
                     **({"n_runs": n_runs} if n_runs > 1 else {})},
    }


def _rejection_sampling_single_run(
    model: BayesianNetworkModel,
    evidence: dict[str, str],
    n_samples: int,
    priors: dict[str, dict[str, float]],
    cpts: dict[str, CPT],
    rng: np.random.Generator,
) -> tuple[dict, dict, dict, int, int]:
    """Single run of rejection sampling. Returns (all_steps, node_curves, marginals, n_accepted, n_total)."""
    node_info = [(n.id, n.parents, n.states) for n in model.nodes]
    counts: dict[str, dict[str, float]] = {
        n.id: {s: 0.0 for s in n.states} for n in model.nodes
    }
    node_curves: dict[str, dict[str, list[float]]] = {
        n.id: {s: [] for s in n.states} for n in model.nodes
    }
    all_steps: list[int] = []
    n_accepted = 0
    n_total = 0

    while n_accepted < n_samples:
        n_total += 1
        sample: dict[str, str] = {}
        for node_id, parents, states in node_info:
            sample[node_id] = _sample_node(node_id, sample, parents, states, priors, cpts, rng)
        if all(sample.get(eid) == eval for eid, eval in evidence.items()):
            n_accepted += 1
            for node_id, _, states in node_info:
                counts[node_id][sample[node_id]] += 1.0
            all_steps.append(n_accepted)
            for node_id, _, states in node_info:
                for s in states:
                    node_curves[node_id][s].append(counts[node_id][s] / n_accepted)

    marginals = {
        nid: {s: counts[nid][s] / n_samples for s in counts[nid]}
        for nid in counts
    }
    return all_steps, node_curves, marginals, n_accepted, n_total


def rejection_sampling(
    model: BayesianNetworkModel,
    evidence: dict[str, str],
    n_samples: int,
    priors: dict[str, dict[str, float]] | None = None,
    cpts: dict[str, CPT] | None = None,
    seed: int | None = None,
    n_runs: int = 1,
) -> dict:
    """
    Rejection sampling: run prior sampling and discard samples inconsistent with evidence.
    Running marginals are computed only over accepted samples.
    """
    priors = priors if priors is not None else model.priors
    cpts = cpts if cpts is not None else model.cpts

    base_seed = seed if seed is not None else 0
    all_marginals = []
    all_curves_list = []
    all_stats = []
    steps_out = []

    for r in range(n_runs):
        rng = np.random.default_rng(base_seed + r)
        all_steps, node_curves, marginals, n_accepted, n_total = _rejection_sampling_single_run(
            model, evidence, n_samples, priors, cpts, rng
        )
        s_out, c_out = _downsample_trajectory(node_curves, all_steps)
        if r == 0:
            steps_out = s_out
        all_marginals.append(marginals)
        all_curves_list.append(c_out)
        all_stats.append((n_accepted, n_total))

    # Aggregate results
    n_accepted, n_total = all_stats[0]
    acceptance_rate = n_accepted / n_total if n_total > 0 else 0.0

    if n_runs == 1:
        marginals = all_marginals[0]
        curves_out = all_curves_list[0]
        ci_bands = None
    else:
        # Mean marginals across runs
        marginals = {}
        for nid in all_marginals[0]:
            marginals[nid] = {}
            for s in all_marginals[0][nid]:
                vals = [run[nid][s] for run in all_marginals]
                marginals[nid][s] = np.mean(vals)

        # Mean curves and CI bands
        curves_out = all_curves_list[0]
        ci_bands = {}
        for nid in all_curves_list[0]:
            ci_bands[nid] = {}
            for s in all_curves_list[0][nid]:
                curves = [run[nid][s] for run in all_curves_list]
                mean = np.mean(curves, axis=0)
                std = np.std(curves, axis=0)
                ci_bands[nid][s] = {
                    "low": np.clip(mean - std, 0, 1).tolist(),
                    "high": np.clip(mean + std, 0, 1).tolist(),
                }
                curves_out[nid][s] = mean.tolist()

    return {
        "method": "rejection",
        "marginals": marginals,
        "trajectory": {"steps": steps_out, "curves": curves_out, **({"ci_bands": ci_bands} if ci_bands else {})},
        "metadata": {
            "n_samples": n_samples,
            "n_accepted": n_accepted,
            "acceptance_rate": round(acceptance_rate, 4),
            "effective_samples": None,
            "n_burn": None,
            **({"n_runs": n_runs} if n_runs > 1 else {}),
        },
    }


def _likelihood_weighting_single_run(
    model: BayesianNetworkModel,
    evidence: dict[str, str],
    n_samples: int,
    priors: dict[str, dict[str, float]],
    cpts: dict[str, CPT],
    rng: np.random.Generator,
) -> tuple[dict, dict, dict, float, float]:
    """Single run of likelihood weighting. Returns (all_steps, node_curves, marginals, total_weight, sum_w2)."""
    node_info = [(n.id, n.parents, n.states) for n in model.nodes]
    weighted_counts: dict[str, dict[str, float]] = {
        n.id: {s: 0.0 for s in n.states} for n in model.nodes
    }
    node_curves: dict[str, dict[str, list[float]]] = {
        n.id: {s: [] for s in n.states} for n in model.nodes
    }
    all_steps: list[int] = []
    total_weight = 0.0
    sum_w2 = 0.0

    for i in range(n_samples):
        sample: dict[str, str] = {}
        w = 1.0
        for node_id, parents, states in node_info:
            if node_id in evidence:
                sample[node_id] = evidence[node_id]
                w *= _lookup_prob(node_id, evidence[node_id], sample, parents, priors, cpts)
            else:
                sample[node_id] = _sample_node(node_id, sample, parents, states, priors, cpts, rng)

        total_weight += w
        sum_w2 += w * w
        for node_id, _, states in node_info:
            weighted_counts[node_id][sample[node_id]] += w

        all_steps.append(i + 1)
        if total_weight > 0:
            for node_id, _, states in node_info:
                for s in states:
                    node_curves[node_id][s].append(weighted_counts[node_id][s] / total_weight)
        else:
            for node_id, _, states in node_info:
                for s in states:
                    node_curves[node_id][s].append(0.0)

    marginals = {
        nid: {s: weighted_counts[nid][s] / total_weight if total_weight > 0 else 0.0
              for s in weighted_counts[nid]}
        for nid in weighted_counts
    }
    return all_steps, node_curves, marginals, total_weight, sum_w2


def likelihood_weighting(
    model: BayesianNetworkModel,
    evidence: dict[str, str],
    n_samples: int,
    priors: dict[str, dict[str, float]] | None = None,
    cpts: dict[str, CPT] | None = None,
    seed: int | None = None,
    n_runs: int = 1,
) -> dict:
    """
    Likelihood weighting: fix evidence variables, sample the rest from prior,
    and weight each sample by P(evidence | parents).

    Weighted marginal: P̂(X=s) = Σ w_i · 1[x_i=s] / Σ w_i
    Effective sample size: Neff = (Σw)² / Σ(w²)
    """
    priors = priors if priors is not None else model.priors
    cpts = cpts if cpts is not None else model.cpts

    base_seed = seed if seed is not None else 0
    all_marginals = []
    all_curves_list = []
    all_stats = []
    steps_out = []

    for r in range(n_runs):
        rng = np.random.default_rng(base_seed + r)
        all_steps, node_curves, marginals, total_weight, sum_w2 = _likelihood_weighting_single_run(
            model, evidence, n_samples, priors, cpts, rng
        )
        s_out, c_out = _downsample_trajectory(node_curves, all_steps)
        if r == 0:
            steps_out = s_out
        all_marginals.append(marginals)
        all_curves_list.append(c_out)
        ess = (total_weight ** 2) / sum_w2 if sum_w2 > 0 else 0.0
        all_stats.append(ess)

    # Aggregate results
    ess = all_stats[0]

    if n_runs == 1:
        marginals = all_marginals[0]
        curves_out = all_curves_list[0]
        ci_bands = None
    else:
        # Mean marginals across runs
        marginals = {}
        for nid in all_marginals[0]:
            marginals[nid] = {}
            for s in all_marginals[0][nid]:
                vals = [run[nid][s] for run in all_marginals]
                marginals[nid][s] = np.mean(vals)

        # Mean curves and CI bands
        curves_out = all_curves_list[0]
        ci_bands = {}
        for nid in all_curves_list[0]:
            ci_bands[nid] = {}
            for s in all_curves_list[0][nid]:
                curves = [run[nid][s] for run in all_curves_list]
                mean = np.mean(curves, axis=0)
                std = np.std(curves, axis=0)
                ci_bands[nid][s] = {
                    "low": np.clip(mean - std, 0, 1).tolist(),
                    "high": np.clip(mean + std, 0, 1).tolist(),
                }
                curves_out[nid][s] = mean.tolist()

    return {
        "method": "likelihood_weighting",
        "marginals": marginals,
        "trajectory": {"steps": steps_out, "curves": curves_out, **({"ci_bands": ci_bands} if ci_bands else {})},
        "metadata": {
            "n_samples": n_samples,
            "n_accepted": None,
            "acceptance_rate": None,
            "effective_samples": round(ess, 2),
            "n_burn": None,
            **({"n_runs": n_runs} if n_runs > 1 else {}),
        },
    }


def _gibbs_single_run(
    model: BayesianNetworkModel,
    evidence: dict[str, str],
    n_samples: int,
    n_burn: int,
    priors: dict[str, dict[str, float]],
    cpts: dict[str, CPT],
    rng: np.random.Generator,
) -> tuple[list[int], dict[str, dict[str, list[float]]], dict[str, dict[str, float]]]:
    """
    One Gibbs chain run. Returns (all_steps, node_curves, marginals).
    node_curves[node_id][state] = running mean at each collected step.
    """
    node_map = {n.id: n for n in model.nodes}
    children_map = _build_children_map(model)
    non_evidence = [n for n in model.nodes if n.id not in evidence]

    sample: dict[str, str] = dict(evidence)
    for node in model.nodes:
        if node.id not in sample:
            sample[node.id] = _sample_node(node.id, sample, node.parents, node.states, priors, cpts, rng)

    counts: dict[str, dict[str, float]] = {n.id: {s: 0.0 for s in n.states} for n in model.nodes}
    node_curves: dict[str, dict[str, list[float]]] = {n.id: {s: [] for s in n.states} for n in model.nodes}
    all_steps: list[int] = []
    n_collected = 0

    for step in range(n_burn + n_samples):
        target = non_evidence[rng.integers(len(non_evidence))]
        tid = target.id
        blanket_probs = []
        for s in target.states:
            sample[tid] = s
            p = _lookup_prob(tid, s, sample, target.parents, priors, cpts)
            for child_id in children_map[tid]:
                child_node = node_map[child_id]
                p *= _lookup_prob(child_id, sample[child_id], sample, child_node.parents, priors, cpts)
            blanket_probs.append(p)

        bp = np.array(blanket_probs, dtype=float)
        total = bp.sum()
        bp = bp / total if total > 0 else np.ones(len(target.states)) / len(target.states)
        sample[tid] = target.states[rng.choice(len(target.states), p=bp)]

        if step >= n_burn:
            n_collected += 1
            for node in model.nodes:
                counts[node.id][sample[node.id]] += 1.0
            all_steps.append(n_collected)
            for node in model.nodes:
                for s in node.states:
                    node_curves[node.id][s].append(counts[node.id][s] / n_collected)

    marginals = {
        nid: {s: counts[nid][s] / n_samples if n_samples > 0 else 0.0 for s in counts[nid]}
        for nid in counts
    }
    return all_steps, node_curves, marginals


def gibbs_sampling(
    model: BayesianNetworkModel,
    evidence: dict[str, str],
    n_samples: int,
    n_burn: int = 100,
    priors: dict[str, dict[str, float]] | None = None,
    cpts: dict[str, CPT] | None = None,
    seed: int | None = None,
    n_runs: int = 1,
) -> dict:
    """
    Gibbs sampling (MCMC): resample one variable at a time from its Markov blanket.

    P(Xi | MB(Xi)) ∝ P(Xi | parents(Xi)) * ∏_{Yj child of Xi} P(Yj | parents(Yj))

    When n_runs > 1, runs n_runs independent chains and returns CI bands on the
    convergence trajectory (mean ± 1 SD across runs).
    """
    priors = priors if priors is not None else model.priors
    cpts = cpts if cpts is not None else model.cpts

    non_evidence = [n for n in model.nodes if n.id not in evidence]
    if not non_evidence:
        marginals = {n.id: {s: 1.0 if s == evidence[n.id] else 0.0
                            for s in n.states} for n in model.nodes}
        empty_curves = {n.id: {s: [] for s in n.states} for n in model.nodes}
        return {
            "method": "gibbs",
            "marginals": marginals,
            "trajectory": {"steps": [], "curves": empty_curves},
            "metadata": {"n_samples": 0, "n_accepted": None,
                         "acceptance_rate": None, "effective_samples": None, "n_burn": n_burn},
        }

    base_seed = seed if seed is not None else 0
    all_marginals: list[dict] = []
    all_curves_list: list[dict] = []  # one entry per run, each is node_curves after downsampling
    steps_out: list[int] = []

    for r in range(n_runs):
        rng = np.random.default_rng(base_seed + r)
        all_steps, node_curves, marginals = _gibbs_single_run(
            model, evidence, n_samples, n_burn, priors, cpts, rng
        )
        s_out, c_out = _downsample_trajectory(node_curves, all_steps)
        if r == 0:
            steps_out = s_out
        all_marginals.append(marginals)
        all_curves_list.append(c_out)

    # Aggregate marginals: mean across runs
    agg_marginals: dict[str, dict[str, float]] = {}
    for nid in all_marginals[0]:
        agg_marginals[nid] = {
            s: float(np.mean([m[nid][s] for m in all_marginals]))
            for s in all_marginals[0][nid]
        }

    # Aggregate curves: mean across runs (always returned)
    mean_curves: dict[str, dict[str, list[float]]] = {}
    for nid in all_curves_list[0]:
        mean_curves[nid] = {}
        for s in all_curves_list[0][nid]:
            vals = np.array([run[nid][s] for run in all_curves_list])  # shape (n_runs, T)
            mean_curves[nid][s] = vals.mean(axis=0).tolist()

    # CI bands: only when n_runs > 1
    ci_bands: dict | None = None
    if n_runs > 1:
        ci_bands = {}
        for nid in all_curves_list[0]:
            ci_bands[nid] = {}
            for s in all_curves_list[0][nid]:
                vals = np.array([run[nid][s] for run in all_curves_list])
                sd = vals.std(axis=0)
                mean = vals.mean(axis=0)
                ci_bands[nid][s] = {
                    "low":  np.clip(mean - sd, 0, 1).tolist(),
                    "high": np.clip(mean + sd, 0, 1).tolist(),
                }

    trajectory: dict = {"steps": steps_out, "curves": mean_curves}
    if ci_bands is not None:
        trajectory["ci_bands"] = ci_bands

    return {
        "method": "gibbs",
        "marginals": agg_marginals,
        "trajectory": trajectory,
        "metadata": {
            "n_samples": n_samples,
            "n_accepted": None,
            "acceptance_rate": None,
            "effective_samples": None,
            "n_burn": n_burn,
            "n_runs": n_runs,
        },
    }
