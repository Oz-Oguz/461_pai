"""Brute-force exact inference engine for Bayesian Networks."""

from __future__ import annotations

from itertools import product

from shared.types import BayesianNetworkModel, CPT, InferenceResult, Node


def generate_truth_table(nodes: list[Node]) -> list[dict[str, str]]:
    """Generate all possible state assignments (cartesian product)."""
    if not nodes:
        return [{}]
    ids = [n.id for n in nodes]
    state_lists = [n.states for n in nodes]
    return [dict(zip(ids, combo)) for combo in product(*state_lists)]


def joint_probability(
    state: dict[str, str],
    model: BayesianNetworkModel,
    priors: dict[str, dict[str, float]],
    cpts: dict[str, CPT],
) -> float:
    """Compute P(state) = product of priors * CPT entries."""
    p = 1.0
    for node in model.nodes:
        val = state[node.id]
        if node.node_type == "root":
            prior = priors.get(node.id, {})
            if val not in prior:
                return 0.0
            p *= prior[val]
        else:
            parent_values = [state[pid] for pid in node.parents]
            condition_key = ",".join(parent_values)
            cpt = cpts.get(node.id, model.cpts.get(node.id))
            if cpt is None:
                return 0.0
            table = cpt.table
            row = table.get(condition_key)
            if row is None and len(parent_values) == 1:
                row = table.get(parent_values[0])
            if row is None or val not in row:
                return 0.0
            p *= row[val]
    return p


def run_inference(
    model: BayesianNetworkModel,
    evidence: dict[str, str],
    priors: dict[str, dict[str, float]] | None = None,
    cpts: dict[str, CPT] | None = None,
) -> InferenceResult | None:
    """Run exact inference by enumeration.

    Returns marginal posteriors for all nodes given the evidence,
    or None if the evidence is impossible.
    """
    priors = priors if priors is not None else model.priors
    cpts = cpts if cpts is not None else model.cpts

    all_states = generate_truth_table(model.nodes)

    # Filter states consistent with evidence
    consistent = [
        s
        for s in all_states
        if all(s.get(k) == v for k, v in evidence.items())
    ]

    # Compute weights
    weighted = [(s, joint_probability(s, model, priors, cpts)) for s in consistent]
    total_weight = sum(w for _, w in weighted)

    if total_weight == 0:
        return None  # Impossible evidence

    # Compute marginals
    marginals: dict[str, dict[str, float]] = {}
    for node in model.nodes:
        marginals[node.id] = {state: 0.0 for state in node.states}

    for state, weight in weighted:
        normalized = weight / total_weight
        for node in model.nodes:
            marginals[node.id][state[node.id]] += normalized

    return InferenceResult(marginals=marginals, total_weight=total_weight)
