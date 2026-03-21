"""Variable Elimination exact inference for Bayesian Networks.

Complements the brute-force enumeration engine (inference.py).
VE exploits conditional independence structure to avoid materialising
the full joint, making it tractable for medium-sized networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product as cart_product
from typing import Optional

from shared.types import BayesianNetworkModel, CPT


@dataclass
class Factor:
    """A factor φ over a subset of variables."""
    variables: list[str]           # ordered list of variable names
    table: dict[tuple, float]      # (state_v1, state_v2, ...) -> value


# ── Factor construction ───────────────────────────────────────────────

def _init_factors(
    model: BayesianNetworkModel,
    priors: dict[str, dict[str, float]],
    cpts: dict[str, CPT],
) -> list[Factor]:
    """Build one factor per node from priors and CPTs."""
    states_map = {n.id: n.states for n in model.nodes}
    factors: list[Factor] = []

    for node in model.nodes:
        if node.node_type == "root":
            p = priors.get(node.id, {})
            table = {(s,): p.get(s, 1.0 / len(node.states)) for s in node.states}
            factors.append(Factor(variables=[node.id], table=table))
        else:
            cpt = cpts.get(node.id) or model.cpts.get(node.id)
            if cpt is None:
                continue
            parent_state_lists = [states_map[pid] for pid in node.parents]
            variables = [node.id] + list(node.parents)
            table: dict[tuple, float] = {}

            for parent_combo in cart_product(*parent_state_lists):
                # Match the key format used by inference.py
                key_csv = ",".join(parent_combo)
                row = cpt.table.get(key_csv)
                if row is None and len(parent_combo) == 1:
                    row = cpt.table.get(parent_combo[0], {})
                row = row or {}
                for child_state in node.states:
                    factor_key = (child_state,) + parent_combo
                    table[factor_key] = row.get(child_state, 0.0)

            factors.append(Factor(variables=variables, table=table))

    return factors


# ── Core factor operations ────────────────────────────────────────────

def _restrict(factor: Factor, var: str, value: str) -> Factor:
    """Fix *var* = *value* in a factor (evidence instantiation)."""
    if var not in factor.variables:
        return factor
    idx = factor.variables.index(var)
    new_vars = [v for v in factor.variables if v != var]
    new_table: dict[tuple, float] = {}
    for key, prob in factor.table.items():
        if key[idx] == value:
            new_key = tuple(k for i, k in enumerate(key) if i != idx)
            new_table[new_key] = prob
    return Factor(variables=new_vars, table=new_table)


def _multiply(f1: Factor, f2: Factor) -> Factor:
    """Pointwise product of two factors."""
    # Union of variables (preserving f1 order then f2 extras)
    seen: set[str] = set()
    all_vars: list[str] = []
    for v in f1.variables + f2.variables:
        if v not in seen:
            seen.add(v)
            all_vars.append(v)

    # Gather known states for each variable from existing table keys
    var_states: dict[str, list[str]] = {v: [] for v in all_vars}
    for factor in (f1, f2):
        for key in factor.table:
            for i, var in enumerate(factor.variables):
                if key[i] not in var_states[var]:
                    var_states[var].append(key[i])

    new_table: dict[tuple, float] = {}
    for combo in cart_product(*[var_states[v] for v in all_vars]):
        state_map = dict(zip(all_vars, combo))
        v1 = f1.table.get(tuple(state_map[v] for v in f1.variables), 0.0)
        v2 = f2.table.get(tuple(state_map[v] for v in f2.variables), 0.0)
        val = v1 * v2
        if val > 1e-15:
            new_table[combo] = val

    return Factor(variables=all_vars, table=new_table)


def _sum_out(factor: Factor, var: str) -> Factor:
    """Marginalise *var* out of a factor."""
    if var not in factor.variables:
        return factor
    idx = factor.variables.index(var)
    new_vars = [v for v in factor.variables if v != var]
    new_table: dict[tuple, float] = {}
    for key, prob in factor.table.items():
        new_key = tuple(k for i, k in enumerate(key) if i != idx)
        new_table[new_key] = new_table.get(new_key, 0.0) + prob
    return Factor(variables=new_vars, table=new_table)


def _normalize(factor: Factor) -> tuple[Factor, float]:
    total = sum(factor.table.values())
    if total == 0:
        return factor, 0.0
    return (
        Factor(variables=factor.variables, table={k: v / total for k, v in factor.table.items()}),
        total,
    )


# ── Main VE routine ───────────────────────────────────────────────────

def variable_elimination(
    query_var: str,
    evidence: dict[str, str],
    model: BayesianNetworkModel,
    priors: dict[str, dict[str, float]],
    cpts: dict[str, CPT],
    elim_order: Optional[list[str]] = None,
) -> tuple[dict[str, float], list[dict]]:
    """Compute P(query_var | evidence) via Variable Elimination.

    Returns
    -------
    posterior : dict  {state -> probability}
    ve_steps  : list  raw VE trace used by the solver to generate LaTeX
    """
    factors = _init_factors(model, priors, cpts)
    ve_steps: list[dict] = []

    # ── Record initial factors ─────────────────────────────────────────
    ve_steps.append({
        "phase": "init",
        "factor_scopes": [f.variables[:] for f in factors],
    })

    # ── Restrict by evidence ───────────────────────────────────────────
    if evidence:
        for var, val in evidence.items():
            factors = [_restrict(f, var, val) for f in factors]
        ve_steps.append({
            "phase": "evidence",
            "evidence": dict(evidence),
        })

    # ── Elimination order ──────────────────────────────────────────────
    if elim_order is None:
        all_vars = [n.id for n in model.nodes]
        elim_order = [v for v in all_vars if v != query_var and v not in evidence]

    # ── Eliminate one variable at a time ───────────────────────────────
    for var in elim_order:
        containing = [f for f in factors if var in f.variables]
        not_containing = [f for f in factors if var not in f.variables]
        if not containing:
            continue

        product_f = containing[0]
        for f in containing[1:]:
            product_f = _multiply(product_f, f)
        new_f = _sum_out(product_f, var)

        ve_steps.append({
            "phase": "eliminate",
            "var": var,
            "input_scopes": [f.variables[:] for f in containing],
            "product_scope": product_f.variables[:],
            "result_scope": new_f.variables[:],
        })
        factors = not_containing + [new_f]

    # ── Multiply remaining factors ─────────────────────────────────────
    result = factors[0]
    for f in factors[1:]:
        result = _multiply(result, f)

    # ── Normalise ──────────────────────────────────────────────────────
    unnorm = {key[0]: val for key, val in result.table.items()}
    result_norm, total = _normalize(result)
    posterior = {key[0]: val for key, val in result_norm.table.items()}

    ve_steps.append({
        "phase": "normalize",
        "query": query_var,
        "unnormalized": unnorm,
        "total": total,
        "posterior": posterior,
    })

    return posterior, ve_steps
