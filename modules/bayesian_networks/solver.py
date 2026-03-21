"""Step-by-step LaTeX derivation generator for Bayesian inference."""

from __future__ import annotations

from shared.types import BayesianNetworkModel, CPT, InferenceResult, Node
from modules.bayesian_networks.inference import generate_truth_table
from modules.bayesian_networks.ve_inference import variable_elimination


def generate_solver_steps(
    model: BayesianNetworkModel,
    evidence: dict[str, str],
    results: InferenceResult,
    priors: dict[str, dict[str, float]],
    cpts: dict[str, CPT],
) -> list[dict[str, str]]:
    """Generate step-by-step Bayesian derivation for the current evidence state."""
    steps = []

    # Step 1: Evidence & Goal
    if evidence:
        ev_parts = [
            rf"\text{{{k}}} = \text{{{v}}}" for k, v in evidence.items()
        ]
        ev_latex = ", ".join(ev_parts)
        steps.append(
            {
                "title": "1. Evidence & Goal",
                "text": "We want to update our beliefs based on the observed evidence:",
                "latex": rf"E = \{{ {ev_latex} \}}",
            }
        )
    else:
        steps.append(
            {
                "title": "1. Evidence & Goal",
                "text": "No evidence observed. Showing prior (marginal) distributions.",
                "latex": r"\text{No evidence observed — displaying prior distributions}",
            }
        )

    # Step 2: Bayesian Update Rule
    steps.append(
        {
            "title": "2. Bayesian Update Rule",
            "text": "For any hypothesis H, compute the joint probability and normalize:",
            "latex": (
                r"P(H \mid E) = \frac{P(H, E)}{P(E)} "
                r"= \frac{P(E \mid H) \cdot P(H)}{\sum_{h} P(E \mid h) \cdot P(h)}"
            ),
        }
    )

    # Step 3: Derivation for each root node
    root_nodes = [n for n in model.nodes if n.node_type == "root"]
    step_num = 3

    for target in root_nodes:
        deriv = _derive_root_node(target, model, evidence, results, priors, cpts)
        steps.append(
            {
                "title": f"{step_num}. Posterior: {target.label}",
                "text": f"Computing posterior probabilities for **{target.label}**:",
                "latex": deriv,
            }
        )
        step_num += 1

    return steps


def _derive_root_node(
    target: Node,
    model: BayesianNetworkModel,
    evidence: dict[str, str],
    results: InferenceResult,
    priors: dict[str, dict[str, float]],
    cpts: dict[str, CPT],
) -> str:
    """Build aligned LaTeX derivation for a single root node."""
    alpha = results.total_weight

    lines = [
        rf"\alpha = P(E) = {alpha:.5f} \quad (\text{{Normalization Constant}})",
        r"\\[12pt]",
        r"\begin{aligned}",
    ]

    for state in target.states:
        marg = results.marginals[target.id][state]
        unnorm = marg * alpha
        lines.append(
            rf"P(\text{{{target.id}={state}}} \mid E) "
            rf"&= \frac{{P(\text{{{target.id}={state}}}, E)}}{{\alpha}} \\"
        )
        lines.append(
            rf"&= \frac{{{unnorm:.5f}}}{{{alpha:.5f}}} \\"
        )
        lines.append(
            rf"&= \mathbf{{{marg * 100:.1f}\%}} \\[10pt]"
        )

    lines.append(r"\end{aligned}")
    return "\n".join(lines)


def generate_marginal_derivation(
    node: Node,
    model: BayesianNetworkModel,
    priors: dict[str, dict[str, float]],
    cpts: dict[str, CPT],
) -> str:
    """Generate Law of Total Probability derivation for a child node's marginal.

    Shows how the base probability (before any evidence on this node)
    is computed from its parents.
    """
    parent_nodes = [n for n in model.nodes if n.id in node.parents]
    parent_truth_table = generate_truth_table(parent_nodes)
    target_state = node.states[0]

    # Header: P(state) = sum over parent states
    parent_label = r", ".join(rf"\text{{{p.id}}}" for p in parent_nodes)
    lines = [
        rf"P(\text{{{target_state}}}) = "
        rf"\sum_{{{parent_label}}} "
        rf"P(\text{{{target_state}}} \mid \text{{parents}}) "
        rf"\times P(\text{{parents}})",
        r"\\[12pt]",
    ]

    terms = []
    calc_sum = 0.0

    for row in parent_truth_table:
        # P(parents) = product of parent priors
        prob_parents = 1.0
        for pid, pval in row.items():
            prob_parents *= priors.get(pid, {}).get(pval, 0.0)

        # P(child | parents)
        condition_key = ",".join(row[p.id] for p in parent_nodes)
        cpt = cpts.get(node.id, model.cpts.get(node.id))
        cpt_table = cpt.table if cpt else {}
        row_probs = cpt_table.get(condition_key) or cpt_table.get(
            list(row.values())[0], {}
        )
        prob_child = row_probs.get(target_state, 0.0)

        term_val = prob_child * prob_parents
        calc_sum += term_val
        terms.append(rf"({prob_child:.2f} \times {prob_parents:.2f})")

    lines.append(rf"P(\text{{{target_state}}}) = " + " + ".join(terms))
    lines.append(
        rf"\\ P(\text{{{target_state}}}) = {calc_sum:.4f} "
        rf"= \mathbf{{{calc_sum * 100:.1f}\%}}"
    )

    return "\n".join(lines)


# ── Variable Elimination solver ────────────────────────────────────────

def generate_ve_solver_steps(
    query_var: str,
    evidence: dict[str, str],
    model: BayesianNetworkModel,
    priors: dict[str, dict[str, float]],
    cpts: dict[str, CPT],
    elim_order: list[str] | None = None,
) -> list[dict[str, str]]:
    """Generate pedagogical VE derivation as a list of SolveStep dicts."""

    posterior, ve_steps = variable_elimination(
        query_var, evidence, model, priors, cpts, elim_order
    )
    steps: list[dict[str, str]] = []
    step_num = 1

    # ── Step 1: Setup ─────────────────────────────────────────────────
    query_node = next(n for n in model.nodes if n.id == query_var)
    if evidence:
        ev_parts = [rf"\text{{{k}}} = \text{{{v}}}" for k, v in evidence.items()]
        ev_latex = ", ".join(ev_parts)
        goal_latex = rf"P(\text{{{query_var}}} \mid {ev_latex})"
    else:
        goal_latex = rf"P(\text{{{query_var}}})"
        ev_latex = r"\emptyset"

    steps.append({
        "title": f"{step_num}. Query & Evidence",
        "text": (
            f"Compute the posterior over **{query_node.label}** "
            f"({'given: ' + ', '.join(f'{k}={v}' for k,v in evidence.items()) if evidence else 'no evidence — prior marginal'})."
        ),
        "latex": (
            rf"\text{{Goal:}} \quad {goal_latex}"
            + (rf" \qquad E = \{{ {ev_latex} \}}" if evidence else "")
        ),
    })
    step_num += 1

    # ── Step 2: Factor initialisation ─────────────────────────────────
    init_step = next(s for s in ve_steps if s["phase"] == "init")
    phi_list = []
    for i, scope in enumerate(init_step["factor_scopes"], start=1):
        node = next(n for n in model.nodes if n.id == scope[0])
        if len(scope) == 1:
            phi_list.append(rf"\phi_{i}(\text{{{scope[0]}}})")
        else:
            parents_tex = r", ".join(rf"\text{{{v}}}" for v in scope[1:])
            phi_list.append(rf"\phi_{i}(\text{{{scope[0]}}} \mid {parents_tex})")

    steps.append({
        "title": f"{step_num}. Initialise Factors",
        "text": "Create one factor per node: root nodes use the prior P(X), child nodes use their CPT P(X | parents).",
        "latex": r",\quad ".join(phi_list),
    })
    step_num += 1

    # ── Step 3: Evidence instantiation ────────────────────────────────
    ev_step = next((s for s in ve_steps if s["phase"] == "evidence"), None)
    if ev_step:
        restrict_parts = []
        for var, val in ev_step["evidence"].items():
            restrict_parts.append(
                rf"\phi(\cdots\mid\text{{{var}}}=\text{{{val}}}) \to \phi'(\cdots)"
            )
        steps.append({
            "title": f"{step_num}. Instantiate Evidence",
            "text": "Restrict every factor that mentions an observed variable by fixing it to the observed value — this removes one variable from those factors' scope.",
            "latex": r"\quad;\quad ".join(restrict_parts),
        })
        step_num += 1

    # ── Step 4…N: Variable elimination ────────────────────────────────
    elim_steps = [s for s in ve_steps if s["phase"] == "eliminate"]
    psi_counter = len(init_step["factor_scopes"]) + 1

    for elim in elim_steps:
        var = elim["var"]
        in_scopes = elim["input_scopes"]
        prod_scope = elim["product_scope"]
        res_scope = elim["result_scope"]

        # Build factor labels for inputs
        in_labels = []
        for sc in in_scopes:
            if len(sc) == 1:
                in_labels.append(rf"\phi(\text{{{sc[0]}}})")
            else:
                parents_tex = r", ".join(rf"\text{{{v}}}" for v in sc[1:])
                in_labels.append(rf"\phi(\text{{{sc[0]}}} \mid {parents_tex})")

        prod_vars_tex = r", ".join(rf"\text{{{v}}}" for v in prod_scope)
        res_vars_tex = r", ".join(rf"\text{{{v}}}" for v in res_scope)
        psi_label = rf"\psi_{{{psi_counter}}}"
        psi_counter += 1

        steps.append({
            "title": f"{step_num}. Eliminate  {var}",
            "text": (
                f"Collect factors involving **{var}**, multiply into a joint factor, "
                f"then sum out **{var}** to create a new reduced factor."
            ),
            "latex": (
                rf"\underbrace{{{' \\times '.join(in_labels)}}}_{{\text{{multiply}}}}"
                rf"\;\longrightarrow\;"
                rf"\psi({prod_vars_tex})"
                rf"\;\xrightarrow{{\sum_{{\text{{{var}}}}}}}\;"
                rf"{psi_label}({res_vars_tex})"
            ),
        })
        step_num += 1

    # ── Final: Normalise ───────────────────────────────────────────────
    norm_step = next(s for s in ve_steps if s["phase"] == "normalize")
    unnorm = norm_step["unnormalized"]
    total = norm_step["total"]
    post = norm_step["posterior"]

    unnorm_lines = []
    result_lines = [r"\begin{aligned}"]
    for state in query_node.states:
        u = unnorm.get(state, 0.0)
        p = post.get(state, 0.0)
        unnorm_lines.append(rf"\psi(\text{{{query_var}}}=\text{{{state}}}) = {u:.5f}")
        result_lines.append(
            rf"P(\text{{{query_var}}}=\text{{{state}}} \mid E) "
            rf"&= \frac{{{u:.5f}}}{{{total:.5f}}} "
            rf"= \mathbf{{{p * 100:.1f}\%}} \\"
        )
    result_lines.append(r"\end{aligned}")

    steps.append({
        "title": f"{step_num}. Normalise",
        "text": (
            f"Multiply any remaining factors to get the unnormalised query factor, "
            f"then divide by the partition function Z = {total:.5f}."
        ),
        "latex": (
            r"\quad;\quad ".join(unnorm_lines)
            + r" \quad (Z = " + f"{total:.5f}" + r")"
            + r"\\[10pt]"
            + "\n".join(result_lines)
        ),
    })

    return steps
