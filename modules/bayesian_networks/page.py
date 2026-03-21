"""Bayesian Networks interactive page for the Probabilistic AI Lab."""

from __future__ import annotations

import copy

import streamlit as st

from shared.types import CPT
from shared.math_display import derivation_panel
from shared.graph_viz import dag_to_dot
from modules.bayesian_networks.models import ALL_MODELS
from modules.bayesian_networks.inference import run_inference, generate_truth_table
from modules.bayesian_networks.solver import (
    generate_solver_steps,
    generate_marginal_derivation,
)


# ── Session state helpers ─────────────────────────────────────────────

def _key(name: str) -> str:
    """Prefix session-state keys to avoid collisions with other modules."""
    return f"bn_{name}"


def _init_model_state(model_id: str):
    """Initialize (or reset) session state for a given model."""
    model = ALL_MODELS[model_id]
    st.session_state[_key("model_id")] = model_id
    st.session_state[_key("evidence")] = {}
    st.session_state[_key("priors")] = copy.deepcopy(model.priors)
    st.session_state[_key("cpts")] = copy.deepcopy(model.cpts)


def _get(name: str):
    return st.session_state.get(_key(name))


# ── Dialogs ───────────────────────────────────────────────────────────

@st.dialog("Mathematical Derivation", width="large")
def _solver_dialog():
    """Show step-by-step Bayesian derivation for the current state."""
    model = ALL_MODELS[_get("model_id")]
    evidence = _get("evidence")
    priors = _get("priors")
    cpts = _get("cpts")
    results = run_inference(model, evidence, priors, cpts)
    if results is None:
        st.error("Impossible evidence — no consistent states.")
        return
    steps = generate_solver_steps(model, evidence, results, priors, cpts)
    derivation_panel(steps)


@st.dialog("Network Structure (DAG)", width="large")
def _graph_dialog():
    """Show the Bayesian Network DAG."""
    model = ALL_MODELS[_get("model_id")]
    evidence = _get("evidence")
    dot = dag_to_dot(model, evidence)
    st.graphviz_chart(dot, use_container_width=True)
    st.caption("Blue-highlighted nodes have observed evidence set.")


@st.dialog("Conditional Probability Table", width="large")
def _cpt_dialog(node_id: str):
    """Show (and optionally edit) the CPT for a node."""
    model = ALL_MODELS[_get("model_id")]
    node = next(n for n in model.nodes if n.id == node_id)
    priors = _get("priors")
    cpts = _get("cpts")

    if node.node_type == "root":
        st.markdown(f"### Prior Distribution: {node.label}")
        st.info("Adjust priors using the sliders in the sidebar.")
        for state, prob in priors.get(node.id, {}).items():
            st.metric(state, f"{prob:.2f}")
    else:
        st.markdown(f"### CPT: {node.label}")
        st.caption("Edit values below. Rows auto-normalize to 1.0.")
        cpt = cpts.get(node.id) or model.cpts[node.id]
        changed = False
        new_table = {}
        for parent_key, prob_row in cpt.table.items():
            st.markdown(f"**Given:** `{parent_key.replace(',', ' + ')}`")
            cols = st.columns(len(node.states))
            new_row = {}
            first_state = None
            first_val = None
            for i, state in enumerate(node.states):
                with cols[i]:
                    val = st.number_input(
                        f"P({state})",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(prob_row.get(state, 0.0)),
                        step=0.05,
                        format="%.2f",
                        key=f"cpt_{node.id}_{parent_key}_{state}",
                    )
                    if i == 0:
                        first_state = state
                        first_val = val
                    new_row[state] = val

            # Auto-normalize: if binary, complement the second state
            if len(node.states) == 2:
                second_state = node.states[1]
                new_row[second_state] = round(1.0 - new_row[node.states[0]], 2)

            if new_row != prob_row:
                changed = True
            new_table[parent_key] = new_row

        if changed:
            if st.button("Apply Changes", type="primary"):
                new_cpts = copy.deepcopy(cpts)
                new_cpts[node.id] = CPT(parents=cpt.parents, table=new_table)
                st.session_state[_key("cpts")] = new_cpts
                st.rerun()


@st.dialog("Marginal Derivation (Law of Total Probability)", width="large")
def _marginal_dialog(node_id: str):
    """Show how the base marginal is computed from parent priors."""
    model = ALL_MODELS[_get("model_id")]
    node = next(n for n in model.nodes if n.id == node_id)
    priors = _get("priors")
    cpts = _get("cpts")

    st.markdown(
        f"How the **base probability** of **{node.label}** is calculated "
        f"from its parents using the *Law of Total Probability*:"
    )
    latex = generate_marginal_derivation(node, model, priors, cpts)
    st.latex(latex)
    st.caption("This calculation uses the current prior values of the parent nodes.")


# ── Node card rendering ───────────────────────────────────────────────

def _render_node_card(node, marginals, evidence, prior_marginals):
    """Render a single node card with probability bars and evidence toggles."""
    is_observed = node.id in evidence
    node_type_label = "Root Cause" if node.node_type == "root" else "Sensor / Effect"

    # Card container
    border_color = "#2563eb" if is_observed else "#e2e8f0"
    st.markdown(
        f"""<div style="border: 2px solid {border_color}; border-radius: 12px;
        padding: 0; margin-bottom: 8px; background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
        <div style="padding: 12px 16px; border-bottom: 1px solid #f1f5f9;
        background: {'#eff6ff' if is_observed else 'white'};
        border-radius: 10px 10px 0 0;">
        <strong>{node.label}</strong>
        <span style="color: #94a3b8; font-size: 0.75rem; margin-left: 8px;">
        {node_type_label}</span>
        </div></div>""",
        unsafe_allow_html=True,
    )

    # Action buttons row
    btn_cols = st.columns([1, 1, 1, 3] if node.node_type == "child" else [1, 1, 4])
    with btn_cols[0]:
        if st.button("📊 CPT", key=f"cpt_btn_{node.id}", use_container_width=True):
            _cpt_dialog(node.id)
    if node.node_type == "child":
        with btn_cols[1]:
            if st.button("❓ Hint", key=f"hint_btn_{node.id}", use_container_width=True):
                _marginal_dialog(node.id)
        clear_col = btn_cols[2]
    else:
        clear_col = btn_cols[1]

    if is_observed:
        with clear_col:
            if st.button(
                "Clear Evidence",
                key=f"clear_{node.id}",
                use_container_width=True,
            ):
                del st.session_state[_key("evidence")][node.id]
                st.rerun()

    # State buttons + probability bars
    for state in node.states:
        prob = marginals.get(node.id, {}).get(state, 0.0)
        prior_prob = prior_marginals.get(node.id, {}).get(state, prob)
        is_selected = evidence.get(node.id) == state
        shift = prob - prior_prob

        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            # Evidence toggle button
            btn_type = "primary" if is_selected else "secondary"
            if st.button(
                state,
                key=f"ev_{node.id}_{state}",
                type=btn_type,
                use_container_width=True,
            ):
                ev = dict(st.session_state[_key("evidence")])
                if ev.get(node.id) == state:
                    ev.pop(node.id, None)
                else:
                    ev[node.id] = state
                st.session_state[_key("evidence")] = ev
                st.rerun()
        with col2:
            st.markdown(
                f"<div style='text-align:right; font-family:monospace; "
                f"font-size:1.1rem; font-weight:bold; padding-top:6px;'>"
                f"{prob * 100:.1f}%</div>",
                unsafe_allow_html=True,
            )
        with col3:
            if abs(shift) > 0.001 and evidence:
                color = "#16a34a" if shift > 0 else "#dc2626"
                sign = "+" if shift > 0 else ""
                st.markdown(
                    f"<div style='text-align:center; font-size:0.75rem; "
                    f"color:{color}; font-weight:600; padding-top:8px;'>"
                    f"{sign}{shift * 100:.1f}%</div>",
                    unsafe_allow_html=True,
                )

        # Probability bar
        st.progress(min(prob, 1.0))


# ── Main page render function ─────────────────────────────────────────

def render():
    """Main render function registered with st.navigation."""

    # ── Initialize state ──
    if _key("model_id") not in st.session_state:
        _init_model_state("fusion")  # default to Level 2

    model_id = _get("model_id")
    model = ALL_MODELS[model_id]
    evidence = _get("evidence") or {}
    priors = _get("priors") or model.priors
    cpts = _get("cpts") or model.cpts

    # ── Header ──
    header_l, header_r = st.columns([3, 2])
    with header_l:
        st.title("Bayesian Fusion Lab")
        st.caption("Interactive Probabilistic Graphical Models")
    with header_r:
        col_sel, col_graph, col_solve = st.columns(3)
        with col_sel:
            new_id = st.selectbox(
                "Model",
                list(ALL_MODELS.keys()),
                format_func=lambda k: ALL_MODELS[k].name,
                index=list(ALL_MODELS.keys()).index(model_id),
                label_visibility="collapsed",
            )
            if new_id != model_id:
                _init_model_state(new_id)
                st.rerun()
        with col_graph:
            if st.button("🔗 Structure", use_container_width=True):
                _graph_dialog()
        with col_solve:
            if st.button("🧮 Solve", use_container_width=True, type="primary"):
                _solver_dialog()

    st.markdown(f"**{model.name}** — {model.description}")
    st.divider()

    # ── Sidebar: Scenarios + Priors ──
    with st.sidebar:
        st.header("Learning Scenarios")
        for scenario in model.scenarios:
            help_text = scenario.description or None
            if st.button(
                scenario.name,
                key=f"scenario_{scenario.name}",
                use_container_width=True,
                help=help_text,
            ):
                st.session_state[_key("evidence")] = dict(scenario.evidence)
                st.rerun()

        st.divider()
        st.header("Model Parameters (Priors)")

        for node in model.nodes:
            if node.node_type != "root":
                continue
            primary_state = node.states[0]
            current_val = priors.get(node.id, {}).get(primary_state, 0.5)

            st.markdown(f"**{node.label}**")
            new_val = st.slider(
                f"P({primary_state})",
                min_value=0.01,
                max_value=0.99,
                value=float(current_val),
                step=0.01,
                key=f"prior_slider_{node.id}_{primary_state}",
            )

            if abs(new_val - current_val) > 0.001:
                new_priors = copy.deepcopy(priors)
                new_priors[node.id][primary_state] = new_val
                new_priors[node.id][node.states[1]] = round(1.0 - new_val, 2)
                st.session_state[_key("priors")] = new_priors
                st.rerun()

            other_state = node.states[1]
            other_val = priors.get(node.id, {}).get(other_state, 0.5)
            st.caption(f"P({primary_state})={current_val:.2f}, P({other_state})={other_val:.2f}")

        st.divider()
        if st.button("🔄 Reset Network", use_container_width=True):
            _init_model_state(model_id)
            st.rerun()

    # ── Run Inference ──
    results = run_inference(model, evidence, priors, cpts)

    # Also compute prior marginals (no evidence) for shift indicators
    prior_results = run_inference(model, {}, priors, cpts)

    if results is None:
        st.error("Impossible evidence combination — no consistent states exist.")
        return

    prior_marginals = prior_results.marginals if prior_results else {}

    # ── Node Cards Grid ──
    cols = st.columns(2)
    for i, node in enumerate(model.nodes):
        with cols[i % 2]:
            _render_node_card(node, results.marginals, evidence, prior_marginals)
