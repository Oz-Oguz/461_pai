"""DAG visualization using Graphviz DOT strings for st.graphviz_chart()."""

from __future__ import annotations

from shared.types import BayesianNetworkModel


def dag_to_dot(
    model: BayesianNetworkModel,
    evidence: dict[str, str] | None = None,
) -> str:
    """Generate a Graphviz DOT string for the Bayesian Network."""
    evidence = evidence or {}
    lines = [
        "digraph BN {",
        "  rankdir=TB;",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica"];',
        '  edge [color="#64748b"];',
    ]

    for node in model.nodes:
        if node.id in evidence:
            color = "#bfdbfe"
            border = "#2563eb"
        elif node.node_type == "root":
            color = "#e0f2fe"
            border = "#0284c7"
        else:
            color = "#f0fdf4"
            border = "#16a34a"

        lines.append(
            f'  {node.id} [label="{node.label}", '
            f'fillcolor="{color}", color="{border}", penwidth=2];'
        )

    for node in model.nodes:
        for parent in node.parents:
            lines.append(f"  {parent} -> {node.id};")

    lines.append("}")
    return "\n".join(lines)
