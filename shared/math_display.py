"""LaTeX rendering helpers for Streamlit."""

from __future__ import annotations

import streamlit as st


def derivation_step(title: str, explanation: str, latex: str):
    """Render one step of a math derivation: title, prose, then equation."""
    st.markdown(f"**{title}**")
    st.markdown(explanation)
    st.latex(latex)
    st.divider()


def derivation_panel(steps: list[dict]):
    """Render a full multi-step derivation."""
    for i, step in enumerate(steps):
        st.markdown(f"**{step['title']}**")
        st.markdown(step["text"])
        st.latex(step["latex"])
        if i < len(steps) - 1:
            st.divider()
