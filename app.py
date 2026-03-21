"""Probabilistic AI Lab — Interactive educational web app.

Run with:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Probabilistic AI Lab",
    page_icon="🎲",
    layout="wide",
)

from modules.bayesian_networks.page import render as bn_render

pages = {
    "Graphical Models": [
        st.Page(bn_render, title="Bayesian Networks", icon=":material/hub:"),
    ],
    # ── Future modules (uncomment as implemented) ──
    # "Regression & GPs": [
    #     st.Page(..., title="Bayesian Linear Regression", icon=":material/show_chart:"),
    #     st.Page(..., title="Gaussian Processes", icon=":material/ssid_chart:"),
    # ],
    # "Approximate Inference": [
    #     st.Page(..., title="Variational Inference", icon=":material/tune:"),
    #     st.Page(..., title="MCMC Methods", icon=":material/shuffle:"),
    # ],
    # "Deep Probabilistic Models": [
    #     st.Page(..., title="Bayesian Neural Networks", icon=":material/neurology:"),
    # ],
    # "Sequential Decisions": [
    #     st.Page(..., title="Active Learning", icon=":material/school:"),
    #     st.Page(..., title="Bayesian Optimization", icon=":material/target:"),
    #     st.Page(..., title="Reinforcement Learning", icon=":material/smart_toy:"),
    # ],
}

pg = st.navigation(pages)
pg.run()
