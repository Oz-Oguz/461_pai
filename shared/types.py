from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    id: str
    label: str
    states: list[str]
    node_type: str  # "root" or "child"
    parents: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class CPT:
    parents: list[str]
    # key = comma-joined parent state values, value = {state: probability}
    table: dict[str, dict[str, float]]


@dataclass
class Scenario:
    name: str
    evidence: dict[str, str]
    description: str = ""


@dataclass
class BayesianNetworkModel:
    id: str
    name: str
    description: str
    nodes: list[Node]
    priors: dict[str, dict[str, float]]
    cpts: dict[str, CPT]
    scenarios: list[Scenario]


@dataclass
class InferenceResult:
    marginals: dict[str, dict[str, float]]
    total_weight: float  # P(Evidence) -- normalization constant
