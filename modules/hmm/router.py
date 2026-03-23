"""FastAPI router for the HMM module."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from modules.hmm.model import (
    forward_algorithm,
    viterbi_algorithm,
    create_robot_world,
    robot_localization_step,
    initialize_robot_belief,
    solver_steps_forward,
    solver_steps_viterbi,
    bayesian_filtering_time_step,
    bayesian_filtering_observe_step,
    solver_steps_filtering,
    build_transition_matrix,
)

router = APIRouter(prefix="/api/hmm", tags=["Hidden Markov Models"])


# ── Request/Response Models ───────────────────────────────────────────


class ForwardRequest(BaseModel):
    observations: list[str]
    states: list[str]
    transition: dict[str, dict[str, float]]
    emission: dict[str, dict[str, float]]
    prior: dict[str, float]


class ViterbiRequest(BaseModel):
    observations: list[str]
    states: list[str]
    transition: dict[str, dict[str, float]]
    emission: dict[str, dict[str, float]]
    prior: dict[str, float]


class RobotWorldRequest(BaseModel):
    grid_size: tuple[int, int] = (10, 10)
    walls: list[tuple[int, int]] | None = None


class RobotStepRequest(BaseModel):
    belief: list[list[float]]
    grid_size: tuple[int, int]
    walls: list[tuple[int, int]]
    action: str | None = None
    observation: dict | None = None
    action_noise: float = 0.1
    sensor_noise: float = 0.2


class RobotInitRequest(BaseModel):
    grid_size: tuple[int, int]
    walls: list[tuple[int, int]]
    uniform: bool = True
    position: tuple[int, int] | None = None


class FilteringTimeRequest(BaseModel):
    belief: list[list[float]]
    grid_size: tuple[int, int]
    walls: list[tuple[int, int]]
    transition_noise: float = 0.2
    transition_model: str = 'uniform'


class FilteringObserveRequest(BaseModel):
    belief: list[list[float]]
    grid_size: tuple[int, int]
    walls: list[tuple[int, int]]
    observation: tuple[int, int]  # (row, col) of sensed position
    sensor_noise: float = 0.3


class TransitionMatrixRequest(BaseModel):
    grid_size: tuple[int, int]
    walls: list[tuple[int, int]]
    transition_noise: float = 0.2
    transition_model: str = 'uniform'


class SolveStep(BaseModel):
    title: str
    text: str
    latex: str


# ── Endpoints ─────────────────────────────────────────────────────────


@router.post("/forward")
def hmm_forward(req: ForwardRequest) -> dict:
    """Run the Forward algorithm for filtering and likelihood computation."""
    return forward_algorithm(
        observations=req.observations,
        states=req.states,
        transition=req.transition,
        emission=req.emission,
        prior=req.prior,
    )


@router.post("/viterbi")
def hmm_viterbi(req: ViterbiRequest) -> dict:
    """Run the Viterbi algorithm to find the most likely state sequence."""
    return viterbi_algorithm(
        observations=req.observations,
        states=req.states,
        transition=req.transition,
        emission=req.emission,
        prior=req.prior,
    )


@router.post("/robot/world")
def robot_world(req: RobotWorldRequest) -> dict:
    """Create a robot world configuration."""
    return create_robot_world(
        grid_size=req.grid_size,
        walls=req.walls,
    )


@router.post("/robot/init")
def robot_init(req: RobotInitRequest) -> dict:
    """Initialize robot belief state."""
    import numpy as np
    belief = initialize_robot_belief(
        grid_size=req.grid_size,
        walls=req.walls,
        uniform=req.uniform,
        position=req.position,
    )
    return {"belief": belief.tolist()}


@router.post("/robot/step")
def robot_step(req: RobotStepRequest) -> dict:
    """Perform one step of robot localization (predict or update)."""
    import numpy as np
    belief = np.array(req.belief)
    return robot_localization_step(
        belief=belief,
        grid_size=req.grid_size,
        walls=req.walls,
        action=req.action,
        observation=req.observation,
        action_noise=req.action_noise,
        sensor_noise=req.sensor_noise,
    )


@router.post("/filtering/time")
def filtering_time(req: FilteringTimeRequest) -> dict:
    """Apply the Passage-of-Time phase: B'(X') = Σ P(X'|x) B(x)."""
    import numpy as np
    belief = np.array(req.belief)
    return bayesian_filtering_time_step(
        belief=belief,
        grid_size=req.grid_size,
        walls=req.walls,
        transition_noise=req.transition_noise,
        transition_model=req.transition_model,
    )


@router.post("/filtering/observe")
def filtering_observe(req: FilteringObserveRequest) -> dict:
    """Apply the Observation phase: B(X) ∝ P(e|X) B'(X)."""
    import numpy as np
    belief = np.array(req.belief)
    return bayesian_filtering_observe_step(
        belief=belief,
        grid_size=req.grid_size,
        walls=req.walls,
        observation=req.observation,
        sensor_noise=req.sensor_noise,
    )


@router.post("/filtering/transition-matrix")
def filtering_transition_matrix(req: TransitionMatrixRequest) -> dict:
    """Return transition matrix T[i][j]=P(X'=j|X=i) over free cells."""
    return build_transition_matrix(
        grid_size=req.grid_size,
        walls=req.walls,
        transition_noise=req.transition_noise,
        transition_model=req.transition_model,
    )


@router.get("/solve/filtering", response_model=list[SolveStep])
def solve_filtering() -> list[SolveStep]:
    """Return step-by-step derivation of Bayesian Filtering (two-phase view)."""
    steps = solver_steps_filtering()
    return [SolveStep(**s) for s in steps]


@router.get("/solve/forward", response_model=list[SolveStep])
def solve_forward() -> list[SolveStep]:
    """Return step-by-step derivation of the Forward algorithm."""
    steps = solver_steps_forward()
    return [SolveStep(**s) for s in steps]


@router.get("/solve/viterbi", response_model=list[SolveStep])
def solve_viterbi() -> list[SolveStep]:
    """Return step-by-step derivation of the Viterbi algorithm."""
    steps = solver_steps_viterbi()
    return [SolveStep(**s) for s in steps]
