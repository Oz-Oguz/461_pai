"""FastAPI router for the Kalman Filter module."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from modules.kalman_filter.model import simulate, solver_steps

router = APIRouter(prefix="/api/kalman", tags=["Kalman Filter"])


class SimulateRequest(BaseModel):
    n_steps: int = 30
    process_noise_q: float = 0.5
    measurement_noise_r: float = 2.0
    x0_true: float = 0.0
    seed: int = 42


class SolveRequest(BaseModel):
    process_noise_q: float = 0.5
    measurement_noise_r: float = 2.0


class SolveStep(BaseModel):
    title: str
    text: str
    latex: str


@router.post("/simulate")
def kalman_simulate(req: SimulateRequest) -> dict:
    """Run the full simulation and return all trajectories."""
    return simulate(
        n_steps=req.n_steps,
        process_noise_q=req.process_noise_q,
        measurement_noise_r=req.measurement_noise_r,
        x0_true=req.x0_true,
        seed=req.seed,
    )


@router.post("/solve", response_model=list[SolveStep])
def kalman_solve(req: SolveRequest) -> list[SolveStep]:
    """Return step-by-step LaTeX derivation of the Kalman filter equations."""
    steps = solver_steps(req.process_noise_q, req.measurement_noise_r)
    return [SolveStep(**s) for s in steps]
