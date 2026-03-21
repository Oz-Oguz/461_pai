"""FastAPI router for Gaussian Process Regression."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from modules.gaussian_processes.model import fit, optimize_hyperparameters, solver_steps

router = APIRouter(prefix="/api/gp", tags=["Gaussian Processes"])


# ── Request / response schemas ─────────────────────────────────────────

class FitRequest(BaseModel):
    x_data: list[float] = []
    y_data: list[float] = []
    kernel: str = "rbf"
    length_scale: float = 1.0
    signal_variance: float = 1.0
    noise_variance: float = 0.3
    period: float = 3.14159
    x_min: float = -5.0
    x_max: float = 5.0
    kernel_ref: float = 0.0


class OptimizeRequest(BaseModel):
    x_data: list[float]
    y_data: list[float]
    kernel: str = "rbf"
    length_scale: float = 1.0
    signal_variance: float = 1.0
    noise_variance: float = 0.3
    period: float = 3.14159


class OptimizeResult(BaseModel):
    length_scale: float
    signal_variance: float
    noise_variance: float
    success: bool


class SolveRequest(BaseModel):
    x_data: list[float] = []
    y_data: list[float] = []
    kernel: str = "rbf"
    length_scale: float = 1.0
    signal_variance: float = 1.0
    noise_variance: float = 0.3
    period: float = 3.14159


class SolveStep(BaseModel):
    title: str
    text: str
    latex: str


# ── Endpoints ──────────────────────────────────────────────────────────

@router.post("/fit")
def gp_fit(req: FitRequest) -> dict:
    """Compute GP posterior predictive distribution."""
    return fit(
        x_data=req.x_data,
        y_data=req.y_data,
        kernel=req.kernel,
        length_scale=req.length_scale,
        signal_variance=req.signal_variance,
        noise_variance=req.noise_variance,
        period=req.period,
        x_min=req.x_min,
        x_max=req.x_max,
        kernel_ref=req.kernel_ref,
    )


@router.post("/optimize", response_model=OptimizeResult)
def gp_optimize(req: OptimizeRequest) -> OptimizeResult:
    """Maximise log marginal likelihood to find optimal hyperparameters."""
    result = optimize_hyperparameters(
        x_data=req.x_data,
        y_data=req.y_data,
        kernel=req.kernel,
        length_scale=req.length_scale,
        signal_variance=req.signal_variance,
        noise_variance=req.noise_variance,
        period=req.period,
    )
    return OptimizeResult(**result)


@router.post("/solve", response_model=list[SolveStep])
def gp_solve(req: SolveRequest) -> list[SolveStep]:
    """Return step-by-step LaTeX derivation of the GP posterior."""
    steps = solver_steps(
        x_data=req.x_data,
        y_data=req.y_data,
        kernel=req.kernel,
        length_scale=req.length_scale,
        signal_variance=req.signal_variance,
        noise_variance=req.noise_variance,
        period=req.period,
    )
    return [SolveStep(**s) for s in steps]
