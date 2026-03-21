"""FastAPI router for Bayesian Linear Regression."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from modules.bayesian_linear_regression.model import fit, solver_steps

router = APIRouter(prefix="/api/blr", tags=["Bayesian Linear Regression"])


class FitRequest(BaseModel):
    x_data: list[float] = []
    y_data: list[float] = []
    prior_variance: float = 1.0
    noise_variance: float = 0.3
    degree: int = 1
    x_min: float = -3.0
    x_max: float = 3.0
    basis: str = "polynomial"


class SolveRequest(BaseModel):
    x_data: list[float] = []
    y_data: list[float] = []
    prior_variance: float = 1.0
    noise_variance: float = 0.3
    degree: int = 1


class SolveStep(BaseModel):
    title: str
    text: str
    latex: str


class EvidenceRequest(BaseModel):
    x_data: list[float]
    y_data: list[float]
    prior_variance: float = 1.0
    noise_variance: float = 0.3
    degrees: list[int] = [1, 2, 3, 4]


class EvidenceResult(BaseModel):
    degree: int
    log_evidence: float | None


@router.post("/fit")
def blr_fit(req: FitRequest) -> dict:
    """Compute posterior and predictive distribution."""
    return fit(
        x_data=req.x_data,
        y_data=req.y_data,
        prior_variance=req.prior_variance,
        noise_variance=req.noise_variance,
        degree=req.degree,
        x_min=req.x_min,
        x_max=req.x_max,
        basis=req.basis,
    )


@router.post("/solve", response_model=list[SolveStep])
def blr_solve(req: SolveRequest) -> list[SolveStep]:
    """Return step-by-step LaTeX derivation."""
    steps = solver_steps(
        x_data=req.x_data,
        y_data=req.y_data,
        prior_variance=req.prior_variance,
        noise_variance=req.noise_variance,
        degree=req.degree,
    )
    return [SolveStep(**s) for s in steps]


@router.post("/evidence", response_model=list[EvidenceResult])
def blr_evidence(req: EvidenceRequest) -> list[EvidenceResult]:
    """Compute log marginal likelihood for multiple polynomial degrees.

    Used by the model evidence chart to compare degrees without re-running
    the full fit for each. Returns log p(y | degree, α, β) for model selection.
    """
    results = []
    for deg in req.degrees:
        result = fit(
            x_data=req.x_data,
            y_data=req.y_data,
            prior_variance=req.prior_variance,
            noise_variance=req.noise_variance,
            degree=deg,
        )
        results.append(
            EvidenceResult(
                degree=deg,
                log_evidence=result["log_marginal_likelihood"],
            )
        )
    return results
