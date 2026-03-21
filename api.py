"""FastAPI backend for Probabilistic AI Lab.

Run with:  uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

# Make sure project root is on the Python path when running from any cwd
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from shared.types import CPT
from modules.bayesian_networks.models import ALL_MODELS
from modules.bayesian_networks.inference import run_inference
from modules.bayesian_networks.sampling import (
    prior_sampling,
    rejection_sampling,
    likelihood_weighting,
    gibbs_sampling,
)


OPTIONAL_ROUTER_ERRORS: dict[str, str] = {}


def _try_import_router(module_path: str, attr_name: str, key: str) -> Any | None:
    """Import optional routers without crashing app startup if one module fails."""
    try:
        module = __import__(module_path, fromlist=[attr_name])
        return getattr(module, attr_name)
    except Exception as exc:  # Keep API alive even if one module has missing deps
        OPTIONAL_ROUTER_ERRORS[key] = f"{type(exc).__name__}: {exc}"
        return None


blr_router = _try_import_router(
    "modules.bayesian_linear_regression.router", "router", "blr"
)
kalman_router = _try_import_router(
    "modules.kalman_filter.router", "router", "kalman"
)
gp_router = _try_import_router(
    "modules.gaussian_processes.router", "router", "gp"
)
hmm_router = _try_import_router(
    "modules.hmm.router", "router", "hmm"
)
from modules.bayesian_networks.solver import (
    generate_solver_steps,
    generate_marginal_derivation,
    generate_ve_solver_steps,
)

app = FastAPI(title="Probabilistic AI Lab API", version="0.1.0")

allowed_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True if "*" not in allowed_origins else False,
)

if blr_router is not None:
    app.include_router(blr_router)
if kalman_router is not None:
    app.include_router(kalman_router)
if gp_router is not None:
    app.include_router(gp_router)
if hmm_router is not None:
    app.include_router(hmm_router)


# ── Pydantic schemas ──────────────────────────────────────────────────

class NodeOut(BaseModel):
    id: str
    label: str
    states: list[str]
    node_type: str
    parents: list[str]
    description: str


class ScenarioOut(BaseModel):
    name: str
    evidence: dict[str, str]
    description: str


class CPTTableOut(BaseModel):
    parents: list[str]
    table: dict[str, dict[str, float]]


class ModelSummary(BaseModel):
    id: str
    name: str
    description: str


class ModelDetail(BaseModel):
    id: str
    name: str
    description: str
    nodes: list[NodeOut]
    priors: dict[str, dict[str, float]]
    cpts: dict[str, CPTTableOut]
    scenarios: list[ScenarioOut]


class InferRequest(BaseModel):
    model_id: str
    evidence: dict[str, str] = {}
    priors: dict[str, dict[str, float]] | None = None
    cpts: dict[str, CPTTableOut] | None = None


class InferResponse(BaseModel):
    marginals: dict[str, dict[str, float]]
    total_weight: float


class SolveRequest(BaseModel):
    model_id: str
    evidence: dict[str, str] = {}
    priors: dict[str, dict[str, float]] | None = None
    cpts: dict[str, CPTTableOut] | None = None


class SolveStep(BaseModel):
    title: str
    text: str
    latex: str


class MarginalRequest(BaseModel):
    model_id: str
    node_id: str
    priors: dict[str, dict[str, float]] | None = None
    cpts: dict[str, CPTTableOut] | None = None


class SampleRequest(BaseModel):
    model_id: str
    method: str  # "prior" | "rejection" | "likelihood_weighting" | "gibbs"
    evidence: dict[str, str] = {}
    priors: dict[str, dict[str, float]] | None = None
    cpts: dict[str, CPTTableOut] | None = None
    n_samples: int = 500
    n_burn: int = 100
    n_runs: int = 1


class SamplingMetadata(BaseModel):
    n_samples: int
    n_accepted: int | None = None
    acceptance_rate: float | None = None
    effective_samples: float | None = None
    n_burn: int | None = None
    n_runs: int | None = None


class CIBand(BaseModel):
    low: list[float]
    high: list[float]


class SamplingTrajectory(BaseModel):
    steps: list[int]
    curves: dict[str, dict[str, list[float]]]
    ci_bands: dict[str, dict[str, CIBand]] | None = None


class SamplingResponse(BaseModel):
    method: str
    marginals: dict[str, dict[str, float]]
    exact_marginals: dict[str, dict[str, float]]
    trajectory: SamplingTrajectory
    metadata: SamplingMetadata


# ── Helpers ───────────────────────────────────────────────────────────

def _resolve_cpts(model, cpts_in: dict[str, CPTTableOut] | None) -> dict[str, CPT]:
    """Convert incoming CPT dicts (from JSON) back to CPT dataclass objects."""
    if cpts_in is None:
        return model.cpts
    resolved = {}
    for node_id, cpt_data in cpts_in.items():
        resolved[node_id] = CPT(
            parents=cpt_data.parents,
            table=cpt_data.table,
        )
    # Fill in any nodes not overridden by the client
    for node_id, cpt in model.cpts.items():
        if node_id not in resolved:
            resolved[node_id] = cpt
    return resolved


def _model_to_detail(model_id: str) -> ModelDetail:
    model = ALL_MODELS[model_id]
    return ModelDetail(
        id=model.id,
        name=model.name,
        description=model.description,
        nodes=[
            NodeOut(
                id=n.id,
                label=n.label,
                states=n.states,
                node_type=n.node_type,
                parents=n.parents,
                description=n.description,
            )
            for n in model.nodes
        ],
        priors=model.priors,
        cpts={
            node_id: CPTTableOut(parents=cpt.parents, table=cpt.table)
            for node_id, cpt in model.cpts.items()
        },
        scenarios=[
            ScenarioOut(name=s.name, evidence=s.evidence, description=s.description)
            for s in model.scenarios
        ],
    )


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict[str, Any]:
    """Report API startup health and optional module import errors."""
    return {
        "ok": len(OPTIONAL_ROUTER_ERRORS) == 0,
        "optional_router_errors": OPTIONAL_ROUTER_ERRORS,
    }


@app.get("/api/models", response_model=list[ModelSummary])
def list_models():
    """List all available models."""
    return [
        ModelSummary(id=m.id, name=m.name, description=m.description)
        for m in ALL_MODELS.values()
    ]


@app.get("/api/models/{model_id}", response_model=ModelDetail)
def get_model(model_id: str):
    """Get full model definition (nodes, priors, CPTs, scenarios)."""
    if model_id not in ALL_MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return _model_to_detail(model_id)


@app.post("/api/infer", response_model=InferResponse)
def infer(req: InferRequest):
    """Run exact Bayesian inference for the given evidence."""
    if req.model_id not in ALL_MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not found")
    model = ALL_MODELS[req.model_id]
    priors = req.priors if req.priors is not None else model.priors
    cpts = _resolve_cpts(model, req.cpts)

    result = run_inference(model, req.evidence, priors, cpts)
    if result is None:
        raise HTTPException(status_code=422, detail="Impossible evidence — no consistent states")

    return InferResponse(marginals=result.marginals, total_weight=result.total_weight)


@app.post("/api/solve", response_model=list[SolveStep])
def solve(req: SolveRequest):
    """Generate step-by-step LaTeX derivation for the current inference state."""
    if req.model_id not in ALL_MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not found")
    model = ALL_MODELS[req.model_id]
    priors = req.priors if req.priors is not None else model.priors
    cpts = _resolve_cpts(model, req.cpts)

    result = run_inference(model, req.evidence, priors, cpts)
    if result is None:
        raise HTTPException(status_code=422, detail="Impossible evidence — no consistent states")

    # For the mission model use Variable Elimination (pedagogically richer).
    # The query variable is the leaf node not set as evidence.
    if model.id == "mission":
        leaf_id = "Mission"
        if leaf_id in req.evidence:
            # Mission is observed → query the most informative root/hidden node
            leaf_id = next(
                n.id for n in reversed(model.nodes)
                if n.id not in req.evidence and n.node_type == "root"
            )
        # Efficient elimination order for this DAG
        elim_order = [
            v for v in ["Battery", "Sensors", "Terrain", "Localization", "Navigation"]
            if v != leaf_id and v not in req.evidence
        ]
        steps = generate_ve_solver_steps(
            leaf_id, req.evidence, model, priors, cpts, elim_order
        )
    else:
        steps = generate_solver_steps(model, req.evidence, result, priors, cpts)

    return [SolveStep(**s) for s in steps]


@app.post("/api/bn/sample", response_model=SamplingResponse)
def sample_inference(req: SampleRequest):
    """Run approximate inference via sampling and return estimated marginals + convergence trajectory."""
    if req.model_id not in ALL_MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not found")
    if req.method not in ("prior", "rejection", "likelihood_weighting", "gibbs"):
        raise HTTPException(status_code=400, detail=f"Unknown method '{req.method}'")
    if not (10 <= req.n_samples <= 10000):
        raise HTTPException(status_code=400, detail="n_samples must be between 10 and 10000")

    model = ALL_MODELS[req.model_id]
    priors = req.priors if req.priors is not None else model.priors
    cpts = _resolve_cpts(model, req.cpts)

    # Run approximate inference
    if req.method == "prior":
        result = prior_sampling(model, req.n_samples, priors, cpts, n_runs=max(1, req.n_runs))
    elif req.method == "rejection":
        result = rejection_sampling(model, req.evidence, req.n_samples, priors, cpts, n_runs=max(1, req.n_runs))
    elif req.method == "likelihood_weighting":
        result = likelihood_weighting(model, req.evidence, req.n_samples, priors, cpts, n_runs=max(1, req.n_runs))
    else:  # gibbs
        result = gibbs_sampling(model, req.evidence, req.n_samples, req.n_burn, priors, cpts,
                                n_runs=max(1, req.n_runs))

    # For prior sampling the estimand is P(X) — no evidence — so compare against that.
    # For all other methods the estimand is P(X|e), so compare against exact P(X|e).
    evidence_for_exact = {} if req.method == "prior" else req.evidence
    exact_result = run_inference(model, evidence_for_exact, priors, cpts)
    exact_marginals = exact_result.marginals if exact_result is not None else {}

    meta = result["metadata"]
    return SamplingResponse(
        method=result["method"],
        marginals=result["marginals"],
        exact_marginals=exact_marginals,
        trajectory=SamplingTrajectory(**result["trajectory"]),
        metadata=SamplingMetadata(**meta),
    )


@app.post("/api/marginal-derivation")
def marginal_derivation(req: MarginalRequest):
    """Get LaTeX for Law of Total Probability derivation of a child node's marginal."""
    if req.model_id not in ALL_MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not found")
    model = ALL_MODELS[req.model_id]
    node = next((n for n in model.nodes if n.id == req.node_id), None)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node '{req.node_id}' not found")
    if node.node_type == "root":
        raise HTTPException(status_code=400, detail="Root nodes have no parents to derive from")

    priors = req.priors if req.priors is not None else model.priors
    cpts = _resolve_cpts(model, req.cpts)
    latex = generate_marginal_derivation(node, model, priors, cpts)
    return {"node_id": req.node_id, "latex": latex}


# ── Serve React frontend ─────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).parent / "frontend" / "dist"

if FRONTEND_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="static")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        """Serve the React SPA; fall back to index.html for client-side routing."""
        file = FRONTEND_DIR / full_path
        if full_path and file.is_file():
            return FileResponse(file)
        return FileResponse(FRONTEND_DIR / "index.html")
