# Probabilistic AI Lab

An interactive web application and slide-deck suite for teaching **Probabilistic Artificial Intelligence** — developed at Bilkent University for use in lectures, labs, and self-study.

The tool lets students build intuition through hands-on exploration: click to add observations, watch posteriors update in real time, pan and zoom weight-space Gaussians, and follow guided "Explore" steps that bridge each concept to the math.

---

## Motivation

Probabilistic reasoning is foundational to modern AI — from sensor fusion in robots to uncertainty quantification in neural networks — yet the core ideas (prior belief, posterior update, epistemic vs. aleatoric uncertainty) remain abstract when only encountered as equations.

This lab was built for a research group that works on **physical and embodied AI**: sequential decision-making, task planning, and physical manipulation. Every module is framed around concrete robotics problems (sensor calibration, state estimation, motion prediction) so students immediately see why principled uncertainty matters in real systems.

The design philosophy is: *show the math, then let students break it interactively*.

---

## Content

Four fully interactive modules, each pairing a lecture slide deck with a live tool:

| # | Module | Key concepts |
|---|--------|-------------|
| 1 | **Bayesian Networks** | Conditional independence, CPTs, Variable Elimination, d-separation, sensor fusion |
| 2 | **Bayesian Linear Regression** | Gaussian prior/posterior, epistemic vs. aleatoric uncertainty, basis functions, model evidence |
| 3 | **Kalman Filter** | Predict–update cycle, 1D Gaussian tracking, process and observation noise |
| 4 | **Gaussian Processes** | Kernel functions (RBF, Matérn, periodic), GP posterior, hyperparameter effects |

Each module includes:
- **Overview** — plain-language description with key equations
- **Algorithm** — step-by-step derivation panel
- **Explore** — guided interactive experiments (8 steps for BLR, 7 for others)
- **References** — primary textbooks and papers

### Lecture slides

Marp-based Markdown slide decks in `slides/`, one per module.
They reference the interactive tool at each key step with `PAI Tool:` callout boxes.

---

## Architecture

```
PAI/
├── api.py                          # FastAPI backend (entry point)
├── modules/
│   ├── bayesian_networks/          # BN inference engine + models
│   ├── bayesian_linear_regression/ # BLR posterior & evidence
│   ├── kalman_filter/              # 1D Kalman filter simulation
│   └── gaussian_processes/         # GP posterior with multiple kernels
├── shared/                         # Shared Python utilities
├── frontend/                       # React + TypeScript + Vite
│   └── src/
│       ├── pages/                  # One page component per module
│       ├── components/             # InfoPanel, RangeSlider, MathBlock, …
│       └── content/                # Structured text content per module
└── slides/                         # Marp Markdown slide decks
```

**Backend:** Python 3.13 · FastAPI · NumPy · SciPy · Uvicorn
**Frontend:** React 19 · TypeScript · Vite · Tailwind CSS 4 · KaTeX · ReactFlow

The frontend dev server proxies all `/api` requests to the backend at `localhost:8000`.

---

## Installation

### Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | ≥ 3.13 | The venv ships with 3.13 |
| Node.js | ≥ 18 | For the frontend dev server |
| npm | ≥ 9 | Bundled with Node |
| uv *(optional)* | latest | Faster Python dependency install |

### 1 — Clone the repository

```bash
git clone <repo-url>
cd PAI
```

### 2 — Python environment

**Option A — uv (recommended, fast):**
```bash
uv venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows (PowerShell)
uv pip install -e .
```

**Option B — standard pip:**
```bash
python3.13 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows (PowerShell)
pip install -e .
```

> **Important:** every time you open a new terminal to run the backend, you must activate the virtual environment first (`source .venv/bin/activate`). Without this step Python will not find the installed packages.

### 3 — Node dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Running the app

Two processes must run simultaneously — open two terminal windows.

**Terminal 1 — Backend:**
```bash
cd PAI
source .venv/bin/activate      # macOS / Linux  |  .venv\Scripts\activate on Windows
uv run uvicorn api:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd PAI/frontend
npm run dev
```

Then open **http://localhost:5173** in your browser.

> The `--reload` flag automatically restarts the backend when Python files change.
> The Vite dev server hot-reloads the frontend on every save.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'fastapi'` | Virtual environment not activated, or `pip install -e .` was not run | `source .venv/bin/activate` then `pip install -e .` |
| `ModuleNotFoundError: No module named 'numpy'` | Same as above | Same fix |
| `uvicorn: command not found` | Venv not activated | Activate the venv; uvicorn is installed inside it |
| Frontend shows "Failed to fetch" / network error | Backend is not running or running on a different port | Start the backend in Terminal 1 first; check it listens on port 8000 |
| `npm: command not found` | Node.js not installed | Install Node.js ≥ 18 from [nodejs.org](https://nodejs.org) |

---

## Lecture slides

Slides are written in [Marp](https://marp.app/) Markdown (`slides/*.md`) and rendered to PDF, HTML, or PPTX.

### Install Marp CLI (once)

```bash
npm install -g @marp-team/marp-cli
```

### Render a single deck to PDF

```bash
marp --pdf --allow-local-files slides/02_bayesian_linear_regression.md -o slides/pdfs/blr.pdf
```

### Render all decks to PDF

```bash
for f in slides/*.md; do
  marp --pdf --allow-local-files "$f" -o "slides/pdfs/$(basename "$f" .md).pdf"
done
```

> `--allow-local-files` is required to embed local images and fonts.
> Equations use **KaTeX** and render natively in PDF and HTML output.

---

## Configuration

### CORS origins

The backend allows requests from `localhost:5173` and `localhost:3000` by default (set in `api.py`).
If you serve the frontend from a different port, add it to the `allow_origins` list in `api.py`:

```python
allow_origins=["http://localhost:5173", "http://localhost:YOUR_PORT"],
```

### API proxy target

The frontend proxies `/api` to `http://localhost:8000` (configured in `frontend/vite.config.ts`).
Change the `target` there if you run the backend on a different port:

```ts
proxy: {
  '/api': {
    target: 'http://localhost:8000',  // ← change if needed
    changeOrigin: true,
  },
},
```

### Production build

To serve everything from the FastAPI backend (single process, no Vite):

```bash
cd frontend
npm run build          # compiles to frontend/dist/
cd ..
uvicorn api:app --port 8000
```

The backend is configured to serve `frontend/dist` as static files and fall back to `index.html` for client-side routing.

---

## Module overview

### Bayesian Networks
Three robot models of increasing complexity (Robot Battery → Sensor Fusion → Mission Planning).
Students set evidence on observed nodes and watch belief propagation update all connected variables.
Includes a Variable Elimination solver with step-by-step derivation.

### Bayesian Linear Regression
Click on the canvas to add observations. The posterior mean and uncertainty band update immediately.
Features: polynomial and RBF basis functions, weight-space posterior ellipse (with pan/zoom), model evidence chart for automatic complexity selection, and a sensor calibration scenario that demonstrates extrapolation danger.

### Kalman Filter
Animated predict–update cycle over a 1D random walk.
Students control process noise Q and observation noise R and watch the Kalman gain and uncertainty band evolve in real time.

### Gaussian Processes
Click to add observations; the GP posterior updates continuously.
Supports RBF, Matérn 3/2, Matérn 5/2, and periodic kernels with adjustable length scale, signal variance, and noise variance.

---

## References

- Krause & Hübotter — *Probabilistic Artificial Intelligence* (ETH Zürich, 2025) · [arxiv.org/abs/2502.05244](https://arxiv.org/abs/2502.05244)
- Bishop — *Pattern Recognition and Machine Learning* (Springer, 2006)
- Murphy — *Machine Learning: A Probabilistic Perspective* (MIT Press, 2012)
- Rasmussen & Williams — *Gaussian Processes for Machine Learning* (MIT Press, 2006)

---

## License

For educational use. Contact the author for redistribution or derivative works.
