import { useState, useEffect, useRef, useCallback } from 'react';
import { Calculator, Trash2, Info, Loader2, Eye, EyeOff, Zap } from 'lucide-react';
import { SolverModal } from '../components/SolverModal';
import { InfoPanel } from '../components/InfoPanel';
import { RangeSlider } from '../components/RangeSlider';
import { gpContent } from '../content/gp';
import type { SolveStep, LearnTab } from '../types';

// ── API helpers ────────────────────────────────────────────────────────

interface GPResult {
  n_data: number;
  x_grid: number[];
  pred_mean: number[];
  pred_std: number[];
  pred_std2: number[];
  posterior_samples: number[][] | null;
  log_marginal_likelihood: number | null;
  kernel_slice: number[];
  is_prior: boolean;
  x_data: number[];
  y_data: number[];
}

interface OptimiseResult {
  length_scale: number;
  signal_variance: number;
  noise_variance: number;
  success: boolean;
}

type KernelId = 'rbf' | 'matern32' | 'linear' | 'periodic';

async function parseApiResponse<T>(res: Response): Promise<T> {
  const body = await res.text();
  if (!res.ok) {
    let detail = res.statusText || 'API error';
    if (body) {
      try {
        const parsed = JSON.parse(body) as { detail?: string };
        detail = parsed.detail ?? detail;
      } catch {
        detail = body;
      }
    }
    throw new Error(detail);
  }
  return JSON.parse(body) as T;
}

async function apiFit(
  xData: number[], yData: number[],
  kernel: KernelId, lengthScale: number, signalVar: number, noiseVar: number,
  period: number, kernelRef: number,
  xMin: number, xMax: number,
): Promise<GPResult> {
  const res = await fetch('/api/gp/fit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      x_data: xData, y_data: yData,
      kernel, length_scale: lengthScale, signal_variance: signalVar,
      noise_variance: noiseVar, period, kernel_ref: kernelRef,
      x_min: xMin, x_max: xMax,
    }),
  });
  return parseApiResponse<GPResult>(res);
}

async function apiOptimise(
  xData: number[], yData: number[],
  kernel: KernelId, lengthScale: number, signalVar: number, noiseVar: number,
  period: number,
): Promise<OptimiseResult> {
  const res = await fetch('/api/gp/optimize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      x_data: xData, y_data: yData,
      kernel, length_scale: lengthScale, signal_variance: signalVar,
      noise_variance: noiseVar, period,
    }),
  });
  return parseApiResponse<OptimiseResult>(res);
}

async function apiSolve(
  xData: number[], yData: number[],
  kernel: KernelId, lengthScale: number, signalVar: number, noiseVar: number,
  period: number,
): Promise<SolveStep[]> {
  const res = await fetch('/api/gp/solve', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      x_data: xData, y_data: yData,
      kernel, length_scale: lengthScale, signal_variance: signalVar,
      noise_variance: noiseVar, period,
    }),
  });
  return parseApiResponse<SolveStep[]>(res);
}

// ── Pre-defined scenarios ─────────────────────────────────────────────

interface Scenario {
  name: string;
  description: string;
  points: { x: number; y: number }[];
  kernel: KernelId;
  lengthScale: number;
  signalVar: number;
  noiseVar: number;
  period: number;
}

const SCENARIOS: Scenario[] = [
  {
    name: 'Clear (Prior)',
    description: 'No data — explore the prior over functions',
    points: [],
    kernel: 'rbf', lengthScale: 1.0, signalVar: 1.0, noiseVar: 0.1, period: 3.14,
  },
  {
    name: 'Posterior Update',
    description: 'Classic GP regression — uncertainty collapses near data',
    points: [
      { x: -3, y: -0.5 }, { x: -1, y: 0.4 }, { x: 0, y: 0.0 },
      { x: 2, y: 1.2 }, { x: 4, y: 0.6 },
    ],
    kernel: 'rbf', lengthScale: 1.0, signalVar: 1.0, noiseVar: 0.1, period: 3.14,
  },
  {
    name: 'Noisy Observations',
    description: 'High σ²_n → smooth interpolation, not exact fit',
    points: [
      { x: -4, y: -1.2 }, { x: -2.5, y: 0.3 }, { x: -1, y: -0.5 },
      { x: 0.5, y: 0.8 }, { x: 1.5, y: -0.2 }, { x: 2.5, y: 1.1 },
      { x: 3.5, y: 0.4 }, { x: 4.5, y: -0.6 },
    ],
    kernel: 'rbf', lengthScale: 1.0, signalVar: 1.0, noiseVar: 0.5, period: 3.14,
  },
  {
    name: 'Short Length Scale',
    description: 'ℓ=0.3 → wiggly fit. Drag ℓ slider to compare',
    points: [
      { x: -3, y: 0.2 }, { x: -1.5, y: -0.8 }, { x: 0, y: 0.5 },
      { x: 1.5, y: -0.3 }, { x: 3, y: 0.7 },
    ],
    kernel: 'rbf', lengthScale: 0.3, signalVar: 1.0, noiseVar: 0.05, period: 3.14,
  },
  {
    name: 'Periodic Pattern',
    description: 'Periodic kernel extrapolates oscillations correctly',
    points: [
      { x: -4.7, y: 0.0 }, { x: -3.1, y: 1.0 }, { x: -1.6, y: 0.0 },
      { x: 0.0, y: -1.0 }, { x: 1.6, y: 0.0 }, { x: 3.1, y: 1.0 },
      { x: 4.7, y: 0.0 },
    ],
    kernel: 'periodic', lengthScale: 0.7, signalVar: 1.2, noiseVar: 0.05, period: 3.14,
  },
  {
    name: 'Extrapolation',
    description: 'Data only on left — prior dominates the right',
    points: [
      { x: -4.5, y: -0.8 }, { x: -3.5, y: 0.2 }, { x: -2.5, y: 0.9 },
      { x: -1.5, y: 0.4 }, { x: -0.5, y: -0.3 },
    ],
    kernel: 'rbf', lengthScale: 1.0, signalVar: 1.0, noiseVar: 0.1, period: 3.14,
  },
];

// ── SVG canvas constants ──────────────────────────────────────────────

const W = 620, H = 420;
const PAD = { top: 20, right: 20, bottom: 40, left: 50 };
const X_DEF = { min: -5, max: 5 }, Y_DEF = { min: -4, max: 4 };

// ── Kernel panel constants ─────────────────────────────────────────────

const KERNEL_COLORS: Record<KernelId, string> = {
  rbf: '#2563eb',
  matern32: '#7c3aed',
  linear: '#d97706',
  periodic: '#0d9488',
};

const KERNEL_LABELS: Record<KernelId, string> = {
  rbf: 'RBF',
  matern32: 'Matérn 3/2',
  linear: 'Linear',
  periodic: 'Periodic',
};

// ── KernelPanel sub-component ─────────────────────────────────────────

function KernelPanel({
  xGrid, kernelSlice, kernelRef, kernel, signalVar, xMin, xMax,
}: {
  xGrid: number[];
  kernelSlice: number[];
  kernelRef: number;
  kernel: KernelId;
  signalVar: number;
  xMin: number;
  xMax: number;
}) {
  const KW = 440, KH = 100;
  const KPAD = { top: 12, right: 16, bottom: 28, left: 44 };
  const kColor = KERNEL_COLORS[kernel];
  const kLabel = KERNEL_LABELS[kernel];

  const kInnerW = KW - KPAD.left - KPAD.right;
  const kInnerH = KH - KPAD.top - KPAD.bottom;

  const kToX = (x: number) =>
    KPAD.left + ((x - xMin) / (xMax - xMin)) * kInnerW;
  const kToY = (v: number) => {
    const yMax = Math.max(signalVar * 1.05, 0.1);
    return KH - KPAD.bottom - (v / yMax) * kInnerH;
  };

  const path = xGrid
    .map((x, i) => `${i === 0 ? 'M' : 'L'} ${kToX(x).toFixed(1)},${kToY(kernelSlice[i]).toFixed(1)}`)
    .join(' ');

  const refX = kToX(kernelRef);
  const yMax = Math.max(signalVar * 1.05, 0.1);

  // y-axis ticks
  const yTicks = [0, yMax / 2, yMax].map(v => ({
    v,
    y: kToY(v),
    label: v.toFixed(1),
  }));

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-4 mb-6">
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-sm font-bold text-slate-700">Kernel Shape</h3>
        <span className="text-xs text-slate-400">
          k(x<sub>ref</sub>, x) — correlation with reference point
        </span>
      </div>
      <div className="flex justify-center">
        <svg viewBox={`0 0 ${KW} ${KH}`} width={KW} height={KH}>
          {/* Background */}
          <rect x={KPAD.left} y={KPAD.top} width={kInnerW} height={kInnerH} fill="#f8fafc" />

          {/* Reference line */}
          <line
            x1={refX} y1={KPAD.top}
            x2={refX} y2={KH - KPAD.bottom}
            stroke="#94a3b8" strokeWidth={1} strokeDasharray="4 3"
          />

          {/* Kernel curve */}
          <path d={path} fill="none" stroke={kColor} strokeWidth={2} strokeLinecap="round" />

          {/* Filled area */}
          <path
            d={`${path} L ${kToX(xGrid[xGrid.length - 1]).toFixed(1)},${KH - KPAD.bottom} L ${kToX(xGrid[0]).toFixed(1)},${KH - KPAD.bottom} Z`}
            fill={kColor} fillOpacity={0.08}
          />

          {/* Axes */}
          <line x1={KPAD.left} y1={KH - KPAD.bottom} x2={KW - KPAD.right} y2={KH - KPAD.bottom} stroke="#94a3b8" strokeWidth={1} />
          <line x1={KPAD.left} y1={KPAD.top} x2={KPAD.left} y2={KH - KPAD.bottom} stroke="#94a3b8" strokeWidth={1} />

          {/* X-axis labels */}
          {Array.from({ length: 5 }, (_, i) => Math.round((xMin + (i / 4) * (xMax - xMin)) * 10) / 10).map(v => (
            <text key={v} x={kToX(v)} y={KH - KPAD.bottom + 12}
              textAnchor="middle" fontSize={9} fill="#94a3b8">{v}</text>
          ))}

          {/* Y-axis ticks */}
          {yTicks.map(({ v, y, label }) => (
            <g key={v}>
              <line x1={KPAD.left - 3} y1={y} x2={KPAD.left} y2={y} stroke="#94a3b8" strokeWidth={1} />
              <text x={KPAD.left - 6} y={y + 3} textAnchor="end" fontSize={8} fill="#94a3b8">{label}</text>
            </g>
          ))}

          {/* Kernel label */}
          <text x={KW - KPAD.right - 4} y={KPAD.top + 10}
            textAnchor="end" fontSize={10} fontWeight={600} fill={kColor}>
            {kLabel}
          </text>

          {/* x_ref annotation */}
          <text x={refX + 4} y={KPAD.top + 10}
            fontSize={9} fill="#94a3b8">
            x_ref={kernelRef.toFixed(1)}
          </text>
        </svg>
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────

interface GPProps {
  learnOpen: boolean;
  learnTab: LearnTab;
  panelWidth: number;
  onLearnToggle: () => void;
  onLearnClose: () => void;
  onLearnWidthChange: (w: number) => void;
  onLearnTabChange: (t: LearnTab) => void;
}

export function GaussianProcesses({
  learnOpen,
  learnTab,
  panelWidth,
  onLearnToggle,
  onLearnClose,
  onLearnWidthChange,
  onLearnTabChange,
}: GPProps) {
  const [points, setPoints] = useState<{ x: number; y: number }[]>([]);
  const [kernel, setKernel] = useState<KernelId>('rbf');
  const [lengthScale, setLengthScale] = useState(1.0);
  const [signalVar, setSignalVar] = useState(1.0);
  const [noiseVar, setNoiseVar] = useState(0.3);
  const [period, setPeriod] = useState(3.14);
  const [showSamples, setShowSamples] = useState(false);
  const [kernelRef, setKernelRef] = useState(0.0);
  const [result, setResult] = useState<GPResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [optimising, setOptimising] = useState(false);
  const [solverSteps, setSolverSteps] = useState<SolveStep[] | null>(null);
  const [solverLoading, setSolverLoading] = useState(false);
  const [xRange, setXRange] = useState(X_DEF);
  const [yRange, setYRange] = useState(Y_DEF);
  const isRangeDefault = xRange === X_DEF && yRange === Y_DEF;

  const svgRef = useRef<SVGSVGElement>(null);

  // Dynamic coordinate functions (close over xRange / yRange)
  const toSvgX = (x: number) => PAD.left + ((x - xRange.min) / (xRange.max - xRange.min)) * (W - PAD.left - PAD.right);
  const toSvgY = (y: number) => H - PAD.bottom - ((y - yRange.min) / (yRange.max - yRange.min)) * (H - PAD.top - PAD.bottom);
  const bandPath = (xs: number[], means: number[], stds: number[], sigma: number): string => {
    const upper = means.map((m, i) => m + sigma * stds[i]);
    const lower = means.map((m, i) => m - sigma * stds[i]);
    const top = xs.map((x, i) => `${toSvgX(x).toFixed(1)},${toSvgY(upper[i]).toFixed(1)}`).join(' ');
    const bot = [...xs].reverse().map((x, i) => {
      const ri = xs.length - 1 - i;
      return `${toSvgX(x).toFixed(1)},${toSvgY(lower[ri]).toFixed(1)}`;
    }).join(' ');
    return `M ${top.split(' ')[0]} L ${top} L ${bot} Z`;
  };
  const linePath = (xs: number[], ys: number[]): string =>
    xs.map((x, i) => `${i === 0 ? 'M' : 'L'} ${toSvgX(x).toFixed(1)},${toSvgY(ys[i]).toFixed(1)}`).join(' ');
  const xTicks = Array.from({ length: 7 }, (_, i) => Math.round((xRange.min + (i / 6) * (xRange.max - xRange.min)) * 10) / 10);
  const yTicks = Array.from({ length: 5 }, (_, i) => Math.round((yRange.min + (i / 4) * (yRange.max - yRange.min)) * 10) / 10);
  const gridLines = () => {
    const els = [];
    for (const x of xTicks) {
      const sx = toSvgX(x);
      els.push(<line key={`vx${x}`} x1={sx} y1={PAD.top} x2={sx} y2={H - PAD.bottom} stroke="#e2e8f0" strokeWidth={1} />);
      els.push(<text key={`lx${x}`} x={sx} y={H - PAD.bottom + 16} textAnchor="middle" fontSize={11} fill="#94a3b8">{x}</text>);
    }
    for (const y of yTicks) {
      const sy = toSvgY(y);
      els.push(<line key={`hy${y}`} x1={PAD.left} y1={sy} x2={W - PAD.right} y2={sy} stroke="#e2e8f0" strokeWidth={1} />);
      els.push(<text key={`ly${y}`} x={PAD.left - 8} y={sy + 4} textAnchor="end" fontSize={11} fill="#94a3b8">{y}</text>);
    }
    return els;
  };

  // Fetch GP fit whenever anything changes
  useEffect(() => {
    setLoading(true);
    apiFit(
      points.map(p => p.x), points.map(p => p.y),
      kernel, lengthScale, signalVar, noiseVar, period, kernelRef,
      xRange.min, xRange.max,
    )
      .then(setResult)
      .finally(() => setLoading(false));
  }, [points, kernel, lengthScale, signalVar, noiseVar, period, kernelRef, xRange]);

  const handleLoadScenario = useCallback((s: Scenario) => {
    setPoints(s.points);
    setKernel(s.kernel);
    setLengthScale(s.lengthScale);
    setSignalVar(s.signalVar);
    setNoiseVar(s.noiseVar);
    setPeriod(s.period);
    setKernelRef(s.points.length > 0 ? s.points[s.points.length - 1].x : 0.0);
  }, []);

  const handleSvgClick = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const svgX = (e.clientX - rect.left) / rect.width * W;
    const svgY = (e.clientY - rect.top) / rect.height * H;
    if (svgX < PAD.left || svgX > W - PAD.right || svgY < PAD.top || svgY > H - PAD.bottom) return;
    const x = parseFloat((xRange.min + ((svgX - PAD.left) / (W - PAD.left - PAD.right)) * (xRange.max - xRange.min)).toFixed(2));
    const y = parseFloat((yRange.min + ((H - PAD.bottom - svgY) / (H - PAD.top - PAD.bottom)) * (yRange.max - yRange.min)).toFixed(2));
    setPoints(prev => [...prev, { x, y }]);
    setKernelRef(x);
  }, [xRange, yRange]);


  const handleOptimise = async () => {
    if (points.length < 2) return;
    setOptimising(true);
    try {
      const opt = await apiOptimise(
        points.map(p => p.x), points.map(p => p.y),
        kernel, lengthScale, signalVar, noiseVar, period,
      );
      setLengthScale(parseFloat(opt.length_scale.toFixed(3)));
      setSignalVar(parseFloat(opt.signal_variance.toFixed(3)));
      setNoiseVar(parseFloat(opt.noise_variance.toFixed(3)));
    } finally {
      setOptimising(false);
    }
  };

  const handleSolve = async () => {
    setSolverLoading(true);
    try {
      const steps = await apiSolve(
        points.map(p => p.x), points.map(p => p.y),
        kernel, lengthScale, signalVar, noiseVar, period,
      );
      setSolverSteps(steps);
    } finally {
      setSolverLoading(false);
    }
  };

  const kernelColor = KERNEL_COLORS[kernel];

  return (
    <div className="flex flex-1 overflow-hidden">
      {/* ── Sidebar ── */}
      <aside className="w-72 bg-white border-r border-slate-200 flex flex-col p-5 gap-6 overflow-y-auto shrink-0">

        {/* Scenarios */}
        <div>
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3">
            Learning Scenarios
          </h3>
          <div className="flex flex-col gap-2">
            {SCENARIOS.map((s) => (
              <button
                key={s.name}
                onClick={() => handleLoadScenario(s)}
                className="text-left px-3 py-2.5 rounded-lg border border-slate-200 hover:border-blue-300 hover:bg-blue-50 transition-colors group"
              >
                <div className="text-sm font-medium text-slate-700 group-hover:text-blue-700 mb-0.5">
                  {s.name}
                </div>
                <div className="text-xs text-slate-400 group-hover:text-blue-600">
                  {s.description}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Kernel selector */}
        <div>
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3">
            Kernel
          </h3>
          <div className="grid grid-cols-2 gap-1.5">
            {(['rbf', 'matern32', 'linear', 'periodic'] as KernelId[]).map(k => (
              <button key={k} onClick={() => setKernel(k)}
                className={`py-1.5 px-2 rounded-lg text-xs font-semibold border transition-colors
                  ${kernel === k
                    ? 'text-white border-transparent'
                    : 'bg-white text-slate-600 border-slate-200 hover:border-blue-300'}`}
                style={kernel === k ? { backgroundColor: KERNEL_COLORS[k], borderColor: KERNEL_COLORS[k] } : {}}>
                {KERNEL_LABELS[k]}
              </button>
            ))}
          </div>
        </div>

        {/* Hyperparameters */}
        <div>
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">
            Hyperparameters
          </h3>

          {kernel !== 'linear' && (
            <label className="block mb-4">
              <div className="flex justify-between mb-1">
                <span className="text-sm font-medium text-slate-700">Length Scale ℓ</span>
                <span className="font-mono text-sm text-slate-500">{lengthScale.toFixed(2)}</span>
              </div>
              <input type="range" min={0.1} max={3.0} step={0.05} value={lengthScale}
                onChange={e => setLengthScale(parseFloat(e.target.value))}
                className="w-full accent-blue-600" />
              <p className="text-xs text-slate-400 mt-1">
                Short = wiggly, long = smooth
              </p>
            </label>
          )}

          <label className="block mb-4">
            <div className="flex justify-between mb-1">
              <span className="text-sm font-medium text-slate-700">Signal Variance σ²_f</span>
              <span className="font-mono text-sm text-slate-500">{signalVar.toFixed(2)}</span>
            </div>
            <input type="range" min={0.1} max={3.0} step={0.05} value={signalVar}
              onChange={e => setSignalVar(parseFloat(e.target.value))}
              className="w-full accent-blue-600" />
            <p className="text-xs text-slate-400 mt-1">
              Prior variance of function values
            </p>
          </label>

          <label className="block mb-4">
            <div className="flex justify-between mb-1">
              <span className="text-sm font-medium text-slate-700">Noise Variance σ²_n</span>
              <span className="font-mono text-sm text-slate-500">{noiseVar.toFixed(2)}</span>
            </div>
            <input type="range" min={0.01} max={2.0} step={0.01} value={noiseVar}
              onChange={e => setNoiseVar(parseFloat(e.target.value))}
              className="w-full accent-blue-600" />
            <p className="text-xs text-slate-400 mt-1">
              Observation noise (0 = exact interpolation)
            </p>
          </label>

          {kernel === 'periodic' && (
            <label className="block mb-2">
              <div className="flex justify-between mb-1">
                <span className="text-sm font-medium text-slate-700">Period p</span>
                <span className="font-mono text-sm text-slate-500">{period.toFixed(2)}</span>
              </div>
              <input type="range" min={0.5} max={8.0} step={0.1} value={period}
                onChange={e => setPeriod(parseFloat(e.target.value))}
                className="w-full accent-teal-600" />
              <p className="text-xs text-slate-400 mt-1">
                Repetition interval (π ≈ 3.14, 2π ≈ 6.28)
              </p>
            </label>
          )}
        </div>

        {/* Data Points */}
        <div>
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3">
            Data Points
          </h3>
          <p className="text-sm text-slate-500 mb-3">
            {points.length === 0
              ? 'Click on the canvas to add data points.'
              : `${points.length} point${points.length > 1 ? 's' : ''} — watch the posterior update.`}
          </p>
          {points.length > 0 && (
            <button onClick={() => { setPoints([]); setKernelRef(0.0); }}
              className="flex items-center gap-2 w-full justify-center px-3 py-2 rounded-lg text-sm text-red-600 border border-red-100 hover:bg-red-50 transition-colors">
              <Trash2 size={14} /> Clear All Points
            </button>
          )}
        </div>

        {/* Visualisation toggles */}
        <div>
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3">
            Visualisation
          </h3>
          <button
            onClick={() => setShowSamples(v => !v)}
            className={`flex items-center gap-2 w-full px-3 py-2 rounded-lg text-sm border transition-colors
              ${showSamples
                ? 'bg-blue-50 text-blue-700 border-blue-200'
                : 'bg-white text-slate-600 border-slate-200 hover:border-blue-300'}`}
          >
            {showSamples ? <Eye size={14} /> : <EyeOff size={14} />}
            Posterior samples
          </button>
          <p className="text-xs text-slate-400 mt-1.5 pl-1">
            10 functions drawn from p(f&nbsp;|&nbsp;data)
          </p>
        </div>

        {/* Tip box */}
        <div className="text-xs text-slate-400 bg-slate-50 rounded-xl p-3 leading-relaxed">
          <Info size={12} className="inline mr-1" />
          <strong>1σ band</strong> (darker) = 68% · <strong>2σ band</strong> (lighter) = 95%
          <br />The dashed line in the kernel panel marks the reference point.
        </div>
      </aside>

      {/* ── Main Canvas ── */}
      <main className="flex-1 overflow-y-auto bg-slate-50 p-8">
        <div className="max-w-3xl mx-auto">
          <div className="mb-6 text-center">
            <h2 className="text-2xl font-bold text-slate-800 mb-1">Gaussian Processes</h2>
            <p className="text-slate-500 text-sm">
              {result?.is_prior
                ? 'Showing GP prior — no data yet. Click the canvas to add observations.'
                : `GP posterior after ${result?.n_data} observation${(result?.n_data ?? 0) > 1 ? 's' : ''}.`}
            </p>
          </div>

          {/* Interactive SVG Plot */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden mb-6">
            <div className="px-5 py-3 border-b border-slate-100 flex items-center justify-between bg-slate-50">
              <span className="text-sm font-medium text-slate-700">
                {result?.is_prior ? 'Prior Predictive' : 'Posterior Predictive'}
                <span className="ml-2 px-2 py-0.5 rounded-full text-xs font-semibold text-white"
                  style={{ backgroundColor: kernelColor }}>
                  {KERNEL_LABELS[kernel]}
                </span>
              </span>
              <div className="flex items-center gap-2 text-xs text-slate-400">
                {loading && <Loader2 size={13} className="animate-spin text-blue-400" />}
                <span className="hidden sm:inline">Click to add points</span>
              </div>
            </div>

            <div className="relative">
            <svg
              ref={svgRef}
              viewBox={`0 0 ${W} ${H}`}
              className="w-full cursor-crosshair"
              onClick={handleSvgClick}
            >
              <defs>
                <clipPath id="gp-plot-clip">
                  <rect x={PAD.left} y={PAD.top}
                    width={W - PAD.left - PAD.right}
                    height={H - PAD.top - PAD.bottom} />
                </clipPath>
              </defs>

              {/* Grid */}
              {gridLines()}

              {/* Axes */}
              <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="#94a3b8" strokeWidth={1.5} />
              <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} stroke="#94a3b8" strokeWidth={1.5} />

              {result && (
                <>
                  {/* Posterior function samples */}
                  {showSamples && result.posterior_samples && (
                    <g clipPath="url(#gp-plot-clip)">
                      {result.posterior_samples.map((sample, i) => (
                        <path
                          key={i}
                          d={linePath(result.x_grid, sample.map(v => Math.max(yRange.min - 1, Math.min(yRange.max + 1, v))))}
                          fill="none"
                          stroke={kernelColor}
                          strokeWidth={1}
                          opacity={0.18}
                          strokeLinecap="round"
                        />
                      ))}
                    </g>
                  )}

                  {/* 2σ band */}
                  <path
                    d={bandPath(result.x_grid, result.pred_mean, result.pred_std, 2)}
                    fill={result.is_prior ? '#f0f9ff' : '#dbeafe'}
                    opacity={0.65}
                  />
                  {/* 1σ band */}
                  <path
                    d={bandPath(result.x_grid, result.pred_mean, result.pred_std, 1)}
                    fill={result.is_prior ? '#e0f2fe' : '#bfdbfe'}
                    opacity={0.8}
                  />
                  {/* Mean line */}
                  <path
                    d={linePath(result.x_grid, result.pred_mean)}
                    fill="none"
                    stroke={result.is_prior ? '#7dd3fc' : kernelColor}
                    strokeWidth={2.5}
                    strokeLinecap="round"
                    clipPath="url(#gp-plot-clip)"
                  />
                </>
              )}

              {/* Data points */}
              {points.map((p, i) => (
                <circle key={i}
                  cx={toSvgX(p.x)} cy={toSvgY(p.y)}
                  r={5} fill="#dc2626" stroke="white" strokeWidth={1.5}
                  style={{ pointerEvents: 'none' }}
                />
              ))}
            </svg>
            </div>

            {/* Axis range sliders */}
            <div className="px-5 py-3 border-t border-slate-100 space-y-2">
              <RangeSlider
                label="X" trackMin={-10} trackMax={10}
                value={xRange} onChange={setXRange} step={0.5}
              />
              <RangeSlider
                label="Y" trackMin={-12} trackMax={12}
                value={yRange} onChange={setYRange} step={0.5}
              />
              {!isRangeDefault && (
                <button
                  onClick={() => { setXRange(X_DEF); setYRange(Y_DEF); }}
                  className="text-xs text-slate-400 hover:text-slate-600 transition-colors"
                >
                  ↺ Reset axes
                </button>
              )}
            </div>
          </div>

          {/* Legend */}
          <div className="flex flex-wrap gap-4 justify-center text-xs text-slate-500 mb-6">
            <div className="flex items-center gap-1.5">
              <div className="w-8 h-2 rounded-full" style={{ backgroundColor: kernelColor }} />
              <span>Predictive mean</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-8 h-3 rounded-full bg-blue-300 opacity-80" />
              <span>±1σ (68%)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-8 h-3 rounded-full bg-blue-200 opacity-70" />
              <span>±2σ (95%)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <span>Data points</span>
            </div>
            {showSamples && (
              <div className="flex items-center gap-1.5">
                <div className="w-8 h-0.5 bg-blue-400 opacity-60" />
                <span>Posterior samples</span>
              </div>
            )}
          </div>

          {/* Kernel panel */}
          {result && result.kernel_slice.length > 0 && (
            <KernelPanel
              xGrid={result.x_grid}
              kernelSlice={result.kernel_slice}
              kernelRef={kernelRef}
              kernel={kernel}
              signalVar={signalVar}
              xMin={xRange.min}
              xMax={xRange.max}
            />
          )}

          {/* Hyperparameter summary card */}
          {result && !result.is_prior && (
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 mb-6">
              <h3 className="text-sm font-bold text-slate-700 mb-3">Hyperparameter Summary</h3>
              <div className="flex flex-wrap gap-4">
                {kernel !== 'linear' && (
                  <div className="bg-slate-50 rounded-xl px-4 py-2 text-center">
                    <div className="text-xs text-slate-400 mb-0.5">ℓ</div>
                    <div className="font-mono font-bold text-slate-800">{lengthScale.toFixed(3)}</div>
                  </div>
                )}
                <div className="bg-slate-50 rounded-xl px-4 py-2 text-center">
                  <div className="text-xs text-slate-400 mb-0.5">σ²_f</div>
                  <div className="font-mono font-bold text-slate-800">{signalVar.toFixed(3)}</div>
                </div>
                <div className="bg-slate-50 rounded-xl px-4 py-2 text-center">
                  <div className="text-xs text-slate-400 mb-0.5">σ²_n</div>
                  <div className="font-mono font-bold text-slate-800">{noiseVar.toFixed(3)}</div>
                </div>
                {kernel === 'periodic' && (
                  <div className="bg-slate-50 rounded-xl px-4 py-2 text-center">
                    <div className="text-xs text-slate-400 mb-0.5">p</div>
                    <div className="font-mono font-bold text-slate-800">{period.toFixed(2)}</div>
                  </div>
                )}
                {result.log_marginal_likelihood != null && (
                  <div className="bg-slate-50 rounded-xl px-4 py-2 text-center">
                    <div className="text-xs text-slate-400 mb-0.5">log p(y|θ)</div>
                    <div className="font-mono font-bold text-slate-800">
                      {result.log_marginal_likelihood.toFixed(2)}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Learn panel */}
      <InfoPanel
        content={gpContent}
        open={learnOpen}
        width={panelWidth}
        activeTab={learnTab}
        onToggle={onLearnToggle}
        onClose={onLearnClose}
        onWidthChange={onLearnWidthChange}
        onActiveTabChange={onLearnTabChange}
      />

      {/* Floating action buttons */}
      <div
        className="fixed bottom-6 flex flex-col gap-3 transition-all duration-300"
        style={{ right: learnOpen ? panelWidth + 24 : 24 }}
      >
        <button
          onClick={handleOptimise}
          disabled={optimising || points.length < 2}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-violet-600 text-white shadow-lg hover:bg-violet-700 transition-all text-sm font-semibold disabled:opacity-50"
        >
          {optimising ? <Loader2 size={15} className="animate-spin" /> : <Zap size={15} />}
          Optimise Hyperparams
        </button>
        <button
          onClick={handleSolve}
          disabled={solverLoading}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-green-600 text-white shadow-lg hover:bg-green-700 transition-all text-sm font-semibold disabled:opacity-60"
        >
          {solverLoading ? <Loader2 size={15} className="animate-spin" /> : <Calculator size={15} />}
          Derive Posterior
        </button>
      </div>

      {solverSteps && (
        <SolverModal steps={solverSteps} onClose={() => setSolverSteps(null)} />
      )}
    </div>
  );
}
