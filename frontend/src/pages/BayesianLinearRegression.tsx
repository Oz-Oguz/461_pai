import { useState, useEffect, useRef, useCallback } from 'react';
import { Calculator, Trash2, Info, Loader2, Eye, EyeOff } from 'lucide-react';
import { SolverModal } from '../components/SolverModal';
import { InfoPanel } from '../components/InfoPanel';
import { RangeSlider } from '../components/RangeSlider';
import { blrContent } from '../content/blr';
import type { SolveStep, LearnTab } from '../types';

// ── API helpers ───────────────────────────────────────────────────────

interface BLRResult {
  n_data: number;
  x_grid: number[];
  pred_mean: number[];
  pred_std: number[];
  pred_std2: number[];
  is_prior: boolean;
  x_data: number[];
  y_data: number[];
  posterior_mean: number[] | null;
  posterior_var_diag: number[] | null;
  posterior_cov: number[][] | null;
  posterior_samples: number[][] | null;
  log_marginal_likelihood: number | null;
}

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
  priorVar: number, noiseVar: number, degree: number,
  basis: string, xMin: number, xMax: number,
): Promise<BLRResult> {
  const res = await fetch('/api/blr/fit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      x_data: xData, y_data: yData,
      prior_variance: priorVar, noise_variance: noiseVar, degree, basis,
      x_min: xMin, x_max: xMax,
    }),
  });
  return parseApiResponse<BLRResult>(res);
}

async function apiSolve(
  xData: number[], yData: number[],
  priorVar: number, noiseVar: number, degree: number,
): Promise<SolveStep[]> {
  const res = await fetch('/api/blr/solve', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      x_data: xData, y_data: yData,
      prior_variance: priorVar, noise_variance: noiseVar, degree,
    }),
  });
  return parseApiResponse<SolveStep[]>(res);
}

interface EvidenceResult {
  degree: number;
  log_evidence: number | null;
}

async function apiEvidence(
  xData: number[], yData: number[],
  priorVar: number, noiseVar: number,
): Promise<EvidenceResult[]> {
  const res = await fetch('/api/blr/evidence', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      x_data: xData, y_data: yData,
      prior_variance: priorVar, noise_variance: noiseVar,
      degrees: [1, 2, 3, 4],
    }),
  });
  return parseApiResponse<EvidenceResult[]>(res);
}

// ── Pre-defined scenarios ─────────────────────────────────────────────

interface Scenario {
  name: string;
  description: string;
  points: { x: number; y: number }[];
  degree: number;
  priorVar: number;
  noiseVar: number;
}

const SCENARIOS: Scenario[] = [
  {
    name: "Clear (Prior)",
    description: "Reset to prior — no observations",
    points: [],
    degree: 1,
    priorVar: 1.0,
    noiseVar: 0.3,
  },
  {
    name: "Sensor Calibration",
    description: "Calibrated only in [−1, 1] — watch the band outside",
    points: [
      { x: -1.0, y: -0.8 },
      { x: -0.5, y: -0.4 },
      { x:  0.0, y:  0.1 },
      { x:  0.5, y:  0.6 },
      { x:  1.0, y:  1.1 },
    ],
    degree: 1,
    priorVar: 1.0,
    noiseVar: 0.1,
  },
  {
    name: "Linear Trend",
    description: "Clean linear data — basic BLR",
    points: [
      { x: -2, y: -1.6 },
      { x: -1, y: -0.8 },
      { x: 0, y: 0.1 },
      { x: 1, y: 0.9 },
      { x: 2, y: 1.7 },
    ],
    degree: 1,
    priorVar: 1.0,
    noiseVar: 0.3,
  },
  {
    name: "Nonlinear Data",
    description: "Sinusoidal — underfitting at d=1",
    points: [
      { x: -2.5, y: -0.6 },
      { x: -1.2, y: -0.9 },
      { x: 0, y: 0 },
      { x: 1.2, y: 0.9 },
      { x: 2.5, y: 0.6 },
    ],
    degree: 1,
    priorVar: 1.0,
    noiseVar: 0.3,
  },
  {
    name: "Dense Observations",
    description: "Many points → uncertainty collapses",
    points: [
      { x: -2.5, y: -2.4 }, { x: -2, y: -1.9 }, { x: -1.5, y: -1.4 },
      { x: -1, y: -0.9 }, { x: -0.5, y: -0.4 }, { x: 0, y: 0.1 },
      { x: 0.5, y: 0.6 }, { x: 1, y: 1.1 }, { x: 1.5, y: 1.6 },
      { x: 2, y: 2.1 }, { x: 2.5, y: 2.6 },
    ],
    degree: 1,
    priorVar: 1.0,
    noiseVar: 0.3,
  },
  {
    name: "Model Selection",
    description: "Cubic data — compare degrees via evidence",
    points: [
      { x: -2, y: -5.8 },
      { x: -1.5, y: -1.7 },
      { x: -1, y: 0.2 },
      { x: -0.5, y: 0.5 },
      { x: 0, y: -0.1 },
      { x: 0.5, y: -0.3 },
      { x: 1, y: 0.1 },
      { x: 1.5, y: 2.0 },
    ],
    degree: 1,  // User should try different degrees and watch evidence
    priorVar: 2.0,
    noiseVar: 0.5,
  },
];

// ── SVG canvas constants ──────────────────────────────────────────────

const W = 620, H = 420;
const PAD = { top: 20, right: 20, bottom: 40, left: 50 };
const X_DEF = { min: -3, max: 3 }, Y_DEF = { min: -4, max: 4 };
const WS_DEF = { min: -4, max: 4 }; // default weight-space axis range

/** Evaluate a polynomial weight vector; clamps to ±50 (SVG clipPath handles visual clipping). */
function evalSample(weights: number[], xGrid: number[]): number[] {
  const degree = weights.length - 1;
  return xGrid.map(x => {
    let y = 0;
    for (let i = 0; i <= degree; i++) y += weights[i] * Math.pow(x, i);
    return Math.max(-50, Math.min(50, y));
  });
}

/** Evaluate RBF weight vector; centers and length-scale must match what the backend used. */
function evalRBFSample(weights: number[], xGrid: number[], centers: number[], ls: number): number[] {
  return xGrid.map(x => {
    const y = weights.reduce((sum, w, k) => {
      return sum + w * Math.exp(-0.5 * ((x - centers[k]) / ls) ** 2);
    }, 0);
    return Math.max(-50, Math.min(50, y));
  });
}

// ── Weight-space panel ────────────────────────────────────────────────

/** Eigendecompose a 2×2 symmetric matrix [[a,b],[b,d]].
 *  Returns the two eigenvalues (λ₁ ≥ λ₂) and the SVG rotation angle
 *  (degrees, clockwise-positive) that aligns the ellipse major axis. */
function eigen2x2(cov: number[][]): { lam1: number; lam2: number; angleDeg: number } {
  const a = cov[0][0], b = cov[0][1], d = cov[1][1];
  const trace = a + d;
  const det   = a * d - b * b;
  const sq    = Math.sqrt(Math.max(0, (trace / 2) ** 2 - det));
  const lam1  = Math.max(0, trace / 2 + sq);
  const lam2  = Math.max(0, trace / 2 - sq);
  // Math angle of eigenvector for λ₁: atan2(λ₁−a, b)
  // SVG angleDeg: negate because SVG y-axis points down
  const mathAngle = Math.abs(b) < 1e-10 ? 0 : Math.atan2(lam1 - a, b);
  const angleDeg  = -(mathAngle * 180) / Math.PI;
  return { lam1, lam2, angleDeg };
}

function WeightSpacePanel({
  cov, mean, samples, isPrior,
}: {
  cov: number[][];
  mean: number[] | null;
  samples: number[][] | null;
  isPrior: boolean;
}) {
  const WS = 280, HS = 280, PADW = 38;
  const mu0 = mean?.[0] ?? 0;
  const mu1 = mean?.[1] ?? 0;
  const { lam1, lam2, angleDeg } = eigen2x2(cov);
  const innerW = WS - 2 * PADW, innerH = HS - 2 * PADW;

  // ── Auto-adjust toggle ────────────────────────────────────────────────
  const [autoAdjust, setAutoAdjust] = useState(false);

  // Auto range: re-centres on posterior mean and scales to 3σ each render
  const spread = Math.max(3.5 * Math.sqrt(Math.max(lam1, 1e-8)), 0.3);
  const autoRange = (centre: number) => ({ min: centre - spread, max: centre + spread });

  // ── Manual axis ranges ────────────────────────────────────────────────
  const [w0Range, setW0Range] = useState(WS_DEF);
  const [w1Range, setW1Range] = useState(WS_DEF);

  // When user switches auto → manual: snap manual ranges to current auto values
  const prevAutoRef = useRef<boolean | null>(null);
  useEffect(() => {
    if (prevAutoRef.current === true && !autoAdjust) {
      const rnd = (v: number) => Math.round(v * 4) / 4; // snap to 0.25
      setW0Range({ min: rnd(mu0 - spread), max: rnd(mu0 + spread) });
      setW1Range({ min: rnd(mu1 - spread), max: rnd(mu1 + spread) });
    }
    prevAutoRef.current = autoAdjust;
  }, [autoAdjust]); // eslint-disable-line react-hooks/exhaustive-deps

  const effectiveW0 = autoAdjust ? autoRange(mu0) : w0Range;
  const effectiveW1 = autoAdjust ? autoRange(mu1) : w1Range;
  const w0Span = effectiveW0.max - effectiveW0.min;
  const w1Span = effectiveW1.max - effectiveW1.min;

  const toX = (w0: number) => PADW + ((w0 - effectiveW0.min) / w0Span) * innerW;
  const toY = (w1: number) => (HS - PADW) - ((w1 - effectiveW1.min) / w1Span) * innerH;
  const cx = toX(mu0), cy = toY(mu1);
  const rx = (Math.sqrt(lam1) / w0Span) * innerW;
  const ry = (Math.sqrt(lam2) / w1Span) * innerH;

  // ── Pan & box-zoom interactions ───────────────────────────────────────
  const svgRef = useRef<SVGSVGElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [boxRect, setBoxRect] = useState<{ x: number; y: number; w: number; h: number } | null>(null);

  // Ref captures stable copies of mutable values needed in event handlers
  const interactionRef = useRef<{
    kind: 'pan' | 'box';
    startClient: { x: number; y: number };
    startRanges: { w0: { min: number; max: number }; w1: { min: number; max: number } };
    svgScaleX: number;
    svgScaleY: number;
  } | null>(null);

  useEffect(() => {
    if (!isDragging) return;

    const onMove = (e: MouseEvent) => {
      const info = interactionRef.current;
      if (!info) return;
      const dx = e.clientX - info.startClient.x;
      const dy = e.clientY - info.startClient.y;
      const { startRanges: sr, svgScaleX: sx, svgScaleY: sy } = info;

      if (info.kind === 'pan') {
        const dw0 = -(dx * sx) / innerW * (sr.w0.max - sr.w0.min);
        const dw1 =  (dy * sy) / innerH * (sr.w1.max - sr.w1.min);
        setW0Range({ min: sr.w0.min + dw0, max: sr.w0.max + dw0 });
        setW1Range({ min: sr.w1.min + dw1, max: sr.w1.max + dw1 });
      } else {
        const el = svgRef.current;
        if (!el) return;
        const r = el.getBoundingClientRect();
        const x1 = (info.startClient.x - r.left) * sx;
        const y1 = (info.startClient.y - r.top)  * sy;
        const x2 = (e.clientX - r.left) * sx;
        const y2 = (e.clientY - r.top)  * sy;
        setBoxRect({ x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1) });
      }
    };

    const onUp = (e: MouseEvent) => {
      const info = interactionRef.current;
      if (info?.kind === 'box') {
        const el = svgRef.current;
        if (el) {
          const r = el.getBoundingClientRect();
          const { startRanges: sr, svgScaleX: sx, svgScaleY: sy, startClient: sc } = info;
          const x1 = (sc.x - r.left) * sx,  y1 = (sc.y - r.top) * sy;
          const x2 = (e.clientX - r.left) * sx, y2 = (e.clientY - r.top) * sy;
          if (Math.abs(x2 - x1) > 5 && Math.abs(y2 - y1) > 5) {
            const svgToW0 = (sx2: number) => sr.w0.min + (sx2 - PADW) / innerW * (sr.w0.max - sr.w0.min);
            const svgToW1 = (sy2: number) => sr.w1.min + (HS - PADW - sy2) / innerH * (sr.w1.max - sr.w1.min);
            setW0Range({ min: Math.min(svgToW0(x1), svgToW0(x2)), max: Math.max(svgToW0(x1), svgToW0(x2)) });
            setW1Range({ min: Math.min(svgToW1(y1), svgToW1(y2)), max: Math.max(svgToW1(y1), svgToW1(y2)) });
          }
        }
      }
      interactionRef.current = null;
      setIsDragging(false);
      setBoxRect(null);
    };

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); };
  }, [isDragging]); // eslint-disable-line react-hooks/exhaustive-deps

  const isWsDefault = !autoAdjust && w0Range === WS_DEF && w1Range === WS_DEF;

  const ellipses = [
    { k: 3, fill: '#eff6ff', stroke: '#bfdbfe' },
    { k: 2, fill: '#dbeafe', stroke: '#93c5fd' },
    { k: 1, fill: '#bfdbfe', stroke: '#3b82f6' },
  ];

  const N_TICKS = 5;
  const xTicks = Array.from({ length: N_TICKS }, (_, i) => effectiveW0.min + (i / (N_TICKS - 1)) * w0Span);
  const yTicks = Array.from({ length: N_TICKS }, (_, i) => effectiveW1.min + (i / (N_TICKS - 1)) * w1Span);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 mb-6">
      {/* Header row: title + auto-adjust toggle */}
      <div className="flex items-center justify-between mb-0.5">
        <h3 className="text-sm font-bold text-slate-700">Weight-Space Posterior</h3>
        <label className="flex items-center gap-1.5 text-xs text-slate-500 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={autoAdjust}
            onChange={e => setAutoAdjust(e.target.checked)}
            className="w-3 h-3 accent-blue-600"
          />
          Auto-adjust
        </label>
      </div>
      <p className="text-xs text-slate-400 mb-3">
        Confidence ellipses of p(w₀, w₁&nbsp;|&nbsp;data) · dots = posterior samples
        {!autoAdjust && <span className="ml-1 opacity-60">· drag to pan · shift+drag to zoom</span>}
      </p>

      <div className="flex justify-center">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${WS} ${HS}`} width={WS} height={HS}
          style={{ cursor: autoAdjust ? 'default' : isDragging ? 'grabbing' : 'grab' }}
        >
          <defs>
            <clipPath id="ws-clip">
              <rect x={PADW} y={PADW} width={innerW} height={innerH} />
            </clipPath>
          </defs>

          {/* Interactive plot background */}
          <rect
            x={PADW} y={PADW} width={innerW} height={innerH} fill="#f8fafc"
            onMouseDown={autoAdjust ? undefined : (e) => {
              e.preventDefault();
              const el = svgRef.current;
              if (!el) return;
              const r = el.getBoundingClientRect();
              interactionRef.current = {
                kind: e.shiftKey ? 'box' : 'pan',
                startClient: { x: e.clientX, y: e.clientY },
                startRanges: { w0: effectiveW0, w1: effectiveW1 },
                svgScaleX: WS / r.width,
                svgScaleY: HS / r.height,
              };
              setIsDragging(true);
            }}
          />

          {/* Confidence ellipses (3σ outermost, drawn back-to-front) */}
          {ellipses.map(({ k, fill, stroke }) => (
            <ellipse
              key={k}
              cx={cx} cy={cy}
              rx={rx * k} ry={ry * k}
              fill={fill} fillOpacity={0.85}
              stroke={stroke} strokeWidth={1.5}
              clipPath="url(#ws-clip)"
              transform={`rotate(${angleDeg}, ${cx}, ${cy})`}
            />
          ))}

          {/* Posterior samples */}
          <g clipPath="url(#ws-clip)">
            {samples?.map((s, i) =>
              s.length >= 2
                ? <circle key={i} cx={toX(s[0])} cy={toY(s[1])} r={3}
                    fill="#1d4ed8" opacity={0.55} />
                : null
            )}
          </g>

          {/* Posterior mean */}
          <circle cx={cx} cy={cy} r={4.5} fill="#1e40af" stroke="white" strokeWidth={2} />

          {/* Box-zoom selection rectangle */}
          {boxRect && (
            <rect
              x={boxRect.x} y={boxRect.y} width={boxRect.w} height={boxRect.h}
              fill="rgba(59,130,246,0.08)" stroke="#3b82f6" strokeWidth={1}
              strokeDasharray="4 2" pointerEvents="none"
            />
          )}

          {/* Axes */}
          <line x1={PADW} y1={HS - PADW} x2={WS - PADW} y2={HS - PADW} stroke="#94a3b8" strokeWidth={1} />
          <line x1={PADW} y1={PADW}       x2={PADW}       y2={HS - PADW} stroke="#94a3b8" strokeWidth={1} />

          {/* X-axis ticks + labels */}
          {xTicks.map((v, i) => {
            const px = toX(v);
            if (px < PADW - 2 || px > WS - PADW + 2) return null;
            return (
              <g key={i}>
                <line x1={px} y1={HS - PADW} x2={px} y2={HS - PADW + 4} stroke="#94a3b8" strokeWidth={1} />
                <text x={px} y={HS - PADW + 14} textAnchor="middle" fontSize={9} fill="#94a3b8">
                  {v.toFixed(1)}
                </text>
              </g>
            );
          })}

          {/* Y-axis ticks + labels */}
          {yTicks.map((v, i) => {
            const py = toY(v);
            if (py < PADW - 2 || py > HS - PADW + 2) return null;
            return (
              <g key={i}>
                <line x1={PADW - 4} y1={py} x2={PADW} y2={py} stroke="#94a3b8" strokeWidth={1} />
                <text x={PADW - 7} y={py + 3} textAnchor="end" fontSize={9} fill="#94a3b8">
                  {v.toFixed(1)}
                </text>
              </g>
            );
          })}

          {/* Axis labels */}
          <text x={WS / 2} y={HS - 4} textAnchor="middle" fontSize={11} fill="#64748b">w₀ (bias)</text>
          <text x={12} y={HS / 2} textAnchor="middle" fontSize={11} fill="#64748b"
            transform={`rotate(-90, 12, ${HS / 2})`}>w₁ (slope)</text>

          {/* σ level labels */}
          {[{ k: 1, label: '1σ' }, { k: 2, label: '2σ' }, { k: 3, label: '3σ' }].map(({ label }, i) => (
            <text key={label} x={WS - PADW - 4} y={PADW + 12 + i * 13}
              textAnchor="end" fontSize={9} fill="#64748b">{label}</text>
          ))}
        </svg>
      </div>

      {isPrior && (
        <p className="text-xs text-slate-400 text-center mt-1">
          Prior distribution N(0, σ²_w I) — add data points to watch posterior contract
        </p>
      )}

      {/* Axis range sliders — only shown in manual mode */}
      {!autoAdjust && (
        <div className="pt-3 mt-1 border-t border-slate-100 space-y-2">
          <RangeSlider
            label="w₀" trackMin={-8} trackMax={8}
            value={w0Range} onChange={setW0Range} step={0.25}
          />
          <RangeSlider
            label="w₁" trackMin={-8} trackMax={8}
            value={w1Range} onChange={setW1Range} step={0.25}
          />
          {!isWsDefault && (
            <button
              onClick={() => { setW0Range(WS_DEF); setW1Range(WS_DEF); }}
              className="text-xs text-slate-400 hover:text-slate-600 transition-colors"
            >
              ↺ Reset axes
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────

interface BLRProps {
  learnOpen: boolean;
  learnTab: LearnTab;
  panelWidth: number;
  onLearnToggle: () => void;
  onLearnClose: () => void;
  onLearnWidthChange: (w: number) => void;
  onLearnTabChange: (t: LearnTab) => void;
}

export function BayesianLinearRegression({
  learnOpen,
  learnTab,
  panelWidth,
  onLearnToggle,
  onLearnClose,
  onLearnWidthChange,
  onLearnTabChange,
}: BLRProps) {
  const [points, setPoints] = useState<{ x: number; y: number }[]>([]);
  const [priorVar, setPriorVar] = useState(1.0);
  const [noiseVar, setNoiseVar] = useState(0.3);
  const [degree, setDegree] = useState(1);
  const [basis, setBasis] = useState<'polynomial' | 'rbf'>('polynomial');
  const [showSamples, setShowSamples] = useState(false);
  const [evidenceData, setEvidenceData] = useState<EvidenceResult[] | null>(null);
  const [result, setResult] = useState<BLRResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [solverSteps, setSolverSteps] = useState<SolveStep[] | null>(null);
  const [solverLoading, setSolverLoading] = useState(false);
  const [xRange, setXRange] = useState(X_DEF);
  const [yRange, setYRange] = useState(Y_DEF);
  const isRangeDefault = xRange === X_DEF && yRange === Y_DEF;
  const svgRef = useRef<SVGSVGElement>(null);

  // ── Dynamic coordinate functions (close over xRange / yRange) ────────
  const toSvgX = (x: number) => PAD.left + ((x - xRange.min) / (xRange.max - xRange.min)) * (W - PAD.left - PAD.right);
  const toSvgY = (y: number) => H - PAD.bottom - ((y - yRange.min) / (yRange.max - yRange.min)) * (H - PAD.top - PAD.bottom);
  // RBF centers must match what the backend computed (linspace over current x range)
  const rbfCenters = Array.from({ length: 7 }, (_, i) => xRange.min + (i / 6) * (xRange.max - xRange.min));
  const rbfLs = (xRange.max - xRange.min) / 8;
  const bandPath = (xs: number[], means: number[], stds: number[], sigma: number): string => {
    const upper = means.map((m, i) => m + sigma * stds[i]);
    const lower = means.map((m, i) => m - sigma * stds[i]);
    const top = xs.map((x, i) => `${toSvgX(x).toFixed(1)},${toSvgY(upper[i]).toFixed(1)}`).join(' ');
    const bot = [...xs].reverse().map((x, i) => `${toSvgX(x).toFixed(1)},${toSvgY(lower[xs.length - 1 - i]).toFixed(1)}`).join(' ');
    return `M ${top.split(' ')[0]} L ${top} L ${bot} Z`;
  };
  const linePath = (xs: number[], ys: number[]): string =>
    xs.map((x, i) => `${i === 0 ? 'M' : 'L'} ${toSvgX(x).toFixed(1)},${toSvgY(ys[i]).toFixed(1)}`).join(' ');
  // Axis ticks — 7 x-ticks, 5 y-ticks, rounded to 1 decimal
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

  // Fetch fit whenever data or params change
  useEffect(() => {
    setLoading(true);
    apiFit(
      points.map(p => p.x), points.map(p => p.y),
      priorVar, noiseVar, degree, basis,
      xRange.min, xRange.max,
    )
      .then(setResult)
      .finally(() => setLoading(false));
  }, [points, priorVar, noiseVar, degree, basis, xRange]);

  // Fetch model evidence for degrees 1-4 when we have data
  useEffect(() => {
    if (points.length < 3) {
      setEvidenceData(null);
      return;
    }
    apiEvidence(
      points.map(p => p.x), points.map(p => p.y),
      priorVar, noiseVar,
    ).then(setEvidenceData);
  }, [points, priorVar, noiseVar]);


  const handleLoadScenario = useCallback((scenario: Scenario) => {
    setPoints(scenario.points);
    setDegree(scenario.degree);
    setPriorVar(scenario.priorVar);
    setNoiseVar(scenario.noiseVar);
  }, []);

  const handleSvgClick = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    // Map screen → SVG pixel (viewBox = "0 0 W H")
    const svgX = (e.clientX - rect.left) / rect.width * W;
    const svgY = (e.clientY - rect.top) / rect.height * H;
    // Reject clicks outside the plot area
    if (svgX < PAD.left || svgX > W - PAD.right || svgY < PAD.top || svgY > H - PAD.bottom) return;
    // Map SVG pixel → data coordinates using current axis ranges
    const x = parseFloat((xRange.min + ((svgX - PAD.left) / (W - PAD.left - PAD.right)) * (xRange.max - xRange.min)).toFixed(2));
    const y = parseFloat((yRange.min + ((H - PAD.bottom - svgY) / (H - PAD.top - PAD.bottom)) * (yRange.max - yRange.min)).toFixed(2));
    setPoints(prev => [...prev, { x, y }]);
  }, [xRange, yRange]);

  const handleSolve = async () => {
    setSolverLoading(true);
    try {
      const steps = await apiSolve(
        points.map(p => p.x), points.map(p => p.y),
        priorVar, noiseVar, degree,
      );
      setSolverSteps(steps);
    } finally {
      setSolverLoading(false);
    }
  };

  const showWeightSpace = basis === 'polynomial' && degree === 1 && result?.posterior_cov != null;

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
            {SCENARIOS.map((scenario) => (
              <button
                key={scenario.name}
                onClick={() => handleLoadScenario(scenario)}
                className="text-left px-3 py-2.5 rounded-lg border border-slate-200 hover:border-blue-300 hover:bg-blue-50 transition-colors group"
              >
                <div className="text-sm font-medium text-slate-700 group-hover:text-blue-700 mb-0.5">
                  {scenario.name}
                </div>
                <div className="text-xs text-slate-400 group-hover:text-blue-600">
                  {scenario.description}
                </div>
              </button>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">
            Model Parameters
          </h3>

          <label className="block mb-4">
            <div className="flex justify-between mb-1">
              <span className="text-sm font-medium text-slate-700">Prior Variance σ²_w</span>
              <span className="font-mono text-sm text-slate-500">{priorVar.toFixed(2)}</span>
            </div>
            <input type="range" min={0.01} max={5} step={0.01} value={priorVar}
              onChange={e => setPriorVar(parseFloat(e.target.value))}
              className="w-full accent-blue-600" />
            <p className="text-xs text-slate-400 mt-1">
              Wide prior → more uncertainty before data
            </p>
          </label>

          <label className="block mb-4">
            <div className="flex justify-between mb-1">
              <span className="text-sm font-medium text-slate-700">Noise Variance σ²_n</span>
              <span className="font-mono text-sm text-slate-500">{noiseVar.toFixed(2)}</span>
            </div>
            <input type="range" min={0.01} max={3} step={0.01} value={noiseVar}
              onChange={e => setNoiseVar(parseFloat(e.target.value))}
              className="w-full accent-blue-600" />
            <p className="text-xs text-slate-400 mt-1">
              How noisy are the observations?
            </p>
          </label>

          <div className="mb-4">
            <span className="text-sm font-medium text-slate-700 block mb-2">Basis Functions</span>
            <div className="flex gap-2">
              {(['polynomial', 'rbf'] as const).map(b => (
                <button key={b} onClick={() => setBasis(b)}
                  className={`flex-1 py-1.5 rounded-lg text-sm font-semibold border transition-colors
                    ${basis === b
                      ? 'bg-blue-600 text-white border-blue-600'
                      : 'bg-white text-slate-600 border-slate-200 hover:border-blue-300'}`}>
                  {b === 'polynomial' ? 'Polynomial' : 'RBF'}
                </button>
              ))}
            </div>
            {basis === 'rbf' && (
              <p className="text-xs text-slate-400 mt-1">7 RBF centres, auto-spaced</p>
            )}
          </div>

          {basis === 'polynomial' && (
            <label className="block mb-2">
              <span className="text-sm font-medium text-slate-700 block mb-2">
                Polynomial Degree
              </span>
              <div className="flex gap-2">
                {[1, 2, 3, 4].map(d => (
                  <button key={d} onClick={() => setDegree(d)}
                    className={`flex-1 py-1.5 rounded-lg text-sm font-semibold border transition-colors
                      ${degree === d
                        ? 'bg-blue-600 text-white border-blue-600'
                        : 'bg-white text-slate-600 border-slate-200 hover:border-blue-300'}`}>
                    {d}
                  </button>
                ))}
              </div>
            </label>
          )}
        </div>

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
            <button onClick={() => setPoints([])}
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
            10 functions drawn from p(w&nbsp;|&nbsp;data)
          </p>
        </div>

        {/* Model Evidence */}
        {evidenceData && evidenceData.length > 0 && (
          <div>
            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2">
              Model Evidence
            </h3>
            <p className="text-xs text-slate-400 mb-3">
              log p(y | degree) — higher is better
            </p>
            <div className="space-y-2">
              {evidenceData.map((ev) => {
                const allLogs = evidenceData
                  .map(e => e.log_evidence)
                  .filter((v): v is number => v != null);
                const minLog = Math.min(...allLogs);
                const maxLog = Math.max(...allLogs);
                const range = maxLog - minLog;
                const normalized = ev.log_evidence != null
                  ? range > 0.01 ? (ev.log_evidence - minLog) / range : 1.0
                  : 0;
                const isCurrent = ev.degree === degree;
                return (
                  <button
                    key={ev.degree}
                    onClick={() => setDegree(ev.degree)}
                    className={`w-full text-left px-3 py-2 rounded-lg border transition-colors
                      ${isCurrent
                        ? 'bg-blue-600 text-white border-blue-600'
                        : 'bg-white text-slate-600 border-slate-200 hover:border-blue-300 hover:bg-blue-50'}`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-semibold">Degree {ev.degree}</span>
                      <span className="text-xs font-mono">
                        {ev.log_evidence != null ? ev.log_evidence.toFixed(2) : '—'}
                      </span>
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-1.5 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all ${
                          isCurrent ? 'bg-white' : 'bg-blue-500'
                        }`}
                        style={{ width: `${normalized * 100}%` }}
                      />
                    </div>
                  </button>
                );
              })}
            </div>
            <p className="text-xs text-slate-400 mt-2 pl-1">
              Click a degree to switch. Wider bar = higher evidence.
            </p>
          </div>
        )}

        <div className="text-xs text-slate-400 bg-slate-50 rounded-xl p-3 leading-relaxed">
          <Info size={12} className="inline mr-1" />
          <strong>1σ band</strong> (dark blue) = 68% of probability mass.<br />
          <strong>2σ band</strong> (light blue) = 95% of probability mass.
        </div>
      </aside>

      {/* ── Main Canvas ── */}
      <main className="flex-1 overflow-y-auto bg-slate-50 p-8">
        <div className="max-w-3xl mx-auto">
          <div className="mb-6 text-center">
            <h2 className="text-2xl font-bold text-slate-800 mb-1">Bayesian Linear Regression</h2>
            <p className="text-slate-500 text-sm">
              {result?.is_prior
                ? 'Showing prior predictive distribution — no data yet. Click the canvas to add points.'
                : `Posterior predictive after ${result?.n_data} observation${(result?.n_data ?? 0) > 1 ? 's' : ''}.`}
            </p>
          </div>

          {/* Interactive SVG Plot */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden mb-6">
            <div className="px-5 py-3 border-b border-slate-100 flex items-center justify-between bg-slate-50">
              <span className="text-sm font-medium text-slate-700">
                {result?.is_prior ? 'Prior Predictive' : 'Posterior Predictive'}
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
                <clipPath id="plot-clip">
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
                  {/* Posterior function samples — drawn before bands */}
                  {showSamples && result.posterior_samples && (
                    <g clipPath="url(#plot-clip)">
                      {result.posterior_samples.map((weights, i) => (
                        <path
                          key={i}
                          d={linePath(result.x_grid, basis === 'rbf'
                            ? evalRBFSample(weights, result.x_grid, rbfCenters, rbfLs)
                            : evalSample(weights, result.x_grid))}
                          fill="none"
                          stroke={result.is_prior ? '#7dd3fc' : '#3b82f6'}
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
                    opacity={0.7}
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
                    stroke={result.is_prior ? '#7dd3fc' : '#2563eb'}
                    strokeWidth={2.5}
                    strokeLinecap="round"
                  />
                </>
              )}

              {/* Data points */}
              {points.map((p, i) => (
                <circle
                  key={i}
                  cx={toSvgX(p.x)}
                  cy={toSvgY(p.y)}
                  r={5}
                  fill="#dc2626"
                  stroke="white"
                  strokeWidth={1.5}
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
              <div className="w-8 h-2 rounded-full bg-blue-600" />
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

          {/* Weight-space posterior — degree=1 only */}
          {showWeightSpace && (
            <WeightSpacePanel
              cov={result!.posterior_cov!}
              mean={result!.posterior_mean}
              samples={showSamples ? result!.posterior_samples : null}
              isPrior={result!.is_prior}
            />
          )}
          {basis === 'polynomial' && degree !== 1 && (
            <p className="text-xs text-slate-400 text-center mb-6">
              Weight-space view available for degree = 1
            </p>
          )}

          {/* Posterior weight summary */}
          {result && !result.is_prior && result.posterior_mean && (
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 mb-6">
              <h3 className="text-sm font-bold text-slate-700 mb-3">Posterior Weight Summary</h3>
              <div className="flex flex-wrap gap-4">
                {result.posterior_mean.map((mu, i) => (
                  <div key={i} className="bg-slate-50 rounded-xl px-4 py-2 text-center">
                    <div className="text-xs text-slate-400 mb-0.5">w<sub>{i}</sub></div>
                    <div className="font-mono font-bold text-slate-800">{mu.toFixed(3)}</div>
                    {result.posterior_var_diag && (
                      <div className="text-xs text-slate-400 mt-0.5">
                        ±{Math.sqrt(result.posterior_var_diag[i]).toFixed(3)}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Learn panel */}
      <InfoPanel
        content={blrContent}
        open={learnOpen}
        width={panelWidth}
        activeTab={learnTab}
        onToggle={onLearnToggle}
        onClose={onLearnClose}
        onWidthChange={onLearnWidthChange}
        onActiveTabChange={onLearnTabChange}
      />

      {/* Floating action button — shifts left when panel is open */}
      <div
        className="fixed bottom-6 flex flex-col gap-3 transition-all duration-300"
        style={{ right: learnOpen ? panelWidth + 24 : 24 }}
      >
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
