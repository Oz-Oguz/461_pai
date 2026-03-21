import { useState, useEffect, useRef } from 'react';
import { Play, StepForward, StepBack, RotateCcw, Calculator, Loader2 } from 'lucide-react';
import { SolverModal } from '../components/SolverModal';
import { InfoPanel } from '../components/InfoPanel';
import { RangeSlider } from '../components/RangeSlider';
import { kalmanContent } from '../content/kalman';
import type { SolveStep, LearnTab } from '../types';

// ── API ───────────────────────────────────────────────────────────────

interface SimResult {
  timesteps: number[];
  true_states: number[];
  measurements: number[];
  estimated_states: number[];
  estimated_stds: number[];
  predicted_states: number[];
  predicted_stds: number[];
  kalman_gains: number[];
  steps_detail: StepDetail[];
}

interface StepDetail {
  t: number;
  true: number;
  measurement: number;
  x_predicted: number;
  p_predicted: number;
  kalman_gain: number;
  x_updated: number;
  p_updated: number;
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

async function apiSimulate(q: number, r: number, n: number, seed: number): Promise<SimResult> {
  const res = await fetch('/api/kalman/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ process_noise_q: q, measurement_noise_r: r, n_steps: n, seed }),
  });
  return parseApiResponse<SimResult>(res);
}

async function apiSolve(q: number, r: number): Promise<SolveStep[]> {
  const res = await fetch('/api/kalman/solve', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ process_noise_q: q, measurement_noise_r: r }),
  });
  return parseApiResponse<SolveStep[]>(res);
}

// ── SVG time-series chart ─────────────────────────────────────────────

const W = 700, H = 300;
const PAD = { top: 20, right: 20, bottom: 40, left: 50 };

function makeScaleY(data: number[][]) {
  const all = data.flat();
  const lo = Math.min(...all) - 1;
  const hi = Math.max(...all) + 1;
  return { lo, hi };
}

function polyline(
  xs: number[], ys: number[],
  toX: (t: number) => number,
  toY: (y: number) => number,
): string {
  return xs.map((t, i) => `${toX(t).toFixed(1)},${toY(ys[i]).toFixed(1)}`).join(' ');
}

function bandPolyline(
  ts: number[], means: number[], stds: number[],
  toX: (t: number) => number,
  toY: (y: number) => number,
): string {
  const upper = ts.map((t, i) => `${toX(t).toFixed(1)},${toY(means[i] + stds[i]).toFixed(1)}`).join(' ');
  const lower = [...ts].reverse().map((t, i) => {
    const ri = ts.length - 1 - i;
    return `${toX(t).toFixed(1)},${toY(means[ri] - stds[ri]).toFixed(1)}`;
  }).join(' ');
  return `M ${upper.split(' ')[0]} L ${upper} L ${lower} Z`;
}

// ── Main component ────────────────────────────────────────────────────

interface KFProps {
  learnOpen: boolean;
  learnTab: LearnTab;
  panelWidth: number;
  onLearnToggle: () => void;
  onLearnClose: () => void;
  onLearnWidthChange: (w: number) => void;
  onLearnTabChange: (t: LearnTab) => void;
}

export function KalmanFilter({
  learnOpen,
  learnTab,
  panelWidth,
  onLearnToggle,
  onLearnClose,
  onLearnWidthChange,
  onLearnTabChange,
}: KFProps) {
  const [q, setQ] = useState(0.5);
  const [r, setR] = useState(2.0);
  const [nSteps, setNSteps] = useState(30);
  const [seed, setSeed] = useState(42);
  const [sim, setSim] = useState<SimResult | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [solverSteps, setSolverSteps] = useState<SolveStep[] | null>(null);
  const [solverLoading, setSolverLoading] = useState(false);
  const [rangeOverride, setRangeOverride] = useState<{ xMin: number; xMax: number; lo: number; hi: number } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  // Reset zoom whenever a new simulation loads
  useEffect(() => {
    setRangeOverride(null);
  }, [sim]);

  const runSim = () => {
    setLoading(true);
    setCurrentStep(0);
    setPlaying(false);
    apiSimulate(q, r, nSteps, seed)
      .then(setSim)
      .finally(() => setLoading(false));
  };

  useEffect(() => { runSim(); }, [q, r, nSteps, seed]);

  // Auto-play
  useEffect(() => {
    if (!playing || !sim) return;
    if (currentStep >= sim.timesteps.length - 1) { setPlaying(false); return; }
    const timer = setTimeout(() => setCurrentStep(s => s + 1), 200);
    return () => clearTimeout(timer);
  }, [playing, currentStep, sim]);

  const handleSolve = async () => {
    setSolverLoading(true);
    try {
      setSolverSteps(await apiSolve(q, r));
    } finally {
      setSolverLoading(false);
    }
  };

  if (!sim) return (
    <div className="flex flex-1 items-center justify-center">
      <Loader2 className="animate-spin text-blue-500" size={32} />
    </div>
  );

  const n = sim.timesteps.length;
  const visible = currentStep + 1;  // show steps 0..currentStep

  // Effective axis ranges: rangeOverride when zoomed, otherwise auto-scale from data
  const dataScale = makeScaleY([sim.true_states, sim.measurements, sim.estimated_states]);
  const xMin = rangeOverride?.xMin ?? 0;
  const xMax = rangeOverride?.xMax ?? (n - 1);
  const lo = rangeOverride?.lo ?? dataScale.lo;
  const hi = rangeOverride?.hi ?? dataScale.hi;
  const isRangeDefault = rangeOverride === null;
  const tToSvgX = (t: number) => PAD.left + ((t - xMin) / (xMax - xMin)) * (W - PAD.left - PAD.right);
  const yToSvg  = (y: number) => H - PAD.bottom - ((y - lo) / (hi - lo)) * (H - PAD.top - PAD.bottom);

  // Partial slices up to currentStep
  const ts  = sim.timesteps.slice(0, visible);
  const tr  = sim.true_states.slice(0, visible);
  const meas = sim.measurements.slice(0, visible);
  const est  = sim.estimated_states.slice(0, visible);
  const estS = sim.estimated_stds.slice(0, visible);

  const detail = sim.steps_detail[currentStep];
  const xTicks = Array.from({ length: 7 }, (_, i) => Math.round(xMin + (i / 6) * (xMax - xMin)));
  const yTicks = Array.from({ length: 5 }, (_, i) => lo + (i / 4) * (hi - lo));


  return (
    <div className="flex flex-1 overflow-hidden">
      {/* ── Sidebar ── */}
      <aside className="w-72 bg-white border-r border-slate-200 flex flex-col p-5 gap-5 overflow-y-auto shrink-0">
        <div>
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">
            Filter Parameters
          </h3>

          <label className="block mb-4">
            <div className="flex justify-between mb-1">
              <span className="text-sm font-medium text-slate-700">Process Noise Q</span>
              <span className="font-mono text-sm text-slate-500">{q.toFixed(2)}</span>
            </div>
            <input type="range" min={0.01} max={5} step={0.01} value={q}
              onChange={e => setQ(parseFloat(e.target.value))}
              className="w-full accent-blue-600" />
            <p className="text-xs text-slate-400 mt-1">How fast does the true state wander?</p>
          </label>

          <label className="block mb-4">
            <div className="flex justify-between mb-1">
              <span className="text-sm font-medium text-slate-700">Measurement Noise R</span>
              <span className="font-mono text-sm text-slate-500">{r.toFixed(2)}</span>
            </div>
            <input type="range" min={0.01} max={10} step={0.01} value={r}
              onChange={e => setR(parseFloat(e.target.value))}
              className="w-full accent-blue-600" />
            <p className="text-xs text-slate-400 mt-1">How noisy is the sensor?</p>
          </label>

          <label className="block mb-4">
            <div className="flex justify-between mb-1">
              <span className="text-sm font-medium text-slate-700">Time Steps</span>
              <span className="font-mono text-sm text-slate-500">{nSteps}</span>
            </div>
            <input type="range" min={10} max={60} step={1} value={nSteps}
              onChange={e => setNSteps(parseInt(e.target.value))}
              className="w-full accent-blue-600" />
          </label>

          <label className="block">
            <div className="flex justify-between mb-1">
              <span className="text-sm font-medium text-slate-700">Seed</span>
              <span className="font-mono text-sm text-slate-500">{seed}</span>
            </div>
            <input type="range" min={1} max={100} step={1} value={seed}
              onChange={e => setSeed(parseInt(e.target.value))}
              className="w-full accent-blue-600" />
            <p className="text-xs text-slate-400 mt-1">Different random trajectory</p>
          </label>
        </div>

        {/* Step-through controls */}
        <div>
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3">
            Step-Through
          </h3>
          <div className="flex gap-2 mb-3">
            <button onClick={() => { setPlaying(false); setCurrentStep(0); }}
              className="p-2 rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-600 transition-colors">
              <RotateCcw size={15} />
            </button>
            <button onClick={() => setCurrentStep(s => Math.max(0, s - 1))}
              disabled={currentStep === 0}
              className="p-2 rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-600 transition-colors disabled:opacity-30">
              <StepBack size={15} />
            </button>
            <button onClick={() => setPlaying(p => !p)}
              className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg text-sm font-semibold transition-colors
                ${playing ? 'bg-amber-500 text-white hover:bg-amber-600' : 'bg-blue-600 text-white hover:bg-blue-700'}`}>
              <Play size={14} className={playing ? 'hidden' : ''} />
              {playing ? 'Pause' : 'Play'}
            </button>
            <button onClick={() => setCurrentStep(s => Math.min(n - 1, s + 1))}
              disabled={currentStep === n - 1}
              className="p-2 rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-600 transition-colors disabled:opacity-30">
              <StepForward size={15} />
            </button>
          </div>

          <div className="text-xs text-slate-400 text-center">
            t = {currentStep} / {n - 1}
          </div>
        </div>

        {/* Legend */}
        <div className="text-xs text-slate-500 space-y-1.5">
          <div className="flex items-center gap-2"><div className="w-6 h-0.5 bg-slate-400" style={{borderTop:'2px dashed #94a3b8'}} /> True state (hidden)</div>
          <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-red-500" /> Measurements</div>
          <div className="flex items-center gap-2"><div className="w-6 h-1 rounded-full bg-blue-600" /> Kalman estimate</div>
          <div className="flex items-center gap-2"><div className="w-6 h-3 rounded-full bg-blue-200 opacity-70" /> ±1σ uncertainty</div>
        </div>
      </aside>

      {/* ── Main ── */}
      <main className="flex-1 overflow-y-auto bg-slate-50 p-8">
        <div className="max-w-3xl mx-auto">
          <div className="mb-6 text-center">
            <h2 className="text-2xl font-bold text-slate-800 mb-1">Kalman Filter</h2>
            <p className="text-slate-500 text-sm">
              1-D random-walk tracking. Press Play or step through manually.
            </p>
          </div>

          {/* Main plot */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden mb-6">
            <div className="px-5 py-3 border-b border-slate-100 bg-slate-50 flex items-center justify-between">
              <span className="text-sm font-medium text-slate-700">State Estimate vs. Ground Truth</span>
              <div className="flex items-center gap-2 text-xs text-slate-400">
                {loading && <Loader2 size={13} className="animate-spin text-blue-400" />}
              </div>
            </div>
            <div className="relative">
            <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="w-full">
              <defs>
                <clipPath id="kalman-clip">
                  <rect x={PAD.left} y={PAD.top} width={W - PAD.left - PAD.right} height={H - PAD.top - PAD.bottom} />
                </clipPath>
              </defs>

              {/* Grid */}
              {xTicks.map(t => (
                <g key={`xt${t}`}>
                  <line x1={tToSvgX(t)} y1={PAD.top} x2={tToSvgX(t)} y2={H - PAD.bottom} stroke="#e2e8f0" strokeWidth={1} />
                  <text x={tToSvgX(t)} y={H - PAD.bottom + 16} textAnchor="middle" fontSize={10} fill="#94a3b8">{t}</text>
                </g>
              ))}
              {yTicks.map((y, i) => (
                <g key={`yt${i}`}>
                  <line x1={PAD.left} y1={yToSvg(y)} x2={W - PAD.right} y2={yToSvg(y)} stroke="#e2e8f0" strokeWidth={1} />
                  <text x={PAD.left - 6} y={yToSvg(y) + 4} textAnchor="end" fontSize={10} fill="#94a3b8">{y.toFixed(1)}</text>
                </g>
              ))}

              <g clipPath="url(#kalman-clip)">
                {/* Uncertainty band */}
                {visible > 0 && (
                  <path d={bandPolyline(ts, est, estS, tToSvgX, yToSvg)} fill="#bfdbfe" opacity={0.5} />
                )}

                {/* True state (dashed) */}
                {visible > 1 && (
                  <polyline points={polyline(ts, tr, tToSvgX, yToSvg)} fill="none" stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="5,4" />
                )}

                {/* Kalman estimate */}
                {visible > 1 && (
                  <polyline points={polyline(ts, est, tToSvgX, yToSvg)} fill="none" stroke="#2563eb" strokeWidth={2.5} />
                )}

                {/* Measurements */}
                {meas.map((z, i) => (
                  <circle key={i} cx={tToSvgX(ts[i])} cy={yToSvg(z)} r={3.5}
                    fill="#dc2626" stroke="white" strokeWidth={1} opacity={0.8} />
                ))}

                {/* Current step indicator */}
                <line
                  x1={tToSvgX(currentStep)} y1={PAD.top}
                  x2={tToSvgX(currentStep)} y2={H - PAD.bottom}
                  stroke="#7c3aed" strokeWidth={1.5} strokeDasharray="3,3" opacity={0.7}
                />
              </g>
            </svg>
            </div>

            {/* Axis range sliders */}
            {(() => {
              const yPad = Math.max(2, (dataScale.hi - dataScale.lo) * 0.6);
              return (
                <div className="px-5 py-3 border-t border-slate-100 space-y-2">
                  <RangeSlider
                    label="X"
                    trackMin={0} trackMax={n - 1}
                    value={{ min: xMin, max: xMax }}
                    onChange={v => setRangeOverride({ xMin: v.min, xMax: v.max, lo, hi })}
                    step={1} decimals={0}
                  />
                  <RangeSlider
                    label="Y"
                    trackMin={dataScale.lo - yPad} trackMax={dataScale.hi + yPad}
                    value={{ min: lo, max: hi }}
                    onChange={v => setRangeOverride({ xMin, xMax, lo: v.min, hi: v.max })}
                    step={0.5}
                  />
                  {!isRangeDefault && (
                    <button
                      onClick={() => setRangeOverride(null)}
                      className="text-xs text-slate-400 hover:text-slate-600 transition-colors"
                    >
                      ↺ Auto-scale
                    </button>
                  )}
                </div>
              );
            })()}
          </div>

          {/* Step detail card */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
            <h3 className="text-sm font-bold text-slate-700 mb-4">
              Step t = {detail.t} — Predict → Update
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-amber-50 border border-amber-100 rounded-xl p-4">
                <p className="text-xs font-bold text-amber-700 uppercase tracking-wide mb-2">Predict</p>
                <div className="space-y-1 text-sm font-mono">
                  <div><span className="text-slate-500">x̂ (pred)</span> = <strong>{detail.x_predicted}</strong></div>
                  <div><span className="text-slate-500">P (pred)</span> = <strong>{detail.p_predicted}</strong></div>
                </div>
              </div>
              <div className="bg-blue-50 border border-blue-100 rounded-xl p-4">
                <p className="text-xs font-bold text-blue-700 uppercase tracking-wide mb-2">Update</p>
                <div className="space-y-1 text-sm font-mono">
                  <div><span className="text-slate-500">z (meas)</span> = <strong>{detail.measurement}</strong></div>
                  <div><span className="text-slate-500">K (gain)</span> = <strong>{detail.kalman_gain}</strong></div>
                  <div><span className="text-slate-500">x̂ (post)</span> = <strong>{detail.x_updated}</strong></div>
                  <div><span className="text-slate-500">P (post)</span> = <strong>{detail.p_updated}</strong></div>
                </div>
              </div>
            </div>
            <div className="mt-3 text-xs text-slate-400 text-center">
              True state (hidden from filter): <strong className="text-slate-600">{detail.true}</strong>
            </div>
          </div>
        </div>
      </main>

      {/* Learn panel */}
      <InfoPanel
        content={kalmanContent}
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
        <button onClick={handleSolve} disabled={solverLoading}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-green-600 text-white shadow-lg hover:bg-green-700 transition-all text-sm font-semibold disabled:opacity-60">
          {solverLoading ? <Loader2 size={15} className="animate-spin" /> : <Calculator size={15} />}
          Show Equations
        </button>
      </div>

      {solverSteps && (
        <SolverModal steps={solverSteps} onClose={() => setSolverSteps(null)} />
      )}
    </div>
  );
}
