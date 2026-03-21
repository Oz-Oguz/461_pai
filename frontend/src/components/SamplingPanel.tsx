/**
 * SamplingPanel — right-side panel for approximate inference via sampling.
 *
 * Features:
 *  - Method selector: Prior | Rejection | Likelihood Weighting | Gibbs
 *  - N samples slider + burn-in slider (Gibbs only)
 *  - SVG convergence chart: estimated P(X=state) vs. sample steps
 *    with dashed exact-answer reference line
 *  - Per-node approximate vs. exact marginal comparison
 *  - Metadata badges: acceptance rate, ESS, burn-in
 */

import { useState, useCallback } from 'react';
import { Dices, Play, Loader2, X, ChevronRight } from 'lucide-react';
import type { ModelDetail, SamplingResult, SamplingMethod, SamplingMetadata, CIBand } from '../types';
import { api } from '../api/client';

// ── Palette ───────────────────────────────────────────────────────────
const METHOD_META: Record<SamplingMethod, { label: string; color: string; desc: string }> = {
  prior: {
    label: 'Prior',
    color: '#6366f1',  // indigo
    desc: 'Samples from the joint P(X) — evidence is ignored by design. Use this as a baseline to see how much evidence shifts the marginals.',
  },
  rejection: {
    label: 'Rejection',
    color: '#f59e0b',  // amber
    desc: 'Discards samples inconsistent with evidence.',
  },
  likelihood_weighting: {
    label: 'Likelihood Weighting',
    color: '#10b981',  // emerald
    desc: 'Fixes evidence, weights samples by P(e|parents).',
  },
  gibbs: {
    label: 'Gibbs',
    color: '#ef4444',  // red
    desc: 'MCMC: resamples one variable at a time from its Markov blanket.',
  },
};

const METHODS: SamplingMethod[] = ['prior', 'rejection', 'likelihood_weighting', 'gibbs'];

// ── Simple SVG line chart ─────────────────────────────────────────────
interface SeriesData {
  method: SamplingMethod;
  estimates: number[];
  ciBand?: CIBand;
}

interface LineChartProps {
  steps: number[];
  exact: number;
  label: string;
  series: SeriesData[];  // One or more series for comparison
}

function LineChart({ steps, exact, label, series }: LineChartProps) {
  if (steps.length === 0 || series.length === 0) return null;

  const W = 320, H = 120;
  const PAD = { top: 10, right: 16, bottom: 28, left: 36 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  const minY = 0, maxY = 1;
  const minX = steps[0], maxX = steps[steps.length - 1];
  const xRange = maxX - minX || 1;

  const toX = (v: number) => PAD.left + ((v - minX) / xRange) * plotW;
  const toY = (v: number) => PAD.top + (1 - (v - minY) / (maxY - minY)) * plotH;

  const exactY = toY(exact);

  // x-axis ticks (up to 5)
  const tickCount = 5;
  const tickStep = Math.ceil(maxX / tickCount / 100) * 100 || 1;
  const ticks: number[] = [];
  for (let t = 0; t <= maxX; t += tickStep) ticks.push(t);

  return (
    <svg width={W} height={H} className="overflow-visible select-none">
      {/* Grid lines */}
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <line key={v} x1={PAD.left} x2={PAD.left + plotW} y1={toY(v)} y2={toY(v)}
          stroke="#f1f5f9" strokeWidth={1} />
      ))}

      {/* Render all series (CI bands first, then lines) */}
      {series.map((s) => {
        const color = METHOD_META[s.method].color;
        const ciPath = s.ciBand && s.ciBand.high.length === steps.length && s.ciBand.low.length === steps.length
          ? (() => {
            const topPts = steps.map((_, i) => `${toX(steps[i])},${toY(s.ciBand!.high[i])}`);
            const botPts = steps.map((_, i) => `${toX(steps[i])},${toY(s.ciBand!.low[i])}`).reverse();
            return `M ${topPts.join(' L ')} L ${botPts.join(' L ')} Z`;
          })()
          : '';
        const pts = steps.map((st, i) => `${toX(st)},${toY(s.estimates[i])}`).join(' ');
        return (
          <g key={s.method}>
            {/* CI band */}
            {ciPath && <path d={ciPath} fill={color} fillOpacity={0.15} stroke="none" />}
            {/* Mean line */}
            {steps.length > 1 && (
              <polyline points={pts} fill="none" stroke={color} strokeWidth={2}
                strokeLinejoin="round" strokeLinecap="round" />
            )}
          </g>
        );
      })}

      {/* Exact reference line (on top) */}
      <line x1={PAD.left} x2={PAD.left + plotW} y1={exactY} y2={exactY}
        stroke="#64748b" strokeWidth={1.5} strokeDasharray="5,3" opacity={0.5} />
      <text x={PAD.left + plotW + 3} y={exactY + 4} fontSize={9} fill="#64748b" opacity={0.7}>
        {exact.toFixed(2)}
      </text>

      {/* Axes */}
      <line x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={PAD.top + plotH}
        stroke="#cbd5e1" strokeWidth={1} />
      <line x1={PAD.left} x2={PAD.left + plotW} y1={PAD.top + plotH} y2={PAD.top + plotH}
        stroke="#cbd5e1" strokeWidth={1} />

      {/* Y-axis labels */}
      {[0, 0.5, 1].map((v) => (
        <text key={v} x={PAD.left - 3} y={toY(v) + 4} fontSize={9} fill="#94a3b8"
          textAnchor="end">{v.toFixed(1)}</text>
      ))}

      {/* X-axis ticks */}
      {ticks.map((t) => (
        <text key={t} x={toX(t)} y={PAD.top + plotH + 14} fontSize={9} fill="#94a3b8"
          textAnchor="middle">{t}</text>
      ))}

      {/* Chart title */}
      <text x={PAD.left} y={PAD.top - 2} fontSize={10} fill="#64748b" fontWeight="600">
        {label}
      </text>
    </svg>
  );
}

// ── Metadata badges ───────────────────────────────────────────────────
function MetaBadges({ meta, method }: { meta: SamplingMetadata; method: SamplingMethod }) {
  return (
    <div className="flex flex-wrap gap-2 mt-2">
      <span className="text-xs bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full font-mono">
        N = {meta.n_samples}
      </span>
      {method === 'rejection' && meta.acceptance_rate !== undefined && (
        <span className={`text-xs px-2 py-0.5 rounded-full font-mono font-semibold
          ${meta.acceptance_rate < 0.1 ? 'bg-red-100 text-red-700' :
            meta.acceptance_rate < 0.3 ? 'bg-amber-100 text-amber-700' :
              'bg-green-100 text-green-700'}`}>
          Accept: {(meta.acceptance_rate * 100).toFixed(1)}%
        </span>
      )}
      {method === 'likelihood_weighting' && meta.effective_samples !== undefined && (
        <span className="text-xs bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded-full font-mono">
          ESS: {meta.effective_samples.toFixed(0)}
        </span>
      )}
      {method === 'gibbs' && meta.n_burn !== undefined && (
        <span className="text-xs bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full font-mono">
          Burn-in: {meta.n_burn}
        </span>
      )}
      {meta.n_runs !== undefined && meta.n_runs > 1 && (
        <span className="text-xs bg-violet-100 text-violet-700 px-2 py-0.5 rounded-full font-mono">
          Runs: {meta.n_runs}
        </span>
      )}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────
interface Props {
  model: ModelDetail;
  evidence: Record<string, string>;
  priors: Record<string, Record<string, number>>;
  cpts: Record<string, { parents: string[]; table: Record<string, Record<string, number>> }>;
}

export function SamplingPanel({ model, evidence, priors, cpts }: Props) {
  const [open, setOpen] = useState(false);
  const [compareMode, setCompareMode] = useState(false);
  const [method, setMethod] = useState<SamplingMethod>('likelihood_weighting');
  const [selectedMethods, setSelectedMethods] = useState<Set<SamplingMethod>>(new Set());
  const [nSamples, setNSamples] = useState(500);
  const [nBurn, setNBurn] = useState(100);
  const [nRuns, setNRuns] = useState(1);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<SamplingResult | null>(null);
  const [comparisonResults, setComparisonResults] = useState<Partial<Record<SamplingMethod, SamplingResult>>>({});
  const [error, setError] = useState<string | null>(null);

  // Which node/state to display in the convergence chart
  const firstNode = model.nodes[0];
  const [chartNodeId, setChartNodeId] = useState(firstNode.id);
  const [chartState, setChartState] = useState(firstNode.states[0]);

  const handleRun = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      if (compareMode && selectedMethods.size > 0) {
        // Run all selected methods in parallel
        const promises = Array.from(selectedMethods).map((m) =>
          api.sample(model.id, m, evidence, nSamples, nBurn, priors, cpts,
            m === 'gibbs' ? nRuns : 1,
          )
        );
        const results = await Promise.all(promises);
        const resultsMap: Partial<Record<SamplingMethod, SamplingResult>> = {};
        Array.from(selectedMethods).forEach((m, i) => {
          resultsMap[m] = results[i];
        });
        setComparisonResults(resultsMap);
      } else {
        // Single method mode
        const res = await api.sample(
          model.id, method, evidence, nSamples, nBurn, priors, cpts,
          method === 'gibbs' ? nRuns : 1,
        );
        setResult(res);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Sampling failed');
    } finally {
      setRunning(false);
    }
  }, [model.id, method, selectedMethods, compareMode, evidence, nSamples, nBurn, nRuns, priors, cpts]);

  const toggleMethod = (m: SamplingMethod) => {
    const newSet = new Set(selectedMethods);
    if (newSet.has(m)) {
      newSet.delete(m);
    } else {
      newSet.add(m);
    }
    setSelectedMethods(newSet);
  };

  const clearComparison = () => {
    setSelectedMethods(new Set());
    setComparisonResults({});
  };

  const meta = METHOD_META[method];

  // Build series data for chart (single or comparison mode)
  const chartSeries: SeriesData[] = compareMode && Object.keys(comparisonResults).length > 0
    ? Array.from(selectedMethods)
      .filter((m) => comparisonResults[m])
      .map((m) => {
        const res = comparisonResults[m]!; // Already filtered above
        return {
          method: m,
          estimates: res.trajectory.curves[chartNodeId]?.[chartState] ?? [],
          ciBand: res.trajectory.ci_bands?.[chartNodeId]?.[chartState],
        };
      })
    : result
      ? [{
        method: result.method as SamplingMethod,
        estimates: result.trajectory.curves[chartNodeId]?.[chartState] ?? [],
        ciBand: result.trajectory.ci_bands?.[chartNodeId]?.[chartState],
      }]
      : [];

  const chartSteps = compareMode && Object.keys(comparisonResults).length > 0
    ? comparisonResults[Array.from(selectedMethods)[0]]?.trajectory.steps ?? []
    : result?.trajectory.steps ?? [];

  const chartExact = compareMode && Object.keys(comparisonResults).length > 0
    ? comparisonResults[Array.from(selectedMethods)[0]]?.exact_marginals[chartNodeId]?.[chartState] ?? 0
    : result?.exact_marginals[chartNodeId]?.[chartState] ?? 0;

  return (
    <aside className="relative shrink-0 flex flex-col transition-all duration-300 ease-in-out"
      style={{ width: open ? 360 : 0 }}>

      {/* ── Pull-tab ── */}
      <button
        onClick={() => setOpen((v) => !v)}
        title={open ? 'Close Sampling panel' : 'Open Sampling panel'}
        className={`
          absolute left-0 top-1/3 -translate-y-1/2 -translate-x-full
          flex flex-col items-center justify-center gap-1.5
          w-7 py-5 rounded-l-xl border border-r-0 shadow-md
          transition-colors z-20 select-none
          ${open
            ? 'bg-violet-600 text-white border-violet-600 hover:bg-violet-700'
            : 'bg-white text-slate-500 border-slate-200 hover:border-violet-300 hover:text-violet-600'}
        `}
      >
        <Dices size={13} />
        <span className="text-[10px] font-bold tracking-widest uppercase"
          style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>
          Sample
        </span>
      </button>

      {/* ── Panel body ── */}
      <div className="flex flex-col h-full bg-white border-l border-slate-200 overflow-hidden"
        style={{ width: open ? 360 : 0, minWidth: open ? 360 : 0 }}>
        {open && (
          <>
            {/* Header */}
            <div className="px-4 pt-4 pb-0 shrink-0">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h2 className="font-bold text-slate-800 text-sm">Approximate Inference</h2>
                  <p className="text-xs text-slate-400 mt-0.5">Sampling-based methods</p>
                </div>
                <button onClick={() => setOpen(false)}
                  className="text-slate-400 hover:text-slate-700 p-1 rounded-lg hover:bg-slate-100 transition-colors">
                  <X size={16} />
                </button>
              </div>

              {/* Comparison mode toggle */}
              <div className="flex items-center gap-2 mb-3 p-2 bg-slate-50 rounded-lg">
                <label className="text-xs font-semibold text-slate-600">Mode:</label>
                <label className="flex items-center gap-1 cursor-pointer text-xs">
                  <input type="radio" checked={!compareMode} onChange={() => setCompareMode(false)}
                    className="w-3 h-3" />
                  <span className="text-slate-600">Single</span>
                </label>
                <label className="flex items-center gap-1 cursor-pointer text-xs">
                  <input type="radio" checked={compareMode} onChange={() => setCompareMode(true)}
                    className="w-3 h-3" />
                  <span className="text-slate-600">Compare</span>
                </label>
              </div>

              {/* Method selector */}
              {!compareMode ? (
                <>
                  <div className="grid grid-cols-2 gap-1.5 mb-4">
                    {METHODS.map((m) => (
                      <button key={m} onClick={() => setMethod(m)}
                        className={`px-2 py-1.5 rounded-lg text-xs font-medium text-left transition-colors border
                          ${method === m
                            ? 'border-violet-400 bg-violet-50 text-violet-700'
                            : 'border-slate-200 text-slate-600 hover:border-slate-300 hover:bg-slate-50'
                          }`}>
                        {METHOD_META[m].label}
                      </button>
                    ))}
                  </div>
                  <p className="text-xs text-slate-400 mb-3 leading-relaxed">{meta.desc}</p>
                </>
              ) : (
                <div className="space-y-2 mb-3">
                  <div className="text-xs font-semibold text-slate-600 mb-2">Select methods:</div>
                  <div className="grid grid-cols-2 gap-2">
                    {METHODS.map((m) => (
                      <label key={m} className="flex items-center gap-2 cursor-pointer p-2 rounded-lg hover:bg-slate-50">
                        <input type="checkbox" checked={selectedMethods.has(m)}
                          onChange={() => toggleMethod(m)}
                          className="w-4 h-4" />
                        <span className="text-xs text-slate-600">{METHOD_META[m].label}</span>
                      </label>
                    ))}
                  </div>
                  {selectedMethods.size > 0 && (
                    <button onClick={clearComparison}
                      className="text-xs text-slate-500 hover:text-slate-700 mt-1">
                      Clear
                    </button>
                  )}
                </div>
              )}
              <div className="border-b border-slate-200 mb-0" />
            </div>

            {/* Scrollable content */}
            <div className="flex-1 overflow-y-auto px-4 py-4 space-y-5">

              {/* Controls */}
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <label className="text-xs font-semibold text-slate-600">Samples (N)</label>
                    <span className="text-xs font-mono bg-slate-100 px-1.5 py-0.5 rounded text-slate-500">
                      {nSamples}
                    </span>
                  </div>
                  <input type="range" min={50} max={10000} step={100} value={nSamples}
                    onChange={(e) => setNSamples(Number(e.target.value))}
                    className="w-full h-1.5 rounded-full accent-violet-600 cursor-pointer" />
                  <div className="flex justify-between mt-0.5 text-xs text-slate-400">
                    <span>50</span><span>10000</span>
                  </div>
                </div>

                {(method === 'gibbs' || (compareMode && selectedMethods.has('gibbs'))) && (
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-xs font-semibold text-slate-600">Burn-in</label>
                      <span className="text-xs font-mono bg-slate-100 px-1.5 py-0.5 rounded text-slate-500">
                        {nBurn}
                      </span>
                    </div>
                    <input type="range" min={0} max={500} step={50} value={nBurn}
                      onChange={(e) => setNBurn(Number(e.target.value))}
                      className="w-full h-1.5 rounded-full accent-violet-600 cursor-pointer" />
                    <div className="flex justify-between mt-0.5 text-xs text-slate-400">
                      <span>0</span><span>500</span>
                    </div>
                  </div>
                )}

                <div>
                  <div className="flex justify-between items-center mb-1">
                    <label className="text-xs font-semibold text-slate-600">
                      Runs
                      <span className="ml-1 font-normal text-slate-400">(mean ± 1σ)</span>
                    </label>
                    <span className="text-xs font-mono bg-slate-100 px-1.5 py-0.5 rounded text-slate-500">
                      {nRuns}
                    </span>
                  </div>
                  <input type="range" min={1} max={10} step={1} value={nRuns}
                    onChange={(e) => setNRuns(Number(e.target.value))}
                    className="w-full h-1.5 rounded-full accent-violet-600 cursor-pointer" />
                  <div className="flex justify-between mt-0.5 text-xs text-slate-400">
                    <span>1</span><span>10</span>
                  </div>
                </div>

                {Object.keys(evidence).length === 0 && ((compareMode && !selectedMethods.has('prior')) || (!compareMode && method !== 'prior')) && (
                  <div className="text-xs bg-amber-50 border border-amber-100 rounded-lg p-2 text-amber-700">
                    No evidence set — load a scenario or click a node to set evidence.
                  </div>
                )}

                <button
                  onClick={handleRun}
                  disabled={running || (compareMode && selectedMethods.size === 0)}
                  className="w-full py-2 rounded-xl bg-violet-600 text-white text-sm font-semibold hover:bg-violet-700 disabled:opacity-60 transition-colors flex items-center justify-center gap-2 shadow-sm"
                >
                  {running
                    ? <><Loader2 size={14} className="animate-spin" /> Running…</>
                    : compareMode
                      ? <><Play size={14} /> Run All Selected</>
                      : <><Play size={14} /> Run {METHOD_META[method].label}</>
                  }
                </button>

                {error && (
                  <p className="text-xs text-red-600 bg-red-50 border border-red-100 rounded-lg px-3 py-2">
                    {error}
                  </p>
                )}
              </div>

              {/* Results */}
              {(result || Object.keys(comparisonResults).length > 0) && (
                <>
                  {!compareMode && result && (
                    <MetaBadges meta={result.metadata} method={result.method as SamplingMethod} />
                  )}
                  {compareMode && Object.keys(comparisonResults).length > 0 && (
                    <div className="space-y-2">
                      {Array.from(selectedMethods)
                        .filter((m) => comparisonResults[m])
                        .map((m) => (
                          <MetaBadges key={m} meta={comparisonResults[m]!.metadata} method={m} />
                        ))}
                    </div>
                  )}

                  {/* Convergence chart */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                        Convergence
                      </span>
                      <div className="flex gap-1">
                        <select
                          value={chartNodeId}
                          onChange={(e) => {
                            const nid = e.target.value;
                            setChartNodeId(nid);
                            const node = model.nodes.find(n => n.id === nid);
                            if (node) setChartState(node.states[0]);
                          }}
                          className="text-xs border border-slate-200 rounded px-1.5 py-0.5 text-slate-600 bg-white">
                          {model.nodes.map((n) => (
                            <option key={n.id} value={n.id}>{n.label}</option>
                          ))}
                        </select>
                        <select
                          value={chartState}
                          onChange={(e) => setChartState(e.target.value)}
                          className="text-xs border border-slate-200 rounded px-1.5 py-0.5 text-slate-600 bg-white">
                          {(model.nodes.find(n => n.id === chartNodeId)?.states ?? []).map((s) => (
                            <option key={s} value={s}>{s}</option>
                          ))}
                        </select>
                      </div>
                    </div>

                    <div className="bg-slate-50 border border-slate-100 rounded-xl p-3 overflow-x-auto">
                      <LineChart
                        steps={chartSteps}
                        exact={chartExact}
                        label={`P(${chartNodeId}=${chartState})`}
                        series={chartSeries}
                      />
                      <div className="flex items-center gap-3 mt-2 text-xs text-slate-400 flex-wrap">
                        <span className="flex items-center gap-1.5">
                          <span className="inline-block w-6 h-0.5 border-t-2 border-dashed"
                            style={{ borderColor: '#64748b', opacity: 0.7 }} />
                          Exact
                        </span>
                        {chartSeries.map((s) => (
                          <span key={s.method} className="flex items-center gap-1.5">
                            <span className="inline-block w-6 h-0.5"
                              style={{ backgroundColor: METHOD_META[s.method].color }} />
                            {METHOD_META[s.method].label}
                            {s.ciBand && (
                              <>
                                {' '}
                                <span className="inline-block w-4 h-2 rounded-sm"
                                  style={{ backgroundColor: METHOD_META[s.method].color, opacity: 0.2 }} />
                              </>
                            )}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Per-node approximate vs. exact */}
                  {!compareMode && result && (
                    <div>
                      <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">
                        Approximate vs. Exact
                      </h3>
                      <div className="space-y-2">
                        {model.nodes.map((node) => {
                          const approx = result.marginals[node.id] ?? {};
                          const exact = result.exact_marginals[node.id] ?? {};
                          return (
                            <div key={node.id}
                              className="border border-slate-100 rounded-xl p-3 bg-slate-50">
                              <div className="text-xs font-semibold text-slate-700 mb-2 flex items-center gap-1">
                                <ChevronRight size={10} className="text-slate-400" />
                                {node.label}
                              </div>
                              {node.states.map((state) => {
                                const ap = approx[state] ?? 0;
                                const ex = exact[state] ?? 0;
                                const diff = ap - ex;
                                return (
                                  <div key={state} className="flex items-center gap-2 mb-1">
                                    <span className="text-xs text-slate-500 w-16 truncate">{state}</span>
                                    <div className="flex-1 h-2 bg-slate-200 rounded-full overflow-hidden relative">
                                      {/* Exact (background) */}
                                      <div className="absolute inset-y-0 left-0 bg-slate-300 rounded-full"
                                        style={{ width: `${ex * 100}%` }} />
                                      {/* Approximate (foreground) */}
                                      <div className="absolute inset-y-0 left-0 rounded-full opacity-80"
                                        style={{ width: `${ap * 100}%`, backgroundColor: meta.color }} />
                                    </div>
                                    <span className="text-xs font-mono text-slate-600 w-10 text-right">
                                      {(ap * 100).toFixed(1)}%
                                    </span>
                                    <span className={`text-xs font-mono w-12 text-right
                                      ${Math.abs(diff) < 0.02 ? 'text-slate-400' :
                                        diff > 0 ? 'text-violet-600' : 'text-amber-600'}`}>
                                      {diff > 0 ? '+' : ''}{(diff * 100).toFixed(1)}%
                                    </span>
                                  </div>
                                );
                              })}
                            </div>
                          );
                        })}
                      </div>
                      <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                        Colored bar = estimate. Grey bar = exact. Diff = estimate − exact.
                      </p>
                    </div>
                  )}
                </>
              )}
            </div>
          </>
        )}
      </div>
    </aside>
  );
}
