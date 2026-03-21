import { useState } from 'react';
import { Table2, HelpCircle } from 'lucide-react';
import type { BayesianNode, InferResult, CPTTable } from '../types';
import { Modal } from './Modal';
import { MathBlock } from './MathBlock';
import { api } from '../api/client';

interface Props {
  node: BayesianNode;
  results: InferResult | null;
  priorMarginals: InferResult | null;
  evidence: Record<string, string>;
  priors: Record<string, Record<string, number>>;
  cpts: Record<string, CPTTable>;
  modelId: string;
  onToggleEvidence: (nodeId: string, state: string) => void;
  onCPTChange: (nodeId: string, parentKey: string, state: string, value: number) => void;
}

export function NodeCard({
  node,
  results,
  priorMarginals,
  evidence,
  priors,
  cpts,
  modelId,
  onToggleEvidence,
  onCPTChange,
}: Props) {
  const [showCPT, setShowCPT] = useState(false);
  const [showHint, setShowHint] = useState(false);
  const [hintLatex, setHintLatex] = useState('');
  const [hintLoading, setHintLoading] = useState(false);

  const isObserved = node.id in evidence;
  const isRoot = node.node_type === 'root';
  const marginals = results?.marginals[node.id] ?? {};
  const priorMarg = priorMarginals?.marginals[node.id] ?? {};

  const handleHint = async () => {
    setHintLoading(true);
    setShowHint(true);
    try {
      const data = await api.marginalDerivation(modelId, node.id, priors, cpts);
      setHintLatex(data.latex);
    } catch (e) {
      setHintLatex('\\text{Error loading derivation}');
    } finally {
      setHintLoading(false);
    }
  };

  return (
    <>
      <div
        className={`
          bg-white rounded-2xl border-2 transition-all duration-300 overflow-hidden
          ${isObserved
            ? 'border-blue-500 shadow-lg ring-4 ring-blue-500/10'
            : 'border-slate-200 shadow-sm hover:shadow-md hover:border-slate-300'}
        `}
      >
        {/* Card Header */}
        <div
          className={`px-5 py-3 border-b flex items-center justify-between
            ${isObserved ? 'bg-blue-50 border-blue-100' : 'bg-slate-50 border-slate-100'}`}
        >
          <div>
            <h3 className="font-bold text-slate-800 text-sm">{node.label}</h3>
            <p className="text-xs text-slate-400 mt-0.5">
              {isRoot ? 'Root Cause' : 'Sensor / Effect'}
            </p>
          </div>
          <div className="flex items-center gap-1">
            {/* CPT Button */}
            <button
              onClick={() => setShowCPT(true)}
              className="p-1.5 rounded-lg text-slate-400 hover:text-blue-600 hover:bg-blue-50 transition-colors"
              title="View Probability Table"
            >
              <Table2 size={14} />
            </button>
            {/* Hint Button (child nodes only) */}
            {!isRoot && (
              <button
                onClick={handleHint}
                className="p-1.5 rounded-lg text-slate-400 hover:text-amber-600 hover:bg-amber-50 transition-colors"
                title="How is this calculated?"
              >
                <HelpCircle size={14} />
              </button>
            )}
            {/* Clear evidence */}
            {isObserved && (
              <button
                onClick={() => onToggleEvidence(node.id, evidence[node.id])}
                className="ml-1 text-xs px-2 py-1 rounded-md bg-blue-100 text-blue-700 hover:bg-blue-200 font-semibold transition-colors"
              >
                CLEAR
              </button>
            )}
          </div>
        </div>

        {/* State Rows */}
        <div className="p-4 space-y-3">
          {node.states.map((state) => {
            const prob = marginals[state] ?? 0;
            const priorProb = priorMarg[state] ?? prob;
            const shift = prob - priorProb;
            const isSelected = evidence[node.id] === state;

            return (
              <div
                key={state}
                onClick={() => onToggleEvidence(node.id, state)}
                className={`
                  cursor-pointer rounded-xl border p-3 transition-all duration-200
                  ${isSelected
                    ? 'bg-blue-50 border-blue-200'
                    : 'bg-slate-50 border-transparent hover:bg-slate-100 hover:border-slate-200'}
                `}
              >
                <div className="flex items-center justify-between mb-2">
                  <span
                    className={`text-sm font-semibold ${isSelected ? 'text-blue-700' : 'text-slate-700'}`}
                  >
                    {state}
                  </span>
                  <div className="flex items-center gap-2">
                    {/* Shift badge */}
                    {Math.abs(shift) > 0.001 && Object.keys(evidence).length > 0 && (
                      <span
                        className={`text-xs font-bold px-1.5 py-0.5 rounded-md
                          ${shift > 0
                            ? 'bg-green-100 text-green-700'
                            : 'bg-red-100 text-red-700'}`}
                      >
                        {shift > 0 ? '+' : ''}{(shift * 100).toFixed(1)}%
                      </span>
                    )}
                    <span className="font-mono font-bold text-base text-slate-800">
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Probability bar */}
                <div className="h-2 w-full bg-slate-200 rounded-full overflow-hidden relative">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ease-out
                      ${isSelected ? 'bg-blue-500' : 'bg-slate-400'}`}
                    style={{ width: `${Math.min(prob * 100, 100)}%` }}
                  />
                  {/* Prior marker */}
                  {isRoot && !isObserved && Math.abs(priorProb - prob) > 0.001 && (
                    <div
                      className="absolute top-0 bottom-0 w-0.5 bg-slate-600/30 z-10"
                      style={{ left: `${priorProb * 100}%` }}
                      title="Original Prior"
                    />
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* CPT Modal */}
      {showCPT && (
        <Modal
          title={<><Table2 size={16} className="text-slate-500" /> {isRoot ? 'Prior Distribution' : 'Conditional Probability Table (CPT)'}: {node.label}</>}
          onClose={() => setShowCPT(false)}
          wide
        >
          {isRoot ? (
            <div>
              <p className="text-sm text-slate-500 mb-4">
                This is a root node. Adjust priors using the sidebar sliders.
              </p>
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="bg-slate-100">
                    <th className="border border-slate-200 px-3 py-2 text-left">State</th>
                    <th className="border border-slate-200 px-3 py-2 text-left">Probability</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(priors[node.id] ?? {}).map(([st, p]) => (
                    <tr key={st}>
                      <td className="border border-slate-200 px-3 py-2 font-medium">{st}</td>
                      <td className="border border-slate-200 px-3 py-2 font-mono text-blue-600">{p.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div>
              <p className="text-sm text-slate-500 mb-1">
                Edit probabilities below.{' '}
                <span className="text-amber-600 font-medium text-xs bg-amber-50 px-2 py-0.5 rounded-full">
                  Auto-normalizes rows to 1.0
                </span>
              </p>
              <table className="w-full text-sm border-collapse mt-3">
                <thead>
                  <tr className="bg-slate-100">
                    <th className="border border-slate-200 px-3 py-2 text-left">Parent State(s)</th>
                    {node.states.length === 2 ? (
                      <th colSpan={2} className="border border-slate-200 px-3 py-2 text-left font-medium text-slate-500">
                        P({node.states[0]}) ↔ P({node.states[1]})
                      </th>
                    ) : (
                      node.states.map((s) => (
                        <th key={s} className="border border-slate-200 px-3 py-2 text-left">
                          P({s})
                        </th>
                      ))
                    )}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(cpts[node.id]?.table ?? {}).map(([parentKey, row]) => (
                    <tr key={parentKey}>
                      <td className="border border-slate-200 px-3 py-2 font-mono text-xs text-slate-500 bg-slate-50">
                        {parentKey.replace(/,/g, ' + ')}
                      </td>
                      {node.states.length === 2 ? (
                        <td colSpan={2} className="border border-slate-200 px-3 py-2">
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-mono text-blue-700 w-10 shrink-0 text-right">
                              {((row[node.states[0]] ?? 0) * 100).toFixed(0)}%
                            </span>
                            <input
                              type="range"
                              min={0.01}
                              max={0.99}
                              step={0.01}
                              value={row[node.states[0]] ?? 0}
                              onChange={(e) =>
                                onCPTChange(node.id, parentKey, node.states[0], parseFloat(e.target.value))
                              }
                              className="flex-1 accent-blue-600 cursor-pointer"
                            />
                            <span className="text-xs font-mono text-blue-700 w-10 shrink-0">
                              {((row[node.states[1]] ?? 0) * 100).toFixed(0)}%
                            </span>
                          </div>
                        </td>
                      ) : (
                        node.states.map((s) => (
                          <td key={s} className="border border-slate-200 px-2 py-1">
                            <input
                              type="number"
                              step={0.05}
                              min={0}
                              max={1}
                              value={row[s] ?? 0}
                              onChange={(e) =>
                                onCPTChange(node.id, parentKey, s, parseFloat(e.target.value))
                              }
                              className="w-24 border border-slate-200 rounded px-2 py-1 font-mono text-blue-700 bg-blue-50 focus:outline-none focus:ring-2 focus:ring-blue-400 text-sm"
                            />
                          </td>
                        ))
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Modal>
      )}

      {/* Marginal Hint Modal */}
      {showHint && (
        <Modal
          title={<><HelpCircle size={16} className="text-amber-500" /> Marginal Derivation: {node.label}</>}
          onClose={() => { setShowHint(false); setHintLatex(''); }}
        >
          <p className="text-sm text-slate-600 mb-4">
            How the <strong>base probability</strong> of <strong>{node.label}</strong> is
            computed from its parents using the <em>Law of Total Probability</em>:
          </p>
          {hintLoading ? (
            <div className="flex items-center justify-center py-8 text-slate-400">
              Loading derivation…
            </div>
          ) : (
            <div className="bg-amber-50 border border-amber-100 rounded-xl p-4 overflow-x-auto">
              <MathBlock latex={hintLatex} />
            </div>
          )}
          <p className="text-xs text-slate-400 mt-3 text-center">
            Uses the current prior values of the parent nodes.
          </p>
        </Modal>
      )}
    </>
  );
}
