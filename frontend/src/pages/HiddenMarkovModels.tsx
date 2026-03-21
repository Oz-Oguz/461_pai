import { useState, useEffect } from 'react';
import { Play, RotateCcw, StepForward, Calculator, Loader2, MoveRight, MoveLeft, MoveUp, MoveDown, Eye } from 'lucide-react';
import { InfoPanel } from '../components/InfoPanel';
import { SolverModal } from '../components/SolverModal';
import { MathBlock } from '../components/MathBlock';
import { hmmContent } from '../content/hmm';
import type { SolveStep, LearnTab } from '../types';

// ── API Interfaces ────────────────────────────────────────────────────

interface ForwardResult {
  states: string[];
  observations: string[];
  beliefs: number[][];
  state_labels: string[];
  log_likelihood: number;
  steps: Array<{
    timestep: number;
    observation: string;
    predicted_belief: Record<string, number>;
    updated_belief: Record<string, number>;
    evidence: number;
    latex: string;
  }>;
}

interface ViterbiResult {
  states: string[];
  observations: string[];
  most_likely_path: string[];
  path_probability: number;
  total_likelihood: number;
  viterbi_table: number[][];
  backpointers: number[][];
}

interface RobotResult {
  belief: number[][];
  step_type: string;
  details: Record<string, any>;
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

// ── API Calls ─────────────────────────────────────────────────────────

async function apiForward(
  observations: string[],
  states: string[],
  transition: Record<string, Record<string, number>>,
  emission: Record<string, Record<string, number>>,
  prior: Record<string, number>
): Promise<ForwardResult> {
  const res = await fetch('/api/hmm/forward', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ observations, states, transition, emission, prior }),
  });
  return parseApiResponse<ForwardResult>(res);
}

async function apiViterbi(
  observations: string[],
  states: string[],
  transition: Record<string, Record<string, number>>,
  emission: Record<string, Record<string, number>>,
  prior: Record<string, number>
): Promise<ViterbiResult> {
  const res = await fetch('/api/hmm/viterbi', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ observations, states, transition, emission, prior }),
  });
  return parseApiResponse<ViterbiResult>(res);
}

async function apiRobotInit(
  gridSize: [number, number],
  walls: [number, number][],
  uniform: boolean
): Promise<{ belief: number[][] }> {
  const res = await fetch('/api/hmm/robot/init', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ grid_size: gridSize, walls, uniform }),
  });
  return parseApiResponse<{ belief: number[][] }>(res);
}

async function apiRobotStep(
  belief: number[][],
  gridSize: [number, number],
  walls: [number, number][],
  action: string | null,
  observation: Record<string, boolean> | null,
  actionNoise: number
): Promise<RobotResult> {
  const res = await fetch('/api/hmm/robot/step', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      belief,
      grid_size: gridSize,
      walls,
      action,
      observation,
      action_noise: actionNoise,
    }),
  });
  return parseApiResponse<RobotResult>(res);
}

async function apiSolveForward(): Promise<SolveStep[]> {
  const res = await fetch('/api/hmm/solve/forward');
  return parseApiResponse<SolveStep[]>(res);
}

async function apiSolveViterbi(): Promise<SolveStep[]> {
  const res = await fetch('/api/hmm/solve/viterbi');
  return parseApiResponse<SolveStep[]>(res);
}

// ── Component Props ───────────────────────────────────────────────────

interface HMMProps {
  learnOpen: boolean;
  learnTab: LearnTab;
  panelWidth: number;
  onLearnToggle: () => void;
  onLearnClose: () => void;
  onLearnWidthChange: (w: number) => void;
  onLearnTabChange: (t: LearnTab) => void;
}

// ── Main Component ────────────────────────────────────────────────────

export function HiddenMarkovModels(props: HMMProps) {
  const [example, setExample] = useState<'forward' | 'viterbi' | 'robot'>('forward');

  return (
    <div className="flex flex-1 overflow-hidden">
      {/* Main content */}
      <main className="flex-1 overflow-auto bg-slate-50">
        <div className="max-w-7xl mx-auto p-6 space-y-6">
          {/* Header */}
          <div className="bg-white rounded-xl shadow-sm p-6 border border-slate-200">
            <div>
              <h1 className="text-3xl font-bold text-slate-900">Hidden Markov Models</h1>
              <p className="text-slate-600 mt-2">
                Temporal reasoning with hidden states and noisy observations
              </p>
            </div>

            {/* Example selector */}
            <div className="mt-6 flex gap-3">
              <button
                onClick={() => setExample('forward')}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  example === 'forward'
                    ? 'bg-blue-600 text-white'
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                }`}
              >
                1. Weather-Mood (Forward)
              </button>
              <button
                onClick={() => setExample('viterbi')}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  example === 'viterbi'
                    ? 'bg-blue-600 text-white'
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                }`}
              >
                2. Viterbi Decoding
              </button>
              <button
                onClick={() => setExample('robot')}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  example === 'robot'
                    ? 'bg-blue-600 text-white'
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                }`}
              >
                3. Robot Localization
              </button>
            </div>
          </div>

          {/* Example content */}
          {example === 'forward' && <WeatherMood />}
          {example === 'viterbi' && <ViterbiExample />}
          {example === 'robot' && <RobotLocalization />}
        </div>
      </main>

      {/* Info Panel */}
      <InfoPanel
        content={hmmContent}
        open={props.learnOpen}
        width={props.panelWidth}
        activeTab={props.learnTab}
        onToggle={props.onLearnToggle}
        onClose={props.onLearnClose}
        onWidthChange={props.onLearnWidthChange}
        onActiveTabChange={props.onLearnTabChange}
      />
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════
// Example 1: Weather-Mood (Forward Algorithm)
// ══════════════════════════════════════════════════════════════════════

function WeatherMood() {
  // Default parameters
  const [transSunToSun, setTransSunToSun] = useState(0.8);
  const [transRainToRain, setTransRainToRain] = useState(0.6);
  const [emitSunHappy, setEmitSunHappy] = useState(0.9);
  const [emitRainHappy, setEmitRainHappy] = useState(0.3);
  const [observations, setObservations] = useState<string[]>(['Happy', 'Sad', 'Happy']);
  const [result, setResult] = useState<ForwardResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [solverSteps, setSolverSteps] = useState<SolveStep[] | null>(null);

  const states = ['Sun', 'Rain'];

  const runForward = async () => {
    setLoading(true);
    try {
      const transition = {
        Sun: { Sun: transSunToSun, Rain: 1 - transSunToSun },
        Rain: { Sun: 1 - transRainToRain, Rain: transRainToRain },
      };
      const emission = {
        Sun: { Happy: emitSunHappy, Sad: 1 - emitSunHappy },
        Rain: { Happy: emitRainHappy, Sad: 1 - emitRainHappy },
      };
      const prior = { Sun: 0.5, Rain: 0.5 };

      const res = await apiForward(observations, states, transition, emission, prior);
      setResult(res);
      setCurrentStep(0);
    } catch (err) {
      console.error('Forward algorithm failed:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runForward();
  }, []);

  const addObservation = (obs: string) => {
    setObservations([...observations, obs]);
  };

  const removeLastObservation = () => {
    if (observations.length > 0) {
      setObservations(observations.slice(0, -1));
    }
  };

  const reset = () => {
    setObservations(['Happy', 'Sad', 'Happy']);
    runForward();
  };

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="bg-white rounded-xl shadow-sm p-6 border border-slate-200">
        <h2 className="text-xl font-bold text-slate-900 mb-4">Example 1: Weather-Mood (Forward Algorithm)</h2>
        <p className="text-slate-600 mb-6">
          Track hidden weather state (Sun/Rain) from mood observations (Happy/Sad).
          Adjust transition and emission probabilities to see how belief propagates.
        </p>

        <div className="grid grid-cols-2 gap-6 mb-6">
          <div>
            <h3 className="font-semibold text-slate-700 mb-3">Transition Probabilities</h3>
            <div className="space-y-3">
              <div>
                <label className="flex justify-between text-sm text-slate-600 mb-1">
                  <span>P(Sun | Sun)</span>
                  <span className="font-mono">{transSunToSun.toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={transSunToSun}
                  onChange={(e) => setTransSunToSun(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
              <div>
                <label className="flex justify-between text-sm text-slate-600 mb-1">
                  <span>P(Rain | Rain)</span>
                  <span className="font-mono">{transRainToRain.toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={transRainToRain}
                  onChange={(e) => setTransRainToRain(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>
          </div>

          <div>
            <h3 className="font-semibold text-slate-700 mb-3">Emission Probabilities</h3>
            <div className="space-y-3">
              <div>
                <label className="flex justify-between text-sm text-slate-600 mb-1">
                  <span>P(Happy | Sun)</span>
                  <span className="font-mono">{emitSunHappy.toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={emitSunHappy}
                  onChange={(e) => setEmitSunHappy(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
              <div>
                <label className="flex justify-between text-sm text-slate-600 mb-1">
                  <span>P(Happy | Rain)</span>
                  <span className="font-mono">{emitRainHappy.toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={emitRainHappy}
                  onChange={(e) => setEmitRainHappy(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Observation sequence */}
        <div className="mb-6">
          <h3 className="font-semibold text-slate-700 mb-3">Observation Sequence</h3>
          <div className="flex gap-2 items-center flex-wrap">
            <span className="text-slate-600">e =</span>
            {observations.map((obs, i) => (
              <span key={i} className="px-3 py-1 bg-blue-100 text-blue-800 rounded font-mono">
                {obs}
              </span>
            ))}
            {observations.length === 0 && (
              <span className="text-slate-400 italic">No observations yet</span>
            )}
          </div>
          <div className="flex gap-2 mt-3">
            <button
              onClick={() => addObservation('Happy')}
              className="px-3 py-1 bg-green-100 text-green-800 rounded hover:bg-green-200 transition-colors"
            >
              + Happy
            </button>
            <button
              onClick={() => addObservation('Sad')}
              className="px-3 py-1 bg-orange-100 text-orange-800 rounded hover:bg-orange-200 transition-colors"
            >
              + Sad
            </button>
            <button
              onClick={removeLastObservation}
              disabled={observations.length === 0}
              className="px-3 py-1 bg-slate-100 text-slate-700 rounded hover:bg-slate-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Remove Last
            </button>
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex gap-3">
          <button
            onClick={runForward}
            disabled={loading || observations.length === 0}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            Run Forward Algorithm
          </button>
          <button
            onClick={reset}
            className="flex items-center gap-2 px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>
          <button
            onClick={async () => {
              const steps = await apiSolveForward();
              setSolverSteps(steps);
            }}
            className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
          >
            <Calculator className="w-4 h-4" />
            Show Derivation
          </button>
        </div>
      </div>

      {/* Results */}
      {result && (
        <>
          {/* Trellis diagram */}
          <div className="bg-white rounded-xl shadow-sm p-6 border border-slate-200">
            <h3 className="text-lg font-bold text-slate-900 mb-4">Trellis Diagram</h3>
            <TrellisDiagram result={result} currentStep={currentStep} />
            
            {/* Step controls */}
            {result.steps.length > 0 && (
              <div className="mt-4 flex gap-3 items-center">
                <button
                  onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
                  disabled={currentStep === 0}
                  className="p-2 bg-slate-100 text-slate-700 rounded hover:bg-slate-200 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <StepForward className="w-4 h-4 rotate-180" />
                </button>
                <span className="text-slate-600">
                  Step {currentStep} / {result.steps.length}
                </span>
                <button
                  onClick={() => setCurrentStep(Math.min(result.steps.length, currentStep + 1))}
                  disabled={currentStep === result.steps.length}
                  className="p-2 bg-slate-100 text-slate-700 rounded hover:bg-slate-200 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <StepForward className="w-4 h-4" />
                </button>
              </div>
            )}
          </div>

          {/* Step-by-step math */}
          {currentStep > 0 && currentStep <= result.steps.length && (
            <div className="bg-white rounded-xl shadow-sm p-6 border border-slate-200">
              <h3 className="text-lg font-bold text-slate-900 mb-4">
                Step {currentStep}: {result.steps[currentStep - 1].observation}
              </h3>
              <MathBlock latex={result.steps[currentStep - 1].latex} />
              <div className="mt-4 grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm font-semibold text-slate-700 mb-2">Predicted Belief</p>
                  {Object.entries(result.steps[currentStep - 1].predicted_belief).map(([state, prob]) => (
                    <div key={state} className="flex justify-between text-sm">
                      <span>{state}:</span>
                      <span className="font-mono">{(prob as number).toFixed(4)}</span>
                    </div>
                  ))}
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-700 mb-2">Updated Belief</p>
                  {Object.entries(result.steps[currentStep - 1].updated_belief).map(([state, prob]) => (
                    <div key={state} className="flex justify-between text-sm">
                      <span>{state}:</span>
                      <span className="font-mono">{(prob as number).toFixed(4)}</span>
                    </div>
                  ))}
                </div>
              </div>
              <p className="mt-4 text-sm text-slate-600">
                Evidence P(e<sub>{currentStep}</sub> | e<sub>1:{currentStep - 1}</sub>) ={' '}
                <span className="font-mono">{result.steps[currentStep - 1].evidence.toFixed(4)}</span>
              </p>
            </div>
          )}

          {/* Summary */}
          <div className="bg-white rounded-xl shadow-sm p-6 border border-slate-200">
            <h3 className="text-lg font-bold text-slate-900 mb-4">Summary</h3>
            <p className="text-slate-600 mb-2">
              Log-Likelihood: <span className="font-mono font-semibold">{result.log_likelihood.toFixed(4)}</span>
            </p>
            <p className="text-slate-600">
              Total Likelihood P(e<sub>1:{result.observations.length}</sub>): <span className="font-mono font-semibold">{Math.exp(result.log_likelihood).toFixed(6)}</span>
            </p>
          </div>
        </>
      )}

      {/* Solver Modal */}
      {solverSteps && (
        <SolverModal steps={solverSteps} onClose={() => setSolverSteps(null)} />
      )}
    </div>
  );
}

// ── Trellis Diagram ───────────────────────────────────────────────────

function TrellisDiagram({ result, currentStep }: { result: ForwardResult; currentStep: number }) {
  const W = 800;
  const H = 300;
  const PAD = { top: 40, right: 40, bottom: 40, left: 100 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;

  const nSteps = result.observations.length + 1;
  const nStates = result.states.length;

  const stepX = (t: number) => PAD.left + (t / (nSteps - 1)) * innerW;
  const stateY = (s: number) => PAD.top + (s / Math.max(1, nStates - 1)) * innerH;

  // Node size based on belief
  const nodeRadius = (belief: number) => 5 + belief * 20;

  return (
    <svg width={W} height={H} className="mx-auto">
      {/* Edges (transitions) */}
      {Array.from({ length: nSteps - 1 }).map((_, t) => (
        <g key={`edges-${t}`}>
          {result.states.map((_, i) =>
            result.states.map((_, j) => {
              const opacity = t < currentStep ? 0.3 : 0.1;
              return (
                <line
                  key={`${i}-${j}`}
                  x1={stepX(t)}
                  y1={stateY(i)}
                  x2={stepX(t + 1)}
                  y2={stateY(j)}
                  stroke="#94a3b8"
                  strokeWidth={1}
                  opacity={opacity}
                />
              );
            })
          )}
        </g>
      ))}

      {/* Nodes (beliefs) */}
      {result.beliefs.slice(0, currentStep + 1).map((belief, t) => (
        <g key={`nodes-${t}`}>
          {belief.map((b, i) => (
            <g key={i}>
              <circle
                cx={stepX(t)}
                cy={stateY(i)}
                r={nodeRadius(b)}
                fill={t === currentStep ? '#3b82f6' : '#60a5fa'}
                stroke="#1e40af"
                strokeWidth={t === currentStep ? 2 : 1}
                opacity={0.8}
              />
              <text
                x={stepX(t)}
                y={stateY(i) - nodeRadius(b) - 5}
                textAnchor="middle"
                fontSize={10}
                fill="#1e293b"
                fontWeight={t === currentStep ? 'bold' : 'normal'}
              >
                {b.toFixed(2)}
              </text>
            </g>
          ))}
        </g>
      ))}

      {/* State labels */}
      {result.states.map((state, i) => (
        <text
          key={i}
          x={PAD.left - 15}
          y={stateY(i)}
          textAnchor="end"
          alignmentBaseline="middle"
          fontSize={14}
          fontWeight="bold"
          fill="#1e293b"
        >
          {state}
        </text>
      ))}

      {/* Time labels */}
      {Array.from({ length: nSteps }).map((_, t) => (
        <g key={t}>
          <text
            x={stepX(t)}
            y={PAD.top - 10}
            textAnchor="middle"
            fontSize={12}
            fill="#64748b"
          >
            t={t}
          </text>
          {t > 0 && (
            <text
              x={stepX(t)}
              y={H - PAD.bottom + 25}
              textAnchor="middle"
              fontSize={11}
              fill="#475569"
            >
              {result.observations[t - 1]}
            </text>
          )}
        </g>
      ))}
    </svg>
  );
}

// ══════════════════════════════════════════════════════════════════════
// Example 2: Viterbi (Most Likely Path)
// ══════════════════════════════════════════════════════════════════════

function ViterbiExample() {
  const [observations, setObservations] = useState<string[]>(['Happy', 'Sad', 'Happy', 'Happy']);
  const [forwardResult, setForwardResult] = useState<ForwardResult | null>(null);
  const [viterbiResult, setViterbiResult] = useState<ViterbiResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [solverSteps, setSolverSteps] = useState<SolveStep[] | null>(null);

  const states = ['Sun', 'Rain'];
  const transition = {
    Sun: { Sun: 0.8, Rain: 0.2 },
    Rain: { Sun: 0.4, Rain: 0.6 },
  };
  const emission = {
    Sun: { Happy: 0.9, Sad: 0.1 },
    Rain: { Happy: 0.3, Sad: 0.7 },
  };
  const prior = { Sun: 0.5, Rain: 0.5 };

  const runAlgorithms = async () => {
    setLoading(true);
    try {
      const [fwd, vit] = await Promise.all([
        apiForward(observations, states, transition, emission, prior),
        apiViterbi(observations, states, transition, emission, prior),
      ]);
      setForwardResult(fwd);
      setViterbiResult(vit);
    } catch (err) {
      console.error('Algorithm failed:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runAlgorithms();
  }, []);

  const addObservation = (obs: string) => {
    setObservations([...observations, obs]);
  };

  const removeLastObservation = () => {
    if (observations.length > 0) {
      setObservations(observations.slice(0, -1));
    }
  };

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="bg-white rounded-xl shadow-sm p-6 border border-slate-200">
        <h2 className="text-xl font-bold text-slate-900 mb-4">Example 2: Viterbi Decoding</h2>
        <p className="text-slate-600 mb-6">
          Find the most likely state sequence (MAP estimate) and compare it with the forward algorithm
          that computes the posterior distribution.
        </p>

        {/* Observation sequence */}
        <div className="mb-6">
          <h3 className="font-semibold text-slate-700 mb-3">Observation Sequence</h3>
          <div className="flex gap-2 items-center flex-wrap">
            {observations.map((obs, i) => (
              <span key={i} className="px-3 py-1 bg-blue-100 text-blue-800 rounded font-mono">
                {obs}
              </span>
            ))}
          </div>
          <div className="flex gap-2 mt-3">
            <button
              onClick={() => addObservation('Happy')}
              className="px-3 py-1 bg-green-100 text-green-800 rounded hover:bg-green-200 transition-colors"
            >
              + Happy
            </button>
            <button
              onClick={() => addObservation('Sad')}
              className="px-3 py-1 bg-orange-100 text-orange-800 rounded hover:bg-orange-200 transition-colors"
            >
              + Sad
            </button>
            <button
              onClick={removeLastObservation}
              disabled={observations.length === 0}
              className="px-3 py-1 bg-slate-100 text-slate-700 rounded hover:bg-slate-200 transition-colors disabled:opacity-50"
            >
              Remove Last
            </button>
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex gap-3">
          <button
            onClick={runAlgorithms}
            disabled={loading || observations.length === 0}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            Run Both Algorithms
          </button>
          <button
            onClick={async () => {
              const steps = await apiSolveViterbi();
              setSolverSteps(steps);
            }}
            className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
          >
            <Calculator className="w-4 h-4" />
            Show Viterbi Derivation
          </button>
        </div>
      </div>

      {/* Results comparison */}
      {viterbiResult && forwardResult && (
        <>
          <div className="bg-white rounded-xl shadow-sm p-6 border border-slate-200">
            <h3 className="text-lg font-bold text-slate-900 mb-4">Most Likely Path (Viterbi)</h3>
            <div className="flex gap-2 mb-4">
              {viterbiResult.most_likely_path.map((state, i) => (
                <div key={i} className="text-center">
                  <div className="text-xs text-slate-500 mb-1">t={i}</div>
                  <div className="px-4 py-2 bg-purple-100 text-purple-900 rounded-lg font-bold">
                    {state}
                  </div>
                  {i > 0 && (
                    <div className="text-xs text-slate-600 mt-1">{observations[i - 1]}</div>
                  )}
                </div>
              ))}
            </div>
            <div className="grid grid-cols-2 gap-4 mt-4">
              <div className="p-3 bg-purple-50 rounded-lg">
                <p className="text-sm font-semibold text-slate-700">Path Probability (MAP)</p>
                <p className="text-xl font-mono font-bold text-purple-900">
                  {viterbiResult.path_probability.toFixed(6)}
                </p>
              </div>
              <div className="p-3 bg-blue-50 rounded-lg">
                <p className="text-sm font-semibold text-slate-700">Total Likelihood (Sum)</p>
                <p className="text-xl font-mono font-bold text-blue-900">
                  {viterbiResult.total_likelihood.toFixed(6)}
                </p>
              </div>
            </div>
            <p className="mt-4 text-sm text-slate-600">
              The Viterbi path probability is always ≤ the total likelihood (which sums over all paths).
            </p>
          </div>

          {/* Forward beliefs comparison */}
          <div className="bg-white rounded-xl shadow-sm p-6 border border-slate-200">
            <h3 className="text-lg font-bold text-slate-900 mb-4">Forward vs Viterbi</h3>
            <p className="text-slate-600 mb-4">
              Forward computes the full posterior distribution at each timestep (summing over all paths).
              Viterbi finds only the single best path (maximizing).
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left p-2">Time</th>
                    <th className="text-left p-2">Observation</th>
                    <th className="text-left p-2">Viterbi Path</th>
                    <th className="text-left p-2">Forward P(Sun)</th>
                    <th className="text-left p-2">Forward P(Rain)</th>
                  </tr>
                </thead>
                <tbody>
                  {viterbiResult.most_likely_path.map((state, t) => (
                    <tr key={t} className="border-b hover:bg-slate-50">
                      <td className="p-2 font-mono">{t}</td>
                      <td className="p-2">{t > 0 ? observations[t - 1] : '—'}</td>
                      <td className="p-2">
                        <span className="px-2 py-1 bg-purple-100 text-purple-900 rounded font-semibold">
                          {state}
                        </span>
                      </td>
                      <td className="p-2 font-mono">{forwardResult.beliefs[t][0].toFixed(3)}</td>
                      <td className="p-2 font-mono">{forwardResult.beliefs[t][1].toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* Solver Modal */}
      {solverSteps && (
        <SolverModal steps={solverSteps} onClose={() => setSolverSteps(null)} />
      )}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════
// Example 3: Robot Localization
// ══════════════════════════════════════════════════════════════════════

function RobotLocalization() {
  const gridSize: [number, number] = [8, 8];
  const [walls, setWalls] = useState<[number, number][]>([]);
  const [belief, setBelief] = useState<number[][]>([]);
  const [actionNoise, setActionNoise] = useState(0.1);
  const [loading, setLoading] = useState(false);

  // Initialize world
  useEffect(() => {
    // Create walls around perimeter
    const newWalls: [number, number][] = [];
    for (let r = 0; r < gridSize[0]; r++) {
      newWalls.push([r, 0]);
      newWalls.push([r, gridSize[1] - 1]);
    }
    for (let c = 0; c < gridSize[1]; c++) {
      newWalls.push([0, c]);
      newWalls.push([gridSize[0] - 1, c]);
    }
    // Add some internal walls
    newWalls.push([3, 3], [3, 4], [4, 3], [4, 4]);
    setWalls(newWalls);

    // Initialize uniform belief
    apiRobotInit(gridSize, newWalls, true).then((res) => {
      setBelief(res.belief);
    });
  }, []);

  const handleMove = async (action: string) => {
    if (!belief.length) return;
    setLoading(true);
    try {
      const res = await apiRobotStep(belief, gridSize, walls, action, null, actionNoise);
      setBelief(res.belief);
    } catch (err) {
      console.error('Move failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSense = async () => {
    if (!belief.length) return;
    setLoading(true);
    try {
      // For demo: sense a corner (top-left free cell)
      const observation = { north: true, south: false, east: false, west: true };
      const res = await apiRobotStep(belief, gridSize, walls, null, observation, actionNoise);
      setBelief(res.belief);
    } catch (err) {
      console.error('Sense failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const reset = async () => {
    const res = await apiRobotInit(gridSize, walls, true);
    setBelief(res.belief);
  };

  // Compute entropy
  const entropy = belief.length
    ? -belief.flat().reduce((sum, p) => (p > 1e-10 ? sum + p * Math.log2(p) : sum), 0)
    : 0;

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="bg-white rounded-xl shadow-sm p-6 border border-slate-200">
        <h2 className="text-xl font-bold text-slate-900 mb-4">Example 3: Robot Localization</h2>
        <p className="text-slate-600 mb-6">
          A robot moves in a 2D grid with walls. Movement diffuses belief (increases entropy).
          Sensing sharpens belief (decreases entropy). This is the "kidnapped robot" problem.
        </p>

        <div className="mb-6">
          <label className="flex justify-between text-sm text-slate-600 mb-1">
            <span>Action Noise</span>
            <span className="font-mono">{actionNoise.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min={0}
            max={0.5}
            step={0.05}
            value={actionNoise}
            onChange={(e) => setActionNoise(parseFloat(e.target.value))}
            className="w-full"
          />
          <p className="text-xs text-slate-500 mt-1">
            Probability that the robot stays in place instead of moving as intended.
          </p>
        </div>

        {/* Movement controls */}
        <div className="grid grid-cols-3 gap-2 max-w-xs mb-6">
          <div></div>
          <button
            onClick={() => handleMove('move_up')}
            disabled={loading}
            className="p-3 bg-blue-100 text-blue-800 rounded hover:bg-blue-200 transition-colors disabled:opacity-50 flex items-center justify-center"
          >
            <MoveUp className="w-5 h-5" />
          </button>
          <div></div>

          <button
            onClick={() => handleMove('move_left')}
            disabled={loading}
            className="p-3 bg-blue-100 text-blue-800 rounded hover:bg-blue-200 transition-colors disabled:opacity-50 flex items-center justify-center"
          >
            <MoveLeft className="w-5 h-5" />
          </button>
          <button
            onClick={handleSense}
            disabled={loading}
            className="p-3 bg-green-600 text-white rounded hover:bg-green-700 transition-colors disabled:opacity-50 flex items-center justify-center gap-1 text-sm font-semibold"
          >
            <Eye className="w-4 h-4" />
            Sense
          </button>
          <button
            onClick={() => handleMove('move_right')}
            disabled={loading}
            className="p-3 bg-blue-100 text-blue-800 rounded hover:bg-blue-200 transition-colors disabled:opacity-50 flex items-center justify-center"
          >
            <MoveRight className="w-5 h-5" />
          </button>

          <div></div>
          <button
            onClick={() => handleMove('move_down')}
            disabled={loading}
            className="p-3 bg-blue-100 text-blue-800 rounded hover:bg-blue-200 transition-colors disabled:opacity-50 flex items-center justify-center"
          >
            <MoveDown className="w-5 h-5" />
          </button>
          <div></div>
        </div>

        <div className="flex gap-3">
          <button
            onClick={reset}
            className="flex items-center gap-2 px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-700 transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Reset to Uniform
          </button>
        </div>

        <div className="mt-6 p-4 bg-slate-50 rounded-lg">
          <p className="text-sm font-semibold text-slate-700 mb-1">Entropy (Uncertainty)</p>
          <p className="text-2xl font-mono font-bold text-slate-900">{entropy.toFixed(2)} bits</p>
          <p className="text-xs text-slate-500 mt-1">
            Lower entropy = more certain about location. Sensing decreases entropy; moving increases it.
          </p>
        </div>
      </div>

      {/* Grid visualization */}
      {belief.length > 0 && (
        <div className="bg-white rounded-xl shadow-sm p-6 border border-slate-200">
          <h3 className="text-lg font-bold text-slate-900 mb-4">Belief State Heatmap</h3>
          <RobotGrid belief={belief} walls={walls} gridSize={gridSize} />
        </div>
      )}
    </div>
  );
}

// ── Robot Grid Heatmap ────────────────────────────────────────────────

function RobotGrid({
  belief,
  walls,
  gridSize,
}: {
  belief: number[][];
  walls: [number, number][];
  gridSize: [number, number];
}) {
  const cellSize = 50;
  const [rows, cols] = gridSize;
  const W = cols * cellSize;
  const H = rows * cellSize;

  const maxBelief = Math.max(...belief.flat());

  const isWall = (r: number, c: number) => walls.some(([wr, wc]) => wr === r && wc === c);

  return (
    <svg width={W} height={H} className="mx-auto border border-slate-300">
      {Array.from({ length: rows }).map((_, r) =>
        Array.from({ length: cols }).map((_, c) => {
          const b = belief[r][c];
          const intensity = maxBelief > 0 ? b / maxBelief : 0;
          const color = isWall(r, c)
            ? '#1e293b'
            : `rgb(${255 - intensity * 200}, ${255 - intensity * 100}, 255)`;

          return (
            <g key={`${r}-${c}`}>
              <rect
                x={c * cellSize}
                y={r * cellSize}
                width={cellSize}
                height={cellSize}
                fill={color}
                stroke="#cbd5e1"
                strokeWidth={0.5}
              />
              {!isWall(r, c) && b > 0.001 && (
                <text
                  x={c * cellSize + cellSize / 2}
                  y={r * cellSize + cellSize / 2}
                  textAnchor="middle"
                  alignmentBaseline="middle"
                  fontSize={9}
                  fill={intensity > 0.5 ? '#fff' : '#1e293b'}
                  fontWeight="bold"
                >
                  {b.toFixed(3)}
                </text>
              )}
            </g>
          );
        })
      )}
    </svg>
  );
}

export default HiddenMarkovModels;
