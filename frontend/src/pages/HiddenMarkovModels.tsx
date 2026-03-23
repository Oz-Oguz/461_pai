import { useState, useEffect } from 'react';
import { Play, RotateCcw, StepForward, Calculator, Loader2, Eye } from 'lucide-react';
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

interface FilteringTimeResult {
  belief_after: number[][];
  entropy_before: number;
  entropy_after: number;
  latex: string;
}

interface FilteringObserveResult {
  belief_after: number[][];
  likelihood_map: number[][];
  entropy_before: number;
  entropy_after: number;
  latex: string;
}

interface TransitionMatrixResult {
  T: number[][];
  states: [number, number][];
  n: number;
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
  observation: Record<string, boolean | number> | null,
  actionNoise: number,
  sensorNoise?: number
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
      sensor_noise: sensorNoise ?? 0.2,
    }),
  });
  return parseApiResponse<RobotResult>(res);
}

async function apiFilteringTime(
  belief: number[][],
  gridSize: [number, number],
  walls: [number, number][],
  transitionNoise: number,
  transitionModel: string,
): Promise<FilteringTimeResult> {
  const res = await fetch('/api/hmm/filtering/time', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      belief,
      grid_size: gridSize,
      walls,
      transition_noise: transitionNoise,
      transition_model: transitionModel,
    }),
  });
  return parseApiResponse<FilteringTimeResult>(res);
}

async function apiFilteringObserve(
  belief: number[][],
  gridSize: [number, number],
  walls: [number, number][],
  observation: [number, number],
  sensorNoise: number,
): Promise<FilteringObserveResult> {
  const res = await fetch('/api/hmm/filtering/observe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ belief, grid_size: gridSize, walls, observation, sensor_noise: sensorNoise }),
  });
  return parseApiResponse<FilteringObserveResult>(res);
}

async function apiTransitionMatrix(
  gridSize: [number, number],
  walls: [number, number][],
  transitionNoise: number,
  transitionModel: string,
): Promise<TransitionMatrixResult> {
  const res = await fetch('/api/hmm/filtering/transition-matrix', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      grid_size: gridSize,
      walls,
      transition_noise: transitionNoise,
      transition_model: transitionModel,
    }),
  });
  return parseApiResponse<TransitionMatrixResult>(res);
}

async function apiSolveFiltering(): Promise<SolveStep[]> {
  const res = await fetch('/api/hmm/solve/filtering');
  return parseApiResponse<SolveStep[]>(res);
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
  const [example, setExample] = useState<'forward' | 'viterbi' | 'robot' | 'filtering'>('forward');

  return (
    <div className="flex flex-1 overflow-hidden">
      {/* Left sidebar: example nav */}
      <aside className="w-52 flex-shrink-0 border-r border-slate-200 bg-white overflow-y-auto">
        <div className="p-4">
          <h1 className="text-base font-bold text-slate-900 leading-tight">Hidden Markov Models</h1>
          <p className="text-xs text-slate-500 mt-1 mb-5">Temporal reasoning with hidden states</p>
          <nav className="space-y-1">
            {(
              [
                ['forward',   '1. Weather-Mood (Forward)'],
                ['viterbi',   '2. Viterbi Decoding'],
                ['robot',     '3. Robot Localization'],
                ['filtering', '4. Bayesian Filtering'],
              ] as [typeof example, string][]
            ).map(([key, label]) => (
              <button
                key={key}
                onClick={() => setExample(key)}
                className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  example === key
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-700 hover:bg-slate-100'
                }`}
              >
                {label}
              </button>
            ))}
          </nav>
        </div>
      </aside>

      {/* Center: example content */}
      <main className="flex-1 overflow-auto bg-slate-50">
        <div className="max-w-6xl mx-auto p-6 space-y-6">
          {example === 'forward' && <WeatherMood />}
          {example === 'viterbi' && <ViterbiExample />}
          {example === 'robot' && <RobotLocalization />}
          {example === 'filtering' && <BayesianFilteringTwoPhase />}
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
  const gridSize: [number, number] = [6, 10];
  const [walls, setWalls] = useState<[number, number][]>([]);
  const [belief, setBelief] = useState<number[][]>([]);
  const [robotPos, setRobotPos] = useState<[number, number] | null>(null);
  const [showRobot, setShowRobot] = useState(false);
  const [actionNoise, setActionNoise] = useState(0.1);
  const [sensorNoise, setSensorNoise] = useState(0.2);
  const [lastObs, setLastObs] = useState<{
    trueN: boolean; trueS: boolean; trueE: boolean; trueW: boolean;
    obsN:  boolean; obsS:  boolean; obsE:  boolean; obsW:  boolean;
  } | null>(null);
  const [loading, setLoading] = useState(false);

  const isWall = (r: number, c: number, wallList: [number, number][]) =>
    wallList.some(([wr, wc]) => wr === r && wc === c);

  const freeCells = (wallList: [number, number][]) => {
    const cells: [number, number][] = [];
    for (let r = 0; r < gridSize[0]; r++)
      for (let c = 0; c < gridSize[1]; c++)
        if (!isWall(r, c, wallList)) cells.push([r, c]);
    return cells;
  };

  const randomFreeCell = (wallList: [number, number][]): [number, number] => {
    const free = freeCells(wallList);
    return free[Math.floor(Math.random() * free.length)];
  };

  // True wall presence in each cardinal direction from (r, c)
  const trueWalls = (r: number, c: number) => ({
    N: r === 0 || isWall(r - 1, c, walls),
    S: r === gridSize[0] - 1 || isWall(r + 1, c, walls),
    E: c === gridSize[1] - 1 || isWall(r, c + 1, walls),
    W: c === 0 || isWall(r, c - 1, walls),
  });

  useEffect(() => {
    const newWalls: [number, number][] = [];
    for (let r = 0; r < gridSize[0]; r++) {
      newWalls.push([r, 0], [r, gridSize[1] - 1]);
    }
    for (let c = 0; c < gridSize[1]; c++) {
      newWalls.push([0, c], [gridSize[0] - 1, c]);
    }
    // Large inner block: rows 2–3, cols 2–5
    for (let c = 2; c <= 5; c++) {
      newWalls.push([2, c], [3, c]);
    }
    // Right inner wall: rows 2–3, col 7 (creates narrow channel at col 6 and pocket at col 8)
    newWalls.push([2, 7], [3, 7]);
    setWalls(newWalls);
    setRobotPos(randomFreeCell(newWalls));
    apiRobotInit(gridSize, newWalls, true).then((res) => setBelief(res.belief));
  }, []);

  useEffect(() => {
    const moveMap: Record<string, string> = {
      ArrowUp: 'move_up', ArrowDown: 'move_down', ArrowLeft: 'move_left', ArrowRight: 'move_right',
      w: 'move_up', W: 'move_up',
      a: 'move_left', A: 'move_left',
      d: 'move_right', D: 'move_right',
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 's' || e.key === 'S') { e.preventDefault(); handleSense(); return; }
      const action = moveMap[e.key];
      if (action) { e.preventDefault(); handleMove(action); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [belief, robotPos, walls, loading, actionNoise, sensorNoise]);

  const handleMove = async (action: string) => {
    if (!belief.length) return;
    setLoading(true);
    try {
      // Move true robot position (noisy — stays with probability actionNoise)
      if (robotPos) {
        const deltas: Record<string, [number, number]> = {
          move_up: [-1, 0], move_down: [1, 0], move_left: [0, -1], move_right: [0, 1],
        };
        const [dr, dc] = deltas[action];
        const [nr, nc] = [robotPos[0] + dr, robotPos[1] + dc];
        const moved = Math.random() > actionNoise;
        if (moved && nr >= 0 && nr < gridSize[0] && nc >= 0 && nc < gridSize[1] && !isWall(nr, nc, walls)) {
          setRobotPos([nr, nc]);
        }
      }
      const res = await apiRobotStep(belief, gridSize, walls, action, null, actionNoise, sensorNoise);
      setBelief(res.belief);
      setLastObs(null);
    } catch (err) {
      console.error('Move failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSense = async () => {
    if (!belief.length || !robotPos) return;
    setLoading(true);
    try {
      const [r, c] = robotPos;
      const tw = trueWalls(r, c);

      // UCB model: exact with (1-noise), else flip exactly one random direction
      let obsN = tw.N, obsS = tw.S, obsE = tw.E, obsW = tw.W;
      if (Math.random() < sensorNoise) {
        const flip = Math.floor(Math.random() * 4);
        if      (flip === 0) obsN = !obsN;
        else if (flip === 1) obsS = !obsS;
        else if (flip === 2) obsE = !obsE;
        else                 obsW = !obsW;
      }

      setLastObs({ trueN: tw.N, trueS: tw.S, trueE: tw.E, trueW: tw.W, obsN, obsS, obsE, obsW });
      const res = await apiRobotStep(belief, gridSize, walls, null,
        { N: obsN, S: obsS, E: obsE, W: obsW }, actionNoise, sensorNoise);
      setBelief(res.belief);
    } catch (err) {
      console.error('Sense failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const reset = async () => {
    const newPos = randomFreeCell(walls);
    setRobotPos(newPos);
    setLastObs(null);
    const res = await apiRobotInit(gridSize, walls, true);
    setBelief(res.belief);
  };

  const entropy = belief.length
    ? -belief.flat().reduce((sum, p) => (p > 1e-10 ? sum + p * Math.log2(p) : sum), 0)
    : 0;

  return (
    <div className="bg-white rounded-xl shadow-sm p-5 border border-slate-200 space-y-3">
      {/* Title row */}
      <div className="flex items-baseline justify-between gap-4 flex-wrap">
        <div>
          <h2 className="text-lg font-bold text-slate-900">Example 3: Robot Localization</h2>
          <p className="text-xs text-slate-500 mt-0.5">
            6×10 grid · sensor counts adjacent walls (±1 noise) · corridors are ambiguous; the channel and pocket resolve it
          </p>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          <label className="flex items-center gap-1.5 text-xs cursor-pointer select-none text-slate-600">
            <input type="checkbox" checked={showRobot} onChange={(e) => setShowRobot(e.target.checked)}
              className="w-3.5 h-3.5 accent-red-600" />
            Show robot
          </label>
          <span className="text-sm font-mono text-slate-700 bg-slate-100 px-2 py-0.5 rounded">
            H&nbsp;{entropy.toFixed(2)} bits
          </span>
        </div>
      </div>

      {/* Controls row */}
      <div className="flex items-center gap-4 flex-wrap">
        <label className="flex items-center gap-1.5 text-xs text-slate-600">
          <span className="whitespace-nowrap">Action noise</span>
          <input type="range" min={0} max={0.5} step={0.05} value={actionNoise}
            onChange={(e) => setActionNoise(parseFloat(e.target.value))} className="w-20 h-1.5" />
          <span className="font-mono w-7 text-right">{actionNoise.toFixed(2)}</span>
        </label>
        <label className="flex items-center gap-1.5 text-xs text-slate-600">
          <span className="whitespace-nowrap">Sensor noise (P flip)</span>
          <input type="range" min={0} max={0.8} step={0.05} value={sensorNoise}
            onChange={(e) => setSensorNoise(parseFloat(e.target.value))} className="w-20 h-1.5" />
          <span className="font-mono w-7 text-right">{sensorNoise.toFixed(2)}</span>
        </label>
        <button onClick={handleSense} disabled={loading || !robotPos}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 text-sm font-semibold">
          {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Eye className="w-3.5 h-3.5" />}
          Sense
        </button>
        <button onClick={reset}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-500 text-white rounded-lg hover:bg-slate-600 transition-colors text-sm">
          <RotateCcw className="w-3.5 h-3.5" /> Reset
        </button>
        <span className="text-xs text-slate-400">↑↓←→ / WAD to move · S to sense</span>
      </div>

      {/* Grid + side panel — always side-by-side so the map never shifts */}
      <div className="flex gap-4 items-start">
        {belief.length > 0 && (
          <RobotGrid belief={belief} walls={walls} gridSize={gridSize} robotPos={showRobot ? robotPos : null} />
        )}

        {/* Persistent side panel — compass rose, no layout shift */}
        <div className="w-36 flex-shrink-0 pt-1 space-y-2">
          <p className="font-semibold text-slate-500 uppercase tracking-wide text-[10px]">Last sensor</p>
          {lastObs ? (() => {
            const flipped = {
              N: lastObs.obsN !== lastObs.trueN,
              S: lastObs.obsS !== lastObs.trueS,
              E: lastObs.obsE !== lastObs.trueE,
              W: lastObs.obsW !== lastObs.trueW,
            };
            const anyFlipped = Object.values(flipped).some(Boolean);
            const cell = (obs: boolean, flip: boolean) => (
              <div className={`w-10 h-10 rounded flex items-center justify-center text-xs font-bold border
                ${flip
                  ? 'bg-amber-100 border-amber-400 text-amber-700'
                  : obs
                    ? 'bg-slate-700 border-slate-800 text-white'
                    : 'bg-white border-slate-300 text-slate-400'}`}>
                {obs ? '■' : '□'}
              </div>
            );
            return (
              <div className="space-y-1">
                <div className="grid grid-cols-3 gap-1 w-fit">
                  <div />
                  {cell(lastObs.obsN, flipped.N)}
                  <div />
                  {cell(lastObs.obsW, flipped.W)}
                  <div className="w-10 h-10 rounded bg-blue-100 border border-blue-300 flex items-center justify-center text-blue-700 font-bold text-xs">R</div>
                  {cell(lastObs.obsE, flipped.E)}
                  <div />
                  {cell(lastObs.obsS, flipped.S)}
                  <div />
                </div>
                <p className="text-[10px] text-slate-500 leading-relaxed">
                  ■ wall &nbsp;□ open
                  {anyFlipped && <span className="text-amber-600"> · <span className="font-semibold">amber = noise flip</span></span>}
                  {!anyFlipped && <span className="text-green-700"> · exact</span>}
                </p>
              </div>
            );
          })() : (
            <p className="text-xs text-slate-400 italic">
              Press <kbd className="bg-slate-100 border border-slate-300 rounded px-1 text-[10px]">S</kbd> to sense.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Robot Grid Heatmap ────────────────────────────────────────────────

function RobotGrid({
  belief,
  walls,
  gridSize,
  robotPos,
}: {
  belief: number[][];
  walls: [number, number][];
  gridSize: [number, number];
  robotPos?: [number, number] | null;
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
          const wall = isWall(r, c);
          const color = wall
            ? '#1e293b'
            : `rgb(${255 - intensity * 200}, ${255 - intensity * 100}, 255)`;
          const isRobot = robotPos && robotPos[0] === r && robotPos[1] === c;

          return (
            <g key={`${r}-${c}`}>
              <rect
                x={c * cellSize} y={r * cellSize}
                width={cellSize} height={cellSize}
                fill={color} stroke="#cbd5e1" strokeWidth={0.5}
              />
              {!wall && b > 0.001 && (
                <text
                  x={c * cellSize + cellSize / 2}
                  y={r * cellSize + cellSize / 2}
                  textAnchor="middle" dominantBaseline="middle"
                  fontSize={9} fontWeight="bold"
                  fill={intensity > 0.5 ? '#fff' : '#1e293b'}
                >
                  {b.toFixed(3)}
                </text>
              )}
              {isRobot && (
                <circle
                  cx={c * cellSize + cellSize / 2}
                  cy={r * cellSize + cellSize / 2}
                  r={cellSize / 2 - 6}
                  fill="#ef4444"
                  stroke="#fff"
                  strokeWidth={2}
                  opacity={0.9}
                />
              )}
            </g>
          );
        })
      )}
    </svg>
  );
}

// ══════════════════════════════════════════════════════════════════════
// Example 4: Bayesian Filtering — Two-Phase
// ══════════════════════════════════════════════════════════════════════

const FILT_ROWS = 5;
const FILT_COLS = 6;
const FILT_WALLS: [number, number][] = []; // No walls — pure diffusion demo

function makeUniformBelief(): number[][] {
  const total = FILT_ROWS * FILT_COLS;
  return Array.from({ length: FILT_ROWS }, () =>
    Array.from({ length: FILT_COLS }, () => 1 / total)
  );
}

function makePeakedBelief(r: number, c: number): number[][] {
  return Array.from({ length: FILT_ROWS }, (_, row) =>
    Array.from({ length: FILT_COLS }, (_, col) => (row === r && col === c ? 1.0 : 0.0))
  );
}

function calcEntropy(belief: number[][]): number {
  return -belief.flat().reduce((s, p) => (p > 1e-10 ? s + p * Math.log2(p) : s), 0);
}

type FilterPhase = 'init' | 'after_time' | 'after_obs';

function BayesianFilteringTwoPhase() {
  const [belief, setBelief] = useState<number[][]>(makeUniformBelief);
  const [prevBelief, setPrevBelief] = useState<number[][] | null>(null);
  const [phase, setPhase] = useState<FilterPhase>('init');
  const [obsPos, setObsPos] = useState<[number, number] | null>(null);
  const [likelihoodMap, setLikelihoodMap] = useState<number[][] | null>(null);
  const [transitionNoise, setTransitionNoise] = useState(0.2);
  const [sensorNoise, setSensorNoise] = useState(0.3);
  const [transitionModel, setTransitionModel] = useState<'uniform' | 'clockwise' | 'vortex'>('uniform');
  const [entropyBefore, setEntropyBefore] = useState<number | null>(null);
  const [entropyAfter, setEntropyAfter] = useState<number | null>(null);
  const [_lastLatex, setLastLatex] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [solverSteps, setSolverSteps] = useState<SolveStep[] | null>(null);
  const [stepCount, setStepCount] = useState(0);
  const [showMatrix, setShowMatrix] = useState(false);
  const [matrixData, setMatrixData] = useState<TransitionMatrixResult | null>(null);
  const [matrixLoading, setMatrixLoading] = useState(false);

  const handleInitUniform = () => {
    const b = makeUniformBelief();
    setBelief(b);
    setPrevBelief(null);
    setPhase('init');
    setObsPos(null);
    setLikelihoodMap(null);
    setEntropyBefore(null);
    setEntropyAfter(null);
    setLastLatex(null);
    setStepCount(0);
  };

  const handleInitPeaked = () => {
    const b = makePeakedBelief(Math.floor(FILT_ROWS / 2), Math.floor(FILT_COLS / 2));
    setBelief(b);
    setPrevBelief(null);
    setPhase('init');
    setObsPos(null);
    setLikelihoodMap(null);
    setEntropyBefore(null);
    setEntropyAfter(null);
    setLastLatex(null);
    setStepCount(0);
  };

  const handleTimeStep = async () => {
    setLoading(true);
    try {
      const res = await apiFilteringTime(belief, [FILT_ROWS, FILT_COLS], FILT_WALLS, transitionNoise, transitionModel);
      setPrevBelief(belief);
      setBelief(res.belief_after);
      setPhase('after_time');
      setObsPos(null);
      setLikelihoodMap(null);
      setEntropyBefore(res.entropy_before);
      setEntropyAfter(res.entropy_after);
      setLastLatex(res.latex);
      setStepCount((n) => n + 1);
    } catch (err) {
      console.error('Filtering time step failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleObserve = async () => {
    if (!obsPos) return;
    setLoading(true);
    try {
      const res = await apiFilteringObserve(
        belief,
        [FILT_ROWS, FILT_COLS],
        FILT_WALLS,
        obsPos,
        sensorNoise,
      );
      setPrevBelief(belief);
      setBelief(res.belief_after);
      setLikelihoodMap(res.likelihood_map);
      setPhase('after_obs');
      setEntropyBefore(res.entropy_before);
      setEntropyAfter(res.entropy_after);
      setLastLatex(res.latex);
    } catch (err) {
      console.error('Filtering observe failed:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!showMatrix) return;
    setMatrixLoading(true);
    apiTransitionMatrix([FILT_ROWS, FILT_COLS], FILT_WALLS, transitionNoise, transitionModel)
      .then(setMatrixData)
      .catch((err) => console.error('Matrix fetch failed:', err))
      .finally(() => setMatrixLoading(false));
  }, [showMatrix, transitionNoise, transitionModel]);

  const entropyNow = calcEntropy(belief);

  // Compact phase strip shown above the grids
  const PhaseStrip = () => {
    if (phase === 'init') return (
      <p className="text-xs text-slate-500 mb-2">
        B(X<sub>t</sub>) — press <strong>Passage of Time</strong> to begin, or click a cell then <strong>Apply Observation</strong>.
      </p>
    );
    const isTime = phase === 'after_time';
    return (
      <div className={`flex items-center gap-3 text-xs rounded-lg px-3 py-1.5 mb-2 border ${isTime ? 'bg-amber-50 border-amber-200 text-amber-800' : 'bg-green-50 border-green-200 text-green-800'}`}>
        <span className="font-semibold">{isTime ? '⏱ Time applied' : '👁 Observation applied'}</span>
        {entropyBefore !== null && entropyAfter !== null && (
          <span className="font-mono text-slate-600">
            H: {entropyBefore.toFixed(2)} →{' '}
            <span className={isTime ? 'text-red-600 font-bold' : 'text-green-700 font-bold'}>
              {entropyAfter.toFixed(2)} {isTime ? '↑' : '↓'}
            </span>
          </span>
        )}
        {isTime && !obsPos && <span className="text-slate-500 ml-auto">← click right grid to choose observation point</span>}
        {isTime && obsPos && <span className="text-slate-500 ml-auto">observing ({obsPos[0]},{obsPos[1]})</span>}
      </div>
    );
  };

  return (
    <div className="flex gap-4 items-start">

      {/* ── Left: grid area ─────────────────────────────────────────── */}
      <div className="flex-1 min-w-0 bg-white rounded-xl shadow-sm p-4 border border-slate-200">
        <PhaseStrip />

        {phase === 'init' && (
          <FilteringGrid belief={belief} rows={FILT_ROWS} cols={FILT_COLS} clickable={true}
            obsPos={obsPos} onCellClick={(r, c) => setObsPos([r, c])} />
        )}

        {phase === 'after_time' && prevBelief && (
          <div className="flex gap-4">
            <div className="flex-1 min-w-0">
              <p className="text-center text-xs font-semibold text-slate-500 mb-1">B(Xₜ) — before</p>
              <FilteringGrid belief={prevBelief} rows={FILT_ROWS} cols={FILT_COLS} clickable={false} />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-center text-xs font-semibold text-amber-700 mb-1">B'(Xₜ₊₁) — after time</p>
              <FilteringGrid belief={belief} rows={FILT_ROWS} cols={FILT_COLS}
                clickable={true} highlight="amber" obsPos={obsPos} onCellClick={(r, c) => setObsPos([r, c])} />
            </div>
          </div>
        )}

        {phase === 'after_obs' && prevBelief && (
          <div className="flex gap-4">
            <div className="flex-1 min-w-0">
              <p className="text-center text-xs font-semibold text-slate-500 mb-1">B'(Xₜ₊₁) — predicted</p>
              <FilteringGrid belief={prevBelief} rows={FILT_ROWS} cols={FILT_COLS} clickable={false} />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-center text-xs font-semibold text-green-700 mb-1">B(Xₜ₊₁) — updated</p>
              <FilteringGrid belief={belief} rows={FILT_ROWS} cols={FILT_COLS}
                clickable={true} highlight="green" obsPos={obsPos}
                likelihoodMap={likelihoodMap ?? undefined} onCellClick={(r, c) => setObsPos([r, c])} />
            </div>
          </div>
        )}

        {showMatrix && matrixData && (
          <MatrixFormView
            T_data={matrixData}
            belief={belief}
            prevBelief={prevBelief}
            likelihoodMap={likelihoodMap}
            phase={phase}
          />
        )}

      </div>

      {/* ── Right: controls sidebar ──────────────────────────────────── */}
      <div className="w-64 flex-shrink-0 space-y-4">

        {/* Title */}
        <div>
          <h2 className="text-sm font-bold text-slate-900 leading-tight">Bayesian Filtering</h2>
          <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">
            Time → spreads belief (H↑) · Observe → sharpens belief (H↓)
          </p>
        </div>

        {/* Status */}
        <div className="bg-slate-50 rounded-lg px-3 py-2 text-xs space-y-0.5">
          <div className="flex justify-between">
            <span className="text-slate-500">Time step</span>
            <span className="font-mono font-bold text-slate-900">{stepCount}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Entropy</span>
            <span className="font-mono font-bold text-slate-900">{entropyNow.toFixed(2)} bits</span>
          </div>
        </div>

        {/* Transition model */}
        <div>
          <p className="text-xs font-semibold text-slate-600 uppercase tracking-wide mb-1.5">Transition Model</p>
          <div className="flex flex-col gap-1">
            {([
              ['uniform',   'Uniform',   'Spreads equally to all free neighbours.'],
              ['clockwise', 'Clockwise', 'Each ring rotates clockwise.'],
              ['vortex',    'Vortex',    'Inward spiral toward the centre.'],
            ] as const).map(([id, label, tip]) => (
              <button key={id} onClick={() => setTransitionModel(id)} title={tip}
                className={`w-full text-left px-2.5 py-1.5 rounded text-xs font-medium transition-colors ${
                  transitionModel === id ? 'bg-amber-500 text-white' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                }`}>
                {label}
                {transitionModel === id && <span className="block font-normal opacity-80 text-[10px] mt-0.5">{tip}</span>}
              </button>
            ))}
          </div>
        </div>

        {/* Current phase equation */}
        {phase !== 'init' && (
          <div className="bg-slate-50 rounded-lg p-3 border border-slate-200">
            <MathBlock latex={
              phase === 'after_time'
                ? "B'(X') = \\sum_{x} P(X'|x)\\, B(x)"
                : "B(X) \\propto P(e|X)\\, B'(X)"
            } />
          </div>
        )}

        {/* Noise sliders */}
        <div className="space-y-2">
          <p className="text-xs font-semibold text-slate-600 uppercase tracking-wide">Noise</p>
          <label className="flex items-center gap-1.5 text-xs text-slate-600">
            <span className="w-20 shrink-0">Transition ε</span>
            <input type="range" min={0} max={0.5} step={0.05} value={transitionNoise}
              onChange={(e) => setTransitionNoise(parseFloat(e.target.value))} className="flex-1 h-1.5" />
            <span className="font-mono w-6 text-right">{transitionNoise.toFixed(2)}</span>
          </label>
          <label className="flex items-center gap-1.5 text-xs text-slate-600">
            <span className="w-20 shrink-0">Sensor</span>
            <input type="range" min={0} max={0.9} step={0.05} value={sensorNoise}
              onChange={(e) => setSensorNoise(parseFloat(e.target.value))} className="flex-1 h-1.5" />
            <span className="font-mono w-6 text-right">{sensorNoise.toFixed(2)}</span>
          </label>
        </div>

        {/* Action buttons */}
        <div className="space-y-1.5">
          <p className="text-xs font-semibold text-slate-600 uppercase tracking-wide">Actions</p>
          <div className="flex gap-1.5">
            <button onClick={handleInitUniform}
              className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 bg-slate-500 text-white rounded text-xs hover:bg-slate-600 transition-colors">
              <RotateCcw className="w-3 h-3" /> Uniform
            </button>
            <button onClick={handleInitPeaked}
              className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 bg-slate-500 text-white rounded text-xs hover:bg-slate-600 transition-colors">
              <RotateCcw className="w-3 h-3" /> Peaked
            </button>
          </div>
          <button onClick={handleTimeStep} disabled={loading}
            className="w-full flex items-center justify-center gap-1.5 px-3 py-2 bg-amber-500 text-white rounded-lg text-sm font-semibold hover:bg-amber-600 transition-colors disabled:opacity-50">
            {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <StepForward className="w-3.5 h-3.5" />}
            Passage of Time
          </button>
          <button onClick={handleObserve} disabled={loading || !obsPos}
            className="w-full flex items-center justify-center gap-1.5 px-3 py-2 bg-green-600 text-white rounded-lg text-sm font-semibold hover:bg-green-700 transition-colors disabled:opacity-50">
            {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Eye className="w-3.5 h-3.5" />}
            {obsPos ? `Observe (${obsPos[0]},${obsPos[1]})` : 'Apply Observation'}
          </button>
          <button onClick={async () => { const steps = await apiSolveFiltering(); setSolverSteps(steps); }}
            className="w-full flex items-center justify-center gap-1.5 px-3 py-1.5 bg-purple-600 text-white rounded-lg text-xs hover:bg-purple-700 transition-colors">
            <Calculator className="w-3 h-3" /> Show Derivation
          </button>
          <button
            onClick={() => setShowMatrix(v => !v)}
            className={`w-full flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-colors ${
              showMatrix
                ? 'bg-indigo-600 text-white hover:bg-indigo-700'
                : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
            }`}
          >
            {matrixLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : null}
            {showMatrix ? 'Hide Matrix Form' : 'Show Matrix Form'}
          </button>
        </div>
      </div>

      {/* Solver Modal */}
      {solverSteps && (
        <SolverModal steps={solverSteps} onClose={() => setSolverSteps(null)} />
      )}
    </div>
  );
}

// ── Matrix Form View ─────────────────────────────────────────────────

function MatrixFormView({
  T_data,
  belief,
  prevBelief,
  likelihoodMap,
  phase,
}: {
  T_data: TransitionMatrixResult;
  belief: number[][];
  prevBelief: number[][] | null;
  likelihoodMap: number[][] | null;
  phase: FilterPhase;
}) {
  const { T, states, n } = T_data;
  const CELL = 6;    // pixels per matrix cell
  const VEC_W = 18;  // width of vector columns

  // Map grid belief → n-vector using states index
  const toVec = (grid: number[][]): number[] =>
    states.map(([r, c]) => grid[r][c]);

  const bPrev = prevBelief ? toVec(prevBelief) : toVec(belief);
  const bCurr = toVec(belief);
  const oDiag = likelihoodMap ? toVec(likelihoodMap) : null;

  // T^T: TT[j][i] = T[i][j]  (row=destination, col=source)
  const TT: number[][] = Array.from({ length: n }, (_, j) =>
    Array.from({ length: n }, (_, i) => T[i][j])
  );

  const maxT = Math.max(...T.flat(), 1e-9);
  const maxB = Math.max(...bPrev, ...bCurr, 1e-9);
  const maxO = oDiag ? Math.max(...oDiag, 1e-9) : 1;

  const matW = n * CELL;
  const matH = n * CELL;

  // Colour helpers
  const blueAlpha = (v: number, mx: number) => `rgba(37,99,235,${(v / mx).toFixed(3)})`;
  const greenAlpha = (v: number, mx: number) => `rgba(22,163,74,${(v / mx).toFixed(3)})`;

  // Render a n×1 vertical vector as SVG
  const VecSVG = ({
    vec, mx, color, label,
  }: { vec: number[]; mx: number; color: 'blue' | 'green'; label: string }) => (
    <div className="flex flex-col items-center gap-1">
      <span className="text-[9px] text-slate-500 font-mono">{label}</span>
      <svg width={VEC_W} height={matH}>
        {vec.map((v, i) => (
          <rect key={i} x={0} y={i * CELL} width={VEC_W} height={CELL}
            fill={color === 'blue' ? blueAlpha(v, mx) : greenAlpha(v, mx)}
            stroke="rgba(0,0,0,0.04)" strokeWidth={0.3}
          />
        ))}
      </svg>
    </div>
  );

  const Op = ({ sym }: { sym: string }) => (
    <span className="text-slate-400 font-mono text-sm self-center mt-4">{sym}</span>
  );

  return (
    <div className="mt-4 pt-4 border-t border-slate-100 space-y-3">
      <p className="text-xs font-semibold text-slate-600 uppercase tracking-wide">
        Matrix Form
      </p>
      <p className="text-[10px] font-mono text-slate-500">
        b_t = (1/Z_t) · O_t · T_t<sup>⊤</sup> · b_&#x7B;t-1&#x7D;
      </p>

      {/* ── Time step: T^T × b_{t-1} = b' ── */}
      <div>
        <p className="text-[10px] text-amber-700 font-medium mb-1.5">
          Passage of Time: b&#x2019; = T_t<sup>⊤</sup> · b_&#x7B;t-1&#x7D;
        </p>
        <div className="flex items-start gap-2 overflow-x-auto">
          {/* T^T heatmap */}
          <div className="flex flex-col items-center gap-1 shrink-0">
            <span className="text-[9px] text-slate-500 font-mono">T_t<sup>⊤</sup> ({n}×{n})</span>
            <svg width={matW} height={matH}>
              {TT.map((row, j) =>
                row.map((val, i) => (
                  <rect key={`${j}-${i}`}
                    x={i * CELL} y={j * CELL}
                    width={CELL} height={CELL}
                    fill={blueAlpha(val, maxT)}
                    stroke="rgba(0,0,0,0.04)" strokeWidth={0.2}
                  />
                ))
              )}
              {/* Axis tick labels every 5 states */}
              {Array.from({ length: Math.floor(n / 5) }, (_, k) => (
                <text key={k} x={(k * 5) * CELL + 1} y={matH + 8}
                  fontSize={5} fill="#94a3b8">{k * 5}</text>
              ))}
            </svg>
            <p className="text-[8px] text-slate-400">col = source state i</p>
          </div>

          <Op sym="×" />

          <VecSVG vec={bPrev} mx={maxB} color="blue" label="b_{t-1}" />

          <Op sym="=" />

          <VecSVG vec={phase === 'after_time' ? bCurr : bPrev} mx={maxB} color="blue" label="b'" />
        </div>
      </div>

      {/* ── Observation: O ⊙ b' / Z = b_t ── */}
      {oDiag && phase === 'after_obs' && (
        <div>
          <p className="text-[10px] text-green-700 font-medium mb-1.5">
            Observation: b_t ∝ diag(O_t) ⊙ b&#x2019;
          </p>
          <div className="flex items-start gap-2 overflow-x-auto">
            <VecSVG vec={oDiag} mx={maxO} color="green" label="diag(O_t)" />
            <Op sym="⊙" />
            <VecSVG vec={bPrev} mx={maxB} color="blue" label="b'" />
            <Op sym="/ Z =" />
            <VecSVG vec={bCurr} mx={maxB} color="blue" label="b_t" />
          </div>
        </div>
      )}

      {/* State index legend */}
      <p className="text-[9px] text-slate-400 leading-relaxed">
        States indexed row-major: state 0 = cell (0,0), state {n-1} = cell ({states[n-1][0]},{states[n-1][1]}).
        Each row of T^⊤ shows where probability flows <em>into</em> that state.
      </p>
    </div>
  );
}

// ── Filtering Grid ────────────────────────────────────────────────────

function FilteringGrid({
  belief,
  rows,
  cols,
  clickable = false,
  highlight,
  obsPos,
  likelihoodMap,
  onCellClick,
}: {
  belief: number[][];
  rows: number;
  cols: number;
  label?: string;
  clickable?: boolean;
  highlight?: 'amber' | 'green';
  obsPos?: [number, number] | null;
  likelihoodMap?: number[][];
  onCellClick?: (r: number, c: number) => void;
}) {
  const cellSize = 48;
  const W = cols * cellSize;
  const H = rows * cellSize;
  const maxB = Math.max(...belief.flat(), 1e-9);

  // Colour helpers
  const cellFill = (r: number, c: number) => {
    const intensity = belief[r][c] / maxB;
    if (highlight === 'amber') {
      // Amber scale: white → amber-600
      const v = Math.round(255 - intensity * 180);
      return `rgb(${255},${v},${Math.round(255 - intensity * 255)})`;
    }
    if (highlight === 'green') {
      const v = Math.round(255 - intensity * 180);
      return `rgb(${Math.round(255 - intensity * 255)},${255},${v})`;
    }
    // Default: blue scale matching existing RobotGrid
    return `rgb(${255 - intensity * 200},${255 - intensity * 100},255)`;
  };

  return (
    <svg
      width={W}
      height={H}
      className={`mx-auto border border-slate-300 rounded ${clickable ? 'cursor-pointer' : ''}`}
    >
      {Array.from({ length: rows }).map((_, r) =>
        Array.from({ length: cols }).map((_, c) => {
          const b = belief[r][c];
          const intensity = b / maxB;
          const isObs = obsPos && obsPos[0] === r && obsPos[1] === c;

          return (
            <g
              key={`${r}-${c}`}
              onClick={clickable && onCellClick ? () => onCellClick(r, c) : undefined}
            >
              <rect
                x={c * cellSize}
                y={r * cellSize}
                width={cellSize}
                height={cellSize}
                fill={cellFill(r, c)}
                stroke={isObs ? '#ca8a04' : '#cbd5e1'}
                strokeWidth={isObs ? 3 : 0.5}
              />
              {/* Likelihood overlay (faint dotted circle) */}
              {likelihoodMap && (
                <circle
                  cx={c * cellSize + cellSize / 2}
                  cy={r * cellSize + cellSize / 2}
                  r={(cellSize / 2 - 4) * (likelihoodMap[r][c])}
                  fill="rgba(22,163,74,0.15)"
                  stroke="none"
                />
              )}
              {b > 0.001 && (
                <text
                  x={c * cellSize + cellSize / 2}
                  y={r * cellSize + cellSize / 2}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fontSize={10}
                  fontWeight="bold"
                  fill={intensity > 0.5 ? '#fff' : '#1e293b'}
                >
                  {b < 0.01 ? b.toFixed(3) : b.toFixed(2)}
                </text>
              )}
              {/* Gold star marker for observed cell */}
              {isObs && (
                <text
                  x={c * cellSize + cellSize - 10}
                  y={r * cellSize + 14}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fontSize={14}
                >
                  ★
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
