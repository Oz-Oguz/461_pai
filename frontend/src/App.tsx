import { useState, useCallback, type ReactElement } from 'react';
import type { LearnTab } from './types';
import { Network, ChevronDown } from 'lucide-react';
import { BayesianNetworks } from './pages/BayesianNetworks';
import { BayesianLinearRegression } from './pages/BayesianLinearRegression';
import { KalmanFilter } from './pages/KalmanFilter';
import { GaussianProcesses } from './pages/GaussianProcesses';
import { HiddenMarkovModels } from './pages/HiddenMarkovModels';

// ── Navigation ────────────────────────────────────────────────────────
// To add a new module: import the page component, add an entry here.

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type PageComponent = (props?: any) => ReactElement | null;

interface NavItem {
  id: string;
  label: string;
  component: PageComponent;
}

const NAV_GROUPS: { label: string; items: NavItem[] }[] = [
  {
    label: 'Graphical Models',
    items: [
      { id: 'bayesian-networks', label: 'Bayesian Networks', component: BayesianNetworks },
    ],
  },
  {
    label: 'Temporal Reasoning',
    items: [
      { id: 'hmm', label: 'Hidden Markov Models', component: HiddenMarkovModels },
      { id: 'kalman', label: 'Kalman Filters', component: KalmanFilter },
    ],
  },
  {
    label: 'Probabilistic ML',
    items: [
      { id: 'blr', label: 'Bayesian Linear Regression', component: BayesianLinearRegression },
      { id: 'gp', label: 'Gaussian Processes', component: GaussianProcesses },
    ],
  },
  // ── Upcoming (uncomment + import when implemented) ──────────────────
  // {
  //   label: 'Approximate Inference',
  //   items: [
  //     { id: 'vi', label: 'Variational Inference', component: VariationalInference },
  //     { id: 'mcmc', label: 'MCMC', component: MCMC },
  //   ],
  // },
  // {
  //   label: 'Deep Models',
  //   items: [
  //     { id: 'bnn', label: 'Bayesian Neural Networks', component: BayesianNeuralNetworks },
  //   ],
  // },
  // {
  //   label: 'Sequential Decisions',
  //   items: [
  //     { id: 'active', label: 'Active Learning', component: ActiveLearning },
  //     { id: 'bo', label: 'Bayesian Optimization', component: BayesianOptimization },
  //     { id: 'rl', label: 'Reinforcement Learning', component: ReinforcementLearning },
  //   ],
  // },
];

const ALL_ITEMS = NAV_GROUPS.flatMap((g) => g.items);

export default function App() {
  const [activeId, setActiveId] = useState(ALL_ITEMS[0].id);
  const [modelId, setModelId] = useState('fusion');

  // ── InfoPanel state (lifted here so it persists across module switches) ──
  const [learnOpen, setLearnOpen] = useState(false);
  const [learnTab, setLearnTab] = useState<LearnTab>('overview');
  const [panelWidth, setPanelWidth] = useState<number>(() => {
    try { return parseInt(localStorage.getItem('infoPanelWidth') ?? '360', 10) || 360; }
    catch { return 360; }
  });
  const handleLearnWidthChange = useCallback((w: number) => {
    setPanelWidth(w);
    try { localStorage.setItem('infoPanelWidth', String(w)); } catch { /* ignore */ }
  }, []);
  const panelProps = {
    learnOpen,
    learnTab,
    panelWidth,
    onLearnToggle: () => setLearnOpen(v => !v),
    onLearnClose: () => setLearnOpen(false),
    onLearnWidthChange: handleLearnWidthChange,
    onLearnTabChange: setLearnTab,
  };

  const ActivePage = ALL_ITEMS.find((i) => i.id === activeId)?.component ?? BayesianNetworks;

  return (
    <div className="flex flex-col min-h-screen bg-slate-50">
      {/* ── Header ── */}
      <header className="bg-white border-b border-slate-200 px-6 py-0 flex items-center gap-4 sticky top-0 z-30 shadow-sm h-14 shrink-0">
        {/* Brand */}
        <div className="flex items-center gap-3 shrink-0">
          <div className="bg-blue-600 p-1.5 rounded-lg text-white">
            <Network size={20} />
          </div>
          <span className="text-base font-bold text-slate-900">Probabilistic AI Lab</span>
        </div>

        {/* Nav tabs — grouped */}
        <nav className="flex items-center gap-0 h-full flex-1 overflow-x-auto">
          {NAV_GROUPS.map((group, gi) => (
            <div key={gi} className="flex items-center">
              {gi > 0 && <div className="w-px h-5 bg-slate-200 mx-2" />}
              <span className="text-xs text-slate-300 mr-1 hidden lg:inline">{group.label}:</span>
              {group.items.map((item) => (
                <button
                  key={item.id}
                  onClick={() => setActiveId(item.id)}
                  className={`
                    px-3 h-14 text-sm font-medium border-b-2 whitespace-nowrap transition-colors
                    ${activeId === item.id
                      ? 'border-blue-600 text-blue-700'
                      : 'border-transparent text-slate-500 hover:text-slate-800'}
                  `}
                >
                  {item.label}
                </button>
              ))}
            </div>
          ))}
          <button disabled
            className="px-3 h-14 text-sm font-medium border-b-2 border-transparent text-slate-300 flex items-center gap-1 cursor-not-allowed ml-2"
            title="More topics coming soon">
            More <ChevronDown size={12} />
          </button>
        </nav>

        {/* Model selector — only for Bayesian Networks */}
        {activeId === 'bayesian-networks' && (
          <select
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            className="bg-slate-100 border-none rounded-lg px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-200 transition-colors cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-400 shrink-0"
          >
            <option value="battery">Level 1: Robot Battery</option>
            <option value="fusion">Level 2: Sensor Fusion</option>
            <option value="mission">Level 3: Robot Mission Planning</option>
            <option value="robot_sampling">Sampling: Robot Sensor Fusion</option>
            <option value="student">Sampling: Student Grades</option>
            <option value="ising">Sampling: Ising Denoising (4×4)</option>
            <option value="medical_diagnosis">Sampling: Medical Diagnosis (15 nodes)</option>
          </select>
        )}
      </header>

      {/* ── Page ── */}
      <div className="flex flex-1 overflow-hidden">
        {activeId === 'bayesian-networks'
          ? <BayesianNetworks key={modelId} initialModelId={modelId} {...panelProps} />
          : <ActivePage key={activeId} {...panelProps} />}
      </div>
    </div>
  );
}

