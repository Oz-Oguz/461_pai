import { useState, useEffect, useCallback, useMemo } from 'react';
import {
  BookOpen,
  Settings,
  Calculator,
  RefreshCcw,
  LayoutGrid,
  ChevronRight,
  AlertCircle,
  Loader2,
  X,
  PanelLeftClose,
  PanelLeftOpen,
} from 'lucide-react';
import {
  ReactFlow,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import type { ModelDetail, InferResult, SolveStep, CPTTable, BayesianNode, Scenario, LearnTab } from '../types';
import { api } from '../api/client';
import { NodeCard } from '../components/NodeCard';
import { SolverModal } from '../components/SolverModal';
import { InfoPanel } from '../components/InfoPanel';
import { BNFlowNode } from '../components/BNFlowNode';
import { SamplingPanel } from '../components/SamplingPanel';
import { bayesianNetworksContent } from '../content/bayesian_networks';

// ── Auto-layout ────────────────────────────────────────────────────────
// depth = longest path from any root; nodes at same depth spread horizontally.

const NODE_TYPES = { bnNode: BNFlowNode };

function computeLayout(nodes: BayesianNode[]): Record<string, { x: number; y: number }> {
  const depth: Record<string, number> = {};
  for (const node of nodes) depth[node.id] = 0;

  let changed = true;
  while (changed) {
    changed = false;
    for (const node of nodes) {
      for (const parentId of node.parents) {
        const candidate = (depth[parentId] ?? 0) + 1;
        if (candidate > depth[node.id]) {
          depth[node.id] = candidate;
          changed = true;
        }
      }
    }
  }

  const byDepth: Record<number, string[]> = {};
  for (const node of nodes) (byDepth[depth[node.id]] ??= []).push(node.id);

  const NODE_W = 220, NODE_H = 170;
  const positions: Record<string, { x: number; y: number }> = {};
  for (const [d, ids] of Object.entries(byDepth)) {
    const count = ids.length;
    ids.forEach((id, i) => {
      positions[id] = {
        x: (i - (count - 1) / 2) * NODE_W,
        y: Number(d) * NODE_H,
      };
    });
  }
  return positions;
}

// ── D-separation highlighting ──────────────────────────────────────────
// Parses the "Structure: A → B ← C" line from d-sep scenario descriptions
// and classifies middle nodes as blocked (chain/fork + observed) or collider.

interface DSepHighlights {
  pathEdgeIds: Set<string>;
  pathNodeIds: Set<string>;
  blockedNodeIds: Set<string>;
  colliderNodeIds: Set<string>;
}

function getDSepHighlights(
  scenario: Scenario,
  evidence: Record<string, string>,
): DSepHighlights | null {
  if (!scenario.name.startsWith('d-sep')) return null;
  const match = scenario.description?.match(/Structure:\s*(.+?)(?:\s*[\(\.\n]|$)/);
  if (!match) return null;

  const tokens = match[1].trim().split(/\s+/);
  const nodeIds: string[] = [];
  const arrows: string[] = [];
  for (const t of tokens) {
    if (t === '→' || t === '←') arrows.push(t);
    else nodeIds.push(t);
  }
  if (nodeIds.length < 2) return null;

  const pathNodeIds = new Set(nodeIds);
  const pathEdgeIds = new Set<string>();
  const blockedNodeIds = new Set<string>();
  const colliderNodeIds = new Set<string>();

  for (let i = 0; i < arrows.length; i++) {
    const a = nodeIds[i], b = nodeIds[i + 1];
    pathEdgeIds.add(arrows[i] === '→' ? `${a}->${b}` : `${b}->${a}`);
  }

  for (let i = 1; i < nodeIds.length - 1; i++) {
    const mid = nodeIds[i];
    const isCollider = arrows[i - 1] === '→' && arrows[i] === '←';
    if (isCollider) {
      colliderNodeIds.add(mid);
    } else if (mid in evidence) {
      blockedNodeIds.add(mid);
    }
  }

  return { pathEdgeIds, pathNodeIds, blockedNodeIds, colliderNodeIds };
}

// ── Component ─────────────────────────────────────────────────────────

interface BNProps {
  initialModelId?: string;
  learnOpen: boolean;
  learnTab: LearnTab;
  panelWidth: number;
  onLearnToggle: () => void;
  onLearnClose: () => void;
  onLearnWidthChange: (w: number) => void;
  onLearnTabChange: (t: LearnTab) => void;
}

export function BayesianNetworks({
  initialModelId = 'fusion',
  learnOpen,
  learnTab,
  panelWidth,
  onLearnToggle,
  onLearnClose,
  onLearnWidthChange,
  onLearnTabChange,
}: BNProps) {
  const [modelId] = useState(initialModelId);
  const [model, setModel] = useState<ModelDetail | null>(null);
  const [evidence, setEvidence] = useState<Record<string, string>>({});
  const [priors, setPriors] = useState<Record<string, Record<string, number>>>({});
  const [cpts, setCpts] = useState<Record<string, CPTTable>>({});
  const [results, setResults] = useState<InferResult | null>(null);
  const [priorResults, setPriorResults] = useState<InferResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [activeScenario, setActiveScenario] = useState<Scenario | null>(null);

  const [sidebarOpen, setSidebarOpen] = useState(() => {
    try { return localStorage.getItem('bn-sidebar-open') !== 'false'; }
    catch { return true; }
  });

  const [solverSteps, setSolverSteps] = useState<SolveStep[] | null>(null);
  const [solverLoading, setSolverLoading] = useState(false);

  // ── React Flow state ──────────────────────────────────────────────
  const [nodes, setNodes, onNodesChange] = useNodesState([] as Node[]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([] as Edge[]);

  // Build layout once when model loads; positions never change after that.
  const flowData = useMemo(() => {
    if (!model) return null;
    const positions = computeLayout(model.nodes);
    const flowNodes: Node[] = model.nodes.map((n) => ({
      id: n.id,
      type: 'bnNode',
      position: positions[n.id] ?? { x: 0, y: 0 },
      data: { node: n, results: null, evidence: {}, selected: false },
    }));
    const flowEdges: Edge[] = [];
    for (const node of model.nodes) {
      for (const parentId of node.parents) {
        flowEdges.push({
          id: `${parentId}->${node.id}`,
          source: parentId,
          target: node.id,
          style: { stroke: '#94a3b8', strokeWidth: 2 },
        });
      }
    }
    return { flowNodes, flowEdges };
  }, [model]);

  useEffect(() => {
    if (!flowData) return;
    setNodes(flowData.flowNodes);
    setEdges(flowData.flowEdges);
  }, [flowData, setNodes, setEdges]);

  // D-sep highlights — recompute whenever active scenario or evidence changes.
  const highlights = useMemo(
    () => (activeScenario ? getDSepHighlights(activeScenario, evidence) : null),
    [activeScenario, evidence],
  );

  // Sync live posteriors, evidence, selection and d-sep highlights into node data.
  useEffect(() => {
    setNodes((nds) =>
      nds.map((nd) => ({
        ...nd,
        data: {
          ...nd.data,
          results,
          priorResults,
          evidence,
          selected: nd.id === selectedNodeId,
          highlight: highlights?.blockedNodeIds.has(nd.id)
            ? 'blocked'
            : highlights?.colliderNodeIds.has(nd.id)
            ? 'collider'
            : undefined,
        },
      })),
    );
  }, [results, priorResults, evidence, selectedNodeId, highlights, setNodes]);

  // Animate edges on the highlighted path; reset when no scenario active.
  useEffect(() => {
    setEdges((eds) =>
      eds.map((e) => {
        const onPath = highlights?.pathEdgeIds.has(e.id) ?? false;
        return {
          ...e,
          animated: onPath,
          style: onPath
            ? { stroke: '#3b82f6', strokeWidth: 3 }
            : { stroke: '#94a3b8', strokeWidth: 2 },
        };
      }),
    );
  }, [highlights, setEdges]);

  // ── API ───────────────────────────────────────────────────────────

  useEffect(() => {
    setLoading(true);
    setError(null);
    setEvidence({});
    setResults(null);
    setSolverSteps(null);
    setSelectedNodeId(null);
    setActiveScenario(null);
    api
      .getModel(modelId)
      .then((m) => {
        setModel(m);
        setPriors(JSON.parse(JSON.stringify(m.priors)));
        setCpts(JSON.parse(JSON.stringify(m.cpts)));
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [modelId]);

  useEffect(() => {
    if (!model) return;
    api.infer(modelId, evidence, priors, cpts).then(setResults).catch(() => setResults(null));
    api.infer(modelId, {}, priors, cpts).then(setPriorResults).catch(() => setPriorResults(null));
  }, [model, modelId, evidence, priors, cpts]);

  // ── Handlers ──────────────────────────────────────────────────────

  const toggleEvidence = useCallback((nodeId: string, state: string) => {
    setEvidence((prev) => {
      const next = { ...prev };
      if (next[nodeId] === state) delete next[nodeId];
      else next[nodeId] = state;
      return next;
    });
  }, []);

  const handlePriorChange = (nodeId: string, primaryState: string, value: number) => {
    if (!model) return;
    const node = model.nodes.find((n) => n.id === nodeId)!;
    const otherState = node.states.find((s) => s !== primaryState)!;
    setPriors((prev) => ({
      ...prev,
      [nodeId]: {
        ...prev[nodeId],
        [primaryState]: value,
        [otherState]: Math.round((1 - value) * 100) / 100,
      },
    }));
  };

  const handleCPTChange = (nodeId: string, parentKey: string, state: string, value: number) => {
    if (!model) return;
    const clamped = Math.max(0, Math.min(1, value));
    const node = model.nodes.find((n) => n.id === nodeId)!;
    const others = node.states.filter((s) => s !== state);
    setCpts((prev) => {
      const prevRow = { ...(prev[nodeId]?.table[parentKey] ?? {}) };
      prevRow[state] = clamped;
      if (others.length === 1) prevRow[others[0]] = Math.round((1 - clamped) * 100) / 100;
      return {
        ...prev,
        [nodeId]: { ...prev[nodeId], table: { ...prev[nodeId]?.table, [parentKey]: prevRow } },
      };
    });
  };

  const handleResetLayout = () => {
    if (!flowData) return;
    setNodes((nds) =>
      nds.map((nd) => {
        const original = flowData.flowNodes.find((fn) => fn.id === nd.id);
        return original ? { ...nd, position: original.position } : nd;
      }),
    );
  };

  const handleReset = () => {
    if (!model) return;
    setEvidence({});
    setSelectedNodeId(null);
    setActiveScenario(null);
    setPriors(JSON.parse(JSON.stringify(model.priors)));
    setCpts(JSON.parse(JSON.stringify(model.cpts)));
  };

  const handleSolve = async () => {
    setSolverLoading(true);
    try {
      const steps = await api.solve(modelId, evidence, priors, cpts);
      setSolverSteps(steps);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Solve failed');
    } finally {
      setSolverLoading(false);
    }
  };

  const handleNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    setSelectedNodeId((prev) => (prev === node.id ? null : node.id));
  }, []);

  const handlePaneClick = useCallback(() => {
    setSelectedNodeId(null);
  }, []);

  // ── Derived ───────────────────────────────────────────────────────

  const selectedNode = model?.nodes.find((n) => n.id === selectedNodeId) ?? null;
  const rootNodes = model?.nodes.filter((n) => n.node_type === 'root') ?? [];

  // ── Render ────────────────────────────────────────────────────────

  if (loading) {
    return (
      <div className="flex items-center justify-center flex-1">
        <Loader2 className="animate-spin text-blue-500" size={32} />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center flex-1">
        <div className="flex items-center gap-2 text-red-600 bg-red-50 border border-red-100 px-6 py-4 rounded-xl">
          <AlertCircle size={18} /> {error}
        </div>
      </div>
    );
  }

  if (!model) return null;

  return (
    <div className="flex flex-1 overflow-hidden">

      {/* ── Left Sidebar ── */}
      <aside
        className="relative shrink-0 flex flex-col transition-all duration-300 ease-in-out"
        style={{ width: sidebarOpen ? 288 : 0 }}
      >
        {/* Pull-tab on right edge */}
        <button
          onClick={() => setSidebarOpen((v) => {
            const next = !v;
            try { localStorage.setItem('bn-sidebar-open', String(next)); } catch { /* ignore */ }
            return next;
          })}
          title={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
          className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-full
            flex items-center justify-center w-5 h-12
            rounded-r-lg border border-l-0 shadow-sm bg-white
            text-slate-400 border-slate-200 hover:border-slate-300 hover:text-slate-700
            transition-colors z-20 select-none"
        >
          {sidebarOpen ? <PanelLeftClose size={13} /> : <PanelLeftOpen size={13} />}
        </button>

        <div
          className="flex flex-col h-full bg-white border-r border-slate-200 overflow-hidden"
          style={{ width: sidebarOpen ? 288 : 0, minWidth: sidebarOpen ? 288 : 0 }}
        >
          {sidebarOpen && (
            <>
              {/* Scenarios */}
              <div className="p-5 border-b border-slate-100 overflow-y-auto">
                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                  <BookOpen size={13} /> Learning Scenarios
                </h3>
                <div className="space-y-1">
                  {model.scenarios.map((s) => (
                    <button
                      key={s.name}
                      onClick={() => { setEvidence({ ...s.evidence }); setActiveScenario(s); }}
                      title={s.description}
                      className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors flex items-center justify-between group
                        ${activeScenario?.name === s.name
                          ? 'bg-blue-50 text-blue-700 border border-blue-200'
                          : 'text-slate-700 hover:bg-blue-50 hover:text-blue-700'}`}
                    >
                      <span>{s.name}</span>
                      <ChevronRight size={14} className="text-slate-300 group-hover:text-blue-500 transition-colors" />
                    </button>
                  ))}
                </div>
              </div>

              {/* Priors */}
              <div className="p-5 flex-1 overflow-y-auto">
                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                  <Settings size={13} /> Model Parameters (Priors)
                </h3>
                {rootNodes.map((node) => {
                  const primaryState = node.states[0];
                  const currentVal = priors[node.id]?.[primaryState] ?? 0.5;
                  return (
                    <div key={node.id} className="mb-6">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-sm font-medium text-slate-700">{node.label}</span>
                        <span className="text-xs font-mono bg-slate-100 px-1.5 py-0.5 rounded text-slate-500">
                          P({primaryState})={currentVal.toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range" min={0.01} max={0.99} step={0.01} value={currentVal}
                        onChange={(e) => handlePriorChange(node.id, primaryState, parseFloat(e.target.value))}
                        className="w-full h-1.5 rounded-full accent-blue-600 cursor-pointer"
                      />
                      <div className="flex justify-between mt-1 text-xs text-slate-400">
                        <span>{node.states[0]}</span>
                        <span>{node.states[1]}</span>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Reset buttons */}
              <div className="p-4 border-t border-slate-100 flex flex-col gap-2 shrink-0">
                <button
                  onClick={handleReset}
                  className="w-full py-2 rounded-lg text-sm text-slate-600 hover:bg-slate-100 hover:text-slate-900 transition-colors flex items-center justify-center gap-2"
                >
                  <RefreshCcw size={14} /> Reset Network
                </button>
                <button
                  onClick={handleResetLayout}
                  className="w-full py-2 rounded-lg text-sm text-slate-500 hover:bg-slate-100 hover:text-slate-800 transition-colors flex items-center justify-center gap-2"
                >
                  <LayoutGrid size={14} /> Reset Layout
                </button>
              </div>
            </>
          )}
        </div>
      </aside>

      {/* ── Main: React Flow Graph ── */}
      <main className="flex-1 bg-slate-50 relative" style={{ minHeight: 0 }}>

        {/* Floating model title */}
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10 text-center pointer-events-none select-none">
          <h2 className="text-base font-bold text-slate-600">{model.name}</h2>
          {!selectedNode && (
            <p className="text-xs text-slate-400 mt-0.5">Click a node to inspect &amp; set evidence</p>
          )}
        </div>

        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          nodeTypes={NODE_TYPES}
          onNodeClick={handleNodeClick}
          onPaneClick={handlePaneClick}
          fitView
          fitViewOptions={{ padding: 0.3 }}
          nodesConnectable={false}
          elementsSelectable={false}
          style={{ width: '100%', height: '100%' }}
        >
          <Background color="#e2e8f0" gap={20} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </main>

      {/* ── Right: Node Detail Panel ── */}
      {selectedNode && (
        <aside className="w-80 bg-white border-l border-slate-200 flex flex-col overflow-y-auto shrink-0">
          <div className="px-5 py-3 border-b border-slate-100 bg-slate-50 flex items-center justify-between shrink-0">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Node Detail</span>
            <button
              onClick={() => setSelectedNodeId(null)}
              className="text-slate-400 hover:text-slate-700 transition-colors p-1 rounded-lg hover:bg-slate-200"
            >
              <X size={15} />
            </button>
          </div>
          <div className="p-4">
            <NodeCard
              node={selectedNode}
              results={results}
              priorMarginals={priorResults}
              evidence={evidence}
              priors={priors}
              cpts={cpts}
              modelId={modelId}
              onToggleEvidence={toggleEvidence}
              onCPTChange={handleCPTChange}
            />
          </div>
          <div className="px-5 pb-4 text-xs text-slate-400 leading-relaxed">
            Click a state row to set or clear evidence. Use the <strong>table icon</strong> to view or edit the CPT.
          </div>
        </aside>
      )}

      {/* ── Sampling Panel (pull-tab from right edge) ── */}
      <SamplingPanel
        model={model}
        evidence={evidence}
        priors={priors}
        cpts={cpts}
      />

      {/* ── Learn Panel (slides from right, fixed) ── */}
      <InfoPanel
        content={bayesianNetworksContent}
        open={learnOpen}
        width={panelWidth}
        activeTab={learnTab}
        onToggle={onLearnToggle}
        onClose={onLearnClose}
        onWidthChange={onLearnWidthChange}
        onActiveTabChange={onLearnTabChange}
      />

      {/* ── Floating actions ── */}
      <div
        className="fixed bottom-6 flex flex-col gap-3 transition-all duration-300"
        style={{ right: learnOpen ? panelWidth + 24 : 24 }}
      >
        <button
          onClick={handleSolve}
          disabled={solverLoading}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-green-600 text-white shadow-lg hover:bg-green-700 hover:shadow-xl transition-all text-sm font-semibold disabled:opacity-60"
        >
          {solverLoading ? <Loader2 size={15} className="animate-spin" /> : <Calculator size={15} />}
          Solve Step-by-Step
        </button>
      </div>

      {solverSteps && (
        <SolverModal steps={solverSteps} onClose={() => setSolverSteps(null)} />
      )}
    </div>
  );
}
