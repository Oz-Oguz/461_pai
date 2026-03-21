import { useEffect, useMemo } from 'react';
import { GitFork } from 'lucide-react';
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
import type { ModelDetail, InferResult } from '../types';
import { Modal } from './Modal';
import { BNFlowNode } from './BNFlowNode';

interface Props {
  model: ModelDetail;
  evidence: Record<string, string>;
  results: InferResult | null;
  onClose: () => void;
}

// Must be defined outside the component to keep a stable reference
const NODE_TYPES = { bnNode: BNFlowNode };

// ── Auto-layout ────────────────────────────────────────────────────────
// Assigns depth = longest path from any root, then centres each depth layer.

function computeLayout(model: ModelDetail): Record<string, { x: number; y: number }> {
  const depth: Record<string, number> = {};
  for (const node of model.nodes) depth[node.id] = 0;

  let changed = true;
  while (changed) {
    changed = false;
    for (const node of model.nodes) {
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
  for (const node of model.nodes) {
    const d = depth[node.id];
    (byDepth[d] ??= []).push(node.id);
  }

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

// ── Component ─────────────────────────────────────────────────────────

export function GraphModal({ model, evidence, results, onClose }: Props) {
  // Layout is fixed for the lifetime of the modal (model doesn't change)
  const initialNodes: Node[] = useMemo(() => {
    const positions = computeLayout(model);
    return model.nodes.map((n) => ({
      id: n.id,
      type: 'bnNode',
      position: positions[n.id] ?? { x: 0, y: 0 },
      data: { node: n, results, evidence },
    }));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model]);

  const initialEdges: Edge[] = useMemo(() => {
    const edges: Edge[] = [];
    for (const node of model.nodes) {
      for (const parentId of node.parents) {
        edges.push({
          id: `${parentId}->${node.id}`,
          source: parentId,
          target: node.id,
          style: { stroke: '#94a3b8', strokeWidth: 2 },
        });
      }
    }
    return edges;
  }, [model]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);

  // Sync live posteriors and evidence into node data without re-running layout
  useEffect(() => {
    setNodes((nds) =>
      nds.map((nd) => ({
        ...nd,
        data: { ...nd.data, results, evidence },
      })),
    );
  }, [results, evidence, setNodes]);

  return (
    <Modal
      title={<><GitFork size={16} className="text-blue-500" /> Network Structure (DAG)</>}
      onClose={onClose}
      wide
    >
      <div style={{ height: 480 }} className="rounded-xl overflow-hidden border border-slate-200">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          nodeTypes={NODE_TYPES}
          fitView
          fitViewOptions={{ padding: 0.25 }}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
        >
          <Background color="#e2e8f0" gap={20} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
      <p className="text-xs text-slate-400 text-center mt-3">
        Blue border = observed &nbsp;·&nbsp; Sky = root cause &nbsp;·&nbsp; Green = sensor/effect
      </p>
    </Modal>
  );
}
