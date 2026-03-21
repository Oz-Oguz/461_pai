import { Handle, Position } from '@xyflow/react';
import { Lock } from 'lucide-react';
import type { BayesianNode, InferResult } from '../types';

interface BNFlowNodeData {
  node: BayesianNode;
  results: InferResult | null;
  priorResults: InferResult | null;
  evidence: Record<string, string>;
  selected: boolean;
  highlight?: 'blocked' | 'collider';
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function BNFlowNode({ data }: { data: any }) {
  const { node, results, priorResults, evidence, selected, highlight } = data as BNFlowNodeData;
  const isObserved = node.id in evidence;
  const isRoot = node.node_type === 'root';
  const marginals = results?.marginals[node.id] ?? {};
  const priorMarginals = priorResults?.marginals[node.id] ?? {};
  const hasEvidence = Object.keys(evidence).length > 0;

  const borderColor = highlight === 'blocked'
    ? '#ef4444'
    : highlight === 'collider'
    ? '#f59e0b'
    : selected
    ? '#1d4ed8'
    : isObserved
    ? '#3b82f6'
    : isRoot
    ? '#7dd3fc'
    : '#4ade80';

  const headerBg = isObserved ? '#eff6ff' : '#f8fafc';
  const headerBorder = isObserved ? '#bfdbfe' : '#f1f5f9';

  const badgeStyle = isObserved
    ? { bg: '#dbeafe', text: '#1d4ed8', label: 'observed' }
    : isRoot
    ? { bg: '#e0f2fe', text: '#0369a1', label: 'root' }
    : { bg: '#dcfce7', text: '#15803d', label: 'child' };

  return (
    <div
      style={{
        width: 176,
        borderRadius: 12,
        border: `2px solid ${borderColor}`,
        background: '#fff',
        boxShadow: (selected || highlight)
          ? `0 0 0 3px ${borderColor}33, 0 4px 16px rgba(0,0,0,0.12)`
          : '0 2px 8px rgba(0,0,0,0.08)',
        fontFamily: 'inherit',
        cursor: 'pointer',
        transition: 'box-shadow 0.15s ease, border-color 0.15s ease',
      }}
    >
      <Handle type="target" position={Position.Top} style={{ background: '#94a3b8' }} />

      {/* Header */}
      <div
        style={{
          padding: '7px 10px',
          background: headerBg,
          borderBottom: `1px solid ${headerBorder}`,
          borderRadius: '10px 10px 0 0',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 4,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 4, overflow: 'hidden' }}>
          <span style={{ fontSize: 12, fontWeight: 700, color: '#1e293b', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {node.label}
          </span>
          {highlight === 'blocked' && (
            <Lock size={10} style={{ color: '#ef4444', flexShrink: 0 }} />
          )}
          {highlight === 'collider' && (
            <span style={{ fontSize: 10, color: '#f59e0b', flexShrink: 0 }}>⚡</span>
          )}
        </div>
        <span style={{
          fontSize: 10,
          fontWeight: 600,
          padding: '1px 6px',
          borderRadius: 999,
          background: badgeStyle.bg,
          color: badgeStyle.text,
          flexShrink: 0,
        }}>
          {badgeStyle.label}
        </span>
      </div>

      {/* State bars */}
      <div style={{ padding: '8px 10px', display: 'flex', flexDirection: 'column', gap: 6 }}>
        {node.states.map((state) => {
          const prob = marginals[state] ?? 0;
          const priorProb = priorMarginals[state] ?? prob;
          const shift = prob - priorProb;
          const showDelta = hasEvidence && Math.abs(shift) > 0.005;
          const isSelectedState = evidence[node.id] === state;
          return (
            <div key={state}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2, alignItems: 'center' }}>
                <span style={{ fontSize: 10, color: '#64748b', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {state}
                </span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 3, flexShrink: 0, marginLeft: 4 }}>
                  {showDelta && (
                    <span style={{
                      fontSize: 9,
                      fontWeight: 700,
                      color: shift > 0 ? '#16a34a' : '#dc2626',
                      lineHeight: 1,
                    }}>
                      {shift > 0 ? '+' : ''}{(shift * 100).toFixed(1)}%
                    </span>
                  )}
                  <span style={{ fontSize: 10, fontFamily: 'monospace', fontWeight: 700, color: '#334155' }}>
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              <div style={{ height: 8, background: '#e2e8f0', borderRadius: 999, overflow: 'hidden' }}>
                <div
                  style={{
                    height: '100%',
                    width: `${Math.min(prob * 100, 100)}%`,
                    borderRadius: 999,
                    background: isSelectedState ? '#3b82f6' : '#94a3b8',
                    transition: 'width 0.3s ease',
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <Handle type="source" position={Position.Bottom} style={{ background: '#94a3b8' }} />
    </div>
  );
}
