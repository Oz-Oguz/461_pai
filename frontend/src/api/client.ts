import type {
  ModelSummary,
  ModelDetail,
  InferResult,
  SolveStep,
  CPTTable,
  SamplingMethod,
  SamplingResult,
} from '../types';

const BASE = '/api';

async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
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
  return res.json() as Promise<T>;
}

export const api = {
  listModels(): Promise<ModelSummary[]> {
    return request('/models');
  },

  getModel(modelId: string): Promise<ModelDetail> {
    return request(`/models/${modelId}`);
  },

  infer(
    modelId: string,
    evidence: Record<string, string>,
    priors?: Record<string, Record<string, number>>,
    cpts?: Record<string, CPTTable>,
  ): Promise<InferResult> {
    return request('/infer', {
      method: 'POST',
      body: JSON.stringify({ model_id: modelId, evidence, priors, cpts }),
    });
  },

  solve(
    modelId: string,
    evidence: Record<string, string>,
    priors?: Record<string, Record<string, number>>,
    cpts?: Record<string, CPTTable>,
  ): Promise<SolveStep[]> {
    return request('/solve', {
      method: 'POST',
      body: JSON.stringify({ model_id: modelId, evidence, priors, cpts }),
    });
  },

  sample(
    modelId: string,
    method: SamplingMethod,
    evidence: Record<string, string>,
    nSamples: number,
    nBurn: number,
    priors?: Record<string, Record<string, number>>,
    cpts?: Record<string, CPTTable>,
    nRuns?: number,
  ): Promise<SamplingResult> {
    return request('/bn/sample', {
      method: 'POST',
      body: JSON.stringify({
        model_id: modelId,
        method,
        evidence,
        n_samples: nSamples,
        n_burn: nBurn,
        priors,
        cpts,
        n_runs: nRuns ?? 1,
      }),
    });
  },

  marginalDerivation(
    modelId: string,
    nodeId: string,
    priors?: Record<string, Record<string, number>>,
    cpts?: Record<string, CPTTable>,
  ): Promise<{ node_id: string; latex: string }> {
    return request('/marginal-derivation', {
      method: 'POST',
      body: JSON.stringify({
        model_id: modelId,
        node_id: nodeId,
        priors,
        cpts,
      }),
    });
  },
};
