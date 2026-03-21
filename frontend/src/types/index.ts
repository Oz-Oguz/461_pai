// TypeScript types mirroring the Python dataclasses in shared/types.py

export type LearnTab = 'overview' | 'algorithm' | 'references' | 'explore';

export interface BayesianNode {
  id: string;
  label: string;
  states: string[];
  node_type: 'root' | 'child';
  parents: string[];
  description: string;
}

export interface CPTTable {
  parents: string[];
  table: Record<string, Record<string, number>>;
}

export interface Scenario {
  name: string;
  evidence: Record<string, string>;
  description: string;
}

export interface ModelSummary {
  id: string;
  name: string;
  description: string;
}

export interface ModelDetail {
  id: string;
  name: string;
  description: string;
  nodes: BayesianNode[];
  priors: Record<string, Record<string, number>>;
  cpts: Record<string, CPTTable>;
  scenarios: Scenario[];
}

export interface InferResult {
  marginals: Record<string, Record<string, number>>;
  total_weight: number;
}

export interface SolveStep {
  title: string;
  text: string;
  latex: string;
}

export type SamplingMethod = 'prior' | 'rejection' | 'likelihood_weighting' | 'gibbs';

export interface SamplingMetadata {
  n_samples: number;
  n_accepted?: number;
  acceptance_rate?: number;
  effective_samples?: number;
  n_burn?: number;
  n_runs?: number;
}

export interface CIBand {
  low: number[];
  high: number[];
}

export interface SamplingResult {
  method: SamplingMethod;
  marginals: Record<string, Record<string, number>>;
  exact_marginals: Record<string, Record<string, number>>;
  trajectory: {
    steps: number[];
    curves: Record<string, Record<string, number[]>>;
    ci_bands?: Record<string, Record<string, CIBand>>;
  };
  metadata: SamplingMetadata;
}
