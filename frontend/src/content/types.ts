export interface Reference {
  label: string;
  authors: string;
  url: string;
  description: string;
  type: 'book' | 'paper' | 'tutorial' | 'wiki';
}

export interface KeyEquation {
  label: string;
  latex: string;
  explanation: string;
}

export interface AlgorithmStep {
  // indent level (0 = no indent, 1 = one level, etc.)
  indent: number;
  // 'header' | 'io' | 'step' | 'comment' | 'return'
  kind: 'header' | 'io' | 'step' | 'comment' | 'return' | 'divider';
  text: string;
  // If true, `text` is rendered as LaTeX via MathBlock
  math?: boolean;
}

export interface ExploreStep {
  title: string;       // Short memorable name, e.g. "Watch the posterior collapse"
  instruction: string; // Imperative: what to click/drag/load
  watch: string;       // What to observe and WHY it matters
}

export interface ModuleContent {
  id: string;
  title: string;
  subtitle: string;
  overview: {
    paragraphs: string[];           // plain text / minimal markdown
    keyInsights: string[];          // bullet points
    equations: KeyEquation[];
  };
  algorithm: {
    name: string;
    complexity?: string;
    steps: AlgorithmStep[];
    note?: string;
  };
  references: Reference[];
  explore?: ExploreStep[];
}
