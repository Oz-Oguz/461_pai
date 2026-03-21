import { useState, useRef, useEffect, useCallback } from 'react';
import {
  BookOpen, Code2, ExternalLink, X, BookMarked,
  ChevronRight, Lightbulb, FlaskConical, GripVertical, Compass,
} from 'lucide-react';
import type { ModuleContent, AlgorithmStep } from '../content/types';
import type { LearnTab } from '../types';
import { MathBlock } from './MathBlock';

// ── Tab type alias ─────────────────────────────────────────────────────
type Tab = LearnTab;

const BASE_TABS: { id: Tab; label: string; icon: typeof BookOpen }[] = [
  { id: 'overview',   label: 'Overview',   icon: BookOpen  },
  { id: 'algorithm',  label: 'Algorithm',  icon: Code2     },
  { id: 'references', label: 'References', icon: BookMarked },
];
const EXPLORE_TAB = { id: 'explore' as Tab, label: 'Explore', icon: Compass };

// ── Reference type badge colours ─────────────────────────────────────
const TYPE_STYLE = {
  book:     'bg-blue-100 text-blue-700',
  paper:    'bg-purple-100 text-purple-700',
  tutorial: 'bg-green-100 text-green-700',
  wiki:     'bg-slate-100 text-slate-600',
};

// ── Resize constraints ────────────────────────────────────────────────
const MIN_WIDTH = 280;
const MAX_WIDTH = 640;

// ── Algorithm step renderer ───────────────────────────────────────────
function AlgoStep({ step }: { step: AlgorithmStep }) {
  const indent = step.indent * 20; // px per indent level

  if (step.kind === 'divider') {
    return <hr className="border-slate-200 my-2" />;
  }

  if (step.kind === 'header') {
    return (
      <div className="font-bold text-slate-700 text-xs uppercase tracking-wider mb-2 mt-1">
        {step.text}
      </div>
    );
  }

  if (step.kind === 'comment') {
    return (
      <div className="text-slate-400 italic text-xs" style={{ paddingLeft: indent }}>
        ▷ {step.text}
      </div>
    );
  }

  return (
    <div className="flex items-start gap-2 py-0.5" style={{ paddingLeft: indent }}>
      {step.kind === 'io' && (
        <span className="text-xs font-bold text-slate-400 shrink-0 mt-0.5 w-12">
          {step.text.startsWith('Input') ? 'INPUT' : 'OUTPUT'}
        </span>
      )}
      {step.kind === 'return' && (
        <span className="text-xs font-bold text-green-600 shrink-0 mt-0.5">return</span>
      )}
      {step.kind === 'step' && (
        <ChevronRight size={12} className="text-slate-300 shrink-0 mt-1" />
      )}
      <div className="text-sm text-slate-700 leading-snug font-mono">
        {step.math ? (
          <MathBlock latex={step.text} display={false} className="inline" />
        ) : (
          step.kind === 'io'
            ? step.text.replace(/^(Input|Output):\s*/, '')
            : step.text
        )}
      </div>
    </div>
  );
}

// ── Explore tab — single-step viewer with dropdown navigation ─────────
function ExploreTab({ steps }: { steps: NonNullable<ModuleContent['explore']> }) {
  const [idx, setIdx] = useState<number>(() => {
    try { return Math.min(parseInt(localStorage.getItem('explore-step-idx') ?? '0', 10), steps.length - 1); }
    catch { return 0; }
  });
  const go = (i: number) => {
    const clamped = Math.max(0, Math.min(steps.length - 1, i));
    setIdx(clamped);
    try { localStorage.setItem('explore-step-idx', String(clamped)); } catch { /* ignore */ }
  };

  const step = steps[idx];
  return (
    <div className="flex flex-col gap-3">
      {/* ── Navigation row ── */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => go(idx - 1)}
          disabled={idx === 0}
          className="p-1 rounded-lg hover:bg-slate-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors shrink-0"
          aria-label="Previous step"
        >
          <ChevronRight size={16} className="text-slate-500 rotate-180" />
        </button>

        <select
          value={idx}
          onChange={e => go(Number(e.target.value))}
          className="flex-1 text-xs bg-slate-50 border border-slate-200 rounded-lg px-2 py-1.5 text-slate-700 font-medium focus:outline-none focus:ring-2 focus:ring-blue-400 cursor-pointer min-w-0"
        >
          {steps.map((s, i) => (
            <option key={i} value={i}>{i + 1}. {s.title}</option>
          ))}
        </select>

        <button
          onClick={() => go(idx + 1)}
          disabled={idx === steps.length - 1}
          className="p-1 rounded-lg hover:bg-slate-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors shrink-0"
          aria-label="Next step"
        >
          <ChevronRight size={16} className="text-slate-500" />
        </button>
      </div>

      {/* ── Step counter ── */}
      <p className="text-xs text-slate-400 text-center -mt-1">
        Step {idx + 1} of {steps.length}
      </p>

      {/* ── Step body ── */}
      <div className="rounded-xl border border-slate-200 overflow-hidden">
        <div className="flex items-center gap-3 px-4 py-2.5 bg-slate-50 border-b border-slate-100">
          <span className="w-6 h-6 rounded-full bg-blue-600 text-white text-xs font-bold flex items-center justify-center shrink-0">
            {idx + 1}
          </span>
          <span className="text-sm font-semibold text-slate-700 leading-snug">
            {step.title}
          </span>
        </div>
        <div className="px-4 py-3 space-y-3">
          <div>
            <span className="text-xs font-bold text-blue-600 uppercase tracking-wide">Try this</span>
            <p className="text-xs text-slate-600 mt-1 leading-relaxed">{step.instruction}</p>
          </div>
          <div>
            <span className="text-xs font-bold text-emerald-600 uppercase tracking-wide">What to notice</span>
            <p className="text-xs text-slate-600 mt-1 leading-relaxed">{step.watch}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Main InfoPanel ────────────────────────────────────────────────────
interface Props {
  content: ModuleContent;
  open: boolean;
  width: number;
  activeTab: LearnTab;
  onToggle: () => void;
  onClose: () => void;
  onWidthChange: (w: number) => void;
  onActiveTabChange: (t: LearnTab) => void;
}

export function InfoPanel({ content, open, width, activeTab, onToggle, onClose, onWidthChange, onActiveTabChange }: Props) {
  const tab = activeTab;
  const setTab = onActiveTabChange;
  const dragRef = useRef<{ startX: number; startW: number } | null>(null);

  // Global mouse handlers for drag-to-resize
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragRef.current) return;
      // Dragging left increases width (panel is on the right)
      const dx = dragRef.current.startX - e.clientX;
      const newW = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, dragRef.current.startW + dx));
      onWidthChange(newW);
    };
    const onUp = () => { dragRef.current = null; };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [onWidthChange]);

  const handleDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragRef.current = { startX: e.clientX, startW: width };
  }, [width]);

  return (
    // outer aside: no overflow:hidden so the pull-tab can poke out
    <aside
      className="relative shrink-0 flex flex-col transition-all duration-300 ease-in-out"
      style={{ width: open ? width : 0 }}
    >
      {/* ── Pull-tab: always visible, sticks out to the left ── */}
      <button
        onClick={onToggle}
        title={open ? 'Close Learn panel' : 'Open Learn panel'}
        className={`
          absolute left-0 top-1/2 -translate-y-1/2 -translate-x-full
          flex flex-col items-center justify-center gap-1.5
          w-7 py-5 rounded-l-xl border border-r-0 shadow-md
          transition-colors z-20 select-none
          ${open
            ? 'bg-blue-600 text-white border-blue-600 hover:bg-blue-700'
            : 'bg-white text-slate-500 border-slate-200 hover:border-blue-300 hover:text-blue-600'}
        `}
      >
        <BookOpen size={13} />
        <span
          className="text-[10px] font-bold tracking-widest uppercase"
          style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}
        >
          Learn
        </span>
      </button>

      {/* ── Inner panel: overflow hidden to clip at width:0 ── */}
      <div
        className="flex flex-col h-full bg-white border-l border-slate-200 overflow-hidden"
        style={{ width: open ? width : 0, minWidth: open ? width : 0 }}
      >
        {open && (
          <>
            {/* Drag-to-resize handle on the left edge */}
            <div
              className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize z-10 group flex items-center"
              onMouseDown={handleDragStart}
            >
              <div className="w-full h-12 bg-slate-200 group-hover:bg-blue-400 transition-colors rounded-r-full opacity-60 group-hover:opacity-100" />
              <GripVertical
                size={12}
                className="absolute left-0 text-slate-400 group-hover:text-blue-500 opacity-0 group-hover:opacity-100 transition-opacity"
              />
            </div>

            {/* Panel header (offset from drag handle) */}
            <div className="pl-3 pr-4 pt-4 pb-0 shrink-0">
              <div className="flex items-start justify-between mb-3">
                <div>
                  <h2 className="font-bold text-slate-800 text-sm leading-tight">
                    {content.title}
                  </h2>
                  <p className="text-xs text-slate-400 mt-0.5">{content.subtitle}</p>
                </div>
                <button
                  onClick={onClose}
                  className="text-slate-400 hover:text-slate-700 p-1 rounded-lg hover:bg-slate-100 transition-colors"
                >
                  <X size={16} />
                </button>
              </div>

              {/* Tabs */}
              <div className="flex gap-0 border-b border-slate-200">
                {[...BASE_TABS, ...(content.explore?.length ? [EXPLORE_TAB] : [])].map(({ id, label, icon: Icon }) => (
                  <button
                    key={id}
                    onClick={() => setTab(id)}
                    className={`
                      flex items-center gap-1.5 px-3 py-2 text-xs font-medium border-b-2
                      transition-colors -mb-px
                      ${tab === id
                        ? 'border-blue-600 text-blue-700'
                        : 'border-transparent text-slate-500 hover:text-slate-700'}
                    `}
                  >
                    <Icon size={12} />
                    {label}
                  </button>
                ))}
              </div>
            </div>

            {/* Tab content */}
            <div className="flex-1 overflow-y-auto pl-3 pr-4 py-4 space-y-5 text-sm">

              {/* ── Overview ── */}
              {tab === 'overview' && (
                <>
                  {content.overview.paragraphs.map((p, i) => (
                    <p key={i} className="text-slate-600 leading-relaxed text-sm">{p}</p>
                  ))}

                  {content.overview.equations.length > 0 && (
                    <div className="space-y-3">
                      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                        Key Equations
                      </h3>
                      {content.overview.equations.map((eq, i) => (
                        <div key={i} className="bg-slate-50 border border-slate-100 rounded-xl p-3">
                          <div className="text-xs font-semibold text-slate-600 mb-2">{eq.label}</div>
                          <div className="overflow-x-auto">
                            <MathBlock latex={eq.latex} display />
                          </div>
                          <p className="text-xs text-slate-400 mt-2 leading-relaxed">{eq.explanation}</p>
                        </div>
                      ))}
                    </div>
                  )}

                  {content.overview.keyInsights.length > 0 && (
                    <div>
                      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1">
                        <Lightbulb size={11} /> What to look for
                      </h3>
                      <ul className="space-y-1.5">
                        {content.overview.keyInsights.map((tip, i) => (
                          <li key={i} className="flex items-start gap-2 text-sm text-slate-600">
                            <span className="text-blue-400 mt-0.5 shrink-0">•</span>
                            {tip}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </>
              )}

              {/* ── Algorithm ── */}
              {tab === 'algorithm' && (
                <>
                  <div className="flex items-center justify-between mb-1">
                    <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1">
                      <FlaskConical size={11} /> {content.algorithm.name}
                    </h3>
                    {content.algorithm.complexity && (
                      <span className="text-xs font-mono bg-amber-50 text-amber-700 border border-amber-100 px-2 py-0.5 rounded-full">
                        {content.algorithm.complexity}
                      </span>
                    )}
                  </div>

                  <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-1 font-mono">
                    {content.algorithm.steps.map((step, i) => (
                      <AlgoStep key={i} step={step} />
                    ))}
                  </div>

                  {content.algorithm.note && (
                    <div className="bg-blue-50 border border-blue-100 rounded-xl p-3 text-xs text-blue-700 leading-relaxed">
                      <strong>Note:</strong> {content.algorithm.note}
                    </div>
                  )}
                </>
              )}

              {/* ── References ── */}
              {tab === 'references' && (
                <div className="space-y-3">
                  {content.references.map((ref, i) => (
                    <a
                      key={i}
                      href={ref.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block border border-slate-200 rounded-xl p-3 hover:border-blue-300 hover:bg-blue-50/40 transition-colors group"
                    >
                      <div className="flex items-start justify-between gap-2 mb-1">
                        <span className="font-semibold text-slate-800 text-sm group-hover:text-blue-700 leading-tight">
                          {ref.label}
                        </span>
                        <div className="flex items-center gap-1 shrink-0">
                          <span className={`text-xs px-1.5 py-0.5 rounded-full font-medium ${TYPE_STYLE[ref.type]}`}>
                            {ref.type}
                          </span>
                          <ExternalLink size={12} className="text-slate-300 group-hover:text-blue-400" />
                        </div>
                      </div>
                      <p className="text-xs text-slate-500 italic mb-1">{ref.authors}</p>
                      <p className="text-xs text-slate-500 leading-relaxed">{ref.description}</p>
                    </a>
                  ))}
                </div>
              )}

              {/* ── Explore ── */}
              {tab === 'explore' && content.explore && (
                <ExploreTab steps={content.explore} />
              )}

            </div>
          </>
        )}
      </div>
    </aside>
  );
}
