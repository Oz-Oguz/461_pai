import { useRef } from 'react';

interface RangeSliderProps {
  label: string;
  trackMin: number;
  trackMax: number;
  value: { min: number; max: number };
  onChange: (v: { min: number; max: number }) => void;
  /** Snap step for dragging (default 0.5) */
  step?: number;
  /** Decimal places shown in the value readout (default 1) */
  decimals?: number;
}

/**
 * Dual-handle range slider.
 * Both handles use Pointer Capture so dragging is smooth even when the
 * cursor leaves the track area.
 */
export function RangeSlider({
  label,
  trackMin,
  trackMax,
  value,
  onChange,
  step = 0.5,
  decimals = 1,
}: RangeSliderProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  // Always-fresh ref so pointer-capture handlers never have stale closures
  const valueRef = useRef(value);
  valueRef.current = value;

  const toPercent = (v: number) =>
    Math.max(0, Math.min(100, ((v - trackMin) / (trackMax - trackMin)) * 100));

  const fromClientX = (clientX: number): number => {
    const rect = trackRef.current!.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    const raw = trackMin + pct * (trackMax - trackMin);
    return Math.round(raw / step) * step;
  };

  const makeHandlers = (which: 'min' | 'max') => ({
    onPointerDown(e: React.PointerEvent<HTMLDivElement>) {
      e.preventDefault();
      e.currentTarget.setPointerCapture(e.pointerId);
    },
    onPointerMove(e: React.PointerEvent<HTMLDivElement>) {
      if (!e.currentTarget.hasPointerCapture(e.pointerId)) return;
      const v = fromClientX(e.clientX);
      const cur = valueRef.current;
      const GAP = Math.max(step * 2, 0.2);
      if (which === 'min') {
        const newMin = Math.max(trackMin, Math.min(v, cur.max - GAP));
        if (newMin !== cur.min) onChange({ min: newMin, max: cur.max });
      } else {
        const newMax = Math.min(trackMax, Math.max(v, cur.min + GAP));
        if (newMax !== cur.max) onChange({ min: cur.min, max: newMax });
      }
    },
    onPointerUp(e: React.PointerEvent<HTMLDivElement>) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    },
  });

  const minPct = toPercent(value.min);
  const maxPct = toPercent(value.max);
  const fmt = (v: number) => v.toFixed(decimals);

  return (
    <div className="flex items-center gap-3 select-none">
      {/* Axis label */}
      <span className="text-xs font-mono font-medium text-slate-500 w-4 text-right shrink-0">
        {label}
      </span>

      {/* Track + thumbs */}
      <div ref={trackRef} className="relative flex-1 h-5 flex items-center">
        {/* Track background */}
        <div className="absolute w-full h-1.5 bg-slate-200 rounded-full" />

        {/* Active fill between thumbs */}
        <div
          className="absolute h-1.5 bg-blue-400 rounded-full pointer-events-none"
          style={{ left: `${minPct}%`, width: `${maxPct - minPct}%` }}
        />

        {/* Min thumb */}
        <div
          className="absolute w-4 h-4 bg-white border-2 border-blue-500 rounded-full shadow-sm
                     cursor-grab active:cursor-grabbing -translate-x-1/2 z-10 touch-none"
          style={{ left: `${minPct}%` }}
          {...makeHandlers('min')}
        />

        {/* Max thumb */}
        <div
          className="absolute w-4 h-4 bg-white border-2 border-blue-500 rounded-full shadow-sm
                     cursor-grab active:cursor-grabbing -translate-x-1/2 z-20 touch-none"
          style={{ left: `${maxPct}%` }}
          {...makeHandlers('max')}
        />
      </div>

      {/* Current value readout */}
      <span className="text-xs font-mono text-slate-400 w-20 text-right tabular-nums shrink-0">
        {fmt(value.min)} – {fmt(value.max)}
      </span>
    </div>
  );
}
