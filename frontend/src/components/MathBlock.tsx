import { useEffect, useRef } from 'react';
import katex from 'katex';

interface Props {
  latex: string;
  display?: boolean;
  className?: string;
}

export function MathBlock({ latex, display = true, className = '' }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    try {
      katex.render(latex, ref.current, {
        throwOnError: false,
        displayMode: display,
        trust: true,
      });
    } catch {
      if (ref.current) ref.current.textContent = latex;
    }
  }, [latex, display]);

  return <div ref={ref} className={className} />;
}
