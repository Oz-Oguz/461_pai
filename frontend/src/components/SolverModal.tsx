import { Calculator } from 'lucide-react';
import type { SolveStep } from '../types';
import { Modal } from './Modal';
import { MathBlock } from './MathBlock';

interface Props {
  steps: SolveStep[];
  onClose: () => void;
}

export function SolverModal({ steps, onClose }: Props) {
  return (
    <Modal
      title={<><Calculator size={18} className="text-green-600" /> Mathematical Derivation</>}
      onClose={onClose}
      wide
    >
      <div className="space-y-8 font-serif">
        {steps.map((step, i) => (
          <div key={i} className={i < steps.length - 1 ? 'pb-8 border-b border-slate-100' : ''}>
            <h4 className="font-sans font-bold text-slate-800 text-base mb-1">{step.title}</h4>
            <p className="font-sans text-sm text-slate-500 mb-4">{step.text}</p>
            <div className="bg-slate-50 border border-slate-100 rounded-xl p-4 overflow-x-auto">
              <MathBlock latex={step.latex} />
            </div>
          </div>
        ))}
      </div>
    </Modal>
  );
}
