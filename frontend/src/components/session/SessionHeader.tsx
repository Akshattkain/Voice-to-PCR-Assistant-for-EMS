import { CheckCircle2, Activity } from 'lucide-react';
import { useSessionStore } from '../../store/sessionStore';
import { CompletenessBar } from './CompletenessBar';
import { MicButton } from './MicButton';
import { GapBadge } from '../gaps/GapBadge';

interface SessionHeaderProps {
  onMicToggle?: () => void;
  onWakeWordToggle?: () => void;
  isWakeWordActive?: boolean;
  onFinalize?: () => void;
}

export function SessionHeader({ onMicToggle, onWakeWordToggle, isWakeWordActive, onFinalize }: SessionHeaderProps) {
  const { sessionId, status } = useSessionStore();

  return (
    <header className="flex h-16 shrink-0 items-center gap-3 border-b border-gray-800 bg-gray-900 px-4">
      <div className="flex items-center gap-2 shrink-0">
        <Activity size={18} className="text-blue-400" />
        <span className="text-sm font-semibold text-gray-100">Voice-to-PCR</span>
        {sessionId && (
          <span className="rounded bg-gray-800 px-2 py-0.5 text-xs text-gray-400 font-mono">
            {sessionId.slice(0, 12)}
          </span>
        )}
      </div>

      <CompletenessBar />

      <div className="flex items-center gap-3 shrink-0">
        <GapBadge />
        {status !== 'finalized' && (
          <button
            onClick={onFinalize}
            className="flex items-center gap-1.5 rounded-lg bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700 transition-colors"
          >
            <CheckCircle2 size={14} />
            Finalize
          </button>
        )}
        {status === 'finalized' && (
          <span className="flex items-center gap-1 rounded-full bg-green-500/20 px-3 py-1 text-xs font-medium text-green-400">
            <CheckCircle2 size={12} />
            Finalized
          </span>
        )}
        <MicButton
          onToggle={onMicToggle}
          onWakeWordToggle={onWakeWordToggle}
          isWakeWordActive={isWakeWordActive}
        />
      </div>
    </header>
  );
}