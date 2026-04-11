import { Mic, MicOff, Loader2, Radio } from 'lucide-react';
import { useUIStore } from '../../store/uiStore';

interface MicButtonProps {
  onToggle?: () => void;
  onWakeWordToggle?: () => void;
  isWakeWordActive?: boolean;
}

export function MicButton({ onToggle, onWakeWordToggle, isWakeWordActive }: MicButtonProps) {
  const micStatus = useUIStore((s) => s.micStatus);
  const wakeWordStatus = useUIStore((s) => s.wakeWordStatus);
  const isActive = micStatus === 'active';
  const isProcessing = micStatus === 'processing';
  const isTriggered = wakeWordStatus === 'triggered';

  return (
    <div className="flex items-center gap-2">
      {/* Wake word toggle */}
      {/* Wake word toggle */}
      <button
        onClick={onWakeWordToggle}
        className={`relative flex h-10 w-10 items-center justify-center rounded-full transition-all
          ${isWakeWordActive
            ? 'bg-green-600 hover:bg-green-700'
            : 'bg-gray-700 hover:bg-gray-600'}
        `}
        title={isWakeWordActive ? 'Disable "Hey MEDIC"' : 'Enable "Hey MEDIC"'}
      >
        <Radio
          size={16}
          className={isWakeWordActive ? 'text-white' : 'text-gray-400'}
        />
      </button>

      {/* Push-to-talk mic */}
      <button
        onClick={onToggle}
        className={`relative flex h-12 w-12 items-center justify-center rounded-full transition-all
          ${isActive || isTriggered
            ? 'bg-red-600 hover:bg-red-700'
            : 'bg-gray-700 hover:bg-gray-600'}
          ${isProcessing ? 'cursor-wait opacity-75' : 'cursor-pointer'}
        `}
        title={
          isActive
            ? 'Stop recording'
            : isProcessing
              ? 'Processing...'
              : 'Start recording (or say "Hey MEDIC")'
        }
      >
        {(isActive || isTriggered) && (
          <span className="absolute inset-0 animate-ping rounded-full bg-red-600 opacity-40" />
        )}
        {isProcessing ? (
          <Loader2 size={20} className="animate-spin text-white" />
        ) : isActive || isTriggered ? (
          <Mic size={20} className="text-white" />
        ) : (
          <MicOff size={20} className="text-gray-300" />
        )}
      </button>
    </div>
  );
}