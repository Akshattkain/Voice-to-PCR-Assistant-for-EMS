import { useEffect } from 'react';
import { useUIStore } from '../store/uiStore';
import { usePCRStore } from '../store/pcrStore';
import { useSession } from '../hooks/useSession';
import { useAudioCapture } from '../hooks/useAudioCapture';
import { useWakeWord } from '../hooks/useWakeWord';
import { AppShell } from '../components/layout/AppShell';
import { TwoColumnLayout } from '../components/layout/TwoColumnLayout';
import { SessionHeader } from '../components/session/SessionHeader';
import { TranscriptPanel } from '../components/transcript/TranscriptPanel';
import { PCRForm } from '../components/pcr/PCRForm';

export function SessionPage() {
  const { sendCorrection, handleFinalize, sessionIdRef } = useSession();
  useAudioCapture(sessionIdRef);
  const { toggleWakeWord, isWakeWordActive } = useWakeWord();
  const { micStatus, setMicStatus } = useUIStore();
  const { setOnCommit } = usePCRStore();

  useEffect(() => {
    setOnCommit((fieldKey, value) => {
      sendCorrection(fieldKey, value);
    });
    return () => setOnCommit(null);
  }, [sendCorrection, setOnCommit]);

  const handleMicToggle = () => {
    if (micStatus === 'idle') setMicStatus('active');
    else if (micStatus === 'active') setMicStatus('processing');
  };

  return (
    <AppShell>
      <SessionHeader
        onMicToggle={handleMicToggle}
        onWakeWordToggle={toggleWakeWord}
        isWakeWordActive={isWakeWordActive}
        onFinalize={handleFinalize}
      />
      <TwoColumnLayout
        left={<TranscriptPanel />}
        right={<PCRForm />}
      />
    </AppShell>
  );
}