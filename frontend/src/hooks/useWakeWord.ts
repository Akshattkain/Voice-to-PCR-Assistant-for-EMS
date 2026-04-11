/**
 * useWakeWord — Web Speech API continuous listener for "Hey MEDIC".
 *
 * Listens in the background for the wake phrase. On detection:
 *   1. Sets wakeWordStatus → 'triggered'
 *   2. Auto-starts recording (sets micStatus → 'active')
 *   3. Resumes listening after processing completes
 *
 * Only the 2-word trigger phrase goes through Web Speech API.
 * All clinical audio goes through local Whisper — no PHI leaves the device.
 */

import { useEffect, useRef, useCallback } from 'react';
import { useUIStore } from '../store/uiStore';

const WAKE_PHRASES = ['hey medic', 'hey medik', 'a medic', 'heymatic'];

export function useWakeWord() {
  const {
    micStatus,
    wakeWordStatus,
    setMicStatus,
    setWakeWordStatus,
  } = useUIStore();

  const recognitionRef = useRef<any>(null);
  const enabledRef = useRef(false);

  const startListening = useCallback(() => {
    const SpeechRecognition =
      window.SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      console.warn('[WakeWord] Web Speech API not supported');
      return;
    }

    if (recognitionRef.current) {
      try { recognitionRef.current.abort(); } catch {}
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      // Check all results for wake phrase
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const text = event.results[i][0].transcript.toLowerCase().trim();

        const detected = WAKE_PHRASES.some(
          (phrase) => text.includes(phrase)
        );

        if (detected && micStatus === 'idle') {
          console.log('[WakeWord] Detected! Triggering recording...');
          setWakeWordStatus('triggered');

          // Brief flash before starting recording
          setTimeout(() => {
            setMicStatus('active');
          }, 200);

          // Pause wake word listening while recording
          try { recognition.stop(); } catch {}
          return;
        }
      }
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      // 'no-speech' and 'aborted' are normal — just restart
      if (event.error === 'no-speech' || event.error === 'aborted') {
        return;
      }
      console.error('[WakeWord] error:', event.error);
    };

    recognition.onend = () => {
      // Auto-restart if still enabled and not recording
      if (enabledRef.current && useUIStore.getState().micStatus === 'idle') {
        setTimeout(() => {
          if (enabledRef.current) {
            try { recognition.start(); } catch {}
          }
        }, 300);
      }
    };

    recognitionRef.current = recognition;

    try {
      recognition.start();
      setWakeWordStatus('listening');
    } catch (err) {
      console.error('[WakeWord] failed to start:', err);
    }
  }, [micStatus, setMicStatus, setWakeWordStatus]);

  const stopListening = useCallback(() => {
    enabledRef.current = false;
    if (recognitionRef.current) {
      try { recognitionRef.current.abort(); } catch {}
      recognitionRef.current = null;
    }
    setWakeWordStatus('off');
  }, [setWakeWordStatus]);

  const toggleWakeWord = useCallback(() => {
    if (enabledRef.current) {
      stopListening();
    } else {
      enabledRef.current = true;
      startListening();
    }
  }, [startListening, stopListening]);

  // Resume listening after processing completes
  useEffect(() => {
    if (micStatus === 'idle' && enabledRef.current && wakeWordStatus !== 'listening') {
      setTimeout(() => {
        if (enabledRef.current) {
          startListening();
        }
      }, 500);
    }
  }, [micStatus, wakeWordStatus, startListening]);

  // Cleanup
  useEffect(() => {
    return () => {
      enabledRef.current = false;
      if (recognitionRef.current) {
        try { recognitionRef.current.abort(); } catch {}
      }
    };
  }, []);

  return {
    toggleWakeWord,
    isWakeWordActive: wakeWordStatus !== 'off',
  };
}