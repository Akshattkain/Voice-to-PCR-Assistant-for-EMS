/**
 * useAudioCapture — connects mic to real audio recording.
 *
 * Supports push-to-talk AND wake-word-triggered auto-recording.
 * When wake word triggers, recording auto-stops after 2s of silence.
 * Push-to-talk still works: click to start, click to stop.
 */

import { useEffect, useRef } from 'react';
import type { RefObject } from 'react';
import { usePCRStore } from '../store/pcrStore';
import { useTranscriptStore } from '../store/transcriptStore';
import { useUIStore } from '../store/uiStore';
import type { PCRStateEnvelope } from '../types/pcr';
import type { GapDetectionResult, TranscriptSegment } from '../types/session';

const SILENCE_THRESHOLD = 0.01;
const SILENCE_DURATION_MS = 2000;

export function useAudioCapture(sessionIdRef: RefObject<string | null>) {
  const { micStatus, setMicStatus, wakeWordStatus, setWakeWordStatus, setGaps, setGapPanelOpen } = useUIStore();
  const { addSegment, setPartial } = useTranscriptStore();
  const { applyServerState } = usePCRStore();

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const wakeTriggeredRef = useRef(false);

  useEffect(() => {
    wakeTriggeredRef.current = wakeWordStatus === 'triggered';
  }, [wakeWordStatus]);

  useEffect(() => {
    if (micStatus === 'active') {
      startRecording();
    } else if (micStatus === 'processing') {
      stopAndProcess();
    } else if (micStatus === 'idle' && recorderRef.current?.state === 'recording') {
      // Cancel — stop recorder, discard audio
      cleanup();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [micStatus]);

  const cleanup = () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    if (recorderRef.current?.state === 'recording') {
      recorderRef.current.stop();
    }
    recorderRef.current = null;
    streamRef.current = null;
    analyserRef.current = null;
    chunksRef.current = [];
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const mimeType = MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : 'audio/mp4';
      const recorder = new MediaRecorder(stream, { mimeType });
      recorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.start();

      // Set up VAD for auto-stop (only when wake word triggered)
      if (wakeTriggeredRef.current) {
        setupVAD(stream);
      }
    } catch (err) {
      console.error('[useAudioCapture] mic access denied', err);
      setMicStatus('idle');
    }
  };

  const setupVAD = (stream: MediaStream) => {
    const audioCtx = new AudioContext();
    const source = audioCtx.createMediaStreamSource(stream);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);
    analyserRef.current = analyser;

    const dataArray = new Float32Array(analyser.fftSize);
    let speaking = false;

    const checkAudio = () => {
      if (!analyserRef.current) return;

      analyser.getFloatTimeDomainData(dataArray);

      // RMS volume
      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) {
        sum += dataArray[i] * dataArray[i];
      }
      const rms = Math.sqrt(sum / dataArray.length);

      if (rms > SILENCE_THRESHOLD) {
        speaking = true;
        // Clear any pending silence timer
        if (silenceTimerRef.current) {
          clearTimeout(silenceTimerRef.current);
          silenceTimerRef.current = null;
        }
      } else if (speaking && !silenceTimerRef.current) {
        // Started being silent after speaking — start countdown
        silenceTimerRef.current = setTimeout(() => {
          console.log('[VAD] Silence detected, auto-stopping recording');
          if (useUIStore.getState().micStatus === 'active') {
            setMicStatus('processing');
          }
        }, SILENCE_DURATION_MS);
      }

      rafRef.current = requestAnimationFrame(checkAudio);
    };

    rafRef.current = requestAnimationFrame(checkAudio);
  };

  const stopAndProcess = async () => {
    // Clean up VAD
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    rafRef.current = null;
    silenceTimerRef.current = null;

    const recorder = recorderRef.current;
    if (!recorder) {
      setMicStatus('idle');
      return;
    }

    await new Promise<void>((resolve) => {
      recorder.onstop = () => resolve();
      recorder.stop();
      streamRef.current?.getTracks().forEach((t) => t.stop());
    });

    const sid = sessionIdRef.current;
    if (!sid || chunksRef.current.length === 0) {
      setMicStatus('idle');
      return;
    }

    const mimeType = chunksRef.current[0]?.type || 'audio/webm';
    const ext = mimeType.includes('mp4') ? 'mp4' : 'webm';
    const audioBlob = new Blob(chunksRef.current, { type: mimeType });
    chunksRef.current = [];

    try {
      setPartial('Transcribing...');

      // 1. Transcribe
      const formData = new FormData();
      formData.append('file', audioBlob, `audio.${ext}`);

      const transcribeRes = await fetch(`/api/v1/sessions/${sid}/transcribe`, {
        method: 'POST',
        body: formData,
      });

      if (!transcribeRes.ok) throw new Error(`Transcribe failed: ${transcribeRes.status}`);
      const transcribeData = await transcribeRes.json();
      const transcript: string = transcribeData.transcript_text ?? '';

      setPartial(null);
      const seg: TranscriptSegment = {
        text: transcript,
        start_time: 0,
        end_time: transcribeData.duration_sec ?? 0,
        timestamp: new Date().toISOString(),
      };
      addSegment(seg);

      // 2. Extract PCR fields
      setPartial('Extracting PCR fields...');
      const extractRes = await fetch(`/api/v1/sessions/${sid}/extract`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript }),
      });

      if (extractRes.ok) {
        const extractData = await extractRes.json();
        if (extractData.pcr_state) {
          applyServerState(extractData.pcr_state as PCRStateEnvelope);
        }
      }

      // 3. Fetch gaps
      const gapRes = await fetch(`/api/v1/sessions/${sid}/gaps`);
      if (gapRes.ok) {
        const gaps: GapDetectionResult = await gapRes.json();
        setGaps(gaps);
        if (gaps.missing_mandatory.length > 0 || gaps.missing_required.length > 0) {
          setGapPanelOpen(true);
        }
      }
    } catch (err) {
      console.error('[useAudioCapture] processing error', err);
    } finally {
      setPartial(null);
      setMicStatus('idle');
      setWakeWordStatus('off');
    }
  };

  useEffect(() => {
    return () => {
      cleanup();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}