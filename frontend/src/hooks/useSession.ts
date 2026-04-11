/**
 * useSession — real session lifecycle hook.
 *
 * On mount:
 *   1. POST /api/v1/sessions  → get session_id + initial PCR state
 *   2. Open WebSocket /ws/session/{id} → stream server-pushed updates
 *
 * Exposes:
 *   - sendMessage: send any WS message to the backend
 *   - sendCorrection: POST /correct for a field edit, syncs backend state
 *   - handleFinalize: POST /finalize
 */

import { useCallback, useEffect, useRef } from 'react';
import { usePCRStore } from '../store/pcrStore';
import { useSessionStore } from '../store/sessionStore';
import { useUIStore } from '../store/uiStore';
import type { PCRStateEnvelope } from '../types/pcr';
import type { GapDetectionResult } from '../types/session';
import type { WSClientMessage, WSServerMessage } from '../types/websocket';

export function useSession() {
  const { applyServerState } = usePCRStore();
  const { setSession, setConnectionState, finalizeSession } = useSessionStore();
  const { setGaps, setGapPanelOpen } = useUIStore();

  const wsRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      setConnectionState('connecting');
      try {
        // Create a new session
        const res = await fetch('/api/v1/sessions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({}),
        });
        if (!res.ok) throw new Error(`Session creation failed: ${res.status}`);
        const data = await res.json();

        if (cancelled) return;

        const sid: string = data.session_id;
        sessionIdRef.current = sid;
        setSession(sid, data.incident_id ?? undefined);
        if (data.pcr_state) applyServerState(data.pcr_state as PCRStateEnvelope);

        // Open WebSocket for real-time server-pushed state updates
        const ws = new WebSocket(`/ws/session/${sid}`);
        wsRef.current = ws;

        ws.onopen = () => setConnectionState('connected');
        ws.onclose = () => { if (!cancelled) setConnectionState('disconnected'); };
        ws.onerror = () => { if (!cancelled) setConnectionState('error'); };

        ws.onmessage = (event: MessageEvent) => {
          try {
            const msg: WSServerMessage = JSON.parse(event.data as string);
            handleServerMessage(msg);
          } catch {
            console.error('[useSession] WS parse error');
          }
        };
      } catch (err) {
        console.error('[useSession] init error', err);
        if (!cancelled) setConnectionState('error');
      }
    }

    function handleServerMessage(msg: WSServerMessage) {
      switch (msg.type) {
        case 'pcr_state':
        case 'correction_applied':
          if (msg.payload) applyServerState(msg.payload as unknown as PCRStateEnvelope);
          break;
        case 'extraction_update':
          // pcr_state follows immediately after; no action needed
          break;
        case 'gap_alert': {
          const gaps = msg.payload as unknown as GapDetectionResult;
          setGaps(gaps);
          if (gaps.missing_mandatory?.length > 0 || gaps.missing_required?.length > 0) {
            setGapPanelOpen(true);
          }
          break;
        }
        case 'transcript_partial':
          // handled by useAudioCapture
          break;
        case 'transcript_final':
          // handled by useAudioCapture
          break;
        case 'error':
          console.error('[useSession] server error:', msg.payload);
          break;
      }
    }

    init();

    return () => {
      cancelled = true;
      wsRef.current?.close();
    };
  }, []);

  /** Send a typed message over WebSocket. */
  const sendMessage = (msg: WSClientMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  };

  /**
   * Send a field correction to the backend via REST and apply the returned state.
   * Called after commitFieldEdit so local state is already updated optimistically.
   */
  const sendCorrection = useCallback(async (fieldKey: string, value: unknown) => {
    const sid = sessionIdRef.current;
    if (!sid) return;

    const utterance = `Update ${fieldKey} to ${JSON.stringify(value)}`;
    try {
      const res = await fetch(`/api/v1/sessions/${sid}/correct`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ utterance }),
      });
      if (!res.ok) return;
      const data = await res.json();
      if (data.pcr_state) applyServerState(data.pcr_state as PCRStateEnvelope);
    } catch (err) {
      console.error('[useSession] correction error', err);
    }
  }, [applyServerState]);

  /** Finalize the session on the backend. */
  /** Finalize the session and download PCR JSON. */
  const handleFinalize = async () => {
    const sid = sessionIdRef.current;
    if (sid) {
      await fetch(`/api/v1/sessions/${sid}/finalize`, { method: 'POST' }).catch(() => {});

      // Download PCR as JSON
      try {
        const res = await fetch(`/api/v1/sessions/${sid}/pcr/json`);
        if (res.ok) {
          const pcr = await res.json();
          const blob = new Blob([JSON.stringify(pcr, null, 2)], { type: 'application/json' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `pcr_${sid.slice(0, 8)}.json`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
        }
      } catch (err) {
        console.error('[useSession] PCR export error', err);
      }
    }
    finalizeSession();
  };

  return { sendMessage, sendCorrection, handleFinalize, sessionIdRef };
}
