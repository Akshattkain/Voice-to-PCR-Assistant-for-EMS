import { create } from 'zustand';
import type { NEMSISSection } from '../types/nemsis';
import type { GapDetectionResult } from '../types/session';

export type MicStatus = 'idle' | 'active' | 'processing';
export type WakeWordStatus = 'off' | 'listening' | 'triggered';

interface UIState {
  micStatus: MicStatus;
  wakeWordStatus: WakeWordStatus;
  activeSection: NEMSISSection | null;
  editingField: string | null;
  gapPanelOpen: boolean;
  gaps: GapDetectionResult | null;

  setMicStatus: (s: MicStatus) => void;
  setWakeWordStatus: (s: WakeWordStatus) => void;
  setActiveSection: (s: NEMSISSection | null) => void;
  setEditingField: (key: string | null) => void;
  toggleGapPanel: () => void;
  setGapPanelOpen: (open: boolean) => void;
  setGaps: (gaps: GapDetectionResult) => void;
}

export const useUIStore = create<UIState>((set) => ({
  micStatus: 'idle',
  wakeWordStatus: 'off',
  activeSection: null,
  editingField: null,
  gapPanelOpen: false,
  gaps: null,

  setMicStatus: (micStatus) => set({ micStatus }),
  setWakeWordStatus: (wakeWordStatus) => set({ wakeWordStatus }),
  setActiveSection: (activeSection) => set({ activeSection }),
  setEditingField: (editingField) => set({ editingField }),
  toggleGapPanel: () => set((s) => ({ gapPanelOpen: !s.gapPanelOpen })),
  setGapPanelOpen: (gapPanelOpen) => set({ gapPanelOpen }),
  setGaps: (gaps) => set({ gaps }),
}));