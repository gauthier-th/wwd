// Messages between the main thread and detector.worker.ts.

import type { DetectorConfig, Tick, WakeEvent } from "@wwd/runtime";

export type ToWorker =
  | {
      type: "init";
      model: ArrayBuffer;
      vad: ArrayBuffer | null;
      gate: boolean;
      overrides?: Partial<DetectorConfig>;
    }
  | { type: "audio"; samples: Float32Array } // 16 kHz, any length
  | { type: "port"; port: MessagePort } // direct line from the capture worklet
  | { type: "reset" };

export type FromWorker =
  | { type: "ready"; config: DetectorConfig; metadata: Record<string, string> }
  | { type: "ticks"; ticks: Tick[]; events: WakeEvent[]; samples: number }
  | { type: "error"; message: string };
