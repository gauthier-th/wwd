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

// Messages between the main thread and recorder.worker.ts.

export type ToRecorder =
  | { type: "init"; vad: ArrayBuffer }
  | { type: "port"; port: MessagePort } // 512-sample 16 kHz frames from the capture worklet
  | { type: "stop" };

export interface RecorderStatus {
  seconds: number; // audio recorded so far
  levelDb: number; // peak over the last tick, dBFS
  vad: number; // max VAD probability over the last tick
  speaking: boolean;
  utterances: number; // closed speech regions, cut as the server will cut them
  speechS: number; // total speech so far
  currentS: number; // length of the utterance in progress
  longestS: number; // longest closed utterance
  sinceSpeechS: number; // time since a frame was last above the VAD threshold
}

export type FromRecorder =
  | { type: "ready" }
  | ({ type: "status" } & RecorderStatus)
  | ({ type: "done"; samples: Float32Array } & RecorderStatus)
  | { type: "error"; message: string };
