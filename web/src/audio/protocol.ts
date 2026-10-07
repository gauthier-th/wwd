// Messages between the main thread and detector.worker.ts.

import type { DetectorConfig, Tick, WakeEvent } from "@wwd/runtime";

// Each init starts a new Detector, whose tick numbering restarts at 0: messages carry the
// generation of the detector that produced them, so the page can drop stale ones.
export type ToWorker =
  | {
      type: "init";
      generation: number;
      model: ArrayBuffer;
      vad: ArrayBuffer | null;
      gate: boolean;
      overrides?: Partial<DetectorConfig>;
      clips?: { beforeS: number; afterS: number }; // send the audio around each detection
    }
  | { type: "audio"; samples: Float32Array } // 16 kHz, any length
  | { type: "port"; port: MessagePort } // direct line from the capture worklet
  | { type: "reset" };

export type FromWorker =
  | { type: "ready"; generation: number; config: DetectorConfig; metadata: Record<string, string> }
  | {
      type: "ticks";
      generation: number;
      ticks: Tick[];
      events: WakeEvent[];
      samples: number;
      processMs: number[]; // per tick: VAD frames + wake model, in this worker
      queued: number; // chunks waiting behind this one: > 0 means falling behind
    }
  | { type: "clip"; generation: number; tick: number; samples: Float32Array }
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
