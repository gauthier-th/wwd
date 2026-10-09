// Messages between the page (local/backend.ts), the compute worker (backend.worker.ts) and
// its embedding workers (embed.worker.ts).

import type { Manifest } from "./assets";
import type { Head } from "./finetune";
import type { Report } from "../api";
import type { TrainJob } from "./train";

export interface ClipOut {
  audio: Float32Array;
  start_s: number;
  speech_s: number;
  accepted: boolean;
  reason: string | null;
}

export type Request =
  | { type: "init" }
  | { type: "checks"; kind: "positive" | "other_speech" | "ambience"; audio: Float32Array }
  | { type: "embed_clean"; clips: Float32Array[] }
  | { type: "train"; trainingId: string; job: TrainJob };

export interface Responses {
  init: { version: string; manifest: Manifest };
  checks: { clips: ClipOut[] };
  embed_clean: { embeddings: Float32Array[] };
  train: { report: Report; head: Head; model: Uint8Array };
}

export type ToBackend = (Request & { id: number }) | { type: "cancel"; trainingId: string };

export type FromBackend =
  | { type: "reply"; id: number; ok: true; result: unknown }
  | { type: "reply"; id: number; ok: false; status: number; detail: string }
  | { type: "progress"; trainingId: string; stage: string; progress: number; message: string }
  | { type: "download"; loaded: number; total: number };

export type ToEmbedder =
  | { type: "init"; root: string; manifest: Manifest }
  | { type: "embed"; id: number; signals: Float32Array[]; first: number; augment: boolean; seed: number; hop: number };

export type FromEmbedder =
  | { type: "ready" }
  | { type: "result"; id: number; embeddings: Float32Array }
  | { type: "error"; id: number | null; message: string };
