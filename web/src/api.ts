// The app's data layer. In this browser-only build there is no server: `local/backend.ts`
// does what server/api.py did, in the page and in Web Workers, and keeps everything in
// IndexedDB. Paths under /api/ still name files (audio, models, bundles) for `fetchBytes`.

import { ApiError } from "./errors";
import { local, localDevice } from "./local/backend";

export { ApiError };

export type SampleKind = "positive" | "other_speech" | "ambience";
export type TrainingStatus = "queued" | "running" | "done" | "failed" | "cancelled";

export interface Language {
  code: string;
  name: string;
  native_name: string;
  rtl: boolean;
  needs_review: boolean;
  paragraph: string;
}

export interface Sample {
  id: string;
  upload_id: string;
  kind: SampleKind;
  start_s: number;
  duration_s: number;
  speech_s: number;
  peak_dbfs: number;
  rms_dbfs: number;
  accepted: boolean;
  reason: "too_long" | "too_short" | "clipped" | "too_quiet" | null;
  similarity: number | null;
  outlier: boolean;
  created_at: string;
}

export interface FileVerification {
  hours: number;
  no_gate: { false_accepts: number; per_hour: number };
  gate: { false_accepts: number; per_hour: number };
  enrollment_detected: number;
  enrollment_total: number;
}

export interface OperatingPoint {
  fa_per_hour: number;
  fp32: { threshold: number; calibration_fa_per_hour: number };
  int8?: { threshold: number; calibration_fa_per_hour: number }; // server builds only
}

export interface Report {
  wake_word: string;
  language: string;
  base_model_version: string;
  recordings: number;
  negative_seconds: Record<string, number>;
  threshold: number;
  target_fa_per_hour: number;
  calibration: {
    hours: number;
    fp32: { threshold: number; fa_per_hour: number; target_met: boolean };
    int8?: { threshold: number; fa_per_hour: number; target_met: boolean; own_cache: boolean };
  };
  operating_points: OperatingPoint[];
  cross_validation: null | {
    folds: number;
    recall_noisy: number;
    recall_clean: number;
    per_recording: { sample_id: string; recall_noisy: number; recall_clean: number; weak: boolean }[];
  };
  verification: { fp32: FileVerification; int8?: FileVerification }; // the browser builds model.onnx only
  timings_s: Record<string, number>;
}

export interface Training {
  id: string;
  wakeword_id: string;
  mode: string;
  target_fa_per_hour: number;
  base_version: string;
  status: TrainingStatus;
  stage: string;
  progress: number;
  error: string | null;
  report: Report | null;
  artifacts: string[];
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface WakeWord {
  id: string;
  name: string;
  language: string;
  created_at: string;
  samples: Sample[];
  trainings: Training[];
}

export interface BaseModelInfo {
  version: string;
  created_at: string;
  default: boolean;
}

export interface ProgressEvent {
  stage: string;
  progress: number;
  message: string;
  at: string;
}

export const api = {
  languages: local.languages,
  baseModels: local.baseModels,
  createWakeWord: local.createWakeWord,
  wakeWord: local.wakeWord,
  deleteWakeWord: local.deleteWakeWord,
  upload: (id: string, kind: SampleKind, file: Blob, _filename: string) => local.upload(id, kind, file),
  deleteSample: local.deleteSample,
  train: local.train,
  training: local.training,
};

export const sampleAudioUrl = (wakeWordId: string, sampleId: string) =>
  `/api/wakewords/${wakeWordId}/samples/${sampleId}/audio`;
export const artifactUrl = (trainingId: string, name: string) => `/api/trainings/${trainingId}/artifacts/${name}`;
export const eventsUrl = (trainingId: string) => `/api/trainings/${trainingId}/events`;
export const bundleUrl = (training: Training, faPerHour: number) =>
  faPerHour === training.target_fa_per_hour
    ? artifactUrl(training.id, "bundle.zip")
    : `/api/trainings/${training.id}/bundle.zip?fa_per_hour=${faPerHour}`;

/** Bytes of a stored file (an /api/ path) or of any other URL. */
export async function fetchBytes(url: string): Promise<ArrayBuffer> {
  if (url.startsWith("/api/")) return local.bytes(url);
  const res = await fetch(url);
  if (!res.ok) throw new ApiError(res.status, `${url}: ${res.status}`);
  return res.arrayBuffer();
}

// Device measurements (PLAN.md M4): benchmarks and field tests, stored with the training.

export interface FieldReport {
  wake_word: string | null;
  model: string;
  host: Record<string, unknown> | null;
  gate: boolean;
  threshold: number;
  hours: number;
  speech_hours: number;
  triggers: number;
  false_accepts: number;
  false_accepts_per_hour: number | null;
  busy_pct: { mean: number | null; max: number | null };
  latency_ms: { p50: number | null; p95: number | null; max: number | null };
  battery: { start: number; end: number; per_hour: number } | null;
  sweep: { factor: number; threshold: number; triggers: number; per_hour: number | null }[];
  events: { n: number; time: number; score: number; clip?: string; false_accept: boolean }[];
}

export interface DeviceRunInfo {
  id: string;
  kind: "benchmark" | "fieldtest";
  finished: boolean;
  created_at: string;
  updated_at: string;
  data?: Record<string, unknown>;
  summary?: Pick<FieldReport, "hours" | "triggers" | "false_accepts_per_hour" | "busy_pct" | "latency_ms" | "battery" | "host">;
}

export const deviceApi = localDevice;

export const fieldClipUrl = (runId: string, clip: string) => `/api/fieldtests/${runId}/clips/${clip.split("/").pop()}`;
