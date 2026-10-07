// Typed client for the server API (server/api.py, PLAN.md §9.1). Same origin: Vite proxies
// /api in development, and the server serves the built app in production.

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
  int8: { threshold: number; calibration_fa_per_hour: number };
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
    int8: { threshold: number; fa_per_hour: number; target_met: boolean; own_cache: boolean };
  };
  operating_points: OperatingPoint[];
  cross_validation: null | {
    folds: number;
    recall_noisy: number;
    recall_clean: number;
    per_recording: { sample_id: string; recall_noisy: number; recall_clean: number; weak: boolean }[];
  };
  verification: { fp32: FileVerification; int8: FileVerification };
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

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(detail);
  }
}

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers = { "content-type": "application/json" };
  }
  const res = await fetch(`/api${path}`, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch {
      // not JSON: keep the status line
    }
    throw new ApiError(res.status, detail);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export const api = {
  languages: () => call<Language[]>("GET", "/languages"),
  baseModels: () => call<BaseModelInfo[]>("GET", "/base-models"),
  createWakeWord: (name: string, language: string) =>
    call<WakeWord>("POST", "/wakewords", { name, language }),
  wakeWord: (id: string) => call<WakeWord>("GET", `/wakewords/${id}`),
  deleteWakeWord: (id: string) => call<void>("DELETE", `/wakewords/${id}`),
  upload: (id: string, kind: SampleKind, file: Blob, filename: string) => {
    const form = new FormData();
    form.append("kind", kind);
    form.append("file", file, filename);
    return call<{ upload_id: string; samples: Sample[] }>("POST", `/wakewords/${id}/samples`, form);
  },
  deleteSample: (id: string, sampleId: string) => call<void>("DELETE", `/wakewords/${id}/samples/${sampleId}`),
  train: (id: string, targetFaPerHour: number) =>
    call<Training>("POST", `/wakewords/${id}/trainings`, { mode: "quick", target_fa_per_hour: targetFaPerHour }),
  training: (id: string) => call<Training>("GET", `/trainings/${id}`),
};

export const sampleAudioUrl = (wakeWordId: string, sampleId: string) =>
  `/api/wakewords/${wakeWordId}/samples/${sampleId}/audio`;
export const artifactUrl = (trainingId: string, name: string) => `/api/trainings/${trainingId}/artifacts/${name}`;
export const eventsUrl = (trainingId: string) => `/api/trainings/${trainingId}/events`;
export const bundleUrl = (training: Training, faPerHour: number) =>
  faPerHour === training.target_fa_per_hour
    ? artifactUrl(training.id, "bundle.zip")
    : `/api/trainings/${training.id}/bundle.zip?fa_per_hour=${faPerHour}`;

export async function fetchBytes(url: string): Promise<ArrayBuffer> {
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

export const deviceApi = {
  runs: (trainingId: string) => call<DeviceRunInfo[]>("GET", `/trainings/${trainingId}/device-runs`),
  saveBenchmark: (trainingId: string, body: unknown) => call<{ id: string }>("POST", `/trainings/${trainingId}/benchmarks`, body),
  startFieldTest: (trainingId: string, body: { model: string; gate: boolean; fa_per_hour: number; device: unknown }) =>
    call<{ id: string; threshold: number }>("POST", `/trainings/${trainingId}/fieldtests`, body),
  chunk: (runId: string, data: unknown, ticks: Float32Array, clips: Blob[]) => {
    const form = new FormData();
    form.append("data", JSON.stringify(data));
    if (ticks.length) form.append("ticks", new Blob([ticks as Float32Array<ArrayBuffer>]), "ticks.f32");
    clips.forEach((clip, i) => form.append("clips", clip, `clip-${i}.wav`));
    return call<{ events: number }>("POST", `/fieldtests/${runId}/chunk`, form);
  },
  finish: (runId: string) => call<FieldReport>("POST", `/fieldtests/${runId}/finish`),
  report: (runId: string) => call<FieldReport>("GET", `/fieldtests/${runId}/report`),
};

export const fieldClipUrl = (runId: string, clip: string) => `/api/fieldtests/${runId}/clips/${clip.split("/").pop()}`;
