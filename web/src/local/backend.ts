// The browser-only backend: what server/api.py and server/device.py did, on IndexedDB and
// Web Workers. Same operations, validations and error statuses, so the pages are unchanged.
// Paths under /api/ name stored files (sample audio, model files, bundles, clips) and resolve
// through `bytes`.

import { configFromMetadata, decodeWav, encodeWav, readOnnxMetadata } from "@wwd/runtime";

import type {
  BaseModelInfo, DeviceRunInfo, FieldReport, Language, ProgressEvent, Report, Sample, SampleKind, Training, WakeWord,
} from "../api";
import { ApiError } from "../errors";
import { bundleZip } from "./bundle";
import * as db from "./db";
import { MAX_UPLOAD_S, outliers } from "./checks";
import { summarize, type FieldEvent, type FieldRun, type FieldStats } from "./fieldreport";
import type { Head } from "./finetune";
import { browserFetcher, loadManifest } from "./fetcher";
import type { Manifest } from "./assets";
import type { FromBackend, Request, Responses, ToBackend } from "./protocol";
import { composeModel } from "./compose";
import { modelMetadata, type TrainJob } from "./train";

const MIN_POSITIVES = 3;
const MAX_UPLOAD_MB = 25;

// ------------------------------------------------------------------ compute worker

let worker: Worker | null = null;
let nextId = 0;
const pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
const downloadListeners = new Set<(loaded: number, total: number) => void>();

function compute(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL("./backend.worker.ts", import.meta.url), { type: "module" });
  worker.onmessage = (e: MessageEvent<FromBackend>) => {
    const msg = e.data;
    if (msg.type === "reply") {
      const p = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.ok) p?.resolve(msg.result);
      else p?.reject(new ApiError(msg.status, msg.detail));
    } else if (msg.type === "progress") {
      emit(msg.trainingId, { stage: msg.stage, progress: msg.progress, message: msg.message, at: now() });
    } else {
      for (const fn of downloadListeners) fn(msg.loaded, msg.total);
    }
  };
  return worker;
}

function rpc<T extends Request["type"]>(req: Extract<Request, { type: T }>, transfer: Transferable[] = []):
  Promise<Responses[T]> {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
    compute().postMessage({ ...req, id } as ToBackend, transfer);
  });
}

/** First-download progress of the base model's files (only reported when not cached). */
export function onDownload(fn: (loaded: number, total: number) => void): () => void {
  downloadListeners.add(fn);
  return () => downloadListeners.delete(fn);
}

// ------------------------------------------------------------------ helpers

const now = () => new Date().toISOString();
const newId = () => (crypto.randomUUID?.() ?? Array.from(crypto.getRandomValues(new Uint8Array(16)),
  (b) => b.toString(16).padStart(2, "0")).join("")).replaceAll("-", "");

let base: Promise<{ root: URL; manifest: Manifest }> | null = null;
const getBase = () => (base ??= loadManifest().catch((err) => {
  base = null;
  throw new ApiError(503, `the base model isn't available: ${err.message}`);
}));
const getManifest = async () => (await getBase()).manifest;
const baseFetcher = async () => {
  const { root, manifest: m } = await getBase();
  return browserFetcher(root, m);
};

interface WakeWordRow { id: string; name: string; language: string; created_at: string }
interface SampleRow extends Sample { wakeword_id: string }
interface TrainingRow extends Training { sample_ids: string[] }
interface RunRow { id: string; training_id: string; kind: "benchmark" | "fieldtest"; data?: Record<string, unknown>;
                   meta?: FieldRun["meta"]; events: number; finished: boolean; created_at: string; updated_at: string }

const sampleKey = (id: string) => `sample/${id}`;
const artifactKey = (tid: string, name: string) => `${tid}/${name}`;
const publicSample = ({ wakeword_id: _, ...s }: SampleRow): Sample => s;
const publicTraining = ({ sample_ids: _, ...t }: TrainingRow): Training => t;

async function getWakeWordRow(id: string): Promise<WakeWordRow> {
  const ww = await db.get<WakeWordRow>("wakewords", id);
  if (!ww) throw new ApiError(404, "wake word not found");
  return ww;
}

// Files are stored as ArrayBuffers, not Blobs: WebKit can't put Blobs in IndexedDB in
// ephemeral sessions (Safari's private browsing, automated tests).
async function putFile(key: string, data: Uint8Array | Blob | string): Promise<void> {
  const bytes = typeof data === "string" ? new TextEncoder().encode(data)
    : data instanceof Blob ? new Uint8Array(await data.arrayBuffer()) : data;
  await db.put("artifacts", bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), key);
}

async function blobBytes(key: string): Promise<Uint8Array> {
  const stored = await db.get<ArrayBuffer | Blob>("artifacts", key);
  if (!stored) throw new ApiError(404, "file not found");
  return new Uint8Array(stored instanceof Blob ? await stored.arrayBuffer() : stored);
}

async function sampleAudio(id: string): Promise<Float32Array> {
  return decodeWav(await blobBytes(sampleKey(id))).samples;
}

/** Any format the browser decodes -> mono 16 kHz (WAV at 16 kHz is read directly). */
async function decodeAudio(file: Blob): Promise<Float32Array> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  try {
    const wav = decodeWav(bytes);
    if (wav.sampleRate === 16000) return wav.samples;
  } catch {
    // not a WAV the runtime reads: let the browser decode it
  }
  try {
    const ctx = new OfflineAudioContext(1, 1, 16000); // decodes and resamples to 16 kHz
    const buffer = await ctx.decodeAudioData(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    const out = new Float32Array(buffer.length);
    for (let c = 0; c < buffer.numberOfChannels; c++) {
      const ch = buffer.getChannelData(c);
      for (let i = 0; i < out.length; i++) out[i] += ch[i] / buffer.numberOfChannels;
    }
    return out.map((v) => Math.max(-1, Math.min(1, v)));
  } catch (err) {
    throw new ApiError(422, `unreadable audio file: ${(err as Error).message}`);
  }
}

// ------------------------------------------------------------------ training events

const events = new Map<string, ProgressEvent[]>();
const listeners = new Map<string, Set<{ progress: (e: ProgressEvent) => void; final: () => void }>>();

function emit(trainingId: string, event: ProgressEvent): void {
  const list = events.get(trainingId) ?? [];
  list.push(event);
  events.set(trainingId, list);
  for (const l of listeners.get(trainingId) ?? []) l.progress(event);
}

function finish(trainingId: string): void {
  for (const l of listeners.get(trainingId) ?? []) l.final();
}

/** Progress of a training run in this tab, replayed from its start; `final` when it ends. */
export function subscribe(trainingId: string, progress: (e: ProgressEvent) => void, final: () => void): () => void {
  const entry = { progress, final };
  const set = listeners.get(trainingId) ?? new Set();
  set.add(entry);
  listeners.set(trainingId, set);
  for (const e of events.get(trainingId) ?? []) progress(e);
  return () => set.delete(entry);
}

// Trainings run one at a time, each holding a Web Lock: another tab sees it's alive.
let queue: Promise<void> = Promise.resolve();
const lockName = (id: string) => `wwd-training-${id}`;

async function recoverInterrupted(): Promise<void> {
  const held = new Set(((await navigator.locks?.query())?.held ?? []).map((l) => l.name));
  for (const t of await db.all<TrainingRow>("trainings")) {
    if ((t.status === "queued" || t.status === "running") && !events.has(t.id) && !held.has(lockName(t.id))) {
      await db.put("trainings", { ...t, status: "failed", error: "Interrupted: the tab was closed during training.",
                                  finished_at: now() });
    }
  }
}
const recovered = recoverInterrupted().catch(() => undefined);

async function runTraining(id: string): Promise<void> {
  const update = async (patch: Partial<TrainingRow>) => {
    const t = await db.get<TrainingRow>("trainings", id);
    if (t) await db.put("trainings", { ...t, ...patch }); // gone: its wake word was deleted meanwhile
  };
  const t = await db.get<TrainingRow>("trainings", id);
  if (!t) return;
  try {
    await update({ status: "running", stage: "loading", started_at: now() });
    const ww = await getWakeWordRow(t.wakeword_id);
    const rows = (await Promise.all(t.sample_ids.map((s) => db.get<SampleRow>("samples", s))))
      .filter((s): s is SampleRow => s !== undefined);
    const positives = rows.filter((s) => s.kind === "positive");
    const negatives = rows.filter((s) => s.kind !== "positive");
    const job: TrainJob = {
      wakeWord: ww.name, language: ww.language, targetFaPerHour: t.target_fa_per_hour,
      positives: await Promise.all(positives.map((s) => sampleAudio(s.id))), positiveIds: positives.map((s) => s.id),
      negatives: await Promise.all(negatives.map((s) => sampleAudio(s.id))),
      negativeSeconds: negatives.reduce<Record<string, number>>((acc, s) =>
        ({ ...acc, [s.kind]: (acc[s.kind] ?? 0) + s.duration_s }), {}),
    };
    const res = await rpc({ type: "train", trainingId: id, job });
    await putFile(artifactKey(id, "model.onnx"), res.model);
    await putFile(artifactKey(id, "head.json"), JSON.stringify({ ...res.head, direction: Array.from(res.head.direction) }));
    await update({ status: "done", stage: "done", progress: 1, report: res.report, finished_at: now(),
                   artifacts: ["model.onnx", "vad.onnx", "report.json", "bundle.zip"] });
    emit(id, { stage: "done", progress: 1, message: "Done", at: now() });
  } catch (err) {
    const cancelled = err instanceof ApiError && err.status === 499;
    await update({ status: cancelled ? "cancelled" : "failed", error: cancelled ? null : (err as Error).message,
                   finished_at: now() });
  } finally {
    finish(id);
  }
}

function enqueue(id: string): void {
  const run = () => (navigator.locks ? navigator.locks.request(lockName(id), () => runTraining(id)) : runTraining(id));
  queue = queue.then(run, run);
}

// ------------------------------------------------------------------ the API

async function refreshOutliers(wakewordId: string): Promise<void> {
  const rows = (await db.byIndex<SampleRow>("samples", "wakeword_id", wakewordId))
    .filter((s) => s.kind === "positive" && s.accepted);
  if (!rows.length) return;
  const clips = await Promise.all(rows.map((r) => sampleAudio(r.id)));
  const { embeddings } = await rpc({ type: "embed_clean", clips });
  const { sims, flags } = outliers(embeddings);
  await Promise.all(rows.map((r, i) => db.put("samples", { ...r, similarity: sims[i], outlier: flags[i] })));
}

async function wakeWordOut(ww: WakeWordRow): Promise<WakeWord> {
  const samples = (await db.byIndex<SampleRow>("samples", "wakeword_id", ww.id))
    .sort((a, b) => a.created_at.localeCompare(b.created_at) || a.start_s - b.start_s);
  const trainings = (await db.byIndex<TrainingRow>("trainings", "wakeword_id", ww.id))
    .sort((a, b) => b.created_at.localeCompare(a.created_at));
  return { ...ww, samples: samples.map(publicSample), trainings: trainings.map(publicTraining) };
}

async function doneTraining(tid: string): Promise<TrainingRow> {
  const t = await db.get<TrainingRow>("trainings", tid);
  if (!t || t.status !== "done") throw new ApiError(404, "no finished training with this id");
  return t;
}

/** model.onnx with another operating point's threshold in its metadata. */
async function modelAt(t: TrainingRow, faPerHour: number): Promise<{ model: Uint8Array; metadata: Record<string, string> }> {
  const stored = await blobBytes(artifactKey(t.id, "model.onnx"));
  const meta = readOnnxMetadata(stored);
  const point = t.report!.operating_points.find((p) => Math.abs(p.fa_per_hour - faPerHour) < 1e-9);
  if (!point) throw new ApiError(422, `no operating point at ${faPerHour} false accepts/h`);
  if (point.fp32.threshold === t.report!.threshold) return { model: stored, metadata: meta };
  const head = JSON.parse(new TextDecoder().decode(await blobBytes(artifactKey(t.id, "head.json"))));
  const { root, manifest: m } = await getBase();
  const template = await browserFetcher(root, m)("model_template.onnx");
  const metadata = modelMetadata({ wakeWord: meta.wake_word, language: meta.language }, meta.base_model_version,
                                 point.fp32.threshold, meta.created_at);
  const h: Head = { direction: Float32Array.from(head.direction), scale: head.scale, bias: head.bias };
  return { model: composeModel(template, m.template, h, metadata), metadata };
}

async function bundle(t: TrainingRow, faPerHour: number): Promise<Uint8Array> {
  const { model, metadata } = await modelAt(t, faPerHour);
  const f = await baseFetcher();
  const m = await getManifest();
  return bundleZip({ model, metadata, vad: await f("vad.onnx"), vadLicense: await f("LICENSE-silero-vad"),
                     report: t.report!, target: faPerHour, teacher: m.base.teacher?.model ?? null });
}

export const local = {
  async languages(): Promise<Language[]> {
    const m = await getManifest();
    return m.languages.map((code) => ({ code, ...m.languages_detail[code] }));
  },

  async baseModels(): Promise<BaseModelInfo[]> {
    const m = await getManifest();
    return [{ version: m.version, created_at: m.base_created_at, default: true }];
  },

  async createWakeWord(name: string, language: string): Promise<WakeWord> {
    const clean = name.split(/\s+/).filter(Boolean).join(" ");
    if (!clean) throw new ApiError(422, "the wake word is empty");
    if (clean.length > 64) throw new ApiError(422, "the wake word is longer than 64 characters");
    if (!(await getManifest()).languages.includes(language)) throw new ApiError(422, `unsupported language ${language}`);
    const ww = { id: newId(), name: clean, language, created_at: now() };
    await db.put("wakewords", ww);
    return { ...ww, samples: [], trainings: [] };
  },

  async wakeWord(id: string): Promise<WakeWord> {
    await recovered;
    return wakeWordOut(await getWakeWordRow(id));
  },

  async deleteWakeWord(id: string): Promise<void> {
    await getWakeWordRow(id);
    for (const t of await db.byIndex<TrainingRow>("trainings", "wakeword_id", id)) {
      if (t.status === "queued" || t.status === "running") compute().postMessage({ type: "cancel", trainingId: t.id });
      for (const run of await db.byIndex<RunRow>("device_runs", "training_id", t.id)) {
        await db.delByIndex("field_chunks", "run_id", run.id);
        await db.delPrefix(`field/${run.id}/`);
        await db.del("device_runs", run.id);
      }
      await db.delPrefix(`${t.id}/`);
      await db.del("trainings", t.id);
    }
    for (const s of await db.byIndex<SampleRow>("samples", "wakeword_id", id)) {
      await db.del("artifacts", sampleKey(s.id));
      await db.del("samples", s.id);
    }
    await db.del("wakewords", id);
  },

  async upload(id: string, kind: SampleKind, file: Blob): Promise<{ upload_id: string; samples: Sample[] }> {
    await getWakeWordRow(id);
    if (file.size > MAX_UPLOAD_MB * 1e6) throw new ApiError(413, `file larger than ${MAX_UPLOAD_MB} MB`);
    const audio = await decodeAudio(file);
    if (!audio.length) throw new ApiError(422, "empty audio file");
    if (audio.length / 16000 > MAX_UPLOAD_S) {
      throw new ApiError(422, `recording too long: ${Math.round(audio.length / 16000)} s > ${MAX_UPLOAD_S} s`);
    }
    const { clips } = await rpc({ type: "checks", kind, audio }, [audio.buffer]);
    const uploadId = newId();
    const created = now();
    const rows: SampleRow[] = [];
    for (const clip of clips) {
      let peak = 0;
      let sq = 0;
      for (const v of clip.audio) {
        peak = Math.max(peak, Math.abs(v));
        sq += v * v;
      }
      const row: SampleRow = {
        id: newId(), wakeword_id: id, upload_id: uploadId, kind, start_s: clip.start_s,
        duration_s: clip.audio.length / 16000, speech_s: clip.speech_s,
        peak_dbfs: 20 * Math.log10(Math.max(peak, 1e-10)),
        rms_dbfs: 20 * Math.log10(Math.max(clip.audio.length ? Math.sqrt(sq / clip.audio.length) : 0, 1e-10)),
        accepted: clip.accepted, reason: clip.reason as Sample["reason"], similarity: null, outlier: false,
        created_at: created,
      };
      await putFile(sampleKey(row.id), encodeWav(clip.audio));
      await db.put("samples", row);
      rows.push(row);
    }
    if (kind === "positive") await refreshOutliers(id);
    const fresh = await Promise.all(rows.map((r) => db.get<SampleRow>("samples", r.id)));
    return { upload_id: uploadId, samples: fresh.map((r) => publicSample(r!)) };
  },

  async deleteSample(id: string, sampleId: string): Promise<void> {
    const row = await db.get<SampleRow>("samples", sampleId);
    if (!row || row.wakeword_id !== id) throw new ApiError(404, "sample not found");
    await db.del("samples", sampleId);
    await db.del("artifacts", sampleKey(sampleId));
    if (row.kind === "positive") await refreshOutliers(id);
  },

  async train(id: string, targetFaPerHour: number): Promise<Training> {
    await getWakeWordRow(id);
    if (!(targetFaPerHour > 0 && targetFaPerHour <= 10)) throw new ApiError(422, "the false-alarm target must be in (0, 10]");
    const trainings = await db.byIndex<TrainingRow>("trainings", "wakeword_id", id);
    const active = trainings.find((t) => t.status === "queued" || t.status === "running");
    if (active) throw new ApiError(409, `training ${active.id} is already ${active.status}`);
    const samples = (await db.byIndex<SampleRow>("samples", "wakeword_id", id)).filter((s) => s.accepted);
    const positives = samples.filter((s) => s.kind === "positive" && !s.outlier);
    if (positives.length < MIN_POSITIVES) {
      throw new ApiError(422, `need at least ${MIN_POSITIVES} accepted recordings of the wake word (10 recommended), ` +
                              `have ${positives.length}`);
    }
    const m = await getManifest();
    const t: TrainingRow = {
      id: newId(), wakeword_id: id, mode: "quick", target_fa_per_hour: targetFaPerHour, base_version: m.version,
      status: "queued", stage: "queued", progress: 0, error: null, report: null, artifacts: [], created_at: now(),
      started_at: null, finished_at: null,
      sample_ids: [...positives.map((s) => s.id), ...samples.filter((s) => s.kind !== "positive").map((s) => s.id)],
    };
    await db.put("trainings", t);
    emit(t.id, { stage: "queued", progress: 0, message: "Queued", at: now() });
    enqueue(t.id);
    return publicTraining(t);
  },

  async training(id: string): Promise<Training> {
    await recovered;
    const t = await db.get<TrainingRow>("trainings", id);
    if (!t) throw new ApiError(404, "training not found");
    return publicTraining(t);
  },

  /** Stored files by their /api/ path. */
  async bytes(path: string): Promise<ArrayBuffer> {
    const url = new URL(path, "http://local");
    const p = url.pathname.split("/").filter(Boolean); // ["api", ...]
    let out: Uint8Array;
    if (p[1] === "wakewords" && p[3] === "samples" && p[5] === "audio") {
      out = await blobBytes(sampleKey(p[4]));
    } else if (p[1] === "trainings" && p[3] === "artifacts") {
      const t = await doneTraining(p[2]);
      if (p[4] === "vad.onnx") out = await (await baseFetcher())("vad.onnx");
      else if (p[4] === "report.json") out = new TextEncoder().encode(JSON.stringify(t.report, null, 2) + "\n");
      else if (p[4] === "bundle.zip") out = await bundle(t, t.target_fa_per_hour);
      else out = await blobBytes(artifactKey(t.id, p[4]));
    } else if (p[1] === "trainings" && p[3] === "bundle.zip") {
      out = await bundle(await doneTraining(p[2]), Number(url.searchParams.get("fa_per_hour")));
    } else if (p[1] === "fieldtests" && p[3] === "clips") {
      out = await blobBytes(`field/${p[2]}/${p[4]}`);
    } else {
      throw new ApiError(404, `nothing at ${path}`);
    }
    return out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength) as ArrayBuffer;
  },
};

// ------------------------------------------------------------------ device runs (M4)

async function getRun(rid: string, kind: RunRow["kind"]): Promise<RunRow> {
  const run = await db.get<RunRow>("device_runs", rid);
  if (!run || run.kind !== kind) throw new ApiError(404, `no ${kind} with this id`);
  return run;
}

interface ChunkRow { run_id: string; stats: FieldStats[]; events: FieldEvent[]; ticks: Float32Array }

async function loadRun(run: RunRow): Promise<FieldRun> {
  const chunks = await db.byIndex<ChunkRow>("field_chunks", "run_id", run.id);
  const n = chunks.reduce((a, c) => a + c.ticks.length / 2, 0);
  const scores = new Float32Array(n);
  const vadMax = new Float32Array(n);
  let off = 0;
  for (const c of chunks) {
    for (let i = 0; i < c.ticks.length / 2; i++) {
      scores[off + i] = c.ticks[2 * i];
      vadMax[off + i] = c.ticks[2 * i + 1];
    }
    off += c.ticks.length / 2;
  }
  return { meta: run.meta!, events: chunks.flatMap((c) => c.events), stats: chunks.flatMap((c) => c.stats),
           scores, vadMax };
}

export const localDevice = {
  async runs(tid: string): Promise<DeviceRunInfo[]> {
    const runs = (await db.byIndex<RunRow>("device_runs", "training_id", tid))
      .sort((a, b) => a.created_at.localeCompare(b.created_at));
    return Promise.all(runs.map(async (run) => {
      const item: DeviceRunInfo = { id: run.id, kind: run.kind, finished: run.finished, created_at: run.created_at,
                                    updated_at: run.updated_at };
      if (run.kind === "benchmark") return { ...item, data: run.data };
      const r = summarize(await loadRun(run));
      return { ...item, summary: { hours: r.hours, triggers: r.triggers, false_accepts_per_hour: r.false_accepts_per_hour,
                                   busy_pct: r.busy_pct, latency_ms: r.latency_ms, battery: r.battery, host: r.host } };
    }));
  },

  async saveBenchmark(tid: string, body: unknown): Promise<{ id: string }> {
    await doneTraining(tid);
    const run: RunRow = { id: newId(), training_id: tid, kind: "benchmark", data: body as Record<string, unknown>,
                          events: 0, finished: true, created_at: now(), updated_at: now() };
    await db.put("device_runs", run);
    return { id: run.id };
  },

  async startFieldTest(tid: string, body: { model: string; gate: boolean; fa_per_hour: number; device: unknown }):
    Promise<{ id: string; threshold: number }> {
    const t = await doneTraining(tid);
    if (body.model !== "fp32") throw new ApiError(422, "the browser builds model.onnx (fp32) only");
    const point = t.report!.operating_points.find((p) => Math.abs(p.fa_per_hour - body.fa_per_hour) < 1e-9);
    if (!point) throw new ApiError(422, `no operating point at ${body.fa_per_hour} false accepts/h`);
    const metadata = readOnnxMetadata(await blobBytes(artifactKey(tid, "model.onnx")));
    const config = { ...configFromMetadata(metadata), threshold: point.fp32.threshold };
    const d = (body.device ?? {}) as Record<string, unknown>;
    const run: RunRow = {
      id: newId(), training_id: tid, kind: "fieldtest", events: 0, finished: false, created_at: now(), updated_at: now(),
      meta: { model: "model.onnx", metadata, config: Object.fromEntries(Object.entries(config).map(([k, v]) => [k, String(v)])),
              gate: body.gate, input: "microphone (browser)",
              host: { machine: d.platform ?? "?", cpu: d.userAgent ?? "?", cores: d.hardwareConcurrency,
                      memory_gb: d.deviceMemory, runtime: "onnxruntime-web (wasm)" } },
    };
    await db.put("device_runs", run);
    return { id: run.id, threshold: config.threshold };
  },

  async chunk(rid: string, chunk: unknown, ticks: Float32Array, clips: Blob[]): Promise<{ events: number }> {
    const data = chunk as { stats?: FieldStats[]; events?: (FieldEvent & { clip?: string })[] };
    const run = await getRun(rid, "fieldtest");
    if (run.finished) throw new ApiError(409, "this field test is finished");
    if (ticks.length % 2) throw new ApiError(422, "ticks: float32 (score, vad) pairs");
    const files = [...clips];
    let count = run.events;
    const events: FieldEvent[] = [];
    for (const e of data.events ?? []) {
      count++;
      const { clip, ...record } = e;
      const event: FieldEvent = { time: record.time, tick: record.tick, sample: record.sample, score: record.score };
      if (clip && files.length) {
        const name = `event-${String(count).padStart(4, "0")}.wav`;
        await putFile(`field/${rid}/${name}`, files.shift()!);
        event.clip = `clips/${name}`;
      }
      events.push(event);
    }
    await db.add("field_chunks", { run_id: rid, stats: data.stats ?? [], events, ticks: Float32Array.from(ticks) });
    await db.put("device_runs", { ...run, events: count, updated_at: now() });
    return { events: count };
  },

  async finish(rid: string): Promise<FieldReport> {
    const run = await getRun(rid, "fieldtest");
    if (!run.finished) await db.put("device_runs", { ...run, finished: true, updated_at: now() });
    return summarize(await loadRun(run));
  },

  async report(rid: string): Promise<FieldReport> {
    return summarize(await loadRun(await getRun(rid, "fieldtest")));
  },
};

export type { Report };
