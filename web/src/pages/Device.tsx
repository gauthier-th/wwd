// Device test (PLAN.md M4): benchmark this phone or computer, and run a long field test with the
// trained files. Field tests upload a minute at a time to the server, in the layout
// `runtime.listen --log` writes on a Pi, so `runtime.fieldreport` reads both.

import { decodeWav, encodeWav } from "@wwd/runtime";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link, useParams } from "react-router";

import speechUrl from "../../../runtime/testdata/audio/speech.wav?url";
import { artifactUrl, deviceApi, fetchBytes, fieldClipUrl, type FieldReport, type Report, type Training } from "../api";
import type { BenchResult } from "../audio/bench.worker";
import { createAudioContext, startDetector, startMic, type DetectorHandle, type MicCapture } from "../audio/pipeline";
import { DetectorSettings, useDetectorChoice } from "../components/DetectorSettings";
import { micError } from "../components/MicSettings";
import type { ModelKind } from "../components/LiveTest";
import { falseAlarmRate, hours } from "../format";
import { useTraining } from "../hooks";

const FILES: Record<ModelKind, string> = { fp32: "model.onnx", int8: "model.int8.onnx" };
// ?upload_s= shortens the interval for the end-to-end tests.
const UPLOAD_EVERY_MS = Number(new URLSearchParams(location.search).get("upload_s") ?? 60) * 1000;
const TICK_MS = 1536 / 16; // runtime/SPEC.md: one tick per 1536 samples at 16 kHz
const BENCH_SECONDS = 30;
// Server limits per upload (server/device.py).
const MAX_TICKS = 20_000;
const MAX_STATS = 50;
const MAX_EVENTS = 90;

type Nav = Navigator & {
  deviceMemory?: number;
  userAgentData?: { platform: string; mobile: boolean };
  getBattery?: () => Promise<BatteryManager>;
};
interface BatteryManager extends EventTarget {
  level: number;
  charging: boolean;
}

export function deviceInfo() {
  const nav = navigator as Nav;
  return {
    userAgent: nav.userAgent,
    platform: nav.userAgentData?.platform ?? nav.platform,
    mobile: nav.userAgentData?.mobile,
    hardwareConcurrency: nav.hardwareConcurrency,
    deviceMemory: nav.deviceMemory,
    secureContext: window.isSecureContext,
  };
}

function useBattery() {
  const [battery, setBattery] = useState<{ level: number; charging: boolean } | null>(null);
  useEffect(() => {
    let manager: BatteryManager | null = null;
    const update = () => manager && setBattery({ level: manager.level, charging: manager.charging });
    void (navigator as Nav).getBattery?.().then((m) => {
      manager = m;
      update();
      m.addEventListener("levelchange", update);
      m.addEventListener("chargingchange", update);
    }).catch(() => undefined);
    return () => {
      manager?.removeEventListener("levelchange", update);
      manager?.removeEventListener("chargingchange", update);
    };
  }, []);
  return battery;
}

const percentile = (values: number[], q: number) => {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
};
const round = (x: number, digits = 3) => Math.round(x * 10 ** digits) / 10 ** digits;

function duration(ms: number): string {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 3600)}:${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

export function DevicePage() {
  const { wid = "", tid = "" } = useParams();
  const training = useTraining(tid);
  if (training.isPending) return <div className="card muted">Loading…</div>;
  const t = training.data;
  if (!t || t.status !== "done" || !t.report) {
    return (
      <div className="card">
        <h1>No finished model here</h1>
        <Link to={`/w/${wid}`} className="btn">Back to the recordings</Link>
      </div>
    );
  }
  return (
    <>
      <div className="card-head">
        <div>
          <h1>Device test</h1>
          <Link to={`/w/${wid}/t/${tid}`} className="small">← “{t.report.wake_word}”</Link>
        </div>
      </div>
      <ThisDevice />
      <Benchmark training={t} />
      <FieldTest training={t} report={t.report} />
      <PastRuns training={t} />
    </>
  );
}

function ThisDevice() {
  const info = deviceInfo();
  const battery = useBattery();
  return (
    <section className="card">
      <h2>This device</h2>
      <p className="small muted" style={{ wordBreak: "break-word" }}>{info.userAgent}</p>
      <div className="row small">
        <span className="badge">{info.hardwareConcurrency ?? "?"} cores</span>
        {info.deviceMemory && <span className="badge">{info.deviceMemory} GB memory</span>}
        {battery && <span className="badge">battery {Math.round(battery.level * 100)}%{battery.charging ? ", charging" : ""}</span>}
      </div>
      {!info.secureContext && (
        <div className="notice notice-warn" style={{ marginTop: 8 }}>
          This page isn't a secure context, so the browser won't open the microphone. Use HTTPS, or on Android
          connect over USB and run <code>adb reverse tcp:8000 tcp:8000</code>, then open http://localhost:8000.
        </div>
      )}
    </section>
  );
}

// ------------------------------------------------------------------------------- benchmark

function benchOnce(model: ArrayBuffer, vad: ArrayBuffer, audio: Float32Array, gate: boolean): Promise<BenchResult> {
  const worker = new Worker(new URL("../audio/bench.worker.ts", import.meta.url), { type: "module" });
  return new Promise((resolve, reject) => {
    worker.onmessage = (e: MessageEvent<{ ok: boolean; result?: BenchResult; error?: string }>) => {
      worker.terminate();
      if (e.data.ok) resolve(e.data.result!);
      else reject(new Error(e.data.error));
    };
    worker.postMessage({ model: model.slice(0), vad: vad.slice(0), audio, gate });
  });
}

type BenchRow = BenchResult & { model: ModelKind };

function Benchmark({ training }: { training: Training }) {
  const client = useQueryClient();
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState("");
  const [rows, setRows] = useState<BenchRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  async function run() {
    setRunning(true);
    setRows([]);
    setError(null);
    setSaved(false);
    try {
      const speech = decodeWav(new Uint8Array(await fetchBytes(speechUrl))).samples;
      const audio = new Float32Array(BENCH_SECONDS * 16000);
      for (let i = 0; i < audio.length; i += speech.length) audio.set(speech.subarray(0, audio.length - i), i);
      const vad = await fetchBytes(artifactUrl(training.id, "vad.onnx"));
      const out: BenchRow[] = [];
      for (const model of ["fp32", "int8"] as const) {
        const bytes = await fetchBytes(artifactUrl(training.id, FILES[model]));
        for (const gate of [true, false]) {
          setStep(`${model === "fp32" ? "Full" : "Small"} model, gate ${gate ? "on" : "off"}…`);
          out.push({ model, ...(await benchOnce(bytes, vad, audio, gate)) });
          setRows([...out]);
        }
      }
      await deviceApi.saveBenchmark(training.id, { device: deviceInfo(), audio_s: BENCH_SECONDS, results: out,
                                                   at: new Date().toISOString() });
      setSaved(true);
      await client.invalidateQueries({ queryKey: ["device-runs", training.id] });
    } catch (err) {
      setError(String(err));
    } finally {
      setRunning(false);
      setStep("");
    }
  }

  return (
    <section className="card" data-testid="benchmark">
      <h2>Benchmark</h2>
      <p className="muted small">
        Streams {BENCH_SECONDS} s of recorded speech through each model file, one 96 ms tick at a time as when listening,
        and times every tick. “Busy” is the share of one core the detector needs to keep up.
      </p>
      <div className="row">
        <button className="btn btn-primary" onClick={() => void run()} disabled={running} data-testid="bench-run">
          {running ? "Running…" : "Run the benchmark"}
        </button>
        <span className="muted small">{step || (saved ? "Saved with this model's results." : "About a minute.")}</span>
      </div>
      {rows.length > 0 && (
        <table className="table small" data-testid="bench-results">
          <thead>
            <tr><th>Model</th><th>Gate</th><th>Per tick (ms) p50 / p95 / max</th><th>Busy</th></tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.model}-${r.gate}`}>
                <td>{r.model === "fp32" ? "Full" : "Small (int8)"}</td>
                <td>{r.gate ? `on (${Math.round(r.gateOpen * 100)}% open)` : "off"}</td>
                <td>{r.latencyMs.p50.toFixed(1)} / {r.latencyMs.p95.toFixed(1)} / {r.latencyMs.max.toFixed(1)}</td>
                <td>{r.busyPct.toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {error && <div className="notice notice-bad" role="alert">{error}</div>}
    </section>
  );
}

// ------------------------------------------------------------------------------- field test

interface PendingEvent {
  n: number; // trigger number in this run
  time: number;
  tick: number;
  sample: number;
  score: number;
  at: number; // Date.now() when it fired: wait a few seconds for its clip
  clip?: Blob;
}

interface Run {
  id: string;
  det: DetectorHandle;
  mic?: MicCapture;
  lock: WakeLockSentinel | null;
  startedAt: number;
  samples: number;
  ticks: number[]; // (score, vad) pairs not uploaded yet
  events: PendingEvent[];
  stats: Record<string, unknown>[];
  minute: { ticks: number; gateOpen: number; events: number; processMs: number[]; queuedMax: number };
  shown: { n: number; time: number; score: number; url?: string }[];
  totals: { ticks: number; events: number };
  last: { busyPct: number; p50: number; p95: number };
  lastUpload: number | null;
  uploading: boolean;
  uploadError: string | null;
}

const emptyMinute = () => ({ ticks: 0, gateOpen: 0, events: 0, processMs: [] as number[], queuedMax: 0 });

async function requestWakeLock(): Promise<WakeLockSentinel | null> {
  try {
    return (await navigator.wakeLock?.request("screen")) ?? null;
  } catch {
    return null; // refused (battery saver, not visible): the run continues, the report notes it
  }
}

function FieldTest({ training, report }: { training: Training; report: Report }) {
  const client = useQueryClient();
  const settings = useDetectorChoice(report);
  const battery = useBattery();
  const batteryRef = useRef(battery);
  batteryRef.current = battery;
  const [phase, setPhase] = useState<"idle" | "starting" | "running" | "stopping">("idle");
  const [, setClock] = useState(0); // re-render once a second while running
  const [error, setError] = useState<ReactNode>(null);
  const [summary, setSummary] = useState<{ id: string; report: FieldReport } | null>(null);
  const runRef = useRef<Run | null>(null);

  function closeMinute(run: Run) {
    const m = run.minute;
    if (!m.ticks) return;
    const busy = (m.processMs.reduce((a, b) => a + b, 0) / (m.ticks * TICK_MS)) * 100;
    run.last = { busyPct: busy, p50: percentile(m.processMs, 0.5), p95: percentile(m.processMs, 0.95) };
    run.stats.push({
      time: Date.now() / 1000,
      audio_s: round(run.samples / 16000),
      ticks: m.ticks,
      gate_open: m.gateOpen,
      events: m.events,
      cpu_pct: null, // a page can't read process CPU; busy_pct is the detector's own time
      busy_pct: round(busy, 2),
      latency_ms: { p50: round(run.last.p50), p95: round(run.last.p95), max: round(Math.max(...m.processMs)) },
      overruns: 0,
      lag_s: round((Date.now() - run.startedAt) / 1000 - run.samples / 16000, 2),
      queued_max: m.queuedMax,
      battery: batteryRef.current?.level ?? null,
      charging: batteryRef.current?.charging ?? null,
      visible: document.visibilityState === "visible",
      wake_lock: run.lock !== null && !run.lock.released,
    });
    run.minute = emptyMinute();
  }

  async function upload(run: Run, final: boolean) {
    if (run.uploading) return;
    closeMinute(run);
    run.uploading = true;
    try {
      do {
        const now = Date.now();
        const ready = run.events.filter((e) => e.clip || final || now - e.at > 5000).slice(0, MAX_EVENTS);
        const stats = run.stats.slice(0, MAX_STATS);
        const nTicks = Math.min(run.ticks.length, MAX_TICKS * 2);
        await deviceApi.chunk(
          run.id,
          { stats, events: ready.map(({ time, tick, sample, score, clip }) =>
            ({ time, tick, sample, score, clip: clip ? "yes" : undefined })) },
          Float32Array.from(run.ticks.slice(0, nTicks)),
          ready.flatMap((e) => (e.clip ? [e.clip] : [])),
        );
        run.ticks.splice(0, nTicks);
        run.stats.splice(0, stats.length);
        run.events = run.events.filter((e) => !ready.includes(e));
        run.lastUpload = Date.now();
        run.uploadError = null;
      } while (run.ticks.length > MAX_TICKS * 2 || run.stats.length > MAX_STATS || (final && run.events.length));
    } catch (err) {
      run.uploadError = String(err); // kept for the next attempt
    } finally {
      run.uploading = false;
    }
  }

  async function start() {
    setError(null);
    setSummary(null);
    const context = createAudioContext(); // synchronously, inside the tap
    setPhase("starting");
    try {
      const { point, kind, gate } = settings.choice;
      const [model, vad] = await Promise.all([fetchBytes(artifactUrl(training.id, FILES[kind])),
                                              fetchBytes(artifactUrl(training.id, "vad.onnx"))]);
      const created = await deviceApi.startFieldTest(training.id, { model: kind, gate, fa_per_hour: point.fa_per_hour,
                                                                    device: deviceInfo() });
      const det = await startDetector(model, vad, { gate, overrides: { threshold: created.threshold },
                                                    clips: { beforeS: 3, afterS: 1 } });
      const run: Run = {
        id: created.id, det, lock: null, startedAt: Date.now(), samples: 0, ticks: [], events: [], stats: [],
        minute: emptyMinute(), shown: [], totals: { ticks: 0, events: 0 }, last: { busyPct: 0, p50: 0, p95: 0 },
        lastUpload: null, uploading: false, uploadError: null,
      };
      det.onMessage((msg) => {
        if (msg.type === "ticks") {
          run.samples += msg.samples;
          for (const t of msg.ticks) {
            run.ticks.push(t.score, t.vadMax);
            run.minute.gateOpen += t.gateOpen ? 1 : 0;
          }
          run.minute.ticks += msg.ticks.length;
          run.totals.ticks += msg.ticks.length;
          run.minute.processMs.push(...msg.processMs);
          run.minute.queuedMax = Math.max(run.minute.queuedMax, msg.queued);
          for (const e of msg.events) {
            const n = ++run.totals.events;
            run.events.push({ n, time: Date.now() / 1000, tick: e.tick, sample: e.sample, score: e.score, at: Date.now() });
            run.minute.events++;
            run.shown.unshift({ n, time: Date.now(), score: e.score });
          }
        } else if (msg.type === "clip") {
          const clip = new Blob([encodeWav(msg.samples) as Uint8Array<ArrayBuffer>], { type: "audio/wav" });
          const event = run.events.find((e) => e.tick === msg.tick);
          if (event) {
            event.clip = clip;
            const shown = run.shown.find((s) => s.n === event.n);
            if (shown) shown.url = URL.createObjectURL(clip);
          }
        } else if (msg.type === "error") {
          setError(msg.message);
        }
      });
      run.mic = await startMic(det, context);
      run.lock = await requestWakeLock();
      runRef.current = run;
      setPhase("running");
    } catch (err) {
      void context.close();
      setError(micError(err) ?? String(err));
      setPhase("idle");
    }
  }

  async function stop() {
    const run = runRef.current;
    if (!run) return;
    setPhase("stopping");
    await run.mic?.stop();
    run.det.terminate();
    await run.lock?.release().catch(() => undefined);
    for (let attempt = 0; attempt < 3; attempt++) {
      await upload(run, true);
      if (!run.uploadError) break;
    }
    try {
      setSummary({ id: run.id, report: await deviceApi.finish(run.id) });
    } catch (err) {
      setError(String(err));
    }
    run.shown.forEach((s) => s.url && URL.revokeObjectURL(s.url));
    runRef.current = null;
    setPhase("idle");
    await client.invalidateQueries({ queryKey: ["device-runs", training.id] });
  }

  // While running: upload every minute, refresh the view every second, keep the screen on.
  useEffect(() => {
    if (phase !== "running") return;
    const tick = setInterval(() => setClock((n) => n + 1), 1000);
    const up = setInterval(() => runRef.current && void upload(runRef.current, false), UPLOAD_EVERY_MS);
    const onVisible = async () => {
      const run = runRef.current;
      if (run && document.visibilityState === "visible" && (!run.lock || run.lock.released)) run.lock = await requestWakeLock();
    };
    const onLeave = (e: BeforeUnloadEvent) => e.preventDefault(); // ask before closing a running test
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("beforeunload", onLeave);
    return () => {
      clearInterval(tick);
      clearInterval(up);
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("beforeunload", onLeave);
    };
  }, [phase]);

  useEffect(() => () => void stop(), []); // leaving the page ends the run cleanly

  const run = runRef.current;
  return (
    <section className="card" data-testid="fieldtest">
      <h2>Field test</h2>
      <p className="muted small">
        Leave the device listening where it will be used, for up to 24 hours: TV on part of the time, people talking,
        no wake word (or note when you said it). Every trigger is kept with the audio around it, and the log is
        uploaded every minute, so closing the page loses at most a minute. Keep this page open and the phone
        plugged in; the screen stays on.
      </p>
      <DetectorSettings state={settings} disabled={phase !== "idle"} />
      <div className="row">
        <button className={`btn ${phase === "running" ? "" : "btn-primary"}`} data-testid="field-toggle"
                disabled={phase === "starting" || phase === "stopping"}
                onClick={() => void (phase === "running" ? stop() : start())}>
          {phase === "running" ? "Stop and upload" : phase === "starting" ? "Starting…"
            : phase === "stopping" ? "Uploading…" : "Start the field test"}
        </button>
      </div>
      {run && phase === "running" && (
        <div className="stack" style={{ marginTop: 12 }}>
          <div className="stats">
            <div className="stat">
              <div className="stat-value" data-testid="field-elapsed">{duration(Date.now() - run.startedAt)}</div>
              <div className="stat-label">listening{run.lock && !run.lock.released ? ", screen kept on" : ""}</div>
            </div>
            <div className="stat">
              <div className="stat-value" data-testid="field-triggers">{run.totals.events}</div>
              <div className="stat-label">triggers ({falseAlarmRate(settings.choice.point.fa_per_hour)} expected)</div>
            </div>
            <div className="stat">
              <div className="stat-value">{run.last.busyPct ? `${run.last.busyPct.toFixed(1)}%` : "…"}</div>
              <div className="stat-label">busy, last minute ({run.last.p50.toFixed(1)} ms per tick, p95 {run.last.p95.toFixed(1)})</div>
            </div>
            {battery && (
              <div className="stat">
                <div className="stat-value">{Math.round(battery.level * 100)}%</div>
                <div className="stat-label">battery{battery.charging ? ", charging" : ""}</div>
              </div>
            )}
          </div>
          <div className="small muted" data-testid="field-upload">
            {run.uploadError ? `Upload failed, will retry: ${run.uploadError}`
              : run.lastUpload ? `Uploaded ${Math.round((Date.now() - run.lastUpload) / 1000)} s ago`
              : "First upload after a minute"}
            {" · "}{run.totals.ticks} ticks
          </div>
          {run.shown.length > 0 && (
            <ul className="list">
              {run.shown.slice(0, 20).map((s) => (
                <li key={s.n}>
                  <span>#{s.n}</span>
                  <span className="muted small">{new Date(s.time).toLocaleTimeString()}</span>
                  <span className="small">score {s.score.toFixed(2)}</span>
                  <span className="spacer" />
                  {s.url && <audio src={s.url} controls preload="none" style={{ height: 32, maxWidth: 180 }} />}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
      {summary && <FieldSummary id={summary.id} report={summary.report} />}
      {error && <div className="notice notice-bad" role="alert" style={{ marginTop: 8 }}>{error}</div>}
    </section>
  );
}

function FieldSummary({ id, report: r }: { id: string; report: FieldReport }) {
  return (
    <div className="stack" data-testid="field-summary" style={{ marginTop: 12 }}>
      <h3>Result</h3>
      <div className="stats">
        <div className="stat">
          <div className="stat-value">{hours(r.hours)}</div>
          <div className="stat-label">listened ({hours(r.speech_hours)} with speech)</div>
        </div>
        <div className="stat">
          <div className="stat-value">{r.triggers}</div>
          <div className="stat-label">triggers, {r.false_accepts_per_hour ?? "–"} per hour</div>
        </div>
        <div className="stat">
          <div className="stat-value">{r.busy_pct.mean ?? "–"}%</div>
          <div className="stat-label">busy on average ({r.latency_ms.p50 ?? "–"} ms per tick)</div>
        </div>
        {r.battery && (
          <div className="stat">
            <div className="stat-value">{(r.battery.per_hour * 100).toFixed(1)}%</div>
            <div className="stat-label">battery per hour</div>
          </div>
        )}
      </div>
      {r.events.length > 0 && (
        <ul className="list">
          {r.events.map((e) => (
            <li key={e.n}>
              <span>#{e.n}</span>
              <span className="muted small">{new Date(e.time * 1000).toLocaleString()}</span>
              <span className="small">score {e.score.toFixed(2)}</span>
              <span className="spacer" />
              {e.clip && <audio src={fieldClipUrl(id, e.clip)} controls preload="none" style={{ height: 32, maxWidth: 180 }} />}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function PastRuns({ training }: { training: Training }) {
  const runs = useQuery({ queryKey: ["device-runs", training.id], queryFn: () => deviceApi.runs(training.id) });
  if (!runs.data?.length) return null;
  return (
    <section className="card" data-testid="past-runs">
      <h2>Measurements of this model</h2>
      <ul className="list">
        {[...runs.data].reverse().map((r) => {
          const ua = String((r.kind === "benchmark" ? (r.data?.device as { userAgent?: string })?.userAgent
            : r.summary?.host?.cpu) ?? "");
          const device = ua.match(/\(([^)]+)\)/)?.[1] ?? ua.slice(0, 40);
          if (r.kind === "benchmark") {
            const rows = (r.data?.results ?? []) as BenchRow[];
            const main = rows.find((x) => x.model === "fp32" && x.gate);
            return (
              <li key={r.id}>
                <span className="badge">benchmark</span>
                <span className="small">{new Date(r.created_at).toLocaleString()}</span>
                <span className="small muted">{device}</span>
                <span className="spacer" />
                {main && <span className="small">{main.latencyMs.p50.toFixed(1)} ms/tick, {main.busyPct.toFixed(1)}% busy</span>}
              </li>
            );
          }
          const s = r.summary!;
          return (
            <li key={r.id}>
              <span className="badge badge-accent">field test</span>
              <span className="small">{new Date(r.created_at).toLocaleString()}</span>
              <span className="small muted">{device}</span>
              <span className="spacer" />
              <span className="small">
                {hours(s.hours)}, {s.triggers} triggers ({s.false_accepts_per_hour ?? "–"}/h){r.finished ? "" : ", running or interrupted"}
              </span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
