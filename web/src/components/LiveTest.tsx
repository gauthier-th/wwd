import type { Tick } from "@wwd/runtime";
import { useEffect, useRef, useState, type ReactNode } from "react";

import { artifactUrl, fetchBytes, type OperatingPoint, type Training } from "../api";
import { createAudioContext, startDetector, startMic, type DetectorHandle, type MicCapture } from "../audio/pipeline";
import { cssVar } from "../format";
import { thresholdOf } from "./DetectorSettings";
import { micError } from "./MicSettings";

const HISTORY = 160; // ticks on screen: about 15 s

export type ModelKind = "fp32" | "int8";
const FILES: Record<ModelKind, string> = { fp32: "model.onnx", int8: "model.int8.onnx" };

interface Trace {
  ticks: Tick[]; // on screen, from the current detector only
  events: number[]; // tick indices
  received: number;
  vadMax: number; // loudest speech probability so far
  version: number;
}

/** A short rising two-note chime, on the microphone's own AudioContext (already allowed to
 * play: it was started by the tap on "Start listening"). */
function chime(context: AudioContext): void {
  const t = context.currentTime;
  const gain = context.createGain();
  gain.gain.setValueAtTime(0.0001, t);
  gain.gain.exponentialRampToValueAtTime(0.12, t + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.28);
  gain.connect(context.destination);
  for (const [freq, at] of [[880, 0], [1320, 0.08]]) {
    const osc = context.createOscillator();
    osc.frequency.value = freq;
    osc.connect(gain);
    osc.start(t + at);
    osc.stop(t + 0.3);
  }
}

function LiveChart({ trace, threshold }: { trace: React.RefObject<Trace>; threshold: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    let frame = 0;
    let drawn = -1;
    let drawnThreshold = Number.NaN;
    const draw = () => {
      frame = requestAnimationFrame(draw);
      const canvas = ref.current;
      const t = trace.current;
      if (!canvas || !t || (t.version === drawn && threshold === drawnThreshold)) return;
      drawn = t.version;
      drawnThreshold = threshold;
      canvas.dataset.ticks = String(t.received);
      canvas.dataset.shown = String(t.ticks.length);
      canvas.dataset.vadMax = t.vadMax.toFixed(3);
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
      }
      const g = canvas.getContext("2d")!;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);
      const pad = 6;
      const x = (i: number) => (i / (HISTORY - 1)) * w;
      const y = (v: number) => pad + (1 - v) * (h - 2 * pad);
      const first = t.ticks.length ? t.ticks[0].index : 0;

      g.fillStyle = cssVar("--vad");
      g.globalAlpha = 0.35;
      t.ticks.forEach((tick, i) => {
        const v = Number.isNaN(tick.vadMax) ? 0 : tick.vadMax;
        g.fillRect(x(i), y(v), Math.max(1, w / HISTORY), h - pad - y(v));
      });
      g.globalAlpha = 1;

      g.strokeStyle = cssVar("--warn");
      g.setLineDash([4, 4]);
      g.beginPath();
      g.moveTo(0, y(threshold));
      g.lineTo(w, y(threshold));
      g.stroke();
      g.setLineDash([]);

      g.strokeStyle = cssVar("--score");
      g.lineWidth = 2;
      g.beginPath();
      t.ticks.forEach((tick, i) => (i ? g.lineTo(x(i), y(tick.score)) : g.moveTo(x(i), y(tick.score))));
      g.stroke();

      g.fillStyle = cssVar("--ok");
      for (const e of t.events) {
        const i = e - first;
        if (i >= 0 && i < t.ticks.length) g.fillRect(x(i) - 1.5, 0, 3, h);
      }
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [trace, threshold]);
  return <canvas ref={ref} className="chart" data-testid="live-chart" data-ticks="0"
                 role="img" aria-label="Wake-word score and speech probability over the last 15 seconds" />;
}

export function LiveTest({ training, point, kind, gate }: {
  training: Training; point: OperatingPoint; kind: ModelKind; gate: boolean;
}) {
  const [running, setRunning] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<ReactNode>(null);
  const [detections, setDetections] = useState(0);
  const [flash, setFlash] = useState(false);
  const [sound, setSound] = useState(true);
  const soundOn = useRef(sound);
  soundOn.current = sound;
  const flashTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const trace = useRef<Trace>({ ticks: [], events: [], received: 0, vadMax: 0, version: 0 });
  const detector = useRef<DetectorHandle | null>(null);
  const mic = useRef<MicCapture | null>(null);
  const bytes = useRef<Partial<Record<ModelKind | "vad", Promise<ArrayBuffer>>>>({});
  const threshold = thresholdOf(point, kind);

  const load = (name: ModelKind | "vad") =>
    (bytes.current[name] ??= fetchBytes(artifactUrl(training.id, name === "vad" ? "vad.onnx" : FILES[name])));

  async function start() {
    setError(null);
    const context = createAudioContext(); // synchronously, inside the tap (Safari)
    setStarting(true);
    try {
      const [model, vad] = await Promise.all([load(kind), load("vad")]);
      const det = await startDetector(model.slice(0), vad.slice(0), { gate, overrides: { threshold } });
      det.onMessage((msg) => {
        if (msg.type === "ticks") {
          if (msg.generation !== det.generation) return; // from the detector before a change
          const t = trace.current;
          t.ticks.push(...msg.ticks);
          if (t.ticks.length > HISTORY) t.ticks.splice(0, t.ticks.length - HISTORY);
          t.received += msg.ticks.length;
          for (const tick of msg.ticks) if (tick.vadMax > t.vadMax) t.vadMax = tick.vadMax;
          if (msg.events.length) {
            t.events.push(...msg.events.map((e) => e.tick));
            setDetections((n) => n + msg.events.length);
            setFlash(true);
            clearTimeout(flashTimer.current);
            flashTimer.current = setTimeout(() => setFlash(false), 1200);
            if (soundOn.current && mic.current) chime(mic.current.context);
          }
          t.version++;
        } else if (msg.type === "error") setError(msg.message);
      });
      detector.current = det;
      mic.current = await startMic(det, context);
      setRunning(true);
    } catch (err) {
      detector.current?.terminate();
      detector.current = null;
      void context.close();
      setError(micError(err) ?? String(err));
    } finally {
      setStarting(false);
    }
  }

  async function stop() {
    await mic.current?.stop();
    detector.current?.terminate();
    mic.current = null;
    detector.current = null;
    setRunning(false);
  }

  // Model, gate or sensitivity changed while listening: start a fresh view (the new detector
  // numbers its ticks from 0) and swap them in without dropping the mic.
  useEffect(() => {
    const det = detector.current;
    if (!running || !det) return;
    const t = trace.current;
    t.ticks = [];
    t.events = [];
    t.version++;
    setDetections(0);
    setFlash(false);
    let cancelled = false;
    void (async () => {
      const [model, vad] = await Promise.all([load(kind), load("vad")]);
      if (!cancelled) await det.reconfigure(model.slice(0), vad.slice(0), { gate, overrides: { threshold } });
    })().catch((err) => setError(String(err)));
    return () => {
      cancelled = true;
    };
  }, [kind, gate, threshold]);

  useEffect(() => () => void stop(), []); // release the mic when leaving the page

  return (
    <div className="stack">
      <div className="row">
        <button className={`btn ${running ? "" : "btn-primary"}`} onClick={() => void (running ? stop() : start())}
                disabled={starting} data-testid="live-toggle">
          {running ? "Stop listening" : starting ? "Starting…" : "Start listening"}
        </button>
        <label className="check small">
          <input type="checkbox" checked={sound} onChange={(e) => setSound(e.target.checked)} />
          Sound
        </label>
        <span className="spacer" />
        <span className="badge badge-accent" data-testid="detections">{detections} detected</span>
      </div>
      <div className="chart-wrap">
        <LiveChart trace={trace} threshold={threshold} />
        <div className="chart-flash" style={{ opacity: flash ? 1 : 0 }} aria-live="assertive">
          {flash ? `“${training.report?.wake_word}” detected` : ""}
        </div>
      </div>
      <div className="legend">
        <span><span className="swatch" style={{ background: "var(--score)" }} />Wake-word score</span>
        <span><span className="swatch" style={{ background: "var(--vad)" }} />Speech (VAD)</span>
        <span><span className="swatch" style={{ background: "var(--warn)" }} />Threshold</span>
        <span><span className="swatch" style={{ background: "var(--ok)" }} />Detection</span>
      </div>
      {error && <div className="notice notice-bad" role="alert">{error}</div>}
    </div>
  );
}
