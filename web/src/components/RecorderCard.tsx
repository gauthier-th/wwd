import { encodeWav } from "@wwd/runtime";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState, type ReactNode } from "react";

import vadUrl from "../../../runtime/assets/silero_vad_16k_op15.onnx?url";
import { api, ApiError, fetchBytes, type Sample, type SampleKind } from "../api";
import { createAudioContext, startRecording, type Recording } from "../audio/pipeline";
import type { RecorderStatus } from "../audio/protocol";
import { REASONS, seconds } from "../format";

let vadBytes: Promise<ArrayBuffer> | null = null;
const loadVad = () => (vadBytes ??= fetchBytes(vadUrl));

const MAX_WORD_S = 1.3; // server/audio.py MAX_WORD_S

export interface RecorderSpec {
  kind: SampleKind;
  title: string;
  description: ReactNode;
  maxS: number; // stop by itself after this long
  autoStopS?: number; // fixed-length takes (ambience)
  targetUtterances?: number; // stop once this many utterances were said (wake word)
  minSpeechS?: number; // refuse to upload with less speech than this
}

type Phase = "idle" | "starting" | "recording" | "uploading";

function summarize(kind: SampleKind, samples: Sample[]): string {
  if (kind !== "positive") return "Saved.";
  const kept = samples.filter((s) => s.accepted).length;
  const rejected = samples.filter((s) => !s.accepted);
  const reasons = [...new Set(rejected.map((s) => REASONS[s.reason ?? ""] ?? "rejected"))].join(", ");
  return `${kept} clip${kept === 1 ? "" : "s"} kept` + (rejected.length ? `, ${rejected.length} rejected (${reasons.toLowerCase()})` : "") + ".";
}

export function RecorderCard({ wakeWordId, spec, intro, children }: {
  wakeWordId: string;
  spec: RecorderSpec;
  intro?: ReactNode; // above the recorder
  children?: ReactNode; // below it: what was recorded
}) {
  const client = useQueryClient();
  const [phase, setPhase] = useState<Phase>("idle");
  const [status, setStatus] = useState<RecorderStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const recording = useRef<Recording | null>(null);
  const stopping = useRef(false);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => () => void recording.current?.cancel(), []);

  async function upload(file: Blob, filename: string) {
    setPhase("uploading");
    try {
      const res = await api.upload(wakeWordId, spec.kind, file, filename);
      setResult(summarize(spec.kind, res.samples));
      await client.invalidateQueries({ queryKey: ["wakeword", wakeWordId] });
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : String(err));
    } finally {
      setPhase("idle");
    }
  }

  async function start() {
    setError(null);
    setResult(null);
    setStatus(null);
    stopping.current = false;
    const context = createAudioContext(); // synchronously, inside the tap (Safari)
    setPhase("starting");
    try {
      recording.current = await startRecording(context, await loadVad(), setStatus);
      setPhase("recording");
    } catch (err) {
      setPhase("idle");
      const name = (err as Error).name;
      setError(name === "NotAllowedError" ? "Microphone access was refused. Allow it in the browser and try again."
        : name === "NotFoundError" ? "No microphone found." : String(err));
    }
  }

  async function stop() {
    if (!recording.current || stopping.current) return;
    stopping.current = true;
    const rec = recording.current;
    recording.current = null;
    setPhase("uploading");
    try {
      const { samples, status: final } = await rec.stop();
      if (spec.kind === "positive" && final.utterances === 0) {
        throw new Error("No speech detected. Check that the right microphone is selected, then try again.");
      }
      if (spec.minSpeechS && final.speechS < spec.minSpeechS) {
        throw new Error(`Only ${seconds(final.speechS)} of speech detected. Read the text out loud, then try again.`);
      }
      await upload(new Blob([encodeWav(samples) as Uint8Array<ArrayBuffer>], { type: "audio/wav" }), `${spec.kind}.wav`);
    } catch (err) {
      setError((err as Error).message);
      setPhase("idle");
    }
  }

  // Stop by itself: fixed length, enough utterances followed by a pause, or the safety limit.
  useEffect(() => {
    if (phase !== "recording" || !status) return;
    const enough = spec.targetUtterances !== undefined && status.utterances >= spec.targetUtterances
      && !status.speaking && status.sinceSpeechS >= 0.8;
    if ((spec.autoStopS && status.seconds >= spec.autoStopS) || enough || status.seconds >= spec.maxS) void stop();
  });

  const recordingNow = phase === "recording";
  const level = status ? Math.max(0, Math.min(100, ((status.levelDb + 60) / 60) * 100)) : 0;
  const tooLong = recordingNow && status !== null && status.currentS > MAX_WORD_S && spec.kind === "positive";

  return (
    <section className="card" data-testid={`recorder-${spec.kind}`}>
      <div className="card-head">
        <div>
          <h2>{spec.title}</h2>
          <div className="muted">{spec.description}</div>
        </div>
      </div>
      {intro}
      <div className="recorder">
        <button
          className={`btn btn-rec${recordingNow ? " recording" : ""}`}
          onClick={recordingNow ? () => void stop() : () => void start()}
          disabled={phase === "starting" || phase === "uploading"}
          aria-label={recordingNow ? "Stop recording" : "Start recording"}
          data-testid={`record-${spec.kind}`}
        />
        <div className="recorder-body">
          {phase === "idle" && !error && (
            <div className="muted small">{result ?? "Tap to record."}</div>
          )}
          {phase === "starting" && <div className="muted small">Starting the microphone…</div>}
          {phase === "uploading" && <div className="muted small">Checking the recording…</div>}
          {recordingNow && status && (
            <div className="stack" aria-live="polite">
              <div className="row">
                <span className={`light${status.vad >= 0.5 ? " on" : ""}`} title="Speech detected" />
                <div className="meter" aria-hidden="true"><div style={{ width: `${level}%` }} /></div>
                <span className="small muted" style={{ fontVariantNumeric: "tabular-nums" }}>
                  {seconds(status.seconds)}{spec.autoStopS ? ` / ${spec.autoStopS} s` : ""}
                </span>
              </div>
              {spec.targetUtterances !== undefined && (
                <div className="row">
                  <span className="count" data-testid="utterances">{status.utterances}</span>
                  <span className="muted">/ {spec.targetUtterances} said</span>
                </div>
              )}
              {tooLong && <div className="notice notice-warn">That's long for one wake word: pause between repetitions.</div>}
            </div>
          )}
          {recordingNow && !status && <div className="muted small">Listening…</div>}
          {error && <div className="notice notice-bad" role="alert" data-testid="recorder-error">{error}</div>}
        </div>
      </div>
      <div className="row small" style={{ marginTop: 8 }}>
        <button className="btn btn-ghost btn-small" onClick={() => fileInput.current?.click()}
                disabled={phase !== "idle"}>
          Import a file instead
        </button>
        <input ref={fileInput} type="file" accept="audio/*,.wav,.flac,.ogg" hidden
               data-testid={`import-${spec.kind}`}
               onChange={(e) => {
                 const file = e.target.files?.[0];
                 e.target.value = "";
                 if (file) {
                   setError(null);
                   void upload(file, file.name);
                 }
               }} />
      </div>
      {children}
    </section>
  );
}
