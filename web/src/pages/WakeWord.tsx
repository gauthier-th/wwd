import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router";

import { api, ApiError, type Training, type WakeWord } from "../api";
import { ClipGrid, SampleRows } from "../components/Clips";
import { RecorderCard, type RecorderSpec } from "../components/RecorderCard";
import { falseAlarmRate, languageLabel, letterCount, seconds, SHORT_WAKE_WORD } from "../format";
import { isFinal, useLanguages, useWakeWord } from "../hooks";
import { forgetWakeWord, rememberWakeWord } from "../recent";

const TARGET_TAKES = 10;
const MIN_POSITIVES = 3; // local/backend.ts MIN_POSITIVES (server/api.py's)
const TARGETS = [0.25, 0.5, 1, 2];

export function Steps({ current }: { current: 0 | 1 | 2 }) {
  const names = ["Record", "Train", "Test & download"];
  return (
    <ol className="steps" aria-label="Progress">
      {names.map((n, i) => (
        <li key={n} className={`step${i < current ? " done" : i === current ? " current" : ""}`}
            aria-current={i === current ? "step" : undefined}>
          {i + 1}. {n}
        </li>
      ))}
    </ol>
  );
}

export function WakeWordPage() {
  const { wid = "" } = useParams();
  const query = useWakeWord(wid);
  const languages = useLanguages();
  const navigate = useNavigate();
  const client = useQueryClient();
  const remove = useMutation({
    mutationFn: () => api.deleteWakeWord(wid),
    onSuccess: () => {
      forgetWakeWord(wid);
      client.removeQueries({ queryKey: ["wakeword", wid] });
      navigate("/");
    },
  });

  const ww = query.data;
  useEffect(() => {
    if (ww) rememberWakeWord({ id: ww.id, name: ww.name, language: ww.language, created_at: ww.created_at });
  }, [ww]);

  if (query.isPending) return <div className="card muted">Loading…</div>;
  if (query.error) {
    const gone = query.error instanceof ApiError && query.error.status === 404;
    return (
      <div className="card">
        <h1>{gone ? "This wake word doesn't exist" : "Couldn't load this wake word"}</h1>
        <p className="muted">{gone ? "It may have been deleted." : String(query.error)}</p>
        <Link to="/" className="btn">Back</Link>
      </div>
    );
  }

  const language = languages.data?.find((l) => l.code === ww!.language);
  const samples = ww!.samples;
  const positives = samples.filter((s) => s.kind === "positive");
  const latestDone = [...ww!.trainings].reverse().find((t) => t.status === "done");
  const weak = new Set(latestDone?.report?.cross_validation?.per_recording.filter((r) => r.weak).map((r) => r.sample_id));

  const specs: Record<string, RecorderSpec> = {
    positive: {
      kind: "positive",
      title: "1. Say the wake word",
      description: <>Say “{ww!.name}” {TARGET_TAKES} times, pausing about a second between each. Recording stops
        by itself after the {TARGET_TAKES}th. Use the microphone you'll use the wake word with.</>,
      maxS: 90,
      targetUtterances: TARGET_TAKES,
    },
    other_speech: {
      kind: "other_speech",
      title: "2. Read this aloud",
      description: <>About 30 seconds of your voice saying other things, so the model learns what isn't the wake
        word. Tap stop when you reach the end.</>,
      maxS: 90,
      minSpeechS: 2,
    },
    ambience: {
      kind: "ambience",
      title: "3. Room sound",
      description: "Stay quiet for 15 seconds so the model hears your room.",
      maxS: 15,
      autoStopS: 15,
    },
  };

  return (
    <>
      <div className="card-head">
        <div>
          <h1>“{ww!.name}”</h1>
          <span className="badge">{language ? languageLabel(language) : ww!.language}</span>
        </div>
        <button className="btn btn-small btn-ghost btn-danger" disabled={remove.isPending}
                onClick={() => {
                  if (confirm(`Delete “${ww!.name}”, its recordings and its models? This can't be undone.`)) remove.mutate();
                }}>
          Delete
        </button>
      </div>
      <Steps current={latestDone ? 2 : 0} />

      <RecorderCard wakeWordId={wid} spec={specs.positive} intro={letterCount(ww!.name) <= SHORT_WAKE_WORD && (
        <div className="notice notice-warn" style={{ marginBottom: 12 }}>
          Short wake words are harder to detect reliably. If you can, use a longer word or a two-word phrase.
        </div>
      )} />
      {positives.length > 0 && (
        <div className="card">
          <h3>Your recordings</h3>
          <ClipGrid wakeWordId={wid} samples={positives} weak={weak} />
        </div>
      )}

      <RecorderCard wakeWordId={wid} spec={specs.other_speech} intro={language && (
        <p className="paragraph" dir={language.rtl ? "rtl" : "ltr"} lang={language.code}>{language.paragraph}</p>
      )}>
        <SampleRows wakeWordId={wid} samples={samples.filter((s) => s.kind === "other_speech")} />
      </RecorderCard>

      <RecorderCard wakeWordId={wid} spec={specs.ambience}>
        <SampleRows wakeWordId={wid} samples={samples.filter((s) => s.kind === "ambience")} />
      </RecorderCard>

      <TrainCard ww={ww!} />
    </>
  );
}

function Check({ ok, warn, children }: { ok: boolean; warn?: boolean; children: React.ReactNode }) {
  const cls = ok ? "badge-ok" : warn ? "badge-warn" : "badge-bad";
  return (
    <li>
      <span className={`badge ${cls}`} aria-hidden="true">{ok ? "✓" : warn ? "!" : "✕"}</span>
      <span>{children}</span>
    </li>
  );
}

function TrainCard({ ww }: { ww: WakeWord }) {
  const navigate = useNavigate();
  const [target, setTarget] = useState(0.5);
  const train = useMutation({
    mutationFn: () => api.train(ww.id, target),
    onSuccess: (t) => navigate(`/w/${ww.id}/t/${t.id}`),
  });
  const usable = ww.samples.filter((s) => s.kind === "positive" && s.accepted && !s.outlier).length;
  const outliers = ww.samples.filter((s) => s.kind === "positive" && s.outlier).length;
  const speechS = ww.samples.filter((s) => s.kind === "other_speech").reduce((a, s) => a + s.speech_s, 0);
  const ambienceS = ww.samples.filter((s) => s.kind === "ambience").reduce((a, s) => a + s.duration_s, 0);
  const active = ww.trainings.find((t) => !isFinal(t));

  return (
    <section className="card" data-testid="train-card">
      <h2>Train</h2>
      <ul className="list">
        <Check ok={usable >= 8} warn={usable >= MIN_POSITIVES}>
          {usable} usable recording{usable === 1 ? "" : "s"} of the wake word ({TARGET_TAKES} recommended, at least{" "}
          {MIN_POSITIVES}){outliers ? `; ${outliers} left out for sounding unlike the others` : ""}
        </Check>
        <Check ok={speechS >= 15} warn>{speechS ? `${seconds(speechS)} of you reading` : "No reading yet"} (optional, recommended)</Check>
        <Check ok={ambienceS >= 10} warn>{ambienceS ? `${seconds(ambienceS)} of room sound` : "No room sound yet"} (optional, recommended)</Check>
      </ul>
      <div className="field" style={{ margin: "12px 0" }}>
        <label id="target-label">False alarms you'd accept</label>
        <div className="segmented" role="group" aria-labelledby="target-label">
          {TARGETS.map((t) => (
            <button key={t} type="button" aria-pressed={t === target} onClick={() => setTarget(t)}>
              {falseAlarmRate(t)}
            </button>
          ))}
        </div>
        <span className="muted small">You can change the sensitivity after training too.</span>
      </div>
      {train.error && (
        <div className="notice notice-bad" role="alert">
          {train.error instanceof ApiError ? train.error.detail : String(train.error)}
        </div>
      )}
      <div className="row" style={{ marginTop: 12 }}>
        {active ? (
          <Link className="btn btn-primary" to={`/w/${ww.id}/t/${active.id}`}>Training in progress…</Link>
        ) : (
          <button className="btn btn-primary" onClick={() => train.mutate()} disabled={usable < MIN_POSITIVES || train.isPending}
                  data-testid="train">
            Train the model
          </button>
        )}
        <span className="muted small">Takes about a minute.</span>
      </div>
      <PastTrainings ww={ww} />
    </section>
  );
}

function PastTrainings({ ww }: { ww: WakeWord }) {
  const done = ww.trainings.filter(isFinal);
  if (!done.length) return null;
  const label = (t: Training) => ({ done: "Ready", failed: "Failed", cancelled: "Cancelled" })[t.status as "done"] ?? t.status;
  return (
    <>
      <h3 style={{ marginTop: 16 }}>Models</h3>
      <ul className="list">
        {[...done].reverse().map((t) => (
          <li key={t.id}>
            <Link to={`/w/${ww.id}/t/${t.id}`}>{new Date(t.created_at).toLocaleString()}</Link>
            <span className={`badge ${t.status === "done" ? "badge-ok" : "badge-bad"}`}>{label(t)}</span>
            <span className="muted small">{falseAlarmRate(t.target_fa_per_hour)}</span>
          </li>
        ))}
      </ul>
    </>
  );
}
