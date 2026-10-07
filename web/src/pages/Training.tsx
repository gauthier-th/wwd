import { Link, useParams } from "react-router";

import { artifactUrl, bundleUrl, type Report, type Training } from "../api";
import { DetectorSettings, useDetectorChoice } from "../components/DetectorSettings";
import { LiveTest } from "../components/LiveTest";
import { hours, pct, seconds } from "../format";
import { useTraining, useTrainingEvents, useWakeWord } from "../hooks";
import { Steps } from "./WakeWord";

const STAGES: [string, string][] = [
  ["queued", "Waiting for the trainer"],
  ["loading", "Loading the base model"],
  ["embedding", "Learning from your recordings"],
  ["cross_validating", "Checking each recording"],
  ["calibrating", "Setting the sensitivity"],
  ["composing", "Building the model files"],
  ["verifying", "Testing them on held-out speech"],
  ["bundling", "Packaging"],
];

export function TrainingPage() {
  const { wid = "", tid = "" } = useParams();
  const ww = useWakeWord(wid);
  const query = useTraining(tid);
  const training = query.data;
  const events = useTrainingEvents(training);

  if (query.isPending) return <div className="card muted">Loading…</div>;
  if (!training) {
    return (
      <div className="card">
        <h1>Training not found</h1>
        <Link to={`/w/${wid}`} className="btn">Back to the recordings</Link>
      </div>
    );
  }
  const name = ww.data?.name ?? training.report?.wake_word ?? "";

  return (
    <>
      <div className="card-head">
        <div>
          <h1>“{name}”</h1>
          <Link to={`/w/${wid}`} className="small">← Recordings</Link>
        </div>
      </div>
      <Steps current={training.status === "done" ? 2 : 1} />
      {training.status === "done" && training.report ? (
        <Done training={training} report={training.report} />
      ) : training.status === "failed" || training.status === "cancelled" ? (
        <div className="card">
          <h2>Training {training.status}</h2>
          {training.error && <div className="notice notice-bad">{training.error}</div>}
          <p style={{ marginTop: 12 }}><Link to={`/w/${wid}`} className="btn">Back to the recordings</Link></p>
        </div>
      ) : (
        <Progress training={training} events={events} />
      )}
    </>
  );
}

function Progress({ training, events }: { training: Training; events: { stage: string; progress: number; message: string }[] }) {
  const last = events[events.length - 1];
  const stage = last?.stage ?? training.stage;
  const progress = Math.max(last?.progress ?? 0, training.progress);
  const current = STAGES.findIndex(([s]) => s === stage);
  return (
    <section className="card" data-testid="progress" aria-busy="true">
      <h2>Training…</h2>
      <div className="progress" role="progressbar" aria-valuenow={Math.round(progress * 100)} aria-valuemin={0} aria-valuemax={100}>
        <div style={{ width: `${Math.round(progress * 100)}%` }} />
      </div>
      <ol className="stage-list">
        {STAGES.map(([s, label], i) => (
          <li key={s} className={i < current ? "done" : i === current ? "current" : ""}>
            <span aria-hidden="true">{i < current ? "✓" : i === current ? "›" : "·"}</span>
            {label}
          </li>
        ))}
      </ol>
      <p className="muted small" style={{ marginTop: 8 }}>You can leave this page: training continues on the server.</p>
    </section>
  );
}

function Done({ training, report }: { training: Training; report: Report }) {
  const settings = useDetectorChoice(report);
  const { point, kind, gate } = settings.choice;
  const verify = report.verification[kind];
  const cv = report.cross_validation;
  const weak = cv?.per_recording.filter((r) => r.weak).length ?? 0;

  return (
    <>
      <section className="card" data-testid="results">
        <h2>Your model is ready</h2>
        <div className="stats">
          <div className="stat">
            <div className="stat-value">{verify.enrollment_detected}/{verify.enrollment_total}</div>
            <div className="stat-label">of your recordings detected</div>
          </div>
          <div className="stat">
            <div className="stat-value">{verify.gate.per_hour.toFixed(2)}</div>
            <div className="stat-label">false alarms per hour, measured on {hours(verify.hours)} of other speech</div>
          </div>
          {cv && (
            <div className="stat">
              <div className="stat-value">{pct(cv.recall_noisy)}</div>
              <div className="stat-label">detected in heavy noise, on recordings held out of training*</div>
            </div>
          )}
          <div className="stat">
            <div className="stat-value">{seconds(report.timings_s.total)}</div>
            <div className="stat-label">training time</div>
          </div>
        </div>
        {weak > 0 && (
          <div className="notice notice-warn">
            {weak} recording{weak === 1 ? " was" : "s were"} hard to detect.{" "}
            <Link to={`/w/${training.wakeword_id}`}>Re-record {weak === 1 ? "it" : "them"}</Link> and train again for a
            better model.
          </div>
        )}
        <p className="muted small" style={{ marginTop: 8 }}>
          * A rough estimate from a few recordings under strong noise; it tends to understate real-world detection.
        </p>
      </section>

      <section className="card" data-testid="settings">
        <h2>Try it</h2>
        <DetectorSettings state={settings} />
        <LiveTest training={training} point={point} kind={kind} gate={gate} />
        <p className="small" style={{ marginTop: 12 }}>
          <Link to={`/w/${training.wakeword_id}/t/${training.id}/device`} data-testid="device-link">
            Measure this device: benchmark and 24 h field test →
          </Link>
        </p>
      </section>

      <section className="card" data-testid="download">
        <h2>Download</h2>
        <p className="muted">
          Both model files, the speech detector and the settings, at the sensitivity chosen above. They run with
          onnxruntime in Python (Raspberry Pi, any computer) or onnxruntime-web in a browser.
        </p>
        <div className="row">
          <a className="btn btn-primary" href={bundleUrl(training, point.fa_per_hour)} download data-testid="bundle-link">
            Download the model (.zip)
          </a>
          <a className="btn btn-ghost btn-small" href={artifactUrl(training.id, "report.json")} download>
            Training report
          </a>
        </div>
        <pre className="paragraph small" style={{ overflowX: "auto" }}>
          pip install -r runtime/requirements.txt{"\n"}python -m runtime.listen model.onnx --vad vad.onnx
        </pre>
      </section>
    </>
  );
}
