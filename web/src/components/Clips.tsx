import { decodeWav } from "@wwd/runtime";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import { api, fetchBytes, sampleAudioUrl, type Sample } from "../api";
import { cssVar, REASONS, seconds } from "../format";

function useClipAudio(wakeWordId: string, sampleId: string) {
  return useQuery({
    queryKey: ["clip", sampleId],
    queryFn: async () => decodeWav(new Uint8Array(await fetchBytes(sampleAudioUrl(wakeWordId, sampleId)))).samples,
    staleTime: Infinity,
  });
}

export function Waveform({ samples, height = 40 }: { samples: Float32Array | undefined; height?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !samples) return;
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const g = canvas.getContext("2d")!;
    g.scale(dpr, dpr);
    g.fillStyle = cssVar("--accent");
    const step = Math.max(1, Math.floor(samples.length / width));
    let peak = 1e-3;
    for (const x of samples) peak = Math.max(peak, Math.abs(x));
    for (let px = 0; px < width; px++) {
      let lo = 0;
      let hi = 0;
      for (let i = px * step; i < Math.min(samples.length, (px + 1) * step); i++) {
        lo = Math.min(lo, samples[i]);
        hi = Math.max(hi, samples[i]);
      }
      const y0 = height / 2 - (hi / peak) * (height / 2 - 1);
      const y1 = height / 2 - (lo / peak) * (height / 2 - 1);
      g.fillRect(px, y0, 1, Math.max(1, y1 - y0));
    }
  }, [samples, height]);
  return <canvas ref={ref} style={{ height }} aria-hidden="true" />;
}

function PlayButton({ url }: { url: string }) {
  const [playing, setPlaying] = useState(false);
  const audio = useRef<HTMLAudioElement | null>(null);
  useEffect(() => () => audio.current?.pause(), []);
  const toggle = () => {
    if (!audio.current) {
      audio.current = new Audio(url);
      audio.current.onended = () => setPlaying(false);
    }
    if (playing) {
      audio.current.pause();
      audio.current.currentTime = 0;
      setPlaying(false);
    } else {
      void audio.current.play();
      setPlaying(true);
    }
  };
  return (
    <button className="btn btn-small" onClick={toggle} aria-label={playing ? "Stop" : "Play"}>
      {playing ? "■" : "▶"}
    </button>
  );
}

function useDeleteSample(wakeWordId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (sampleId: string) => api.deleteSample(wakeWordId, sampleId),
    onSuccess: () => client.invalidateQueries({ queryKey: ["wakeword", wakeWordId] }),
  });
}

export function ClipGrid({ wakeWordId, samples, weak }: { wakeWordId: string; samples: Sample[]; weak?: Set<string> }) {
  const remove = useDeleteSample(wakeWordId);
  if (!samples.length) return null;
  return (
    <div className="clips" data-testid="clips-positive">
      {samples.map((s) => (
        <Clip key={s.id} wakeWordId={wakeWordId} sample={s} weak={weak?.has(s.id)}
              onDelete={() => remove.mutate(s.id)} deleting={remove.isPending && remove.variables === s.id} />
      ))}
    </div>
  );
}

function Clip({ wakeWordId, sample, weak, onDelete, deleting }: {
  wakeWordId: string; sample: Sample; weak?: boolean; onDelete: () => void; deleting: boolean;
}) {
  const audio = useClipAudio(wakeWordId, sample.id);
  const cls = ["clip", !sample.accepted && "rejected", sample.outlier && "outlier"].filter(Boolean).join(" ");
  return (
    <div className={cls} data-testid="clip" data-accepted={sample.accepted} data-outlier={sample.outlier}>
      <Waveform samples={audio.data} />
      <div className="row small">
        {!sample.accepted ? (
          <span className="badge badge-bad">{REASONS[sample.reason ?? ""] ?? "Rejected"}</span>
        ) : sample.outlier ? (
          <span className="badge badge-warn" title="Sounds unlike your other recordings: left out of training">
            Unlike the others
          </span>
        ) : weak ? (
          <span className="badge badge-warn" title="The model found this one hard to detect">Weak</span>
        ) : (
          <span className="badge badge-ok">Good</span>
        )}
        <span className="muted">{seconds(sample.speech_s)}</span>
      </div>
      <div className="clip-actions">
        <PlayButton url={sampleAudioUrl(wakeWordId, sample.id)} />
        <span className="spacer" />
        <button className="btn btn-small btn-ghost btn-danger" onClick={onDelete} disabled={deleting}
                aria-label="Delete this recording">
          Delete
        </button>
      </div>
    </div>
  );
}

export function SampleRows({ wakeWordId, samples }: { wakeWordId: string; samples: Sample[] }) {
  const remove = useDeleteSample(wakeWordId);
  if (!samples.length) return null;
  return (
    <ul className="list" data-testid={`samples-${samples[0].kind}`}>
      {samples.map((s) => (
        <li key={s.id} data-testid="sample-row">
          <PlayButton url={sampleAudioUrl(wakeWordId, s.id)} />
          <span>{seconds(s.duration_s)}</span>
          {s.kind === "other_speech" && <span className="muted small">{seconds(s.speech_s)} of speech</span>}
          <span className="spacer" />
          <button className="btn btn-small btn-ghost btn-danger" onClick={() => remove.mutate(s.id)}
                  disabled={remove.isPending} aria-label="Delete this recording">
            Delete
          </button>
        </li>
      ))}
    </ul>
  );
}
