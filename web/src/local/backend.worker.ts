// The compute half of the local backend: upload checks (VAD), embeddings for the outlier
// check, and trainings. The base model's large files load on the first training and stay.

import { SileroVad } from "@wwd/runtime";

import { EMBED_FILES, finetuneConfig, loadTrainAssets, TRAIN_FILES, type Manifest, type TrainAssets } from "./assets";
import { centerWindow, prepareOther, splitPositives, UploadError } from "./checks";
import { EncoderSession } from "./embed";
import { browserFetcher, cachedBytes, loadManifest, pruneCache } from "./fetcher";
import { PoolEmbedder } from "./pool";
import type { FromBackend, Request, Responses, ToBackend } from "./protocol";
import { Cancelled, train } from "./train";

const post = (msg: FromBackend, transfer: Transferable[] = []) => self.postMessage(msg, transfer);

class HttpError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
  }
}

let base: Promise<{ root: URL; manifest: Manifest }> | null = null;
let vad: Promise<SileroVad> | null = null;
let encoder: Promise<EncoderSession> | null = null;
let assets: Promise<TrainAssets> | null = null;
const cancelled = new Set<string>();

const getBase = () => (base ??= loadManifest().then(async (b) => {
  void pruneCache(b.root, b.manifest);
  return b;
}));
const fetcherOf = async () => {
  const { root, manifest } = await getBase();
  return browserFetcher(root, manifest);
};
const getVad = () => (vad ??= fetcherOf().then(async (f) => SileroVad.create(await f("vad.onnx"))));
const getEncoder = () => (encoder ??= getBase().then(async ({ manifest }) =>
  EncoderSession.create(await (await fetcherOf())("encoder.onnx"), manifest.embedding_dim)));

/** The large files, with download progress for the first time. */
function getAssets(): Promise<TrainAssets> {
  return (assets ??= (async () => {
    const { root, manifest } = await getBase();
    const names = TRAIN_FILES(manifest);
    const total = names.reduce((n, f) => n + (manifest.files[f]?.bytes ?? 0), 0);
    const have = await cachedBytes(root, manifest, names);
    const seen = new Map<string, number>();
    let last = 0;
    const progress = (name: string, n: number) => {
      seen.set(name, n);
      const loaded = [...seen.values()].reduce((a, b) => a + b, 0);
      if (have < total && (loaded - last > 2e6 || loaded === total)) {
        last = loaded;
        post({ type: "download", loaded, total });
      }
    };
    try {
      const fetcher = browserFetcher(root, manifest);
      await Promise.all(EMBED_FILES(manifest).map((f) => fetcher(f, (n) => progress(f, n))));
      return await loadTrainAssets(fetcher, manifest, progress);
    } catch (err) {
      assets = null; // try again next time
      throw err;
    }
  })());
}

async function handle(req: Request): Promise<[unknown, Transferable[]]> {
  switch (req.type) {
    case "init": {
      const { manifest } = await getBase();
      return [{ version: manifest.version, manifest } satisfies Responses["init"], []];
    }
    case "checks": {
      try {
        const v = await getVad();
        const clips = req.kind === "positive" ? await splitPositives(req.audio, v)
          : [await prepareOther(req.audio, req.kind, v)];
        const out = clips.map((c) => ({ audio: c.audio, start_s: c.startS, speech_s: c.speechS,
                                         accepted: c.accepted, reason: c.reason }));
        return [{ clips: out } satisfies Responses["checks"], out.map((c) => c.audio.buffer)];
      } catch (err) {
        if (err instanceof UploadError) throw new HttpError(422, err.message);
        throw err;
      }
    }
    case "embed_clean": {
      const session = await getEncoder();
      const emb = await session.run(req.clips.map((c) => centerWindow(c)));
      const embeddings = req.clips.map((_, i) => emb.slice(i * session.dim, (i + 1) * session.dim));
      return [{ embeddings } satisfies Responses["embed_clean"], []];
    }
    case "train": {
      const progress = (stage: string, value: number, message: string) =>
        post({ type: "progress", trainingId: req.trainingId, stage, progress: value, message });
      progress("loading", 0.01, "Loading the base model");
      const a = await getAssets();
      const { root, manifest } = await getBase();
      const pool = new PoolEmbedder(root, manifest);
      try {
        const res = await train(req.job, a, pool, progress, () => cancelled.has(req.trainingId),
                                finetuneConfig(manifest));
        return [res satisfies Responses["train"], [res.model.buffer]];
      } finally {
        pool.close();
        cancelled.delete(req.trainingId);
      }
    }
  }
}

self.onmessage = async (e: MessageEvent<ToBackend>) => {
  const msg = e.data;
  if (msg.type === "cancel") {
    cancelled.add(msg.trainingId);
    return;
  }
  try {
    const [result, transfer] = await handle(msg);
    post({ type: "reply", id: msg.id, ok: true, result }, transfer);
  } catch (err) {
    const status = err instanceof HttpError ? err.status : err instanceof Cancelled ? 499 : 500;
    post({ type: "reply", id: msg.id, ok: false, status, detail: err instanceof Error ? err.message : String(err) });
  }
};
