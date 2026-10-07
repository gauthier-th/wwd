// Runs @wwd/runtime (onnxruntime-web, wasm) off the main thread. Audio arrives either from the
// main thread or straight from the capture worklet through a MessagePort.

import { Detector, type WakeEvent } from "@wwd/runtime";

import type { FromWorker, ToWorker } from "./protocol";

let detector: Detector | null = null;
let generation = 0;
let queue: Promise<void> = Promise.resolve(); // process() is async: keep chunks in order
let waiting = 0; // chunks enqueued and not processed yet
let pendingMs = 0; // processing time not yet attributed to a tick

/** The last few seconds of audio, so a detection can be sent with what was said around it. */
class Ring {
  readonly buf: Float32Array;
  end = 0; // samples pushed since the detector started
  constructor(readonly size: number) {
    this.buf = new Float32Array(size);
  }
  push(x: Float32Array): void {
    for (let i = 0; i < x.length; i++) this.buf[(this.end + i) % this.size] = x[i];
    this.end += x.length;
  }
  slice(start: number, stop: number): Float32Array {
    start = Math.max(start, this.end - this.size, 0);
    stop = Math.min(stop, this.end);
    const out = new Float32Array(Math.max(0, stop - start));
    for (let i = 0; i < out.length; i++) out[i] = this.buf[(start + i) % this.size];
    return out;
  }
}
let clips: { before: number; after: number; ring: Ring; pending: WakeEvent[] } | null = null;

const post = (msg: FromWorker, transfer: Transferable[] = []) => self.postMessage(msg, transfer);

function enqueue(samples: Float32Array): void {
  waiting++;
  queue = queue.then(async () => {
    waiting--;
    // An init may swap the detector while this chunk is being scored: stick to this one.
    const [det, gen, clip] = [detector, generation, clips];
    if (!det) return;
    const started = performance.now();
    const events = await det.process(samples);
    pendingMs += performance.now() - started;
    const ticks = det.ticks.splice(0);
    const processMs = ticks.map(() => pendingMs / ticks.length);
    if (ticks.length) pendingMs = 0;
    if (ticks.length || events.length) {
      post({ type: "ticks", generation: gen, ticks, events, samples: samples.length, processMs, queued: waiting });
    }
    if (clip) {
      clip.ring.push(samples);
      clip.pending.push(...events);
      while (clip.pending.length && clip.pending[0].sample + clip.after <= clip.ring.end) {
        const e = clip.pending.shift()!;
        const audio = clip.ring.slice(e.sample - clip.before, e.sample + clip.after);
        post({ type: "clip", generation: gen, tick: e.tick, samples: audio }, [audio.buffer]);
      }
    }
  }).catch((err) => post({ type: "error", message: String(err) }));
}

self.onmessage = async (e: MessageEvent<ToWorker>) => {
  const msg = e.data;
  try {
    switch (msg.type) {
      case "init":
        detector = await Detector.create(msg.model, msg.vad, {
          gate: msg.gate,
          overrides: msg.overrides,
          recordTicks: true,
        });
        generation = msg.generation;
        pendingMs = 0;
        clips = msg.clips
          ? {
              before: Math.round(msg.clips.beforeS * 16000),
              after: Math.round(msg.clips.afterS * 16000),
              ring: new Ring(Math.round((msg.clips.beforeS + msg.clips.afterS + 1) * 16000)),
              pending: [],
            }
          : null;
        post({ type: "ready", generation, config: detector.config, metadata: detector.metadata });
        break;
      case "audio":
        enqueue(msg.samples);
        break;
      case "port":
        msg.port.onmessage = (ev: MessageEvent<Float32Array>) => enqueue(ev.data);
        break;
      case "reset":
        queue = queue.then(() => detector?.reset());
        break;
    }
  } catch (err) {
    post({ type: "error", message: String(err) });
  }
};
