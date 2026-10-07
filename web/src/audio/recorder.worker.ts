// Recording: frames from the capture worklet, the Silero VAD on each one (the same file the
// server cuts uploads with), and the speech segmenter behind the live utterance counter.

import { SileroVad } from "@wwd/runtime";

import type { FromRecorder, RecorderStatus, ToRecorder } from "./protocol";
import { FRAME, regionSeconds, SAMPLE_RATE, SpeechSegmenter } from "./segments";

const STATUS_EVERY = 3; // frames: one update per 96 ms

let vad: Promise<SileroVad> | null = null;
let queue: Promise<void> = Promise.resolve();
const chunks: Float32Array[] = [];
const segmenter = new SpeechSegmenter();
let lastSpeechFrame = -1;
let peak = 0;
let vadMax = 0;

const post = (msg: FromRecorder, transfer: Transferable[] = []) => self.postMessage(msg, transfer);

function status(): RecorderStatus {
  const frames = segmenter.frames;
  const regions = segmenter.regions;
  return {
    seconds: (frames * FRAME) / SAMPLE_RATE,
    levelDb: 20 * Math.log10(Math.max(peak, 1e-5)),
    vad: vadMax,
    speaking: segmenter.speaking,
    utterances: regions.length,
    speechS: regions.reduce((s, r) => s + regionSeconds(r), 0) + segmenter.currentS,
    currentS: segmenter.currentS,
    longestS: Math.max(0, ...regions.map(regionSeconds)),
    sinceSpeechS: ((frames - 1 - lastSpeechFrame) * FRAME) / SAMPLE_RATE,
  };
}

function onFrame(frame: Float32Array): void {
  queue = queue
    .then(async () => {
      const p = await (await vad!).process(frame);
      chunks.push(frame);
      if (p >= segmenter.options.threshold) lastSpeechFrame = segmenter.frames;
      segmenter.push(p);
      for (const x of frame) peak = Math.max(peak, Math.abs(x));
      vadMax = Math.max(vadMax, p);
      if (segmenter.frames % STATUS_EVERY === 0) {
        post({ type: "status", ...status() });
        peak = 0;
        vadMax = 0;
      }
    })
    .catch((err) => post({ type: "error", message: String(err) }));
}

self.onmessage = (e: MessageEvent<ToRecorder>) => {
  const msg = e.data;
  switch (msg.type) {
    case "init":
      vad = SileroVad.create(msg.vad);
      vad.then(() => post({ type: "ready" })).catch((err) => post({ type: "error", message: String(err) }));
      break;
    case "port":
      msg.port.onmessage = (ev: MessageEvent<Float32Array>) => onFrame(ev.data);
      break;
    case "stop":
      queue = queue.then(() => {
        segmenter.finish();
        const samples = new Float32Array(chunks.length * FRAME);
        chunks.forEach((c, i) => samples.set(c, i * FRAME));
        post({ type: "done", samples, ...status() }, [samples.buffer]);
      });
      break;
  }
};
