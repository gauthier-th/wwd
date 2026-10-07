// Runs @wwd/runtime (onnxruntime-web, wasm) off the main thread. Audio arrives either from the
// main thread or straight from the capture worklet through a MessagePort.

import { Detector } from "@wwd/runtime";

import type { FromWorker, ToWorker } from "./protocol";

let detector: Detector | null = null;
let generation = 0;
let queue: Promise<void> = Promise.resolve(); // process() is async: keep chunks in order

const post = (msg: FromWorker) => self.postMessage(msg);

function enqueue(samples: Float32Array): void {
  queue = queue.then(async () => {
    // An init may swap the detector while this chunk is being scored: stick to this one.
    const [det, gen] = [detector, generation];
    if (!det) return;
    const events = await det.process(samples);
    const ticks = det.ticks.splice(0);
    if (ticks.length || events.length) post({ type: "ticks", generation: gen, ticks, events, samples: samples.length });
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
