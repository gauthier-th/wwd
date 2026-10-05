// Main-thread glue: detector worker + microphone capture worklet.

import type { DetectorConfig } from "@wwd/runtime";

import workletUrl from "./capture-worklet.js?url";
import type { FromWorker, ToWorker } from "./protocol";

export interface DetectorHandle {
  worker: Worker;
  config: DetectorConfig;
  metadata: Record<string, string>;
  send(msg: ToWorker, transfer?: Transferable[]): void;
  onMessage(fn: (msg: FromWorker) => void): () => void;
  terminate(): void;
}

export async function startDetector(
  model: ArrayBuffer,
  vad: ArrayBuffer | null,
  options: { gate?: boolean; overrides?: Partial<DetectorConfig> } = {},
): Promise<DetectorHandle> {
  const worker = new Worker(new URL("./detector.worker.ts", import.meta.url), { type: "module" });
  const send = (msg: ToWorker, transfer: Transferable[] = []) => worker.postMessage(msg, transfer);
  const onMessage = (fn: (msg: FromWorker) => void) => {
    const listener = (e: MessageEvent<FromWorker>) => fn(e.data);
    worker.addEventListener("message", listener);
    return () => worker.removeEventListener("message", listener);
  };
  const ready = await new Promise<Extract<FromWorker, { type: "ready" }>>((resolve, reject) => {
    const off = onMessage((msg) => {
      if (msg.type === "ready") resolve(msg);
      else if (msg.type === "error") reject(new Error(msg.message));
      if (msg.type !== "ticks") off();
    });
    send({ type: "init", model, vad, gate: options.gate ?? true, overrides: options.overrides });
  });
  return {
    worker,
    config: ready.config,
    metadata: ready.metadata,
    send,
    onMessage,
    terminate: () => worker.terminate(),
  };
}

export interface MicCapture {
  context: AudioContext;
  stream: MediaStream;
  stop(): Promise<void>;
}

/** Route the microphone, resampled to 16 kHz in an AudioWorklet, straight into the worker. */
export async function startMic(detector: DetectorHandle): Promise<MicCapture> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      // Raw signal, like a Pi mic hears it: no browser voice processing.
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });
  const context = new AudioContext(); // device rate; the worklet resamples
  await context.audioWorklet.addModule(workletUrl);
  const source = context.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(context, "wwd-capture", { numberOfInputs: 1, numberOfOutputs: 0 });
  const channel = new MessageChannel();
  node.port.postMessage({ port: channel.port1 }, [channel.port1]);
  detector.send({ type: "port", port: channel.port2 }, [channel.port2]);
  source.connect(node);
  return {
    context,
    stream,
    async stop() {
      source.disconnect();
      node.disconnect();
      stream.getTracks().forEach((t) => t.stop());
      await context.close();
    },
  };
}
