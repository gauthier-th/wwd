// Main-thread glue: microphone -> capture worklet (16 kHz frames) -> a worker, either the
// detector (live test) or the recorder (enrollment takes).

import type { DetectorConfig } from "@wwd/runtime";

import workletUrl from "./capture-worklet.js?url";
import type { FromRecorder, FromWorker, RecorderStatus, ToRecorder, ToWorker } from "./protocol";

export interface DetectorOptions {
  gate?: boolean;
  overrides?: Partial<DetectorConfig>;
  clips?: { beforeS: number; afterS: number };
}

export interface DetectorHandle {
  worker: Worker;
  /** Bumped by each reconfigure: tick messages from older generations are stale. */
  generation: number;
  config: DetectorConfig;
  metadata: Record<string, string>;
  send(msg: ToWorker, transfer?: Transferable[]): void;
  onMessage(fn: (msg: FromWorker) => void): () => void;
  /** Swap model or settings without dropping the audio stream. */
  reconfigure(model: ArrayBuffer, vad: ArrayBuffer | null, options?: DetectorOptions): Promise<void>;
  terminate(): void;
}

export async function startDetector(
  model: ArrayBuffer,
  vad: ArrayBuffer | null,
  options: DetectorOptions = {},
): Promise<DetectorHandle> {
  const worker = new Worker(new URL("./detector.worker.ts", import.meta.url), { type: "module" });
  const send = (msg: ToWorker, transfer: Transferable[] = []) => worker.postMessage(msg, transfer);
  const onMessage = (fn: (msg: FromWorker) => void) => {
    const listener = (e: MessageEvent<FromWorker>) => fn(e.data);
    worker.addEventListener("message", listener);
    return () => worker.removeEventListener("message", listener);
  };
  const init = (generation: number, m: ArrayBuffer, v: ArrayBuffer | null, o: DetectorOptions) =>
    new Promise<Extract<FromWorker, { type: "ready" }>>((resolve, reject) => {
      const off = onMessage((msg) => {
        if (msg.type === "ready" && msg.generation === generation) resolve(msg);
        else if (msg.type === "error") reject(new Error(msg.message));
        else return;
        off();
      });
      send({ type: "init", generation, model: m, vad: v, gate: o.gate ?? true, overrides: o.overrides, clips: o.clips });
    });
  const ready = await init(0, model, vad, options);
  const handle: DetectorHandle = {
    worker,
    generation: 0,
    config: ready.config,
    metadata: ready.metadata,
    send,
    onMessage,
    async reconfigure(m, v, o = {}) {
      const next = await init(++handle.generation, m, v, o);
      handle.config = next.config;
      handle.metadata = next.metadata;
    },
    terminate: () => worker.terminate(),
  };
  return handle;
}

/**
 * Call synchronously from the click (or tap) handler: Safari only lets an AudioContext run if
 * it was created during a user gesture, and an `await` before it ends the gesture.
 */
export function createAudioContext(): AudioContext {
  return new AudioContext(); // device rate; the worklet resamples to 16 kHz
}

export interface MicCapture {
  context: AudioContext;
  stream: MediaStream;
  stop(): Promise<void>;
}

/** Microphone -> capture worklet; `connect` receives the port the 512-sample frames come out of. */
export async function openMic(
  context: AudioContext,
  connect: (port: MessagePort) => void,
): Promise<MicCapture> {
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        // Raw signal, like a Pi mic hears it: no browser voice processing.
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
  } catch (err) {
    await context.close();
    throw err;
  }
  if (context.state === "suspended") await context.resume();
  await context.audioWorklet.addModule(workletUrl);
  const source = context.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(context, "wwd-capture", { numberOfInputs: 1, numberOfOutputs: 0 });
  const channel = new MessageChannel();
  node.port.postMessage({ port: channel.port1 }, [channel.port1]);
  connect(channel.port2);
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

/** Route the microphone straight into a detector worker. */
export function startMic(detector: DetectorHandle, context = createAudioContext()): Promise<MicCapture> {
  return openMic(context, (port) => detector.send({ type: "port", port }, [port]));
}

export interface Recording {
  /** Stop and get the 16 kHz recording. */
  stop(): Promise<{ samples: Float32Array; status: RecorderStatus }>;
  /** Stop and throw the audio away. */
  cancel(): Promise<void>;
}

/** Record from the microphone, with the VAD running live for feedback. */
export async function startRecording(
  context: AudioContext,
  vad: ArrayBuffer,
  onStatus: (status: RecorderStatus) => void,
): Promise<Recording> {
  const worker = new Worker(new URL("./recorder.worker.ts", import.meta.url), { type: "module" });
  const send = (msg: ToRecorder, transfer: Transferable[] = []) => worker.postMessage(msg, transfer);
  let done: ((value: { samples: Float32Array; status: RecorderStatus }) => void) | null = null;
  let failed: ((err: Error) => void) | null = null;
  const ready = new Promise<void>((resolve, reject) => {
    worker.onmessage = (e: MessageEvent<FromRecorder>) => {
      const msg = e.data;
      if (msg.type === "ready") resolve();
      else if (msg.type === "status") onStatus(msg);
      else if (msg.type === "done") done?.({ samples: msg.samples, status: msg });
      else if (msg.type === "error") {
        const err = new Error(msg.message);
        reject(err);
        failed?.(err);
      }
    };
  });
  send({ type: "init", vad: vad.slice(0) });
  let mic: MicCapture;
  try {
    await ready;
    mic = await openMic(context, (port) => send({ type: "port", port }, [port]));
  } catch (err) {
    worker.terminate();
    throw err;
  }
  return {
    async stop() {
      await mic.stop(); // no more frames
      const result = await new Promise<{ samples: Float32Array; status: RecorderStatus }>((resolve, reject) => {
        done = resolve;
        failed = reject;
        send({ type: "stop" });
      });
      worker.terminate();
      return result;
    },
    async cancel() {
      await mic.stop();
      worker.terminate();
    },
  };
}
