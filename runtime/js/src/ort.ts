// The wasm-only build (ort-wasm-simd-threaded.wasm, ~14 MB) instead of the default WebGPU/WebNN
// one (~28 MB). Under Node, vitest aliases this to onnxruntime-web's Node entry, which loads the
// same .wasm binary, so the tests exercise the exact kernels browsers run.
import * as ort from "onnxruntime-web/wasm";

export { ort };

/** Raw model bytes, or a URL to fetch them from. */
export type ModelSource = Uint8Array | ArrayBuffer | string;

let configured = false;

/** wasm only, single-threaded: needs no COOP/COEP headers and is plenty for these models. */
export function configureOrt(): void {
  if (configured) return;
  ort.env.wasm.numThreads = 1;
  ort.env.logLevel = "error"; // Silero ships unused initializers; silence the warnings
  configured = true;
}

export async function loadBytes(source: ModelSource): Promise<Uint8Array> {
  if (source instanceof Uint8Array) return source;
  if (source instanceof ArrayBuffer) return new Uint8Array(source);
  const res = await fetch(source);
  if (!res.ok) throw new Error(`fetching ${source}: HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

export async function createSession(bytes: Uint8Array): Promise<ort.InferenceSession> {
  configureOrt();
  return ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
}
