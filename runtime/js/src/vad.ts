import { createSession, loadBytes, ort, type ModelSource } from "./ort.js";
import { SAMPLE_RATE, VAD_CONTEXT, VAD_FRAME } from "./spec.js";

/**
 * Streaming Silero VAD (16 kHz). One call per 512-sample frame. Like the official wrapper,
 * each call prepends the last 64 samples of the previous call and carries the state forward.
 */
export class SileroVad {
  private state = new Float32Array(2 * 128);
  private context = new Float32Array(VAD_CONTEXT);
  private readonly input = new Float32Array(VAD_CONTEXT + VAD_FRAME);
  private readonly hasSr: boolean;

  private constructor(private readonly session: ort.InferenceSession) {
    this.hasSr = session.inputNames.includes("sr");
  }

  static async create(model: ModelSource): Promise<SileroVad> {
    return new SileroVad(await createSession(await loadBytes(model)));
  }

  reset(): void {
    this.state = new Float32Array(2 * 128);
    this.context = new Float32Array(VAD_CONTEXT);
  }

  async process(frame: Float32Array): Promise<number> {
    if (frame.length !== VAD_FRAME) throw new Error(`expected ${VAD_FRAME} samples, got ${frame.length}`);
    this.input.set(this.context, 0);
    this.input.set(frame, VAD_CONTEXT);
    const feeds: Record<string, ort.Tensor> = {
      input: new ort.Tensor("float32", this.input.slice(), [1, VAD_CONTEXT + VAD_FRAME]),
      state: new ort.Tensor("float32", this.state, [2, 1, 128]),
    };
    if (this.hasSr) feeds.sr = new ort.Tensor("int64", BigInt64Array.from([BigInt(SAMPLE_RATE)]), []);
    const out = await this.session.run(feeds);
    this.state = (out.stateN.data as Float32Array).slice();
    this.context = this.input.slice(VAD_FRAME); // last 64 samples of this call's input
    return (out.output.data as Float32Array)[0];
  }
}
