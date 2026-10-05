// AudioWorklet: mic input at the context's native rate -> 16 kHz mono -> 512-sample frames.
// Plain JS with no imports: worklet modules load outside the bundler's module graph.
//
// Resampling uses a Blackman-windowed sinc low-pass (cutoff 7.2 kHz) evaluated at each output
// instant. The context keeps the device rate, because Firefox refuses to connect a mic stream
// to an AudioContext running at a different sample rate.

const OUT_RATE = 16000;
const FRAME = 512;
const HALF_WIDTH = 16; // kernel half-width, in output-sample periods
const CUTOFF = 0.45; // of the output rate (7.2 kHz)
const TABLE_RES = 128; // kernel table entries per input sample

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / OUT_RATE; // input samples per output sample
    this.support = Math.ceil(HALF_WIDTH * Math.max(this.ratio, 1)); // input samples each side
    this.scale = Math.min(1, 1 / this.ratio); // low-pass at the lower of the two Nyquists
    this.table = this.buildTable();
    this.buf = new Float32Array(0);
    this.base = 0; // absolute input index of buf[0]
    this.next = 0; // absolute (fractional) input index of the next output sample
    this.frame = new Float32Array(FRAME);
    this.fill = 0;
    this.target = this.port; // replaced by a direct MessagePort to the detector worker
    this.port.onmessage = (e) => {
      if (e.data && e.data.port) this.target = e.data.port;
    };
  }

  buildTable() {
    // kernel(x) for x in [-support, support] input samples, sampled every 1/TABLE_RES
    const span = this.support;
    const fc = 2 * CUTOFF * this.scale;
    const table = new Float32Array(2 * span * TABLE_RES + 1);
    for (let k = 0; k < table.length; k++) {
      const x = k / TABLE_RES - span;
      const t = fc * x;
      const sinc = t === 0 ? 1 : Math.sin(Math.PI * t) / (Math.PI * t);
      const w = (x / span + 1) / 2; // 0..1 across the window
      const blackman = 0.42 - 0.5 * Math.cos(2 * Math.PI * w) + 0.08 * Math.cos(4 * Math.PI * w);
      table[k] = fc * sinc * blackman;
    }
    return table;
  }

  kernel(x) {
    // x in input samples relative to the output instant
    const k = Math.round((x + this.support) * TABLE_RES);
    return k >= 0 && k < this.table.length ? this.table[k] : 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const chunk = input[0]; // first channel; mics are mono in practice

    const merged = new Float32Array(this.buf.length + chunk.length);
    merged.set(this.buf, 0);
    merged.set(chunk, this.buf.length);
    this.buf = merged;

    const end = this.base + this.buf.length; // absolute index one past the newest sample
    while (this.next + this.support < end) {
      const center = this.next;
      const lo = Math.max(Math.ceil(center - this.support), this.base);
      const hi = Math.floor(center + this.support);
      let acc = 0;
      for (let i = lo; i <= hi; i++) acc += this.buf[i - this.base] * this.kernel(i - center);
      this.frame[this.fill++] = acc;
      if (this.fill === FRAME) {
        const out = this.frame;
        this.target.postMessage(out, [out.buffer]);
        this.frame = new Float32Array(FRAME);
        this.fill = 0;
      }
      this.next += this.ratio;
    }

    // Drop input no future output sample can reach.
    const keepFrom = Math.max(Math.floor(this.next - this.support) - 1, this.base);
    if (keepFrom > this.base) {
      this.buf = this.buf.slice(keepFrom - this.base);
      this.base = keepFrom;
    }
    return true;
  }
}

registerProcessor("wwd-capture", CaptureProcessor);
