// Speech regions from per-frame VAD probabilities, cut exactly as server/audio.py
// (`speech_regions`) cuts uploads, so the app's live utterance counter matches what the
// server keeps: a region starts at p >= threshold and ends after `minSilenceS` below
// `negThreshold`, so the short pause inside a phrase such as "hey jarvis" doesn't split it.

export const FRAME = 512;
export const SAMPLE_RATE = 16000;

export interface SegmenterOptions {
  threshold: number;
  negThreshold: number;
  minSpeechS: number;
  minSilenceS: number;
}

export const DEFAULT_SEGMENTER: SegmenterOptions = {
  threshold: 0.5,
  negThreshold: 0.35,
  minSpeechS: 0.15,
  minSilenceS: 0.35,
};

/** Half-open range of VAD frames. */
export interface Region {
  start: number;
  end: number;
}

export class SpeechSegmenter {
  readonly regions: Region[] = []; // closed and long enough
  private frame = 0;
  private start: number | null = null;
  private quietFrom: number | null = null;
  private readonly minSilence: number;

  constructor(readonly options: SegmenterOptions = DEFAULT_SEGMENTER) {
    this.minSilence = Math.ceil((options.minSilenceS * SAMPLE_RATE) / FRAME);
  }

  /** Feed one frame's probability; returns the region it closes, if any. */
  push(p: number): Region | null {
    const i = this.frame++;
    if (this.start === null) {
      if (p >= this.options.threshold) [this.start, this.quietFrom] = [i, null];
      return null;
    }
    if (p < this.options.negThreshold) {
      this.quietFrom ??= i;
      if (i - this.quietFrom + 1 >= this.minSilence) return this.close(this.quietFrom);
    } else {
      this.quietFrom = null;
    }
    return null;
  }

  /** Close a region still open at the end of the recording. */
  finish(): Region[] {
    if (this.start !== null) this.close(this.quietFrom ?? this.frame);
    return this.regions;
  }

  get speaking(): boolean {
    return this.start !== null && this.quietFrom === null;
  }

  /** Length of the region in progress, in seconds (0 when silent). */
  get currentS(): number {
    return this.start === null ? 0 : ((this.frame - this.start) * FRAME) / SAMPLE_RATE;
  }

  get frames(): number {
    return this.frame;
  }

  private close(end: number): Region | null {
    const region = { start: this.start!, end };
    this.start = null;
    this.quietFrom = null;
    if ((region.end - region.start) * FRAME < this.options.minSpeechS * SAMPLE_RATE) return null;
    this.regions.push(region);
    return region;
  }
}

/** Batch form, as server/audio.py `speech_regions`: sample ranges. */
export function speechRegions(probs: ArrayLike<number>, options = DEFAULT_SEGMENTER): [number, number][] {
  const seg = new SpeechSegmenter(options);
  for (let i = 0; i < probs.length; i++) seg.push(probs[i]);
  return seg.finish().map((r) => [r.start * FRAME, r.end * FRAME]);
}

export const regionSeconds = (r: Region) => ((r.end - r.start) * FRAME) / SAMPLE_RATE;
