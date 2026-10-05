// Detector contract shared by every runtime. Mirrors runtime/SPEC.md and runtime/spec.py.
// Config keys stay snake_case on purpose: they are the ONNX metadata keys, identical across
// languages.

export const FORMAT_VERSION = "1";
export const SAMPLE_RATE = 16000;
export const VAD_FRAME = 512; // new samples per VAD call (32 ms)
export const VAD_CONTEXT = 64; // samples of the previous call prepended to each VAD call

export interface DetectorConfig {
  readonly sample_rate: number;
  readonly window_samples: number;
  readonly hop_samples: number;
  readonly threshold: number;
  readonly smoothing_k: number;
  readonly smoothing_n: number;
  readonly refractory_ms: number;
  readonly vad_threshold: number;
  readonly vad_hangover_ms: number;
}

export const DEFAULT_CONFIG: DetectorConfig = Object.freeze({
  sample_rate: SAMPLE_RATE,
  window_samples: 24000,
  hop_samples: 1536,
  threshold: 0.5,
  smoothing_k: 2,
  smoothing_n: 3,
  refractory_ms: 1500,
  vad_threshold: 0.5,
  vad_hangover_ms: 1500,
});

const CONFIG_KEYS = Object.keys(DEFAULT_CONFIG) as (keyof DetectorConfig)[];

export function makeConfig(overrides: Partial<DetectorConfig> = {}, base = DEFAULT_CONFIG): DetectorConfig {
  const defined = Object.fromEntries(Object.entries(overrides).filter(([, v]) => v !== undefined));
  const config = Object.freeze({ ...base, ...defined });
  if (config.sample_rate !== SAMPLE_RATE) throw new Error(`sample_rate must be ${SAMPLE_RATE}`);
  if (config.hop_samples % VAD_FRAME) throw new Error(`hop_samples must be a multiple of ${VAD_FRAME}`);
  if (!(1 <= config.smoothing_k && config.smoothing_k <= config.smoothing_n)) {
    throw new Error("need 1 <= smoothing_k <= smoothing_n");
  }
  return config;
}

/** Build from ONNX metadata_props / model.json; unknown keys are ignored. */
export function configFromMetadata(meta: Record<string, string>): DetectorConfig {
  const overrides: Record<string, number> = {};
  for (const key of CONFIG_KEYS) {
    if (key in meta) overrides[key] = Number(meta[key]);
  }
  return makeConfig(overrides);
}

// Same operation order as runtime/spec.py, so float rounding is identical.
export const framesPerTick = (c: DetectorConfig) => c.hop_samples / VAD_FRAME;
export const gateFrames = (c: DetectorConfig) =>
  Math.ceil((c.vad_hangover_ms * c.sample_rate) / 1000 / VAD_FRAME);
export const refractoryTicks = (c: DetectorConfig) =>
  Math.ceil((c.refractory_ms * c.sample_rate) / 1000 / c.hop_samples);
