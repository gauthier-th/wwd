"""Upload checks (PLAN.md §9.2): decode, resample to 16 kHz, cut with the runtime VAD (the
same file the browser runs), normalize loudness, reject unusable clips, flag outliers."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import soxr

from runtime.spec import SAMPLE_RATE as SR
from runtime.spec import VAD_FRAME
from runtime.vad import SileroVad

WINDOW_S = 1.5
MAX_WORD_S = 1.3  # speech longer than this can't sit inside the 1.5 s window with margins
MIN_WORD_S = 0.2
PAD_S = 0.2  # kept around the VAD's speech region (its onset lags by about a frame)
FADE_S = 0.01
TARGET_DBFS = -20.0  # speech RMS after normalization
PEAK_LIMIT = 10 ** (-1 / 20)
CLIP_LEVEL = 0.999
CLIP_FRACTION = 1e-3  # this share of samples at full scale = clipped
QUIET_DBFS = -55.0
SILENT_DBFS = -70.0
MIN_OTHER_SPEECH_S = 2.0
# Outliers among positives (mispronounced, wrong word, noise burst): cosine to the prototype
# of the other recordings below an absolute floor, or well below the typical recording.
OUTLIER_FLOOR = 0.5
OUTLIER_DROP = 0.2

KINDS = ("positive", "other_speech", "ambience")


class UploadError(ValueError):
    """The whole upload is unusable (surfaced as HTTP 422)."""


@dataclass
class Clip:
    audio: np.ndarray  # float32, 16 kHz
    start_s: float  # position in the upload
    speech_s: float
    accepted: bool
    reason: str | None = None

    @property
    def duration_s(self) -> float:
        return len(self.audio) / SR

    @property
    def peak_dbfs(self) -> float:
        return dbfs(np.abs(self.audio).max(initial=0.0))

    @property
    def rms_dbfs(self) -> float:
        return dbfs(rms(self.audio))


def dbfs(value: float) -> float:
    return float(20 * np.log10(max(float(value), 1e-10)))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def decode(data: bytes, max_s: float) -> np.ndarray:
    """Any format libsndfile reads (WAV, FLAC, OGG) -> mono float32 at 16 kHz."""
    try:
        audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError, TypeError) as e:
        raise UploadError(f"unreadable audio file: {e}") from e
    if len(audio) == 0:
        raise UploadError("empty audio file")
    if len(audio) / rate > max_s:
        raise UploadError(f"recording too long: {len(audio) / rate:.0f} s > {max_s:.0f} s")
    audio = audio.mean(axis=1)
    if rate != SR:
        audio = soxr.resample(audio, rate, SR, quality="HQ")
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def encode_wav(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1.0, 1.0), SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def vad_probs(audio: np.ndarray, vad: SileroVad) -> np.ndarray:
    vad.reset()
    n = len(audio) // VAD_FRAME
    return np.array([vad(audio[i * VAD_FRAME : (i + 1) * VAD_FRAME]) for i in range(n)])


def speech_regions(probs: np.ndarray, *, threshold: float = 0.5, neg_threshold: float = 0.35,
                   min_speech_s: float = 0.15, min_silence_s: float = 0.35) -> list[tuple[int, int]]:
    """Speech as sample ranges, with Silero's hysteresis: a region starts at p >= threshold and
    ends after `min_silence_s` below `neg_threshold`, so the short pause inside a phrase such as
    "hey jarvis" doesn't split it."""
    min_silence = int(np.ceil(min_silence_s * SR / VAD_FRAME))
    regions, start, quiet_from = [], None, None
    for i, p in enumerate(probs):
        if start is None:
            if p >= threshold:
                start, quiet_from = i, None
        elif p < neg_threshold:
            quiet_from = i if quiet_from is None else quiet_from
            if i - quiet_from + 1 >= min_silence:
                regions.append((start, quiet_from))
                start = None
        else:
            quiet_from = None
    if start is not None:
        regions.append((start, quiet_from if quiet_from is not None else len(probs)))
    return [(a * VAD_FRAME, b * VAD_FRAME) for a, b in regions
            if (b - a) * VAD_FRAME >= min_speech_s * SR]


def _fade(x: np.ndarray) -> np.ndarray:
    n = min(int(FADE_S * SR), len(x) // 2)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n, dtype=np.float32))
    x = x.copy()
    x[:n] *= ramp
    x[len(x) - n :] *= ramp[::-1]
    return x


def _normalize(x: np.ndarray, speech: np.ndarray) -> np.ndarray:
    gain = 10 ** (TARGET_DBFS / 20) / max(rms(speech), 1e-6)
    gain = min(gain, PEAK_LIMIT / max(np.abs(x).max(initial=0.0), 1e-6))
    return (x * gain).astype(np.float32)


def split_positives(audio: np.ndarray, vad: SileroVad) -> list[Clip]:
    """One take holding several utterances of the wake word -> one clip per utterance."""
    if dbfs(np.abs(audio).max(initial=0.0)) < SILENT_DBFS:
        raise UploadError("the recording is silent")
    regions = speech_regions(vad_probs(audio, vad))
    if not regions:
        raise UploadError("no speech detected")
    clips = []
    for k, (a, b) in enumerate(regions):
        speech_s = (b - a) / SR
        pad = int(max(0.05, min(PAD_S, (WINDOW_S - speech_s) / 2)) * SR)
        lo = max(a - pad, (regions[k - 1][1] + a) // 2 if k else 0)
        hi = min(b + pad, (b + regions[k + 1][0]) // 2 if k + 1 < len(regions) else len(audio))
        x, speech = audio[lo:hi], audio[a:b]
        reason = None
        if speech_s > MAX_WORD_S:
            reason = "too_long"
        elif speech_s < MIN_WORD_S:
            reason = "too_short"
        elif np.mean(np.abs(x) >= CLIP_LEVEL) >= CLIP_FRACTION:
            reason = "clipped"
        elif dbfs(rms(speech)) < QUIET_DBFS:
            reason = "too_quiet"
        if reason is None:
            x = _fade(_normalize(x, speech))
        clips.append(Clip(x, lo / SR, speech_s, reason is None, reason))
    return clips


def prepare_other(audio: np.ndarray, kind: str, vad: SileroVad) -> Clip:
    """Other speech (a paragraph read aloud) or room ambience: kept whole, as negatives."""
    peak = dbfs(np.abs(audio).max(initial=0.0))
    if kind == "ambience":
        if peak < -90:
            raise UploadError("the recording is digital silence")
        return Clip(audio, 0.0, 0.0, True)
    if peak < SILENT_DBFS:
        raise UploadError("the recording is silent")
    regions = speech_regions(vad_probs(audio, vad))
    speech_s = sum(b - a for a, b in regions) / SR
    if speech_s < MIN_OTHER_SPEECH_S:
        raise UploadError(f"only {speech_s:.1f} s of speech detected")
    speech = np.concatenate([audio[a:b] for a, b in regions])
    return Clip(_normalize(audio, speech), 0.0, speech_s, True)


def center_window(clip: np.ndarray, window: int = int(WINDOW_S * SR)) -> np.ndarray:
    clip = clip[:window]
    out = np.zeros(window, np.float32)
    start = (window - len(clip)) // 2
    out[start : start + len(clip)] = clip
    return out


def outliers(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out cosine of each recording to the others' prototype, and outlier flags."""
    e = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    if len(e) < 2:
        return np.ones(len(e)), np.zeros(len(e), bool)
    others = e.sum(0, keepdims=True) - e
    sims = np.sum(e * others / np.linalg.norm(others, axis=1, keepdims=True), axis=1)
    flags = sims < OUTLIER_FLOOR
    if len(e) >= 4:
        flags |= sims < np.median(sims) - OUTLIER_DROP
    return sims, flags
