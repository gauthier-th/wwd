"""Fast Ogg Opus -> 16 kHz decoding with the system libopus (ctypes).

libopus decodes natively at 16 kHz, so there is no 48 kHz decode and no resampling step:
about 5x faster than soundfile + soxr, which matters because decoding is what limits training
throughput. Falls back to None when libopus isn't installed; callers then use soundfile.
"""

from __future__ import annotations

import ctypes
import ctypes.util

import numpy as np

RATE = 16000
_MAX_FRAME = 5760  # 120 ms at 48 kHz: the largest Opus frame
_OPUS_RESET_STATE = 4028


def _load() -> ctypes.CDLL | None:
    name = ctypes.util.find_library("opus") or "libopus.so.0"
    try:
        lib = ctypes.CDLL(name)
    except OSError:
        return None
    lib.opus_decoder_create.restype = ctypes.c_void_p
    lib.opus_decoder_create.argtypes = [ctypes.c_int32, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.opus_decode_float.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
                                      ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
    lib.opus_decoder_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int]
    return lib


_lib = _load()
AVAILABLE = _lib is not None


def _packets(data: bytes) -> tuple[list[bytes], int]:
    """Ogg demux: all packets, plus the granule position of the last page."""
    pos, packet, packets, granule = 0, b"", [], 0
    while pos + 27 <= len(data):
        if data[pos : pos + 4] != b"OggS":
            raise ValueError(f"bad Ogg page at byte {pos}")
        granule = int.from_bytes(data[pos + 6 : pos + 14], "little", signed=True)
        nseg = data[pos + 26]
        segments = data[pos + 27 : pos + 27 + nseg]
        pos += 27 + nseg
        for size in segments:
            packet += data[pos : pos + size]
            pos += size
            if size < 255:
                packets.append(packet)
                packet = b""
    return packets, granule


class OpusDecoder:
    """Mono Ogg Opus bytes -> float32 at 16 kHz, trimmed like any conforming decoder
    (pre-skip at the start, final granule position at the end)."""

    def __init__(self) -> None:
        if _lib is None:
            raise OSError("libopus not found")
        err = ctypes.c_int()
        self._dec = _lib.opus_decoder_create(RATE, 1, ctypes.byref(err))
        if err.value != 0:
            raise OSError(f"opus_decoder_create failed: {err.value}")
        self._buf = (ctypes.c_float * _MAX_FRAME)()

    def __call__(self, data: bytes) -> np.ndarray:
        packets, granule = _packets(data)
        head = packets[0]
        if not head.startswith(b"OpusHead") or head[9] != 1:
            raise ValueError("not a mono Ogg Opus stream")
        pre_skip = int.from_bytes(head[10:12], "little")  # in 48 kHz samples
        _lib.opus_decoder_ctl(self._dec, _OPUS_RESET_STATE)
        out = []
        for p in packets[2:]:  # skip OpusHead and OpusTags
            n = _lib.opus_decode_float(self._dec, p, len(p), self._buf, _MAX_FRAME, 0)
            if n < 0:
                raise ValueError(f"opus_decode_float failed: {n}")
            out.append(np.frombuffer(self._buf, np.float32, n).copy())
        audio = np.concatenate(out)
        return audio[pre_skip // 3 : (granule // 3) if granule > 0 else None]
