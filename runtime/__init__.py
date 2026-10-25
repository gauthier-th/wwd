"""Wake-word runtime: onnxruntime + numpy only. Must never import torch or other repo packages."""

from .detector import Detector, Event, Gate, MultiDetector, Tick, Trigger, Word
from .spec import DetectorConfig
from .vad import SileroVad

__all__ = [
    "Detector", "DetectorConfig", "Event", "Gate", "MultiDetector", "SileroVad", "Tick", "Trigger",
    "Word",
]
