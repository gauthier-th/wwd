"""Wake-word runtime: onnxruntime + numpy only. Must never import torch or other repo packages."""

from runtime.detector import Detector, Event, Gate, Tick, Trigger
from runtime.spec import DetectorConfig
from runtime.vad import SileroVad

__all__ = ["Detector", "DetectorConfig", "Event", "Gate", "SileroVad", "Tick", "Trigger"]
