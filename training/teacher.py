"""XLS-R teacher (PLAN.md §7 step 4): frozen XLS-R 300M -> learned layer mix -> small head.

XLS-R only runs forward passes (bf16, compiled); the head and its Sub-center ArcFace are
the only trained parameters. The teacher never ships: it exists to give the student
(core.encoder) better embeddings to learn from (training.train_student --teacher).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from core.encoder import AttentivePooling, l2_normalize
from training.paths import DATA

XLSR_REPO = "facebook/wav2vec2-xls-r-300m"
XLSR_REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"
XLSR_DIR = DATA / "xlsr-300m"
FEATURE_DIM = 1024


def xlsr_path() -> Path:
    """The pinned XLS-R checkpoint (1.3 GB, Apache-2.0), downloaded on first use."""
    if not (XLSR_DIR / "pytorch_model.bin").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(XLSR_REPO, revision=XLSR_REVISION, local_dir=XLSR_DIR,
                          allow_patterns=["config.json", "preprocessor_config.json",
                                          "pytorch_model.bin"])
    return XLSR_DIR


@dataclass(frozen=True)
class TeacherConfig:
    layers: int = 24  # XLS-R transformer layers kept; the ones above are never run
    dim: int = 512
    context_layers: int = 2
    heads: int = 8
    embedding_dim: int = 256
    dropout: float = 0.1


class XlsrFeatures(nn.Module):
    """Frozen XLS-R: [N, samples] -> hidden states of every kept layer, each layer-normalized
    without affine, as bf16 [N, layers + 1, frames, 1024]. Always in eval mode."""

    def __init__(self, layers: int = 24, chunk: int = 128, compile: bool = True) -> None:
        super().__init__()
        from transformers import Wav2Vec2Model

        model = Wav2Vec2Model.from_pretrained(xlsr_path(), attn_implementation="sdpa")
        model.encoder.layers = model.encoder.layers[:layers]
        self.model = model.eval().requires_grad_(False)
        self.n_states, self.chunk = layers + 1, chunk
        # One input shape (chunk, samples): the compiled graph never recompiles.
        self._run = torch.compile(self._states) if compile else self._states

    def train(self, mode: bool = True) -> XlsrFeatures:
        return super().train(False)

    def _states(self, audio: torch.Tensor) -> torch.Tensor:
        # XLS-R's own input normalization (preprocessor_config.json: do_normalize).
        x = (audio - audio.mean(1, keepdim=True)) / torch.sqrt(audio.var(1, keepdim=True) + 1e-7)
        states = torch.stack(self.model(x, output_hidden_states=True).hidden_states, 1)
        return F.layer_norm(states.float(), states.shape[-1:]).to(torch.bfloat16)

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        out = []
        for i in range(0, len(audio), self.chunk):
            x = audio[i : i + self.chunk].float()
            n = len(x)
            if n < self.chunk:
                x = F.pad(x, (0, 0, 0, self.chunk - n))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(self._run(x)[:n])
        return torch.cat(out)


class TeacherHead(nn.Module):
    """Softmax-weighted layer mix -> projection -> small transformer -> attentive pooling."""

    def __init__(self, config: TeacherConfig) -> None:
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(config.layers + 1))
        self.proj = nn.Sequential(nn.Linear(FEATURE_DIM, config.dim), nn.GELU(),
                                  nn.Dropout(config.dropout))
        layer = nn.TransformerEncoderLayer(config.dim, config.heads, 2 * config.dim,
                                           config.dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.context = nn.TransformerEncoder(layer, config.context_layers, enable_nested_tensor=False)
        self.pool = AttentivePooling(config.dim)
        self.out = nn.Linear(config.dim, config.embedding_dim)

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits.float(), 0)

    def forward(self, states: torch.Tensor) -> torch.Tensor:  # [N, S, T, F]
        x = torch.einsum("s,nstf->ntf", self.layer_weights().to(states.dtype), states)
        x = self.context(self.proj(x))
        return l2_normalize(self.out(self.pool(x)).float())


class Teacher(nn.Module):
    """[N, samples] audio -> [N, embedding_dim] L2-normalized embedding (CUDA only)."""

    def __init__(self, config: TeacherConfig | None = None, compile: bool = True) -> None:
        super().__init__()
        self.config = config = config or TeacherConfig()
        self.features = XlsrFeatures(config.layers, compile=compile)
        self.head = TeacherHead(config)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.head(self.features(audio))

    def checkpoint(self) -> dict:
        """What identifies a trained teacher: config + head (XLS-R itself is pinned)."""
        return {"teacher_config": asdict(self.config), "xlsr_revision": XLSR_REVISION,
                "head": self.head.state_dict()}


def is_teacher(checkpoint: str | Path) -> bool:
    return "teacher_config" in torch.load(checkpoint, map_location="cpu", weights_only=False)


def load_teacher(checkpoint: str | Path, device: str = "cuda", compile: bool = True) -> Teacher:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    teacher = Teacher(TeacherConfig(**ckpt["teacher_config"]), compile=compile)
    teacher.head.load_state_dict(ckpt["head"])
    return teacher.to(device).eval()
