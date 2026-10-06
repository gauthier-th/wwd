"""Per-wake-word fine-tuning, quick mode (PLAN.md §9.3): frozen encoder, small MLP head.

Shared by the server (M2) and the base-model end-to-end evaluation (training.eval_e2e), so the
number we evaluate is the number users get.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.augment import Augmenter
from core.encoder import Encoder
from core.model import Head, PrototypeHead

WINDOW = 24000


@dataclass(frozen=True)
class FinetuneConfig:
    augment_per_clip: int = 300
    partial_per_clip: int = 40
    speed: tuple[float, float] = (0.9, 1.1)
    partial_fraction: tuple[float, float] = (0.4, 0.7)
    hard_negatives: int = 3000
    random_negatives: int = 30000
    same_language_share: float = 0.7
    exclude_similarity: float = 0.8  # bank clips this close to the prototype are the same word
    user_hop_s: float = 0.5
    # Scorer: "prototype" (default) = enrollment prototype pushed away from the mean of the
    # hardest negatives (Rocchio); "mlp" = small MLP head trained on augmented embeddings.
    scorer: str = "prototype"
    rocchio: float = 0.5  # 0.4-0.6 measured best; >= 0.8 collapses onto the negatives
    score_scale: float = 20.0  # logit slope per unit cosine (the threshold is calibrated anyway)
    hidden: int = 64
    steps: int = 400
    lr: float = 3e-3
    weight_decay: float = 1e-4
    hard_weight: float = 2.0
    embed_batch: int = 256
    seed: int = 0


@dataclass
class Bank:
    """Negative bank (PLAN.md §8): L2-normalized embeddings of single-word clips."""

    embeddings: np.ndarray  # [M, D] float16
    words: np.ndarray  # [M] str
    langs: np.ndarray  # [M] str

    @classmethod
    def load(cls, path: str | Path) -> Bank:
        z = np.load(path, allow_pickle=False)
        return cls(z["embeddings"], z["words"], z["langs"])

    def save(self, path: str | Path) -> None:
        np.savez(path, embeddings=self.embeddings.astype(np.float16),
                 words=self.words.astype(str), langs=self.langs.astype(str))


@dataclass
class Enrollment:
    wake_word: str  # label only; used to drop the same word from the bank
    language: str
    positives: list[np.ndarray]  # 16 kHz float32, trimmed to the wake word
    negatives: list[np.ndarray] = field(default_factory=list)  # user's other speech, ambience


@dataclass
class FinetuneResult:
    head: torch.nn.Module  # embedding [N, D] -> logit [N]
    prototype: np.ndarray
    report: dict


def normalize_word(word: str) -> str:
    return unicodedata.normalize("NFC", word).casefold().strip()


def _speed(clip: torch.Tensor, factor: float) -> torch.Tensor:
    n = max(1, round(len(clip) / factor))
    return F.interpolate(clip[None, None], size=n, mode="linear", align_corners=True)[0, 0]


def _place(clip: torch.Tensor, rng: np.random.Generator, center: bool = False) -> torch.Tensor:
    clip = clip[:WINDOW]
    out = torch.zeros(WINDOW)
    start = (WINDOW - len(clip)) // 2 if center else int(rng.integers(0, WINDOW - len(clip) + 1))
    out[start : start + len(clip)] = clip
    return out


def positive_windows(clips: list[np.ndarray], n: int, cfg: FinetuneConfig,
                     rng: np.random.Generator) -> torch.Tensor:
    out = []
    for clip in clips:
        c = torch.from_numpy(np.asarray(clip, np.float32))
        out += [_place(_speed(c, rng.uniform(*cfg.speed)), rng) for _ in range(n)]
    return torch.stack(out)


def partial_windows(clips: list[np.ndarray], n: int, cfg: FinetuneConfig,
                    rng: np.random.Generator) -> torch.Tensor:
    """First or last 40-70% of the wake word: must not trigger (no early firing)."""
    out = []
    for clip in clips:
        c = torch.from_numpy(np.asarray(clip, np.float32))
        for _ in range(n):
            keep = int(len(c) * rng.uniform(*cfg.partial_fraction))
            part = c[:keep] if rng.random() < 0.5 else c[-keep:]
            out.append(_place(_speed(part, rng.uniform(*cfg.speed)), rng))
    return torch.stack(out)


def sliding_windows(audio: list[np.ndarray], hop_s: float) -> torch.Tensor:
    hop, out = int(hop_s * 16000), []
    for a in audio:
        a = torch.from_numpy(np.asarray(a, np.float32))
        if len(a) < WINDOW:
            a = F.pad(a, (0, WINDOW - len(a)))
        out += [a[s : s + WINDOW] for s in range(0, len(a) - WINDOW + 1, hop)]
    return torch.stack(out) if out else torch.zeros(0, WINDOW)


@torch.no_grad()
def embed(encoder: Encoder, windows: torch.Tensor, augmenter: Augmenter | None,
          batch: int = 256, device: str | torch.device = "cpu") -> np.ndarray:
    encoder.eval()
    out = []
    for i in range(0, len(windows), batch):
        x = windows[i : i + batch].to(device)
        if augmenter is not None:
            x = augmenter(x)
        out.append(encoder(x).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, encoder.config.embedding_dim), np.float32)


def select_bank(bank: Bank, prototype: np.ndarray, enrollment: Enrollment, cfg: FinetuneConfig,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    """Bank negatives without the wake word itself; the nearest ones weighted up as hard."""
    emb = bank.embeddings.astype(np.float32)
    sims = emb @ prototype
    target = normalize_word(enrollment.wake_word)
    same_word = np.array([normalize_word(w) == target for w in bank.words])
    keep = ~same_word & (sims < cfg.exclude_similarity)
    idx = np.flatnonzero(keep)
    hard = idx[np.argsort(-sims[idx])[: cfg.hard_negatives]]
    same_lang = idx[bank.langs[idx] == enrollment.language]
    other_lang = idx[bank.langs[idx] != enrollment.language]
    n_same = min(len(same_lang), int(cfg.random_negatives * cfg.same_language_share))
    n_other = min(len(other_lang), cfg.random_negatives - n_same)
    rand = np.concatenate([rng.choice(same_lang, n_same, replace=False),
                           rng.choice(other_lang, n_other, replace=False)])
    rand = np.setdiff1d(rand, hard)
    chosen = np.concatenate([hard, rand])
    weights = np.concatenate([np.full(len(hard), cfg.hard_weight), np.ones(len(rand))])
    info = {"bank_same_word": int(same_word.sum()),
            "bank_too_similar": int((~same_word & ~keep).sum()),
            "bank_hard": len(hard), "bank_random": len(rand),
            "hardest_similarity": float(sims[hard[0]]) if len(hard) else None}
    return emb[chosen], weights, info


def train_head(pos: np.ndarray, neg: np.ndarray, neg_weight: np.ndarray, cfg: FinetuneConfig) -> Head:
    torch.manual_seed(cfg.seed)
    x = torch.from_numpy(np.concatenate([pos, neg]).astype(np.float32))
    x = x / x.norm(dim=1, keepdim=True)
    y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
    w = torch.cat([torch.full((len(pos),), float(neg_weight.sum() / len(pos))),
                   torch.from_numpy(neg_weight.astype(np.float32))])
    head = Head(x.shape[1], cfg.hidden)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for _ in range(cfg.steps):
        loss = F.binary_cross_entropy_with_logits(head(x), y, weight=w) / w.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head.eval()


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def prototype_head(proto: np.ndarray, pos: np.ndarray, neg: np.ndarray,
                   cfg: FinetuneConfig) -> PrototypeHead:
    """Direction = prototype minus `rocchio` x the mean of the hardest negatives; the bias puts
    the sigmoid midpoint between typical positives and the hardest negatives."""
    neg = _unit(neg)
    hard = neg[np.argsort(-(neg @ proto))[: cfg.hard_negatives]]
    direction = _unit(proto - cfg.rocchio * _unit(hard.mean(0)))
    pos_cos = np.median(_unit(pos) @ direction)
    neg_cos = np.quantile(neg @ direction, 0.999)
    midpoint = float(pos_cos + neg_cos) / 2
    return PrototypeHead(torch.from_numpy(direction.astype(np.float32)), cfg.score_scale,
                         -cfg.score_scale * midpoint).eval()


def finetune(encoder: Encoder, enrollment: Enrollment, bank: Bank, augmenter: Augmenter,
             cfg: FinetuneConfig = FinetuneConfig(), device: str | torch.device = "cpu") -> FinetuneResult:  # noqa: B008
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    clean = torch.stack([_place(torch.from_numpy(np.asarray(c, np.float32)), rng, center=True)
                         for c in enrollment.positives])
    proto = embed(encoder, clean, None, device=device).mean(0)
    proto /= np.linalg.norm(proto)

    pos = embed(encoder, positive_windows(enrollment.positives, cfg.augment_per_clip, cfg, rng),
                augmenter, cfg.embed_batch, device)
    partial = embed(encoder, partial_windows(enrollment.positives, cfg.partial_per_clip, cfg, rng),
                    augmenter, cfg.embed_batch, device)
    user = embed(encoder, sliding_windows(enrollment.negatives, cfg.user_hop_s), augmenter,
                 cfg.embed_batch, device)
    bank_neg, bank_w, info = select_bank(bank, proto, enrollment, cfg, rng)

    neg = np.concatenate([bank_neg, partial, user])
    if cfg.scorer == "prototype":
        head = prototype_head(proto, pos, neg, cfg)
    elif cfg.scorer == "mlp":
        neg_w = np.concatenate([bank_w, np.full(len(partial) + len(user), cfg.hard_weight)])
        head = train_head(pos, neg, neg_w, cfg)
    else:
        raise ValueError(f"unknown scorer {cfg.scorer!r}")
    report = {**info, "scorer": cfg.scorer, "positives": len(pos),
              "partial_negatives": len(partial), "user_negatives": len(user)}
    return FinetuneResult(head, proto, report)
