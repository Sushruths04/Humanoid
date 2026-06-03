"""Frozen text encoder for language-conditioned navigation (P1).

Commands are encoded OFFLINE into fixed embeddings and cached to disk; the Isaac
Lab env then looks up the cached vector, so there is no text-encoder cost in the
training loop. Uses a small frozen SentenceTransformer (MiniLM, 384-dim).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_MODELS: dict[str, object] = {}


def _get_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    if model_name not in _MODELS:
        _MODELS[model_name] = SentenceTransformer(model_name, device="cpu")
    return _MODELS[model_name]


def encode_commands(texts, model_name: str = DEFAULT_MODEL) -> torch.Tensor:
    """Encode command strings into (N, dim) unit-normalized float embeddings."""
    model = _get_model(model_name)
    emb = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
    return torch.tensor(emb, dtype=torch.float32)


def build_command_cache(texts, out_path, model_name: str = DEFAULT_MODEL) -> dict:
    """Encode commands once and persist {text: embedding} as JSON for the env."""
    emb = encode_commands(texts, model_name)
    cache = {t: emb[i].tolist() for i, t in enumerate(texts)}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cache))
    return cache


def load_command_cache(path) -> dict:
    """Load a {text: embedding} cache written by build_command_cache."""
    return json.loads(Path(path).read_text())
