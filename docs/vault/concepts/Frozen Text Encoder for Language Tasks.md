---
tags: [concepts, language, text-encoder, miniLM, embedding]
---

# Frozen Text Encoder for Language Tasks

## What It Is

A pre-trained sentence transformer (MiniLM-L6) that converts text commands into fixed-length vectors. It is **frozen during RL training** — we use its embeddings as features, not fine-tune it.

## Why Freeze It

RL reward signals are extremely noisy. Backpropagating through a 384-dim text encoder with a scalar reward signal would destroy the encoder's semantic structure. Instead:
- Encoder is fixed — it produces stable, consistent embeddings
- The RL policy **learns to use** those embeddings
- This is analogous to how vision RL often uses frozen ImageNet features

## The Cache

Pre-compute embeddings offline, save to JSON:

```json
{
  "red marker":   [0.12, -0.04, 0.31, ...],   // 384 dims
  "blue marker":  [0.08,  0.11, 0.27, ...],
  "green marker": [-0.03, 0.19, 0.22, ...]
}
```

Location: `programs/common/cache/nav_command_embeddings.json`

At training time: load the cache, index by command id — zero overhead during rollout.

## Testing Embeddings (the gotcha)

When writing tests for the text encoder, do NOT assert exact ordering of similarities (e.g., "red should be closest to red, then blue"). MiniLM weights PHRASING over color — "red marker" and "blue marker" are syntactically very similar so their cosine similarities are high. The correct test:

```python
# Assert distinguishability, not exact ordering:
for i, j in combinations(range(len(keys)), 2):
    sim = cosine_similarity(embeddings[i], embeddings[j])
    assert sim < 0.95, "embeddings too similar to distinguish"
```

## Related

- [[P1.2 - LangNav]]
- [programs/common/text_encoder.py](../../programs/common/text_encoder.py)
