import torch
from programs.common.text_encoder import encode_commands


def test_encoder_deterministic_and_normalized():
    a = encode_commands(["go to the red marker"])
    b = encode_commands(["go to the red marker"])
    assert a.shape[0] == 1
    assert a.shape[1] >= 128          # MiniLM is 384-dim
    assert torch.allclose(a, b)
    assert abs(float(a.norm(dim=-1)[0]) - 1.0) < 1e-4   # unit-normalized


def test_different_target_commands_are_distinguishable():
    # Distinct color commands must yield distinct embeddings so the policy can
    # condition on them (MiniLM weights phrasing heavily, but colors still separate).
    emb = encode_commands(["go to the red marker", "go to the blue marker"])
    assert float(emb[0] @ emb[1]) < 0.95
    # Identical commands must yield (near) identical embeddings.
    same = encode_commands(["go to the red marker", "go to the red marker"])
    assert float(same[0] @ same[1]) > 0.999
