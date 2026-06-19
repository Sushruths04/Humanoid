"""AMP (Adversarial Motion Priors) — style reward (PLAN.md §5.3, §9)."""
from .discriminator import AMPDiscriminator
from .reference_motion import ReferenceMotionBuffer, build_keyframe_reference
