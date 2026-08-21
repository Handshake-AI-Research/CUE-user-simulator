"""User-simulator baselines for MirrorBench-style evaluation.

This top-level package implements four user-simulator baselines (UserLM, USP,
PPol, RealUserSim). Each baseline can ``train`` (where needed) and
``rollout`` a MirrorBench-compatible JSONL for the three evaluation domains
(coding, customer service, writing) via ``python -m cue_training.baselines.main``.
"""
