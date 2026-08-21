"""Top-level evaluation harnesses for CUE + baseline user simulators.

Each subpackage integrates an external evaluation checkout under
``<harness>/external/`` and reuses the shared, harness-agnostic utilities in
``cue_training.evaluation.common`` (model loading, per-turn conditioning, the next-user-turn
interface, and MirrorBench rollout export). Heavy/external imports are lazy so
importing this package stays cheap and does not require the external trees to be present.
"""
