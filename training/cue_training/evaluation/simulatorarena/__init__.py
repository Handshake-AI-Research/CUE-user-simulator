"""SimulatorArena document-creation proxy eval with CUE/baseline user simulators.

SimulatorArena (microsoft/SimulatorArena) lives as an external checkout under
``external/SimulatorArena`` and is imported lazily. Because it has no user-plugin hook,
we reimplement the thin document-creation conversation loop (substituting our user-turn
generator) and reuse its data, assistant model, termination, and evaluation.
"""
