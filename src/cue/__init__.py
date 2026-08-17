"""Production runtime for CUE user simulation.

Load a published checkpoint from Hugging Face, build a persona manual from a
dialogue history or from the diffusion prior, and steer any chat simulator with
it. Training and evaluation live elsewhere; this package is application-only.
"""

from cue.__about__ import __version__
from cue.manual import PersonaManual
from cue.runtime import Cue
from cue.simulator import HuggingFaceSimulator, OpenAICompatSimulator, Simulator

__all__ = [
    "Cue",
    "HuggingFaceSimulator",
    "OpenAICompatSimulator",
    "PersonaManual",
    "Simulator",
    "__version__",
]
