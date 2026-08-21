"""Transformers-compatible inference for CUE user models."""

from cue_hf.configuration_cue import CueConfig
from cue_hf.modeling_cue import CueModel
from cue_hf.processing_cue import CueProcessor

__all__ = ["CueConfig", "CueModel", "CueProcessor"]
__version__ = "0.1.0"


def _register_auto_classes() -> None:
    """Make AutoConfig/AutoModel resolve ``model_type: cue`` in this process."""

    from transformers import AutoConfig, AutoModel

    try:
        AutoConfig.register("cue", CueConfig)
        AutoModel.register(CueConfig, CueModel)
    except ValueError:
        # Already registered (re-import in the same interpreter).
        pass


_register_auto_classes()
