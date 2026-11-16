"""Configuration primitives. ``LLMConfig`` is exported from :mod:`llm_blocks.nn`."""

from .spec import BuildContext, Spec, SpecError, spec_from_dict

__all__ = ["Spec", "BuildContext", "SpecError", "spec_from_dict"]
