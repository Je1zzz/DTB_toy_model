"""Source-to-sensor observation operators."""

from .seeg import bipolarize_gain, project_to_seeg

__all__ = ["bipolarize_gain", "project_to_seeg"]
