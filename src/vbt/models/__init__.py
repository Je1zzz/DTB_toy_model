"""Neural mass models."""

from .epileptor import Epileptor6D

__all__ = ["Epileptor6D"]
from .repo_spatepi_stim import RepoSpatEpiStim7D, repo_source

__all__ = ["RepoSpatEpiStim7D", "repo_source"]
