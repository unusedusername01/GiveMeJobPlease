"""GiveMeJobPlease package exposing runtime SemVer version."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("give-me-job-please")
except PackageNotFoundError:  # Editable / uninstalled fallback
    __version__ = "0.1.0-dev"

__all__ = ["__version__"]
