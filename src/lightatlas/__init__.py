"""LightAtlas: High-throughput Astronomical Light Curve Analysis & Photometric Classification."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lightatlas")
except PackageNotFoundError:  # pragma: no cover
    try:
        from lightatlas._version import __version__  # type: ignore[import-not-found]
    except ImportError:
        __version__ = "0.0.0"

from lightatlas.pipeline import PipelineConfig, run_pipeline

__all__ = ["PipelineConfig", "__version__", "run_pipeline"]
