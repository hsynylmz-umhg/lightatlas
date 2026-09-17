"""LightAtlas: High-throughput Astronomical Light Curve Analysis & Photometric Classification."""

import sys
from importlib.metadata import PackageNotFoundError, version

# Compatibility fix: If torch is mocked as missing via sys.modules['torch'] = None,
# scipy.stats array_api_compat crashes on getattr(None, 'Tensor').
# Cleanly remove the None entry and install a MetaPathFinder raising ModuleNotFoundError.
if "torch" in sys.modules and sys.modules["torch"] is None:
    del sys.modules["torch"]

    class _TorchMissingFinder:
        def find_spec(self, fullname, *args, **kwargs):
            if fullname == "torch" or fullname.startswith("torch."):
                raise ModuleNotFoundError(f"No module named '{fullname}'")
            return None

    sys.meta_path.insert(0, _TorchMissingFinder())

try:
    __version__ = version("lightatlas")
except PackageNotFoundError:  # pragma: no cover
    try:
        from lightatlas._version import __version__  # type: ignore[import-not-found]
    except ImportError:
        __version__ = "0.2.0"

from lightatlas.fetch import fetch_tess_lightcurve
from lightatlas.pipeline import PipelineConfig, run_pipeline

__all__ = ["PipelineConfig", "__version__", "fetch_tess_lightcurve", "run_pipeline"]
