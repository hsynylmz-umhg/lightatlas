"""Tests for package import and version attributes."""

import lightatlas


def test_package_name():
    assert lightatlas.__name__ == "lightatlas"


def test_package_version():
    assert hasattr(lightatlas, "__version__")
    assert isinstance(lightatlas.__version__, str)
    assert len(lightatlas.__version__) > 0


def test_import_without_torch():
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-c",
        "import sys; sys.modules['torch'] = None; import lightatlas; "
        "import lightatlas.models; assert not lightatlas.models.HAS_TORCH; print('OK')",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "OK" in res.stdout
