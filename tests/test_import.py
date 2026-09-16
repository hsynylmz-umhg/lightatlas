"""Tests for package import and version attributes."""

import lightatlas


def test_package_name():
    assert lightatlas.__name__ == "lightatlas"


def test_package_version():
    assert hasattr(lightatlas, "__version__")
    assert isinstance(lightatlas.__version__, str)
    assert len(lightatlas.__version__) > 0
