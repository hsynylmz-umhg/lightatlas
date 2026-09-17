"""Unit tests for TESS archival light curve ingestion via MAST and lightkurve."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lightatlas.fetch.mast import fetch_tess_lightcurve


class MockQuantity:
    """Mock an Astropy Quantity or Time object with a .value attribute."""

    def __init__(self, value: np.ndarray) -> None:
        self.value = value


class MockLightCurve:
    """Mock LightCurve object mimicking lightkurve.TessLightCurve."""

    def __init__(
        self,
        time_arr: np.ndarray,
        flux_arr: np.ndarray,
        quality_arr: np.ndarray | None = None,
        meta: dict | None = None,
    ) -> None:
        self.time = MockQuantity(time_arr)
        self.flux = MockQuantity(flux_arr)
        self.pdcsap_flux = MockQuantity(flux_arr)
        self.colnames = ["time", "flux", "pdcsap_flux"]
        if quality_arr is not None:
            self.quality = MockQuantity(quality_arr)
            self.colnames.append("quality")
        else:
            self.quality = None
        self.meta = meta or {}

    def __getitem__(self, key: str) -> MockQuantity:
        if key == "time":
            return self.time
        if key in ("flux", "pdcsap_flux"):
            return self.pdcsap_flux
        if key == "quality" and self.quality is not None:
            return self.quality
        raise KeyError(key)


def test_fetch_tess_lightcurve_success():
    """Verify that fetch_tess_lightcurve cleans NaNs, infinite values, and filters quality."""
    raw_time = np.array([10.0, 10.1, 10.2, 10.3, 10.4, 10.5])
    raw_flux = np.array([100.0, np.nan, 102.0, 101.5, np.inf, 100.8])
    raw_quality = np.array([0, 0, 1, 0, 0, 0])  # index 2 has quality=1 (bad data)
    meta = {
        "SECTOR": 14,
        "AUTHOR": "SPOC",
        "CAMERA": 1,
        "CCD": 2,
        "RA_OBJ": 120.5,
        "DEC_OBJ": -45.2,
        "MISSION": "TESS",
    }

    mock_lc = MockLightCurve(raw_time, raw_flux, raw_quality, meta)
    mock_search_item = MagicMock()
    mock_search_item.download.return_value = mock_lc
    mock_search_result = [mock_search_item]

    with patch("lightkurve.search_lightcurve", return_value=mock_search_result) as mock_search:
        t, f, out_meta = fetch_tess_lightcurve(tic_id=261136679, sector=14)

        mock_search.assert_called_with("TIC 261136679", mission="TESS", sector=14, author="SPOC")

        # Kept indices:
        # 0: t=10.0, f=100.0 (quality=0, finite) -> kept
        # 1: NaN -> dropped
        # 2: quality=1 -> dropped
        # 3: t=10.3, f=101.5 (quality=0, finite) -> kept
        # 4: Inf -> dropped
        # 5: t=10.5, f=100.8 (quality=0, finite) -> kept
        assert len(t) == 3
        assert len(f) == 3
        np.testing.assert_allclose(t, [10.0, 10.3, 10.5])
        np.testing.assert_allclose(f, [100.0, 101.5, 100.8])

        assert out_meta["tic_id"] == 261136679
        assert out_meta["sector"] == 14
        assert out_meta["author"] == "SPOC"
        assert out_meta["camera"] == 1
        assert out_meta["ccd"] == 2
        assert out_meta["n_raw_points"] == 6
        assert out_meta["n_clean_points"] == 3


def test_fetch_tess_lightcurve_fallback_search():
    """Verify fallback search if author='SPOC' returns no results."""
    mock_lc = MockLightCurve(
        np.array([1.0, 2.0]),
        np.array([50.0, 51.0]),
        quality_arr=np.array([0, 0]),
        meta={"SECTOR": 1},
    )
    mock_item = MagicMock()
    mock_item.download.return_value = mock_lc

    # First call (author="SPOC") returns empty; second call (no author) returns mock_item
    with patch("lightkurve.search_lightcurve", side_effect=[[], [mock_item]]):
        t, f, meta = fetch_tess_lightcurve(tic_id=12345, sector=1)
        assert len(t) == 2
        assert meta["sector"] == 1


def test_fetch_tess_lightcurve_invalid_tic_id():
    """Verify ValueError is raised for non-positive or non-integer TIC IDs."""
    with pytest.raises(ValueError, match="Invalid TIC ID"):
        fetch_tess_lightcurve(-1)

    with pytest.raises(ValueError, match="Invalid TIC ID"):
        fetch_tess_lightcurve(0)

    with pytest.raises(ValueError, match="Invalid TIC ID"):
        fetch_tess_lightcurve("invalid_id")  # type: ignore[arg-type]


def test_fetch_tess_lightcurve_invalid_sector():
    """Verify ValueError is raised for non-positive or invalid sector numbers."""
    with pytest.raises(ValueError, match="Invalid sector"):
        fetch_tess_lightcurve(12345, sector=-1)

    with pytest.raises(ValueError, match="Invalid sector"):
        fetch_tess_lightcurve(12345, sector=0)

    with pytest.raises(ValueError, match="Invalid sector"):
        fetch_tess_lightcurve(12345, sector="two")  # type: ignore[arg-type]


def test_fetch_tess_lightcurve_target_not_found():
    """Verify ValueError is raised when MAST search returns no products."""
    with (
        patch("lightkurve.search_lightcurve", return_value=[]),
        pytest.raises(ValueError, match="No TESS light curve found"),
    ):
        fetch_tess_lightcurve(999999999, sector=1)


def test_fetch_tess_lightcurve_download_fails():
    """Verify ValueError is raised when .download() returns None."""
    mock_item = MagicMock()
    mock_item.download.return_value = None

    with (
        patch("lightkurve.search_lightcurve", return_value=[mock_item]),
        pytest.raises(ValueError, match="Failed to download"),
    ):
        fetch_tess_lightcurve(123456, sector=2)


def test_fetch_tess_lightcurve_empty_after_cleaning():
    """Verify ValueError is raised when all points are filtered out."""
    mock_lc = MockLightCurve(
        np.array([1.0, 2.0]),
        np.array([np.nan, np.nan]),
        quality_arr=np.array([0, 0]),
    )
    mock_item = MagicMock()
    mock_item.download.return_value = mock_lc

    with (
        patch("lightkurve.search_lightcurve", return_value=[mock_item]),
        pytest.raises(ValueError, match="no valid flux points after cleaning"),
    ):
        fetch_tess_lightcurve(123456, sector=3)


def test_fetch_tess_lightcurve_missing_columns():
    """Verify ValueError is raised when light curve lacks time or flux columns."""
    bad_lc = MagicMock()
    del bad_lc.time

    mock_item = MagicMock()
    mock_item.download.return_value = bad_lc

    with (
        patch("lightkurve.search_lightcurve", return_value=[mock_item]),
        pytest.raises(ValueError, match="lacks a 'time' column"),
    ):
        fetch_tess_lightcurve(123456)


def test_fetch_tess_lightcurve_missing_lightkurve():
    """Verify ImportError is raised with informative message if lightkurve is not available."""
    with (
        patch.dict(sys.modules, {"lightkurve": None}),
        pytest.raises(ImportError, match="lightkurve is required"),
    ):
        fetch_tess_lightcurve(123456)
