"""MAST archival data fetcher for TESS photometric light curves using lightkurve."""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def fetch_tess_lightcurve(
    tic_id: int,
    sector: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fetch and preprocess a TESS light curve from the MAST archive.

    Downloads 2-min / 20-sec cadence PDCSAP_FLUX light curves from MAST,
    applies quality bitmask filtering (quality == 0), and strips all NaN
    and infinite values.

    Parameters
    ----------
    tic_id : int
        TESS Input Catalog (TIC) identifier (must be positive integer).
    sector : int, optional
        Observation sector number. If None, retrieves the first available sector.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict[str, Any]]
        A 3-element tuple:
        - t: 1D float64 array of Barycentric TESS Julian Dates (BTJD).
        - f: 1D float64 array of clean, quality-filtered PDCSAP fluxes.
        - meta: Dictionary containing observation metadata (TIC ID, sector, etc.).

    Raises
    ------
    ValueError
        If tic_id or sector is invalid, if no light curve is found,
        if downloading fails, or if no valid data points remain.
    ImportError
        If lightkurve is not installed.
    """
    # 1. Input validation
    if not isinstance(tic_id, (int, np.integer)) or tic_id <= 0:
        raise ValueError(f"Invalid TIC ID: {tic_id}. Expected a positive integer.")

    if sector is not None and (not isinstance(sector, (int, np.integer)) or sector <= 0):
        raise ValueError(f"Invalid sector: {sector}. Expected a positive integer or None.")

    # 2. Dependency check
    try:
        import lightkurve as lk
    except ImportError as exc:
        raise ImportError(
            "lightkurve is required to fetch TESS data from MAST. "
            "Install it via 'pip install lightatlas[mast]'."
        ) from exc

    target_name = f"TIC {tic_id}"
    logger.info("Searching MAST archive for %s (sector=%s)...", target_name, sector)

    # 3. Search MAST archive (prefer official SPOC 2-min / 20-sec cadence pipeline)
    search_result = lk.search_lightcurve(
        target_name,
        mission="TESS",
        sector=sector,
        author="SPOC",
    )

    if len(search_result) == 0:
        search_result = lk.search_lightcurve(
            target_name,
            mission="TESS",
            sector=sector,
        )

    if len(search_result) == 0:
        raise ValueError(
            f"No TESS light curve found on MAST for target '{target_name}' (sector={sector})."
        )

    # 4. Download light curve
    logger.info(
        "Downloading light curve for %s (found %d products)...",
        target_name,
        len(search_result),
    )
    lc = search_result[0].download()
    if lc is None:
        raise ValueError(
            f"Failed to download light curve from MAST for target '{target_name}' "
            f"(sector={sector})."
        )

    # 5. Extract time
    if hasattr(lc, "time") and hasattr(lc.time, "value"):
        t_raw = np.asarray(lc.time.value, dtype=np.float64)
    elif hasattr(lc, "time"):
        t_raw = np.asarray(lc.time, dtype=np.float64)
    else:
        raise ValueError(f"Light curve for {target_name} lacks a 'time' column.")

    # 6. Extract flux (prioritize PDCSAP_FLUX over standard flux)
    if "pdcsap_flux" in lc.colnames:
        col = lc["pdcsap_flux"]
    elif hasattr(lc, "pdcsap_flux") and lc.pdcsap_flux is not None:
        col = lc.pdcsap_flux
    elif hasattr(lc, "flux"):
        col = lc.flux
    else:
        raise ValueError(f"No flux column found in light curve for {target_name}.")

    if hasattr(col, "value"):
        f_raw = np.asarray(col.value, dtype=np.float64)
    else:
        f_raw = np.asarray(col, dtype=np.float64)

    # 7. Quality bitmask filtering (quality == 0)
    if "quality" in lc.colnames:
        q_col = lc["quality"]
        q_raw = np.asarray(q_col.value if hasattr(q_col, "value") else q_col)
        quality_mask = q_raw == 0
    elif hasattr(lc, "quality") and lc.quality is not None:
        q_raw = np.asarray(lc.quality.value if hasattr(lc.quality, "value") else lc.quality)
        quality_mask = q_raw == 0
    else:
        quality_mask = np.ones(len(t_raw), dtype=bool)

    # 8. Clean NaN and Inf values
    finite_mask = np.isfinite(t_raw) & np.isfinite(f_raw)
    valid_mask = quality_mask & finite_mask

    t_clean = t_raw[valid_mask]
    f_clean = f_raw[valid_mask]

    if len(t_clean) == 0:
        raise ValueError(
            f"Light curve for target '{target_name}' contains no valid flux points after cleaning."
        )

    # 9. Extract metadata
    lc_meta = getattr(lc, "meta", {}) or {}
    meta: dict[str, Any] = {
        "tic_id": int(tic_id),
        "target": target_name,
        "sector": lc_meta.get("SECTOR", sector),
        "author": lc_meta.get("AUTHOR", "SPOC"),
        "camera": lc_meta.get("CAMERA"),
        "ccd": lc_meta.get("CCD"),
        "ra": lc_meta.get("RA_OBJ"),
        "dec": lc_meta.get("DEC_OBJ"),
        "mission": lc_meta.get("MISSION", "TESS"),
        "n_raw_points": len(t_raw),
        "n_clean_points": len(t_clean),
    }

    # Include additional metadata keys in lowercase
    for k, v in lc_meta.items():
        k_lower = str(k).lower()
        if k_lower not in meta and not isinstance(v, (dict, list)):
            meta[k_lower] = v

    return t_clean, f_clean, meta
