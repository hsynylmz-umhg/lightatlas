"""Astrophysical feature extraction module for light curves (40 features)."""

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    # FFT Power & Frequency Metrics (6)
    "fft_peak_period",
    "fft_peak_power",
    "fft_power_snr",
    "fft_harmonic_ratio",
    "fft_band_ratio_low",
    "fft_band_ratio_high",
    # Autocorrelation Metrics (4)
    "acf_first_zero_crossing",
    "acf_fwhm",
    "acf_first_local_peak_lag",
    "acf_first_local_peak_val",
    # Statistical Moments & Spread (5)
    "moment_skew",
    "moment_kurtosis",
    "moment_iqr",
    "moment_cv",
    "moment_std",
    # Quantiles & Distribution Widths (5)
    "quantile_range",
    "quantile_p95_p05",
    "quantile_q01",
    "quantile_q99",
    "quantile_median",
    # Spikes & Flare Activity (5)
    "spike_count",
    "spike_max",
    "spike_width",
    "flare_index",
    "spike_rise_fall_ratio",
    # Dips & Transit Activity (5)
    "dip_count",
    "dip_depth",
    "dip_width",
    "dip_fraction",
    "dip_symmetry",
    # Z-scores & Surface Roughness (4)
    "zscore_max",
    "zscore_min",
    "zscore_fraction_outside_3sigma",
    "roughness",
    # Trends, Entropy, Sampling Proxy & Extrema (6)
    "trend_slope",
    "histogram_entropy",
    "gap_proxy",
    "peak_count",
    "trough_count",
    "peak_trough_ratio",
)


def compute_features(t: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Compute 40 astrophysical features from a single light curve.

    Parameters
    ----------
    t : np.ndarray
        1D array of observation timestamps.
    f : np.ndarray
        1D array of observed fluxes.

    Returns
    -------
    np.ndarray
        Array of shape (40,) with dtype float64 strictly matching FEATURE_NAMES.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    f_arr = np.asarray(f, dtype=np.float64)
    n = len(f_arr)

    # Basic time step
    dt = float(np.median(np.diff(t_arr))) if n > 1 else 0.0304
    if dt <= 0.0:
        dt = 0.0304

    # Mean and variance
    f_mean = float(np.mean(f_arr))
    f_var = float(np.var(f_arr))
    f_std = float(np.sqrt(f_var))
    f_detrend = f_arr - f_mean

    # 1. FFT Features (6)
    fft_vals = np.fft.rfft(f_detrend)
    freqs = np.fft.rfftfreq(n, d=dt)
    powers = np.abs(fft_vals) ** 2

    # Skip index 0 (DC component)
    if len(powers) > 1:
        powers_pos = powers[1:]
        freqs_pos = freqs[1:]
        peak_idx = int(np.argmax(powers_pos))
        peak_freq = float(freqs_pos[peak_idx])
        fft_peak_period = float(1.0 / peak_freq) if peak_freq > 0.0 else 0.0
        fft_peak_power = float(powers_pos[peak_idx])
        mean_power = float(np.mean(powers_pos))
        fft_power_snr = float(fft_peak_power / (mean_power + 1e-12))

        # First harmonic at ~ 2 * peak_freq
        harmonic_idx = int(np.argmin(np.abs(freqs_pos - 2.0 * peak_freq)))
        fft_harmonic_ratio = float(powers_pos[harmonic_idx] / (fft_peak_power + 1e-12))

        # Low vs high band ratio
        mid_idx = len(powers_pos) // 2
        total_power = float(np.sum(powers_pos)) + 1e-12
        fft_band_ratio_low = float(np.sum(powers_pos[:mid_idx]) / total_power)
        fft_band_ratio_high = float(np.sum(powers_pos[mid_idx:]) / total_power)
    else:
        fft_peak_period = 0.0
        fft_peak_power = 0.0
        fft_power_snr = 0.0
        fft_harmonic_ratio = 0.0
        fft_band_ratio_low = 0.0
        fft_band_ratio_high = 0.0

    # 2. Autocorrelation Function (ACF) Features (4)
    if f_var > 1e-12:
        corr = np.correlate(f_detrend, f_detrend, mode="full")
        acf = corr[n - 1 :] / (f_var * n)
    else:
        acf = np.zeros(n, dtype=np.float64)
        acf[0] = 1.0

    # First zero crossing
    zero_crossings = np.where(acf <= 0.0)[0]
    first_zc_lag = int(zero_crossings[0]) if len(zero_crossings) > 0 else n
    acf_first_zero_crossing = float(first_zc_lag * dt)

    # FWHM of central peak
    below_half = np.where(acf <= 0.5)[0]
    fwhm_lag = int(below_half[0]) if len(below_half) > 0 else n
    acf_fwhm = float(2.0 * fwhm_lag * dt)

    # First local peak after first drop/zero crossing
    search_start = max(1, min(first_zc_lag, n - 2))
    diff_acf = np.diff(acf)
    local_peaks = np.where((diff_acf[:-1] > 0.0) & (diff_acf[1:] <= 0.0))[0] + 1
    valid_peaks = local_peaks[local_peaks >= search_start]

    if len(valid_peaks) > 0:
        first_peak_idx = int(valid_peaks[0])
        acf_first_local_peak_lag = float(first_peak_idx * dt)
        acf_first_local_peak_val = float(acf[first_peak_idx])
    else:
        acf_first_local_peak_lag = 0.0
        acf_first_local_peak_val = 0.0

    # 3. Moments & Statistical Spread (5)
    if f_std > 1e-12:
        z_norm = f_detrend / f_std
        moment_skew = float(np.mean(z_norm**3))
        moment_kurtosis = float(np.mean(z_norm**4) - 3.0)
        moment_cv = float(f_std / (np.abs(f_mean) + 1e-12))
    else:
        moment_skew = 0.0
        moment_kurtosis = 0.0
        moment_cv = 0.0

    moment_std = float(f_std)
    q25, q75 = np.percentile(f_arr, [25.0, 75.0])
    moment_iqr = float(q75 - q25)

    # 4. Quantiles (5)
    q01, p05, median_val, p95, q99 = np.percentile(f_arr, [1.0, 5.0, 50.0, 95.0, 99.0])
    quantile_range = float(np.max(f_arr) - np.min(f_arr))
    quantile_p95_p05 = float(p95 - p05)
    quantile_q01 = float(q01)
    quantile_q99 = float(q99)
    quantile_median = float(median_val)

    # 5. Spikes & Flare Features (5)
    med = float(median_val)
    mad = float(np.median(np.abs(f_arr - med)))
    robust_sigma = float(1.4826 * mad) if mad > 1e-12 else (f_std if f_std > 1e-12 else 1.0)

    spike_thresh = med + 3.0 * robust_sigma
    is_spike = f_arr > spike_thresh
    spike_count = float(np.sum(is_spike))
    spike_max = float(np.max(f_arr) - med)
    spike_width = float(spike_count * dt)
    flare_index = float(np.sum(f_arr[is_spike] - spike_thresh)) if np.any(is_spike) else 0.0

    peak_idx = int(np.argmax(f_arr))
    rise_pts = float(np.sum(f_arr[:peak_idx] > med + robust_sigma))
    fall_pts = float(np.sum(f_arr[peak_idx + 1 :] > med + robust_sigma))
    spike_rise_fall_ratio = float((rise_pts + 1.0) / (fall_pts + 1.0))

    # 6. Dips & Transit Features (5)
    dip_thresh = med - 3.0 * robust_sigma
    is_dip = f_arr < dip_thresh
    dip_count = float(np.sum(is_dip))
    dip_depth = float(med - np.min(f_arr))
    dip_width = float(dip_count * dt)
    dip_fraction = float(np.mean(is_dip))

    trough_idx = int(np.argmin(f_arr))
    left_pts = float(np.sum(f_arr[:trough_idx] < med - robust_sigma))
    right_pts = float(np.sum(f_arr[trough_idx + 1 :] < med - robust_sigma))
    dip_symmetry = float((left_pts + 1.0) / (right_pts + 1.0))

    # 7. Z-scores & Roughness (4)
    z_robust = (f_arr - med) / robust_sigma
    zscore_max = float(np.max(z_robust))
    zscore_min = float(np.min(z_robust))
    zscore_fraction_outside_3sigma = float(np.mean(np.abs(z_robust) > 3.0))

    diffs = np.diff(f_arr)
    roughness = float(np.mean(diffs**2) / (f_var + 1e-12))

    # 8. Trend, Entropy, Gap Proxy & Peaks/Troughs (6)
    var_t = float(np.var(t_arr))
    cov_tf = float(np.mean((t_arr - np.mean(t_arr)) * f_detrend))
    trend_slope = float(cov_tf / (var_t + 1e-12))

    counts, _ = np.histogram(f_arr, bins=20)
    probs = counts / (np.sum(counts) + 1e-12)
    probs = probs[probs > 0.0]
    histogram_entropy = float(-np.sum(probs * np.log2(probs)))

    gap_proxy = float(np.max(np.abs(diffs))) if len(diffs) > 0 else 0.0

    # Local extrema
    diff_f = np.diff(f_arr)
    local_peaks = np.where((diff_f[:-1] > 0.0) & (diff_f[1:] <= 0.0))[0]
    local_troughs = np.where((diff_f[:-1] < 0.0) & (diff_f[1:] >= 0.0))[0]
    peak_count = float(len(local_peaks))
    trough_count = float(len(local_troughs))
    peak_trough_ratio = float((peak_count + 1.0) / (trough_count + 1.0))

    features = np.array(
        [
            fft_peak_period,
            fft_peak_power,
            fft_power_snr,
            fft_harmonic_ratio,
            fft_band_ratio_low,
            fft_band_ratio_high,
            acf_first_zero_crossing,
            acf_fwhm,
            acf_first_local_peak_lag,
            acf_first_local_peak_val,
            moment_skew,
            moment_kurtosis,
            moment_iqr,
            moment_cv,
            moment_std,
            quantile_range,
            quantile_p95_p05,
            quantile_q01,
            quantile_q99,
            quantile_median,
            spike_count,
            spike_max,
            spike_width,
            flare_index,
            spike_rise_fall_ratio,
            dip_count,
            dip_depth,
            dip_width,
            dip_fraction,
            dip_symmetry,
            zscore_max,
            zscore_min,
            zscore_fraction_outside_3sigma,
            roughness,
            trend_slope,
            histogram_entropy,
            gap_proxy,
            peak_count,
            trough_count,
            peak_trough_ratio,
        ],
        dtype=np.float64,
    )

    return features


def feature_matrix(t: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Extract the (n, 40) feature matrix for a batch of light curves.

    Parameters
    ----------
    t : np.ndarray
        1D array of timestamps of length m.
    f : np.ndarray
        Either 1D array of shape (m,) or 2D array of shape (n, m).

    Returns
    -------
    np.ndarray
        2D feature matrix of shape (n, 40) with dtype float64.
    """
    f_arr = np.asarray(f, dtype=np.float64)
    if f_arr.ndim == 1:
        return compute_features(t, f_arr).reshape(1, len(FEATURE_NAMES))

    n_curves = f_arr.shape[0]
    matrix = np.empty((n_curves, len(FEATURE_NAMES)), dtype=np.float64)
    for i in range(n_curves):
        matrix[i] = compute_features(t, f_arr[i])

    return matrix
