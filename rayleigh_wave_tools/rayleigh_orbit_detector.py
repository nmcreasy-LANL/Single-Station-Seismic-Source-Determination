#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rayleigh wave orbit detector."""
from __future__ import annotations

import os

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr
from .prem_dispersion import rayleigh_group_velocity_prem


def calculate_snr_in_window(energy_trace, peak_idx, window_half_width=50):
    """
    Calculate signal-to-noise ratio around a peak.
    
    Parameters
    ----------
    energy_trace : np.ndarray
        Energy time series
    peak_idx : int
        Index of peak
    window_half_width : int
        Half-width of signal window in samples
    
    Returns
    -------
    snr : float
        Signal-to-noise ratio (linear scale)
    """
    n = len(energy_trace)
    sig_start = max(0, peak_idx - window_half_width)
    sig_end = min(n, peak_idx + window_half_width)
    signal = energy_trace[sig_start:sig_end]
    signal_power = np.mean(signal)
    noise_start1 = max(0, sig_start - 2 * window_half_width)
    noise_end1 = sig_start
    noise_start2 = sig_end
    noise_end2 = min(n, sig_end + 2 * window_half_width)
    noise_parts = []
    if noise_end1 > noise_start1:
        noise_parts.append(energy_trace[noise_start1:noise_end1])
    if noise_end2 > noise_start2:
        noise_parts.append(energy_trace[noise_start2:noise_end2])
    if len(noise_parts) > 0:
        noise = np.concatenate(noise_parts)
        noise_power = np.mean(noise)
    else:
        noise_power = np.min(energy_trace)
    if noise_power > 0:
        snr = signal_power / noise_power
    else:
        snr = signal_power / (np.mean(energy_trace) * 0.01)
    return snr


def calculate_dispersion_shape_correlation(periods_obs, arrivals_obs, distance_deg, orbit='R1'):
    """
    Calculate correlation between observed and theoretical dispersion curve SHAPE.
    This is more robust than RMS because it's insensitive to absolute time shifts.
    
    Parameters
    ----------
    periods_obs : np.ndarray
        Observed periods (seconds)
    arrivals_obs : np.ndarray
        Observed arrival times (minutes)
    distance_deg : float
        Distance in degrees
    orbit : str
        Orbit type: "R1", "R2", or "R3"
    
    Returns
    -------
    correlation : float
        Pearson correlation coefficient (0-1, higher is better)
    rms_residual : float
        RMS residual in seconds
    """
    R_earth = 6371
    if orbit == 'R1':
        path_length_km = distance_deg * R_earth * np.pi / 180
    elif orbit == 'R2':
        path_length_km = (360 - distance_deg) * R_earth * np.pi / 180
    elif orbit == 'R3':
        path_length_km = (360 + distance_deg) * R_earth * np.pi / 180
    else:
        return (0.0, np.inf)
    U_pred = rayleigh_group_velocity_prem(periods_obs)
    arrivals_pred = path_length_km / U_pred / 60
    if len(periods_obs) >= 3:
        corr, _ = pearsonr(arrivals_obs, arrivals_pred)
        corr = max(0, corr)
    else:
        corr = 0.0
    residuals_sec = (arrivals_obs - arrivals_pred) * 60
    rms = np.sqrt(np.mean(residuals_sec ** 2))
    return (corr, rms)


def calculate_arrival_consistency(periods_obs, arrivals_obs):
    """
    Measure how consistent arrival times are across periods.
    Good picks should show smooth dispersion, not erratic jumps.
    
    Returns
    -------
    consistency_score : float
        0-1, where 1 is perfectly smooth
    """
    if len(periods_obs) < 3:
        return 0.0
    sort_idx = np.argsort(periods_obs)
    T_sorted = periods_obs[sort_idx]
    arrivals_sorted = arrivals_obs[sort_idx]
    d2_arrivals = np.diff(arrivals_sorted, n=2)
    d2_periods = np.diff(T_sorted, n=2)
    if len(d2_periods) > 0:
        curvature = np.abs(d2_arrivals / (d2_periods + 1e-06))
        consistency = np.exp(-np.mean(curvature) / 0.5)
    else:
        consistency = 0.5
    return consistency


def get_expected_bandwidth_factor(orbit, periods_obs):
    """
    Calculate expected bandwidth factor for orbit-aware confidence scoring.
    
    Physics reasoning:
    - R2/R3 travel longer paths -> more attenuation -> narrower usable bandwidth
    - Short-period energy (50-80s) is heavily attenuated for R2/R3
    - This is EXPECTED physics, not poor detection quality
    
    The bandwidth factor normalizes the number of frequency points by the
    fraction of the full 50-150s band that is actually usable for each orbit.
    
    Parameters
    ----------
    orbit : str
        Orbit type: 'R1', 'R2', or 'R3'
    periods_obs : np.ndarray
        Observed periods (seconds) - used to calculate actual bandwidth
    
    Returns
    -------
    bandwidth_factor : float
        Expected usable bandwidth fraction (0-1)
        - R1: ~1.0 (full 50-150s band usable)
        - R2: ~0.70 (mainly 80-150s survives)
        - R3: ~0.50 (mainly 100-150s survives)
    actual_bandwidth_fraction : float
        Actual bandwidth fraction from observed periods (for diagnostics)
    
    Examples
    --------
    R1 with 11 points across 50-150s:
        bandwidth_factor = 1.0
        normalized_points = 11 / 1.0 = 11 (no penalty)
    
    R2 with 6 points across 80-150s:
        bandwidth_factor = 0.70
        normalized_points = 6 / 0.70 = 8.6 (fair comparison!)
    
    R3 with 5 points across 110-150s:
        bandwidth_factor = 0.50
        normalized_points = 5 / 0.50 = 10 (excellent!)
    """
    if orbit == 'R1':
        bandwidth_factor = 1.0
    elif orbit == 'R2':
        bandwidth_factor = 0.7
    elif orbit == 'R3':
        bandwidth_factor = 0.5
    else:
        bandwidth_factor = 1.0
    if len(periods_obs) > 0:
        actual_min = np.min(periods_obs)
        actual_max = np.max(periods_obs)
        actual_bandwidth_fraction = (actual_max - actual_min) / 100.0
    else:
        actual_bandwidth_fraction = 0.0
    return (bandwidth_factor, actual_bandwidth_fraction)


def calculate_weighted_rms(periods_obs, arrivals_obs, arrivals_pred, orbit):
    """
    Calculate amplitude-weighted RMS for orbit-aware scoring.
    
    For R2/R3, long-period measurements are more reliable (higher SNR)
    and should be weighted more heavily in the RMS calculation.
    
    Parameters
    ----------
    periods_obs : np.ndarray
        Observed periods (seconds)
    arrivals_obs : np.ndarray
        Observed arrival times (minutes)
    arrivals_pred : np.ndarray
        Predicted arrival times (minutes)
    orbit : str
        Orbit type: 'R1', 'R2', or 'R3'
    
    Returns
    -------
    weighted_rms : float
        Weighted RMS residual in seconds
    weights : np.ndarray
        Weights used (for diagnostics)
    """
    residuals_sec = (arrivals_obs - arrivals_pred) * 60.0
    if orbit in ['R2', 'R3']:
        weights = np.exp((periods_obs - 50.0) / 50.0)
    else:
        weights = np.ones_like(periods_obs)
    weights = weights * len(weights) / np.sum(weights)
    weighted_rms = np.sqrt(np.sum(weights * residuals_sec ** 2) / np.sum(weights))
    return (weighted_rms, weights)


def calculate_rms_score_adaptive(rms, orbit, n_points):
    """
    Adaptive RMS scoring that accounts for orbit-specific dispersion characteristics.
    
    Key insights:
    - R2/R3 travel longer paths -> more scattering -> broader dispersion
    - Fewer measurement points -> less overfitting concern -> more lenient
    - R1 is shortest path -> tightest fit expected
    
    Parameters
    ----------
    rms : float
        RMS residual in seconds
    orbit : str
        Orbit type: 'R1', 'R2', or 'R3'
    n_points : int
        Number of dispersion curve points measured
    
    Returns
    -------
    rms_score : float
        Score from 0-1, where 1 is perfect fit
    """
    rms_tolerance = {'R1': 60.0, 'R2': 90.0, 'R3': 120.0}
    tolerance = rms_tolerance.get(orbit, 60.0)
    if n_points < 5:
        tolerance *= 1.5
    elif n_points < 8:
        tolerance *= 1.2
    rms_score = np.exp(-rms / tolerance)
    return rms_score


def calculate_confidence_score(periods_obs, arrivals_obs, amps_obs, distance_deg, orbit, energy_trace, peak_idx):
    """
    Calculate comprehensive confidence score for orbit detection.
    
    **BANDWIDTH-AWARE VERSION** - Fairly treats R2/R3 orbits!
    
    Components:
    1. Dispersion shape correlation (0-1)
    2. RMS residual quality (0-1, exponential decay) - ORBIT-ADAPTIVE with optional weighting
    3. SNR in detection window (0-1, sigmoid)
    4. Arrival time consistency (0-1)
    5. Number of frequency points (0-1) - **BANDWIDTH-NORMALIZED**
    
    Key improvement: R2/R3 are no longer penalized for having fewer frequency
    points due to physics (attenuation) rather than poor detection quality.
    
    Parameters
    ----------
    periods_obs : np.ndarray
        Observed periods (seconds)
    arrivals_obs : np.ndarray
        Observed arrival times (minutes)
    amps_obs : np.ndarray
        Observed amplitudes
    distance_deg : float
        Distance in degrees
    orbit : str
        Orbit type: 'R1', 'R2', or 'R3'
    energy_trace : np.ndarray
        Energy time series for SNR calculation
    peak_idx : int
        Peak index in energy trace
    
    Returns
    -------
    confidence : float
        Overall confidence score (0-1)
    components : dict
        Individual component scores for diagnostics
    """
    corr, rms = calculate_dispersion_shape_correlation(periods_obs, arrivals_obs, distance_deg, orbit)
    n_points = len(periods_obs)
    R_earth = 6371
    if orbit == 'R1':
        path_length_km = distance_deg * R_earth * np.pi / 180
    elif orbit == 'R2':
        path_length_km = (360 - distance_deg) * R_earth * np.pi / 180
    elif orbit == 'R3':
        path_length_km = (360 + distance_deg) * R_earth * np.pi / 180
    else:
        path_length_km = distance_deg * R_earth * np.pi / 180
    U_pred = rayleigh_group_velocity_prem(periods_obs)
    arrivals_pred = path_length_km / U_pred / 60
    if orbit in ['R2', 'R3']:
        rms_weighted, weights_used = calculate_weighted_rms(periods_obs, arrivals_obs, arrivals_pred, orbit)
        rms_for_scoring = rms_weighted
    else:
        rms_for_scoring = rms
        weights_used = np.ones_like(periods_obs)
    rms_score = calculate_rms_score_adaptive(rms_for_scoring, orbit, n_points)
    snr = calculate_snr_in_window(energy_trace, peak_idx)
    snr_score = 1.0 / (1.0 + np.exp(-0.5 * (snr - 3.0)))
    consistency = calculate_arrival_consistency(periods_obs, arrivals_obs)
    bandwidth_factor, actual_bw_frac = get_expected_bandwidth_factor(orbit, periods_obs)
    n_points_normalized = n_points / bandwidth_factor
    n_points_score = min(1.0, n_points_normalized / 10.0)
    weights = {'shape_correlation': 0.3, 'rms_quality': 0.25, 'snr': 0.2, 'consistency': 0.15, 'n_points': 0.1}
    confidence = weights['shape_correlation'] * corr + weights['rms_quality'] * rms_score + weights['snr'] * snr_score + weights['consistency'] * consistency + weights['n_points'] * n_points_score
    components = {'shape_correlation': corr, 'rms_residual_sec': rms, 'rms_weighted_sec': rms_for_scoring if orbit in ['R2', 'R3'] else rms, 'rms_quality': rms_score, 'snr': snr, 'snr_score': snr_score, 'consistency': consistency, 'n_points': n_points, 'n_points_normalized': n_points_normalized, 'n_points_score': n_points_score, 'bandwidth_factor': bandwidth_factor, 'actual_bandwidth_fraction': actual_bw_frac, 'overall_confidence': confidence}
    return (confidence, components)


def targeted_R2_search(peak_dispersions, S_energy_band, freqs_band, time_min, energy_time_integrated, R1_arrival_time, distance_est, verbose=True):
    """
    Targeted search for R2 when initial peak detection misses it.
    
    Uses R1 detection to predict R2 location via geometric relationship:
    t_R2 = t_R1 x (360 - Delta) / Delta
    
    Parameters
    ----------
    peak_dispersions : list
        Already detected peaks (may not include R2)
    S_energy_band : np.ndarray
        Stockwell energy in frequency band
    freqs_band : np.ndarray
        Frequencies in band
    time_min : np.ndarray
        Time array in minutes
    energy_time_integrated : np.ndarray
        Frequency-integrated energy time series
    R1_arrival_time : float
        Detected R1 arrival in minutes
    distance_est : float
        Estimated distance from R1 in degrees
    verbose : bool
        Print search details
    
    Returns
    -------
    R2_candidate : dict or None
        Dispersion curve data if R2 found, else None
    """

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    if distance_est <= 0 or distance_est >= 180:
        vprint(f'   Warning: Invalid distance estimate: {distance_est} deg')
        return None
    R2_expected = R1_arrival_time * (360 - distance_est) / distance_est
    vprint(f'   Expected R2 arrival: {R2_expected:.1f} min (based on R1 @ {R1_arrival_time:.1f} min)')
    search_start = R2_expected - 20
    search_end = R2_expected + 20
    search_start = max(time_min[0], search_start)
    search_end = min(time_min[-1], search_end)
    vprint(f'   Search window: {search_start:.1f} - {search_end:.1f} min')
    time_mask = (time_min >= search_start) & (time_min <= search_end)
    if np.sum(time_mask) < 50:
        vprint(f'   Warning: Search window too small: {np.sum(time_mask)} points')
        return None
    energy_local = energy_time_integrated[time_mask]
    time_local = time_min[time_mask]
    prominence_threshold = 0.02 * np.max(energy_time_integrated)
    vprint(f'   Using relaxed prominence: {prominence_threshold:.2e} (2% of global max)')
    peaks_idx, peak_props = find_peaks(energy_local, prominence=prominence_threshold, distance=10)
    if len(peaks_idx) == 0:
        vprint('   No peaks found with relaxed criteria; using global maximum in window.')
        peaks_idx = [np.argmax(energy_local)]
    vprint(f'   Found {len(peaks_idx)} candidate peak(s) in search window')
    best_peak_local_idx = peaks_idx[np.argmax(energy_local[peaks_idx])]
    peak_time = time_local[best_peak_local_idx]
    peak_energy = energy_local[best_peak_local_idx]
    global_time_idx = np.where(time_min == peak_time)[0]
    if len(global_time_idx) == 0:
        vprint('   Warning: Could not map local peak to global time array')
        return None
    peak_idx_global = global_time_idx[0]
    vprint(f'   Strongest peak at: {peak_time:.1f} min (energy: {peak_energy:.2e})')
    vprint('   Extracting dispersion curve with R2 parameters.')
    disp_result = extract_dispersion_curve_robust(S_energy_band, freqs_band, time_min, peak_time, orbit_hint='R2')
    if disp_result is None:
        vprint('   Failed to extract a valid dispersion curve')
        return None
    vprint(f"   OK: Extracted {len(disp_result['periods'])} frequency points")
    R2_candidate = {'peak_time': peak_time, 'peak_energy': peak_energy, 'peak_idx': peak_idx_global, 'periods': disp_result['periods'], 'arrivals': disp_result['arrivals'], 'amps': disp_result['amps'], 'is_targeted_search': True}
    return R2_candidate


def get_adaptive_confidence_threshold(orbit, snr, base_threshold=0.2):
    """
    Orbit-specific confidence thresholds.
    
    R2/R3 get relaxed thresholds because:
    - Longer paths = lower SNR
    - More scattering = less coherent waveforms
    - Broader dispersion = harder to fit
    """
    orbit_thresholds = {'R1': base_threshold, 'R2': base_threshold * 0.7, 'R3': base_threshold * 0.75}
    threshold = orbit_thresholds.get(orbit, base_threshold)
    if snr > 2.0:
        threshold *= 0.85
    elif snr > 3.0:
        threshold *= 0.7
    return threshold


def joint_fit_orbits(peak_dispersions, distance_range_deg=(50, 110)):
    """
    Jointly fit R1, R2, R3 with a SHARED distance constraint.
    
    This is more robust than independent fitting because:
    - All orbits must be consistent with the same source distance
    - Reduces false positives from picking wrong peaks
    
    Parameters
    ----------
    peak_dispersions : list of dict
        Each dict contains 'periods', 'arrivals', 'amps', 'peak_time', 'peak_energy'
    distance_range_deg : tuple
        (min, max) distance in degrees
    
    Returns
    -------
    best_solution : dict
        Contains distance and orbit assignments with confidence scores
    """
    n_peaks = len(peak_dispersions)
    if n_peaks == 0:
        return None
    best_score = -np.inf
    best_solution = None
    orbit_options = ['R1', 'R2', 'R3', None]
    if n_peaks > 5:
        print(f'Warning: Warning: {n_peaks} peaks detected. Limiting joint fit to top 5 by energy.')
        sorted_peaks = sorted(peak_dispersions, key=lambda x: x['peak_energy'], reverse=True)
        peak_dispersions = sorted_peaks[:5]
        n_peaks = 5
    distances_to_test = np.linspace(distance_range_deg[0], distance_range_deg[1], 30)
    for distance_deg in distances_to_test:
        orbit_scores = {}
        for orbit in ['R1', 'R2', 'R3']:
            orbit_scores[orbit] = []
            for peak_idx, disp_data in enumerate(peak_dispersions):
                periods_obs = disp_data['periods']
                arrivals_obs = disp_data['arrivals']
                corr, rms = calculate_dispersion_shape_correlation(periods_obs, arrivals_obs, distance_deg, orbit)
                score = corr * np.exp(-rms / 100.0)
                orbit_scores[orbit].append({'peak_idx': peak_idx, 'score': score, 'corr': corr, 'rms': rms})
        assignment = {}
        used_peaks = set()
        for orbit in ['R1', 'R2', 'R3']:
            sorted_peaks = sorted(orbit_scores[orbit], key=lambda x: x['score'], reverse=True)
            for peak_info in sorted_peaks:
                peak_idx = peak_info['peak_idx']
                if peak_idx not in used_peaks and peak_info['score'] > 0.3:
                    assignment[orbit] = peak_info
                    used_peaks.add(peak_idx)
                    break
        total_score = sum((info['score'] for info in assignment.values()))
        n_orbits_detected = len(assignment)
        total_score *= 1.0 + 0.2 * n_orbits_detected
        if total_score > best_score:
            best_score = total_score
            best_solution = {'distance_deg': distance_deg, 'assignments': assignment, 'total_score': total_score, 'n_orbits': n_orbits_detected}
    return best_solution


def _orbit_path_length_km(distance_deg, orbit):
    """Return the geometric Rayleigh orbit path length for a distance."""
    radius_earth_km = 6371.0
    if orbit == "R1":
        path_deg = float(distance_deg)
    elif orbit == "R2":
        path_deg = 360.0 - float(distance_deg)
    elif orbit == "R3":
        path_deg = 360.0 + float(distance_deg)
    else:
        raise ValueError(f"Unknown orbit: {orbit}")
    return path_deg * np.pi / 180.0 * radius_earth_km


def _predicted_arrivals_min(periods_sec, distance_deg, orbit):
    """Predict period-dependent arrival times for an orbit and distance."""
    periods_sec = np.asarray(periods_sec, dtype=float)
    velocities_km_s = rayleigh_group_velocity_prem(periods_sec)
    return _orbit_path_length_km(distance_deg, orbit) / velocities_km_s / 60.0


def _observed_arrivals_for_distance_fit(disp_data):
    """Return measured arrivals for distance fitting, preferring raw Stockwell picks."""
    periods = np.asarray(disp_data.get("periods", []), dtype=float)
    arrivals = np.asarray(
        disp_data.get("arrivals_picked_raw", disp_data.get("arrivals", [])),
        dtype=float,
    )
    amps = np.asarray(disp_data.get("amps", np.ones_like(periods)), dtype=float)

    n = min(periods.size, arrivals.size, amps.size)
    if n == 0:
        return None, None, None

    periods = periods[:n]
    arrivals = arrivals[:n]
    amps = amps[:n]
    finite = np.isfinite(periods) & np.isfinite(arrivals) & np.isfinite(amps)
    if not np.any(finite):
        return None, None, None
    return periods[finite], arrivals[finite], amps[finite]


def _orbit_distance_log_likelihood(disp_data, orbit, distance_deg, sigma_sec):
    """Gaussian residual log likelihood for one orbit at one distance."""
    periods, arrivals_obs, amps = _observed_arrivals_for_distance_fit(disp_data)
    if periods is None or periods.size < 2:
        return -np.inf

    arrivals_pred = _predicted_arrivals_min(periods, distance_deg, orbit)
    residual_sec = (arrivals_obs - arrivals_pred) * 60.0

    weights = np.maximum(amps, 0.0) ** 2
    if np.sum(weights) <= 0.0 or not np.all(np.isfinite(weights)):
        weights = np.ones_like(periods, dtype=float)
    weights = weights * (weights.size / np.sum(weights))

    sigma_sec = max(float(sigma_sec), 1e-6)
    return float(-0.5 * np.sum(weights * (residual_sec / sigma_sec) ** 2))


def _distance_posterior_moments(distance_grid_deg, posterior_pdf):
    """Return MAP, mean, and standard deviation for a 1-D distance posterior."""
    grid = np.asarray(distance_grid_deg, dtype=float)
    pdf = np.asarray(posterior_pdf, dtype=float)
    if grid.size == 0 or pdf.size != grid.size:
        return np.nan, np.nan, np.nan
    map_distance = float(grid[int(np.nanargmax(pdf))])
    if grid.size == 1:
        return map_distance, map_distance, 0.0
    area = float(np.trapz(pdf, grid))
    if not np.isfinite(area) or area <= 0.0:
        pdf = np.ones_like(grid, dtype=float) / grid.size
        area = float(np.trapz(pdf, grid))
    mean = float(np.trapz(grid * pdf, grid) / area)
    variance = float(np.trapz((grid - mean) ** 2 * pdf, grid) / area)
    return map_distance, mean, float(np.sqrt(max(variance, 0.0)))


def _branch_diagnostics_for_distance(disp_data, orbit, distance_deg):
    """Return RMS/correlation diagnostics against measured picks for one distance."""
    periods, arrivals_obs, _ = _observed_arrivals_for_distance_fit(disp_data)
    if periods is None or periods.size < 2:
        return {"corr": 0.0, "rms": np.inf, "n_points": 0}
    arrivals_pred = _predicted_arrivals_min(periods, distance_deg, orbit)
    residual_sec = (arrivals_obs - arrivals_pred) * 60.0
    rms = float(np.sqrt(np.mean(residual_sec ** 2)))
    if periods.size >= 3:
        try:
            corr, _ = pearsonr(arrivals_obs, arrivals_pred)
            corr = float(max(0.0, corr))
        except Exception:
            corr = 0.0
    else:
        corr = 0.0
    return {"corr": corr, "rms": rms, "n_points": int(periods.size)}


def joint_fit_guided_orbits(
    peak_dispersions,
    distance_range_deg,
    *,
    guide_distance_deg=None,
    prior_sigma_deg=None,
    grid_step_deg=0.1,
    orbit_residual_sigma_sec=None,
):
    """Estimate shared distance for guided R1/R2/R3 picks on a distance grid.

    The guided distance is used only as a broad prior center and to establish
    the original extraction windows.  The returned ``distance_deg`` and
    ``map_distance_deg`` are the MAP of a normalized distance posterior built
    from the measured local Stockwell maxima, preferring ``arrivals_picked_raw``
    over model branch times whenever raw picks are available.
    """
    if orbit_residual_sigma_sec is None:
        orbit_residual_sigma_sec = {"R1": 90.0, "R2": 150.0, "R3": 210.0}

    d0, d1 = distance_range_deg
    d0 = max(1.0, float(d0))
    d1 = min(179.0, float(d1))
    if d1 <= d0:
        return None

    grid_step_deg = max(float(grid_step_deg), 1e-3)
    distance_grid = np.arange(d0, d1 + 0.5 * grid_step_deg, grid_step_deg)
    if distance_grid.size == 0:
        return None

    candidates = []
    seen_orbits = set()
    for peak_idx, disp_data in enumerate(peak_dispersions):
        orbit = disp_data.get("orbit_hint")
        if orbit not in {"R1", "R2", "R3"}:
            continue
        if orbit in seen_orbits:
            # Keep the first guided extraction for each geometric orbit.  The
            # distance-guided path normally produces one candidate per orbit.
            continue
        periods, arrivals_obs, amps = _observed_arrivals_for_distance_fit(disp_data)
        if periods is None or periods.size < 2:
            continue
        seen_orbits.add(orbit)
        candidates.append((orbit, peak_idx, disp_data))

    if not candidates:
        return None

    log_likelihood = np.zeros_like(distance_grid, dtype=float)
    orbit_log_likelihood = {}
    for orbit, _, disp_data in candidates:
        sigma_sec = orbit_residual_sigma_sec.get(orbit, 150.0)
        ll = np.array([
            _orbit_distance_log_likelihood(disp_data, orbit, d, sigma_sec)
            for d in distance_grid
        ], dtype=float)
        if not np.any(np.isfinite(ll)):
            continue
        ll = np.where(np.isfinite(ll), ll, -1e300)
        orbit_log_likelihood[orbit] = ll
        log_likelihood += ll

    if not orbit_log_likelihood:
        return None

    log_prior = np.zeros_like(distance_grid, dtype=float)
    if guide_distance_deg is not None and prior_sigma_deg is not None and float(prior_sigma_deg) > 0.0:
        sigma = float(prior_sigma_deg)
        log_prior = -0.5 * ((distance_grid - float(guide_distance_deg)) / sigma) ** 2

    log_posterior = log_likelihood + log_prior
    finite = np.isfinite(log_posterior)
    if not np.any(finite):
        return None
    log_posterior = np.where(finite, log_posterior, -1e300)
    log_posterior_shifted = log_posterior - np.nanmax(log_posterior)
    posterior = np.exp(log_posterior_shifted)
    if distance_grid.size > 1:
        area = float(np.trapz(posterior, distance_grid))
        if np.isfinite(area) and area > 0.0:
            posterior = posterior / area
        else:
            posterior = posterior / np.sum(posterior)
    else:
        posterior = np.ones_like(distance_grid, dtype=float)

    map_distance, mean_distance, std_distance = _distance_posterior_moments(distance_grid, posterior)
    map_idx = int(np.nanargmax(posterior))

    assignments = {}
    for orbit, peak_idx, disp_data in candidates:
        diag = _branch_diagnostics_for_distance(disp_data, orbit, map_distance)
        assignments[orbit] = {
            "peak_idx": peak_idx,
            "score": float(np.exp(orbit_log_likelihood[orbit][map_idx] - np.nanmax(orbit_log_likelihood[orbit]))),
            "corr": diag["corr"],
            "rms": diag["rms"],
            "n_points": diag["n_points"],
        }

    return {
        "distance_deg": map_distance,
        "map_distance_deg": map_distance,
        "mean_distance_deg": mean_distance,
        "distance_std_deg": std_distance,
        "assignments": assignments,
        "total_score": float(log_likelihood[map_idx]),
        "n_orbits": len(assignments),
        "is_distance_guided": True,
        "guide_distance_deg": None if guide_distance_deg is None else float(guide_distance_deg),
        "distance_prior_center_deg": None if guide_distance_deg is None else float(guide_distance_deg),
        "distance_prior_sigma_deg": None if prior_sigma_deg is None else float(prior_sigma_deg),
        "distance_grid_step_deg": float(grid_step_deg),
        "distance_grid_deg": distance_grid,
        "distance_log_likelihood": log_likelihood,
        "distance_log_prior": log_prior,
        "distance_log_posterior": log_posterior,
        "distance_posterior_pdf": posterior,
    }


def _dispersion_with_distance_branch(disp_data, orbit, distance_deg):
    """Return a copy whose model branch is evaluated at the shared distance."""
    out = dict(disp_data)
    periods = np.asarray(out.get("periods", []), dtype=float)
    if periods.size == 0:
        return out
    arrivals_model = _predicted_arrivals_min(periods, distance_deg, orbit)
    out["arrivals"] = arrivals_model
    out["distance"] = float(distance_deg)
    out["best_distance_deg"] = float(distance_deg)

    if "arrivals_picked_raw" in out:
        arrivals_picked = np.asarray(out["arrivals_picked_raw"], dtype=float)
        n = min(arrivals_picked.size, arrivals_model.size)
        if n > 0:
            residual_sec = (arrivals_picked[:n] - arrivals_model[:n]) * 60.0
            out["raw_pick_rms_to_model_sec"] = float(np.sqrt(np.mean(residual_sec ** 2)))
            out["raw_pick_median_abs_to_model_sec"] = float(np.median(np.abs(residual_sec)))
    return out


def fit_dispersion_curve_model(
    S_energy_band,
    freqs_band,
    time_min,
    peak_time,
    orbit,
    distance_range=(50, 120),
    window_half_width=10.0,
    pick_half_width_min=1.0,
    proximity_sigma_fraction=0.50,
    match_threshold=None,
    grid_step_deg=0.1,
):
    """
    Fit a PREM-like branch to the 2D Stockwell energy distribution, then
    return the actual local Stockwell maxima near that branch.

    Important distinction:
      arrivals             = model branch times, minutes
      arrivals_picked_raw  = actual local Stockwell maxima, minutes

    The returned result keeps the model branch and measured local maxima
    separate so diagnostics can distinguish prediction from observation.
    """
    time_mask = (time_min >= peak_time - window_half_width) & (
        time_min <= peak_time + window_half_width
    )
    if np.sum(time_mask) < 50:
        return None

    time_local = time_min[time_mask]
    S_local = S_energy_band[:, time_mask]
    periods = 1.0 / freqs_band
    R_earth = 6371.0

    if orbit in ["R2", "R3"]:
        min_freqs = 4
        period_weights = np.exp((periods - 50.0) / 50.0)
    else:
        min_freqs = 5
        period_weights = np.ones_like(periods)

    if match_threshold is None:
        # Threshold applied to normalized energy multiplied by model proximity.
        match_threshold = {"R1": 0.25, "R2": 0.18, "R3": 0.15}.get(orbit, 0.20)

    freq_max = np.max(S_local, axis=1)
    valid_freq = np.isfinite(freq_max) & (freq_max > 0.0)

    pick_half_width_min = float(pick_half_width_min)
    proximity_sigma_min = max(
        1e-6, pick_half_width_min * float(proximity_sigma_fraction)
    )

    def path_km_for_distance(distance_deg):
        if orbit == "R1":
            path_deg = distance_deg
        elif orbit == "R2":
            path_deg = 360.0 - distance_deg
        elif orbit == "R3":
            path_deg = 360.0 + distance_deg
        else:
            return None
        return path_deg * R_earth * np.pi / 180.0

    def best_local_pick(i, t_pred):
        """
        Pick the local Stockwell maximum near the model time for one period row.

        Returns the measured peak time, model offset, normalized energy, and
        proximity-weighted score.
        """
        if not np.isfinite(t_pred):
            return None
        if t_pred < time_local[0] or t_pred > time_local[-1]:
            return None

        local_pick_mask = np.abs(time_local - t_pred) <= pick_half_width_min
        if not np.any(local_pick_mask):
            return None

        idxs = np.where(local_pick_mask)[0]
        offsets = time_local[idxs] - t_pred
        norm_energy = S_local[i, idxs] / (freq_max[i] + 1e-300)
        proximity = np.exp(-0.5 * (offsets / proximity_sigma_min) ** 2)
        score = norm_energy * proximity

        jj = int(np.nanargmax(score))
        j = int(idxs[jj])

        return {
            "j_local": j,
            "time": float(time_local[j]),
            "offset_min": float(offsets[jj]),
            "energy": float(S_local[i, j]),
            "norm_energy": float(norm_energy[jj]),
            "proximity": float(proximity[jj]),
            "score": float(score[jj]),
        }

    def model_score_at_distance(distance_deg):
        path_km = path_km_for_distance(distance_deg)
        if path_km is None:
            return -np.inf

        U_pred = rayleigh_group_velocity_prem(periods)
        arrivals_pred = path_km / U_pred / 60.0

        scores = []
        weights = []

        for i, t_pred in enumerate(arrivals_pred):
            if not valid_freq[i]:
                continue

            local = best_local_pick(i, t_pred)
            if local is None:
                continue

            scores.append(local["score"])
            weights.append(period_weights[i])

        if len(scores) < min_freqs:
            return -np.inf

        scores = np.asarray(scores, dtype=float)
        weights = np.asarray(weights, dtype=float)
        weights = weights / np.sum(weights)

        coverage = min(1.0, len(scores) / max(8.0, min_freqs))
        return float(np.sum(scores * weights) * (0.75 + 0.25 * coverage))

    d0, d1 = distance_range
    d0 = max(1.0, float(d0))
    d1 = min(179.0, float(d1))
    if d1 <= d0:
        return None

    distances_test = np.linspace(d0, d1, 61)
    scores_test = np.array([model_score_at_distance(d) for d in distances_test])

    if not np.any(np.isfinite(scores_test)):
        return None

    best_idx = int(np.nanargmax(scores_test))
    best_distance = float(distances_test[best_idx])
    best_score = float(scores_test[best_idx])

    refine_lo = max(d0, best_distance - 3.0)
    refine_hi = min(d1, best_distance + 3.0)
    if refine_hi > refine_lo and np.isfinite(best_score):
        result_opt = minimize_scalar(
            lambda d: -model_score_at_distance(d),
            bounds=(refine_lo, refine_hi),
            method="bounded",
        )
        if result_opt.success and np.isfinite(result_opt.fun):
            best_distance = float(result_opt.x)
            best_score = float(-result_opt.fun)

    # Compute grid-based posterior for single-orbit distance estimation
    grid_step_deg = max(float(grid_step_deg), 1e-3)
    distance_grid = np.arange(d0, d1 + 0.5 * grid_step_deg, grid_step_deg)
    
    log_likelihood = np.array([
        model_score_at_distance(d) for d in distance_grid
    ], dtype=float)
    
    # Normalize to posterior (flat prior for independent mode)
    log_likelihood = np.where(np.isfinite(log_likelihood), log_likelihood, -1e300)
    log_likelihood_shifted = log_likelihood - np.nanmax(log_likelihood)
    posterior = np.exp(log_likelihood_shifted)
    
    if distance_grid.size > 1:
        area = float(np.trapz(posterior, distance_grid))
        if np.isfinite(area) and area > 0.0:
            posterior = posterior / area
        else:
            posterior = posterior / np.sum(posterior)
    else:
        posterior = np.ones_like(distance_grid, dtype=float)
    
    # Compute moments using helper function
    map_distance, mean_distance, std_distance = _distance_posterior_moments(distance_grid, posterior)

    path_km = path_km_for_distance(best_distance)
    if path_km is None:
        return None

    U_pred = rayleigh_group_velocity_prem(periods)
    arrivals_pred = path_km / U_pred / 60.0

    candidate_periods = []
    candidate_arrivals_model = []
    candidate_arrivals_picked_raw = []
    candidate_amps = []
    candidate_norm_energy = []
    candidate_proximity = []
    candidate_score = []
    candidate_time_offset_min = []
    candidate_accepted = []
    candidate_reason = []

    periods_good = []
    arrivals_model_good = []
    arrivals_picked_good = []
    amps_good = []
    pick_scores_good = []
    pick_offsets_good = []

    for i, (T, t_pred) in enumerate(zip(periods, arrivals_pred)):
        if not valid_freq[i]:
            continue

        local = best_local_pick(i, t_pred)
        if local is None:
            continue

        accepted = local["score"] >= float(match_threshold)

        candidate_periods.append(float(T))
        candidate_arrivals_model.append(float(t_pred))
        candidate_arrivals_picked_raw.append(local["time"])
        candidate_amps.append(np.sqrt(max(local["energy"], 0.0)))
        candidate_norm_energy.append(local["norm_energy"])
        candidate_proximity.append(local["proximity"])
        candidate_score.append(local["score"])
        candidate_time_offset_min.append(local["offset_min"])
        candidate_accepted.append(bool(accepted))
        candidate_reason.append(
            "accepted" if accepted else "below_energy_proximity_threshold"
        )

        if not accepted:
            continue

        periods_good.append(float(T))
        arrivals_model_good.append(float(t_pred))
        arrivals_picked_good.append(local["time"])
        amps_good.append(np.sqrt(max(local["energy"], 0.0)))
        pick_scores_good.append(local["score"])
        pick_offsets_good.append(local["offset_min"])

    if len(periods_good) < min_freqs:
        return None

    def sort_by_period(*arrays):
        arrays = [np.asarray(a) for a in arrays]
        order = np.argsort(arrays[0].astype(float))
        return [a[order] for a in arrays]

    (
        candidate_periods,
        candidate_arrivals_model,
        candidate_arrivals_picked_raw,
        candidate_amps,
        candidate_norm_energy,
        candidate_proximity,
        candidate_score,
        candidate_time_offset_min,
        candidate_accepted,
        candidate_reason,
    ) = sort_by_period(
        candidate_periods,
        candidate_arrivals_model,
        candidate_arrivals_picked_raw,
        candidate_amps,
        candidate_norm_energy,
        candidate_proximity,
        candidate_score,
        candidate_time_offset_min,
        candidate_accepted,
        candidate_reason,
    )

    (
        periods_good,
        arrivals_model_good,
        arrivals_picked_good,
        amps_good,
        pick_scores_good,
        pick_offsets_good,
    ) = sort_by_period(
        periods_good,
        arrivals_model_good,
        arrivals_picked_good,
        amps_good,
        pick_scores_good,
        pick_offsets_good,
    )

    periods_good = periods_good.astype(float)
    arrivals_model_good = arrivals_model_good.astype(float)
    arrivals_picked_good = arrivals_picked_good.astype(float)
    amps_good = amps_good.astype(float)
    pick_scores_good = pick_scores_good.astype(float)
    pick_offsets_good = pick_offsets_good.astype(float)

    raw_pick_rms_to_model_sec = float(
        np.sqrt(np.mean(((arrivals_picked_good - arrivals_model_good) * 60.0) ** 2))
    )
    raw_pick_median_abs_to_model_sec = float(
        np.median(np.abs((arrivals_picked_good - arrivals_model_good) * 60.0))
    )

    reference_period_sec = 100.0
    if periods_good.min() <= reference_period_sec <= periods_good.max():
        reference_arrival_model_min = float(
            np.interp(reference_period_sec, periods_good, arrivals_model_good)
        )
        reference_arrival_picked_min = float(
            np.interp(reference_period_sec, periods_good, arrivals_picked_good)
        )
    else:
        weights = np.maximum(amps_good.astype(float) ** 2, 1e-300)
        weights = weights / np.sum(weights)
        reference_arrival_model_min = float(np.sum(weights * arrivals_model_good))
        reference_arrival_picked_min = float(np.sum(weights * arrivals_picked_good))

    peak_idx = int(np.argmin(np.abs(time_min - reference_arrival_picked_min)))

    return {
        "distance": best_distance,
        "best_distance_deg": best_distance,
        "periods": periods_good,

        # Model branch.
        "arrivals": arrivals_model_good,

        # Actual local Stockwell maxima.
        "arrivals_picked_raw": arrivals_picked_good,

        "amps": amps_good,
        "model_score": best_score,
        "n_good_freqs": len(periods_good),
        "raw_pick_rms_to_model_sec": raw_pick_rms_to_model_sec,
        "raw_pick_median_abs_to_model_sec": raw_pick_median_abs_to_model_sec,
        "mean_pick_score": float(np.mean(pick_scores_good)),
        "median_abs_pick_offset_min": float(np.median(np.abs(pick_offsets_good))),
        "reference_period_sec": reference_period_sec,
        "reference_arrival_model_min": reference_arrival_model_min,
        "reference_arrival_picked_min": reference_arrival_picked_min,
        "peak_time": reference_arrival_picked_min,
        "peak_idx": peak_idx,
        "extraction_method": "independent_model_guided_local_max",

        # Grid-based posterior fields (new)
        "distance_grid_deg": distance_grid,
        "distance_log_likelihood": log_likelihood,
        "distance_posterior_pdf": posterior,
        "map_distance_deg": map_distance,
        "mean_distance_deg": mean_distance,
        "distance_std_deg": std_distance,

        "candidate_periods": candidate_periods.astype(float),
        "candidate_arrivals_model": candidate_arrivals_model.astype(float),
        "candidate_arrivals_picked_raw": candidate_arrivals_picked_raw.astype(float),
        "candidate_amps": candidate_amps.astype(float),
        "candidate_norm_energy": candidate_norm_energy.astype(float),
        "candidate_proximity": candidate_proximity.astype(float),
        "candidate_score": candidate_score.astype(float),
        "candidate_time_offset_min": candidate_time_offset_min.astype(float),
        "candidate_accepted": candidate_accepted.astype(bool),
        "candidate_reason": candidate_reason.astype(object),
        "candidate_match_threshold": float(match_threshold),
        "candidate_pick_tolerance_min": float(pick_half_width_min),
        "candidate_proximity_sigma_min": float(proximity_sigma_min),
    }

def extract_dispersion_curve_model_guided(
    S_energy_band,
    freqs_band,
    time_min,
    orbit,
    distance_range_deg,
    search_start_min,
    search_end_min,
    pick_tolerance_min=None,
    min_points=None,
    match_threshold=None,
    proximity_sigma_fraction=0.50,
):
    """
    Extract a Rayleigh dispersion curve using expected PREM orbit geometry.

    The picker searches along candidate PREM branches.  Within the local time
    tolerance around a branch it no longer takes the absolute strongest point;
    it maximizes ``normalized_energy * proximity_to_model``.  This prevents a
    strong neighboring burst from stealing the pick and makes the raw picked
    maxima a more honest test of whether energy actually lies on the model
    branch.
    """
    R_earth = 6371.0
    if pick_tolerance_min is None:
        # R2/R3 use wider gates because longer paths broaden and attenuate
        # the Rayleigh packet.
        pick_tolerance_min = {'R1': 2.0, 'R2': 4.0, 'R3': 5.5}.get(orbit, 3.0)
    if min_points is None:
        min_points = 5 if orbit == 'R1' else 4
    if match_threshold is None:
        # Threshold is applied to normalized_energy * proximity.  Therefore it
        # is stricter than a pure local-energy threshold.
        match_threshold = {'R1': 0.35, 'R2': 0.22, 'R3': 0.18}.get(orbit, 0.25)
    pick_tolerance_min = float(pick_tolerance_min)
    proximity_sigma_min = max(1e-6, pick_tolerance_min * float(proximity_sigma_fraction))

    time_mask = (time_min >= search_start_min) & (time_min <= search_end_min)
    if np.sum(time_mask) < 20:
        return None
    time_local = time_min[time_mask]
    S_local = S_energy_band[:, time_mask]
    periods = 1.0 / freqs_band
    freq_max = np.max(S_local, axis=1)
    valid_freq = freq_max > 0
    if np.sum(valid_freq) < min_points:
        return None
    if orbit in ['R2', 'R3']:
        period_weights = np.exp((periods - 50.0) / 50.0)
    else:
        period_weights = np.ones_like(periods)

    def path_km(distance_deg):
        if orbit == 'R1':
            path_deg = distance_deg
        elif orbit == 'R2':
            path_deg = 360.0 - distance_deg
        elif orbit == 'R3':
            path_deg = 360.0 + distance_deg
        else:
            return None
        return path_deg * R_earth * np.pi / 180.0

    def best_local_branch_score(i, t_pred):
        if t_pred < time_local[0] or t_pred > time_local[-1]:
            return None
        local_pick_mask = np.abs(time_local - t_pred) <= pick_tolerance_min
        if np.sum(local_pick_mask) == 0:
            return None
        idxs = np.where(local_pick_mask)[0]
        offsets = time_local[idxs] - t_pred
        norm_energy = S_local[i, idxs] / (freq_max[i] + 1e-300)
        proximity = np.exp(-0.5 * (offsets / proximity_sigma_min) ** 2)
        score = norm_energy * proximity
        j = int(idxs[np.nanargmax(score)])
        jj = int(np.where(idxs == j)[0][0])
        return {
            'j_local': j,
            'offset_min': float(offsets[jj]),
            'norm_energy': float(norm_energy[jj]),
            'proximity': float(proximity[jj]),
            'score': float(score[jj]),
            'energy': float(S_local[i, j]),
            'time': float(time_local[j]),
        }

    def score_distance(distance_deg):
        pk = path_km(distance_deg)
        if pk is None:
            return -np.inf
        U_pred = rayleigh_group_velocity_prem(periods)
        arrivals_pred = pk / U_pred / 60.0
        scores = []
        weights = []
        for i, t_pred in enumerate(arrivals_pred):
            if not valid_freq[i]:
                continue
            local = best_local_branch_score(i, t_pred)
            if local is None:
                continue
            scores.append(local['score'])
            weights.append(period_weights[i])
        if len(scores) < min_points:
            return -np.inf
        scores = np.asarray(scores)
        weights = np.asarray(weights)
        weights = weights / np.sum(weights)
        coverage = min(1.0, len(scores) / max(8.0, min_points))
        return float(np.sum(scores * weights) * (0.75 + 0.25 * coverage))

    d0, d1 = distance_range_deg
    d0 = max(1.0, float(d0))
    d1 = min(179.0, float(d1))
    if d1 <= d0:
        return None
    distances = np.linspace(d0, d1, 61)
    scores = np.array([score_distance(d) for d in distances])
    if not np.any(np.isfinite(scores)):
        return None
    best_distance = float(distances[np.nanargmax(scores)])
    best_score = float(np.nanmax(scores))
    if np.isfinite(best_score):
        refine_lo = max(d0, best_distance - 3.0)
        refine_hi = min(d1, best_distance + 3.0)
        opt = minimize_scalar(lambda d: -score_distance(d), bounds=(refine_lo, refine_hi), method='bounded')
        if opt.success and np.isfinite(opt.fun):
            best_distance = float(opt.x)
            best_score = float(-opt.fun)

    pk = path_km(best_distance)
    U_pred = rayleigh_group_velocity_prem(periods)
    arrivals_pred = pk / U_pred / 60.0

    candidate_periods = []
    candidate_arrivals_model = []
    candidate_arrivals_picked_raw = []
    candidate_amps = []
    candidate_norm_energy = []
    candidate_proximity = []
    candidate_score = []
    candidate_time_offset_min = []
    candidate_accepted = []
    candidate_reason = []

    periods_good = []
    arrivals_good = []
    amps_good = []
    pred_good = []
    arrivals_picked_raw = []
    pick_scores_good = []
    pick_offsets_good = []

    for i, (T, t_pred) in enumerate(zip(periods, arrivals_pred)):
        if not valid_freq[i]:
            continue
        local = best_local_branch_score(i, t_pred)
        if local is None:
            continue
        accepted = local['score'] >= float(match_threshold)

        candidate_periods.append(T)
        candidate_arrivals_model.append(t_pred)
        candidate_arrivals_picked_raw.append(local['time'])
        candidate_amps.append(np.sqrt(max(local['energy'], 0.0)))
        candidate_norm_energy.append(local['norm_energy'])
        candidate_proximity.append(local['proximity'])
        candidate_score.append(local['score'])
        candidate_time_offset_min.append(local['offset_min'])
        candidate_accepted.append(bool(accepted))
        candidate_reason.append('accepted' if accepted else 'below_energy_proximity_threshold')

        if not accepted:
            continue
        periods_good.append(T)
        arrivals_good.append(t_pred)
        arrivals_picked_raw.append(local['time'])
        amps_good.append(np.sqrt(max(local['energy'], 0.0)))
        pred_good.append(t_pred)
        pick_scores_good.append(local['score'])
        pick_offsets_good.append(local['offset_min'])

    if len(periods_good) < min_points:
        return None

    def as_sorted(*arrays):
        arrays = [np.asarray(a) for a in arrays]
        if arrays[0].size == 0:
            return arrays
        order = np.argsort(arrays[0].astype(float))
        return [a[order] for a in arrays]

    (candidate_periods, candidate_arrivals_model, candidate_arrivals_picked_raw,
     candidate_amps, candidate_norm_energy, candidate_proximity, candidate_score,
     candidate_time_offset_min, candidate_accepted, candidate_reason) = as_sorted(
        candidate_periods,
        candidate_arrivals_model,
        candidate_arrivals_picked_raw,
        candidate_amps,
        candidate_norm_energy,
        candidate_proximity,
        candidate_score,
        candidate_time_offset_min,
        candidate_accepted,
        candidate_reason,
    )

    (periods_good, arrivals_good, arrivals_picked_raw, amps_good,
     pred_good, pick_scores_good, pick_offsets_good) = as_sorted(
        periods_good,
        arrivals_good,
        arrivals_picked_raw,
        amps_good,
        pred_good,
        pick_scores_good,
        pick_offsets_good,
    )

    periods_good = periods_good.astype(float)
    arrivals_good = arrivals_good.astype(float)
    arrivals_picked_raw = arrivals_picked_raw.astype(float)
    amps_good = amps_good.astype(float)
    pred_good = pred_good.astype(float)
    pick_scores_good = pick_scores_good.astype(float)
    pick_offsets_good = pick_offsets_good.astype(float)

    rms_to_model_sec = float(np.sqrt(np.mean(((arrivals_good - pred_good) * 60.0) ** 2)))
    raw_pick_rms_to_model_sec = float(np.sqrt(np.mean(((arrivals_picked_raw - pred_good) * 60.0) ** 2)))
    raw_pick_median_abs_to_model_sec = float(np.median(np.abs((arrivals_picked_raw - pred_good) * 60.0)))

    # Representative arrival for display, SNR, and summaries. Prefer the
    # measured 100 s branch time when available.
    reference_period_sec = 100.0
    if periods_good.min() <= reference_period_sec <= periods_good.max():
        reference_arrival_model_min = float(np.interp(reference_period_sec, periods_good, arrivals_good))
        reference_arrival_picked_min = float(np.interp(reference_period_sec, periods_good, arrivals_picked_raw))
    else:
        weights = np.maximum(amps_good.astype(float) ** 2, 1e-300)
        weights = weights / np.sum(weights)
        reference_arrival_model_min = float(np.sum(weights * arrivals_good))
        reference_arrival_picked_min = float(np.sum(weights * arrivals_picked_raw))

    return {
        'periods': periods_good,
        'arrivals': arrivals_good,
        'arrivals_picked_raw': arrivals_picked_raw,
        'amps': amps_good,
        'best_distance_deg': best_distance,
        'model_score': best_score,
        'rms_to_model_sec': rms_to_model_sec,
        'raw_pick_rms_to_model_sec': raw_pick_rms_to_model_sec,
        'raw_pick_median_abs_to_model_sec': raw_pick_median_abs_to_model_sec,
        'mean_pick_score': float(np.mean(pick_scores_good)),
        'median_abs_pick_offset_min': float(np.median(np.abs(pick_offsets_good))),
        'reference_period_sec': reference_period_sec,
        'reference_arrival_model_min': reference_arrival_model_min,
        'reference_arrival_picked_min': reference_arrival_picked_min,
        'extraction_method': 'model_guided_energy_proximity',
        'candidate_periods': candidate_periods.astype(float),
        'candidate_arrivals_model': candidate_arrivals_model.astype(float),
        'candidate_arrivals_picked_raw': candidate_arrivals_picked_raw.astype(float),
        'candidate_amps': candidate_amps.astype(float),
        'candidate_norm_energy': candidate_norm_energy.astype(float),
        'candidate_proximity': candidate_proximity.astype(float),
        'candidate_score': candidate_score.astype(float),
        'candidate_time_offset_min': candidate_time_offset_min.astype(float),
        'candidate_accepted': candidate_accepted.astype(bool),
        'candidate_reason': candidate_reason.astype(object),
        'candidate_match_threshold': float(match_threshold),
        'candidate_pick_tolerance_min': float(pick_tolerance_min),
        'candidate_proximity_sigma_min': float(proximity_sigma_min),
    }


def check_spectral_coherence(periods_sorted, arrivals_sorted, max_erratic_fraction=0.3):
    """
    Check if dispersion curve has spectral coherence (smooth variation across periods).
    
    Rejects curves where adjacent periods show wild jumps in arrival time,
    indicating noise or incorrect energy maxima were tracked.
    
    Parameters
    ----------
    periods_sorted : np.ndarray
        Periods in ascending order (seconds)
    arrivals_sorted : np.ndarray
        Corresponding arrival times (minutes)
    max_erratic_fraction : float
        Maximum fraction of "steep" slopes allowed (default: 30%)
    
    Returns
    -------
    is_coherent : bool
        True if curve passes coherence check
    erratic_fraction : float
        Fraction of slopes that are erratic (for diagnostics)
    """
    if len(periods_sorted) < 3:
        return (True, 0.0)
    d_arrivals = np.diff(arrivals_sorted)
    d_periods = np.diff(periods_sorted)
    slopes = d_arrivals / (d_periods + 1e-06)
    erratic_mask = np.abs(slopes) > 0.5
    erratic_fraction = np.sum(erratic_mask) / len(slopes)
    is_coherent = erratic_fraction <= max_erratic_fraction
    return (is_coherent, erratic_fraction)


def check_prem_correlation(periods_sorted, arrivals_sorted, min_correlation=0.3, test_distances=None):
    """
    Check if dispersion curve correlates with PREM theory at ANY plausible distance.
    
    Rejects curves that don't match theoretical dispersion at any distance,
    indicating the extracted points are noise or artifacts.
    
    Parameters
    ----------
    periods_sorted : np.ndarray
        Periods (seconds)
    arrivals_sorted : np.ndarray
        Arrival times (minutes)
    min_correlation : float
        Minimum acceptable correlation (default: 0.3 = weak but present)
    test_distances : list or None
        Distances to test in degrees (default: [60, 80, 100, 120])
    
    Returns
    -------
    passes_check : bool
        True if curve correlates with PREM at some distance
    best_corr : float
        Best correlation found (for diagnostics)
    best_distance : float
        Distance that gave best correlation
    """
    if len(periods_sorted) < 3:
        return (True, 0.0, 0.0)
    if test_distances is None:
        test_distances = [60, 70, 80, 90, 100, 110, 120]
    best_corr = -1.0
    best_distance = 0.0
    for distance in test_distances:
        for orbit in ['R1', 'R2', 'R3']:
            corr, _ = calculate_dispersion_shape_correlation(periods_sorted, arrivals_sorted, distance, orbit)
            if corr > best_corr:
                best_corr = corr
                best_distance = distance
    passes_check = best_corr >= min_correlation
    return (passes_check, best_corr, best_distance)


def extract_dispersion_curve_robust(S_energy_band, freqs_band, time_min, peak_time, window_half_width=None, smooth_window=None, orbit_hint=None):
    """
    Extract dispersion curve - ORBIT-AWARE VERSION with RELAXED R2/R3 gates.
    
    Parameters
    ----------
    orbit_hint : str or None
        Orbit type hint: 'R1', 'R2', 'R3', or None
        Adjusts extraction parameters based on expected signal characteristics
    """
    if orbit_hint == 'R2':
        if window_half_width is None:
            window_half_width = 7.5
        energy_threshold = 0.03
        max_negative_slopes = 0.8
        outlier_tolerance = 7.0
    elif orbit_hint == 'R3':
        if window_half_width is None:
            window_half_width = 10.0
        energy_threshold = 0.04
        max_negative_slopes = 0.75
        outlier_tolerance = 8.0
    else:
        if window_half_width is None:
            window_half_width = 5.0
        energy_threshold = 0.05
        max_negative_slopes = 0.7
        outlier_tolerance = 5.0
    time_mask = (time_min >= peak_time - window_half_width) & (time_min <= peak_time + window_half_width)
    if np.sum(time_mask) < 10:
        return None
    time_local = time_min[time_mask]
    periods_local = []
    arrivals_local = []
    amps_local = []
    global_max_energy = np.max(S_energy_band)
    for i, f in enumerate(freqs_band):
        energy_trace = S_energy_band[i, time_mask]
        if np.max(energy_trace) == 0:
            continue
        idx_max = np.argmax(energy_trace)
        max_energy = energy_trace[idx_max]
        if max_energy <= energy_threshold * np.max(S_energy_band[i, :]):
            continue
        if orbit_hint in ['R2', 'R3']:
            amp_threshold = 0.002
        else:
            amp_threshold = 0.005
        if max_energy < amp_threshold * global_max_energy:
            continue
        arrivals_local.append(time_local[idx_max])
        periods_local.append(1.0 / f)
        amps_local.append(np.sqrt(max_energy))
    if len(periods_local) < 3:
        return None
    periods_arr = np.array(periods_local)
    arrivals_arr = np.array(arrivals_local)
    amps_arr = np.array(amps_local)
    sort_idx = np.argsort(periods_arr)
    periods_sorted = periods_arr[sort_idx]
    arrivals_sorted = arrivals_arr[sort_idx]
    amps_sorted = amps_arr[sort_idx]
    d_arrivals = np.diff(arrivals_sorted)
    d_periods = np.diff(periods_sorted)
    slopes = d_arrivals / (d_periods + 1e-06)
    if len(slopes) >= 4:
        slope_sign = np.sign(slopes)
        nonzero = slope_sign != 0
        if np.sum(nonzero) >= 4:
            slope_sign = slope_sign[nonzero]
            sign_changes = np.sum(slope_sign[1:] != slope_sign[:-1])
            if sign_changes > 0.75 * max(1, len(slope_sign) - 1):
                return None
    valid_mask = np.ones(len(periods_sorted), dtype=bool)
    for i in range(1, len(arrivals_sorted)):
        if abs(arrivals_sorted[i] - arrivals_sorted[i - 1]) > outlier_tolerance:
            valid_mask[i] = False
    periods_clean = periods_sorted[valid_mask]
    arrivals_clean = arrivals_sorted[valid_mask]
    amps_clean = amps_sorted[valid_mask]
    if len(periods_clean) < 3:
        return None
    if orbit_hint == 'R2':
        max_erratic = 0.4
    elif orbit_hint == 'R3':
        max_erratic = 0.45
    else:
        max_erratic = 0.3
    is_coherent, erratic_frac = check_spectral_coherence(periods_clean, arrivals_clean, max_erratic_fraction=max_erratic)
    if not is_coherent:
        return None
    min_corr_threshold = 0.2 if orbit_hint in ['R2', 'R3'] else 0.25
    passes_prem, best_corr, best_dist = check_prem_correlation(periods_clean, arrivals_clean, min_correlation=min_corr_threshold)
    if not passes_prem:
        return None
    min_points = 4 if orbit_hint in ['R2', 'R3'] else 3
    if len(periods_clean) < min_points:
        return None
    return {'periods': periods_clean, 'arrivals': arrivals_clean, 'amps': amps_clean}


def distance_guided_extraction(
    S_energy_band,
    freqs_band,
    time_min,
    energy_time_integrated,
    expected_distance_deg,
    velocity_uncertainty=0.15,
    distance_uncertainty_deg=10.0,
    orbits_to_search=['R1', 'R2', 'R3'],
    S_R_raw=None,
    S_Z_raw=None,
    masks=None,
    freq_mask=None,
    verbose=True,
):
    """Distance-guided orbit extraction with orbit-specific polarization masks."""

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    vprint(f'\nDISTANCE-GUIDED EXTRACTION')
    vprint(f'   Expected distance: {expected_distance_deg:.1f} deg (+/-{distance_uncertainty_deg:.1f} deg)')
    vprint(f'   Velocity uncertainty: +/-{velocity_uncertainty * 100:.0f}%')
    R_earth = 6371
    reference_velocity = 3.77
    path_lengths_min = {}
    path_lengths_max = {}
    expected_arrivals_min = {}
    expected_arrivals_max = {}
    for orbit in orbits_to_search:
        dist_min = max(1.0, expected_distance_deg - distance_uncertainty_deg)
        dist_max = min(179.0, expected_distance_deg + distance_uncertainty_deg)
        if orbit == 'R1':
            path_lengths_min[orbit] = dist_min * R_earth * np.pi / 180
            path_lengths_max[orbit] = dist_max * R_earth * np.pi / 180
        elif orbit == 'R2':
            path_lengths_min[orbit] = (360 - dist_max) * R_earth * np.pi / 180
            path_lengths_max[orbit] = (360 - dist_min) * R_earth * np.pi / 180
        elif orbit == 'R3':
            path_lengths_min[orbit] = (360 + dist_min) * R_earth * np.pi / 180
            path_lengths_max[orbit] = (360 + dist_max) * R_earth * np.pi / 180
        v_min = reference_velocity * (1 - velocity_uncertainty)
        v_max = reference_velocity * (1 + velocity_uncertainty)
        expected_arrivals_min[orbit] = path_lengths_min[orbit] / v_max / 60
        expected_arrivals_max[orbit] = path_lengths_max[orbit] / v_min / 60

    vprint(f'\n   Expected arrival ranges (@ 100s period, with uncertainties):')
    for orbit in orbits_to_search:
        window_width = expected_arrivals_max[orbit] - expected_arrivals_min[orbit]
        center = (expected_arrivals_min[orbit] + expected_arrivals_max[orbit]) / 2
        vprint(f'   {orbit}: {expected_arrivals_min[orbit]:.1f} - {expected_arrivals_max[orbit]:.1f} min (center: {center:.1f}, window: {window_width:.1f} min)')

    guided_dispersions = []
    for orbit in orbits_to_search:
        expected_time_min = expected_arrivals_min[orbit]
        expected_time_max = expected_arrivals_max[orbit]
        if expected_time_max < time_min[0] or expected_time_min > time_min[-1]:
            vprint(f'\n   Warning: {orbit} window [{expected_time_min:.1f}-{expected_time_max:.1f} min] outside data range [{time_min[0]:.1f}-{time_min[-1]:.1f} min]')
            continue
        vprint(f'\n   Searching for {orbit}...')
        vprint(f'   Search window: {expected_time_min:.1f} - {expected_time_max:.1f} min (width: {expected_time_max - expected_time_min:.1f} min)')

        if masks and S_R_raw is not None and (S_Z_raw is not None) and (freq_mask is not None):
            # Do not enforce retro/pro by orbit.  Use any R-Z elliptical
            # polarization that passed the upstream filter.
            if 'elliptic' in masks:
                orbit_mask = masks['elliptic']
            else:
                retro_mask = masks.get('retro', np.ones_like(S_R_raw))
                pro_mask = masks.get('pro', np.ones_like(S_R_raw))
                orbit_mask = np.maximum(retro_mask, pro_mask)

            vprint(f'   {orbit}: using ELLIPTIC mask')
            S_energy_orbit = (
                np.abs(orbit_mask * S_R_raw) ** 2
                + np.abs(orbit_mask * S_Z_raw) ** 2
            )
            S_energy_band_orbit = S_energy_orbit[freq_mask, :]
        else:
            S_energy_band_orbit = S_energy_band

        search_start = max(time_min[0], expected_time_min)
        search_end = min(time_min[-1], expected_time_max)
        time_mask = (time_min >= search_start) & (time_min <= search_end)
        energy_orbit_time_integrated = np.sum(S_energy_band_orbit, axis=0)
        energy_local = energy_orbit_time_integrated[time_mask]
        time_local = time_min[time_mask]
        if len(energy_local) < 50:
            vprint(f'   Warning: Search window too small: {len(energy_local)} points')
            continue

        window_peak_idx_local = int(np.argmax(energy_local))
        window_peak_time = float(time_local[window_peak_idx_local])
        window_peak_energy = float(energy_local[window_peak_idx_local])
        expected_center = (expected_time_min + expected_time_max) / 2
        vprint(f'   Window energy maximum: {window_peak_time:.1f} min (offset from center: {window_peak_time - expected_center:+.1f} min)')

        dist_range_for_orbit = (expected_distance_deg - distance_uncertainty_deg, expected_distance_deg + distance_uncertainty_deg)
        disp_result = extract_dispersion_curve_model_guided(
            S_energy_band_orbit,
            freqs_band,
            time_min,
            orbit=orbit,
            distance_range_deg=dist_range_for_orbit,
            search_start_min=search_start,
            search_end_min=search_end,
        )
        if disp_result is None:
            full_half_width = max(5.0, (search_end - search_start) / 2.0)
            window_center = (search_start + search_end) / 2.0
            disp_result = extract_dispersion_curve_robust(S_energy_band_orbit, freqs_band, time_min, window_center, orbit_hint=orbit, window_half_width=full_half_width)
        if disp_result is None:
            vprint(f'   Failed to extract dispersion curve')
            continue

        # Use a branch-based representative arrival, not the unrelated window max.
        peak_time = float(disp_result.get('reference_arrival_picked_min', np.nan))
        if not np.isfinite(peak_time):
            peak_time = float(np.median(disp_result.get('arrivals_picked_raw', disp_result['arrivals'])))
        peak_idx_global = int(np.argmin(np.abs(time_min - peak_time)))
        peak_energy = float(energy_orbit_time_integrated[peak_idx_global])
        time_offset = peak_time - expected_center

        if 'best_distance_deg' in disp_result:
            vprint(f"   Model-guided distance: {disp_result['best_distance_deg']:.1f} deg | score={disp_result.get('model_score', 0):.3f} | raw-pick RMS={disp_result.get('raw_pick_rms_to_model_sec', np.nan):.1f}s")
        vprint(f"   Representative {disp_result.get('reference_period_sec', 100):.0f}s picked arrival: {peak_time:.1f} min (offset from center: {time_offset:+.1f} min)")
        n_points = len(disp_result['periods'])
        vprint(f'   OK: Extracted {n_points} frequency points')
        corr, rms = calculate_dispersion_shape_correlation(disp_result['periods'], disp_result['arrivals'], expected_distance_deg, orbit)
        vprint(f'   Model-branch shape correlation: {corr:.3f}, RMS to expected distance: {rms:.1f}s')

        guided_item = {
            'orbit': orbit,
            'peak_time': peak_time,
            'peak_energy': peak_energy,
            'peak_idx': peak_idx_global,
            'window_peak_time': window_peak_time,
            'window_peak_energy': window_peak_energy,
            'periods': disp_result['periods'],
            'arrivals': disp_result['arrivals'],
            'amps': disp_result['amps'],
            'expected_time_min': expected_time_min,
            'expected_time_max': expected_time_max,
            'expected_time_center': expected_center,
            'time_offset': time_offset,
            'is_distance_guided': True,
            'extraction_method': disp_result.get('extraction_method', 'robust_local'),
            'best_distance_deg': disp_result.get('best_distance_deg', expected_distance_deg),
            'model_score': disp_result.get('model_score', None),
            'rms_to_model_sec': disp_result.get('rms_to_model_sec', None),
            'raw_pick_rms_to_model_sec': disp_result.get('raw_pick_rms_to_model_sec', None),
        }
        for key in (
            'arrivals_picked_raw',
            'raw_pick_median_abs_to_model_sec',
            'mean_pick_score',
            'median_abs_pick_offset_min',
            'reference_period_sec',
            'reference_arrival_model_min',
            'reference_arrival_picked_min',
            'candidate_periods',
            'candidate_arrivals_model',
            'candidate_arrivals_picked_raw',
            'candidate_amps',
            'candidate_norm_energy',
            'candidate_proximity',
            'candidate_score',
            'candidate_time_offset_min',
            'candidate_accepted',
            'candidate_reason',
            'candidate_match_threshold',
            'candidate_pick_tolerance_min',
            'candidate_proximity_sigma_min',
        ):
            if key in disp_result:
                guided_item[key] = disp_result[key]
        guided_dispersions.append(guided_item)
        vprint(f'   Accepted: {orbit} candidate added to guided extractions')
    vprint(f'\n   Total guided extractions: {len(guided_dispersions)}')
    return guided_dispersions


def build_particleman_orbit_energies(result_dict, detected_orbits, time_min, fmin_hz, fmax_hz):
    """
    Build orbit energy traces from the elliptic Particleman-style NIP energy.

    Orbit labels are geometric/PREM labels and are not tied to a fixed
    polarization sense.
    """
    freqs = result_dict['freqs']
    freq_mask = (freqs >= fmin_hz) & (freqs <= fmax_hz)
    out = {}

    if 'S_energy_elliptic_nip' in result_dict:
        energy_key = 'S_energy_elliptic_nip'
        sense = 'elliptic'
    elif 'S_energy_filtered' in result_dict:
        energy_key = 'S_energy_filtered'
        sense = 'filtered'
    else:
        energy_key = 'S_energy_raw'
        sense = 'raw'

    for orbit in ('R1', 'R2', 'R3'):
        if orbit not in detected_orbits:
            continue
        if energy_key not in result_dict:
            continue

        energy_tf = result_dict[energy_key]
        energy_time = np.sum(energy_tf[freq_mask, :], axis=0)
        peak_time = detected_orbits[orbit]['dispersion']['peak_time']

        out[orbit] = {
            'sense': sense,
            'peak_time': peak_time,
            'energy': energy_time,
            'total_energy': float(np.sum(energy_time)),
            'max_energy': float(np.max(energy_time)),
            'source': energy_key,
            'time': time_min,
        }

    return out


def apply_model_guided_confidence_penalties(
    confidence,
    components,
    disp_data,
    shared_distance,
    orbit,
    distance_uncertainty_deg=10.0,
):
    """Penalize model-guided detections when raw picks do not track the branch."""
    base_confidence = float(confidence)
    components = dict(components)
    components['overall_confidence_before_guided_penalties'] = base_confidence

    raw_rms = disp_data.get('raw_pick_rms_to_model_sec')
    if raw_rms is not None and np.isfinite(float(raw_rms)):
        raw_rms = float(raw_rms)
        tol = {'R1': 150.0, 'R2': 240.0, 'R3': 300.0}.get(orbit, 220.0)
        raw_quality = float(np.exp(-raw_rms / tol))
        components['raw_pick_rms_to_model_sec'] = raw_rms
        components['raw_pick_rms_tolerance_sec'] = tol
        components['raw_pick_quality'] = raw_quality
        confidence *= 0.75 + 0.25 * raw_quality

    best_distance = disp_data.get('best_distance_deg')
    if best_distance is not None and np.isfinite(float(best_distance)):
        best_distance = float(best_distance)
        mismatch = abs(best_distance - float(shared_distance))
        sigma = max(float(distance_uncertainty_deg), 1e-6)
        distance_quality = float(np.exp(-0.5 * (mismatch / sigma) ** 2))
        components['model_guided_distance_deg'] = best_distance
        components['model_guided_distance_mismatch_deg'] = mismatch
        components['model_guided_distance_quality'] = distance_quality
        confidence *= 0.85 + 0.15 * distance_quality

    if disp_data.get('model_score') is not None:
        components['model_score'] = float(disp_data['model_score'])
    if disp_data.get('mean_pick_score') is not None:
        components['mean_pick_score'] = float(disp_data['mean_pick_score'])
    if disp_data.get('median_abs_pick_offset_min') is not None:
        components['median_abs_pick_offset_min'] = float(disp_data['median_abs_pick_offset_min'])
    confidence = float(confidence)
    components['overall_confidence'] = confidence
    return confidence, components

def detect_rayleigh_orbits_from_stockwell(result_dict, time_array, fmin_hz, fmax_hz, distance_range_deg=(50, 110), event_offset_sec=0, use_joint_fitting=True, min_confidence=0.4, expected_distance_deg=None, velocity_uncertainty=0.15, distance_uncertainty_deg=10.0, distance_prior_sigma_deg=None, distance_grid_step_deg=0.1, output_dir=None, verbose=True):
    """
    Automatically detect R1, R2, and R3 with confidence scoring and joint fitting.
    
    1. Confidence scoring based on:
       - Dispersion curve shape correlation (not just RMS)
       - SNR in detection window
       - Consistency of arrival times across periods
    2. Joint fitting option: fit R1+R2+R3 with shared distance constraint
    3. Diagnostic plots showing WHY each orbit was chosen
    4. Robust handling of cases with only 1-2 visible orbits
    
    Parameters
    ----------
    result_dict : dict
        Output from Stockwell filter (must contain 'S_R', 'S_Z', 'freqs')
    time_array : np.ndarray
        Time array in seconds from trace start
    fmin_hz, fmax_hz : float
        Frequency band limits
    distance_range_deg : tuple
        (min, max) distance range to search (degrees)
    event_offset_sec : float
        Offset to add to time array to make it relative to event
    use_joint_fitting : bool
        If True, use joint fitting with shared distance constraint
    min_confidence : float
        Minimum confidence score to accept an orbit detection (0-1)
    expected_distance_deg : float or None
        If provided, uses distance-guided extraction instead of peak detection.
        The value defines broad R1/R2/R3 search windows and, by default, a
        broad Gaussian prior for the shared-distance posterior. It is not
        copied into the final distance estimate.
    distance_prior_sigma_deg : float or None
        Standard deviation of the Gaussian distance prior centered on
        expected_distance_deg. If None, distance_uncertainty_deg is used. Set
        to 0 or a negative value for a flat prior during validation.
    distance_grid_step_deg : float
        Distance grid spacing, in degrees, used for guided joint fitting.
    output_dir : str or None
        Directory to save diagnostic plots (None = no plots)
    verbose : bool
        Print detailed progress information
    
    Returns
    -------
    detected_orbits : dict
        {
            'R1': {
                'distance_deg': float,
                'dispersion': {'periods': array, 'arrivals': array, 'amps': array},
                'confidence': float,
                'confidence_components': dict,
                'rms_residual_sec': float
            },
            'R2': {...},
            'R3': {...},
            'metadata': {
                'method': 'joint' or 'independent',
                'shared_distance_deg': float (if joint),
                'n_orbits_detected': int
            }
        }
    """

    def vprint(*args, **kwargs):
        """Verbose print"""
        if verbose:
            print(*args, **kwargs)
    S_R_raw = result_dict.get('S_R_raw', result_dict.get('S_R'))
    S_Z_raw = result_dict.get('S_Z_raw', result_dict.get('S_Z'))
    freqs = result_dict.get('freqs')
    if S_R_raw is None or freqs is None:
        vprint('Warning: Error: Missing Stockwell transform data')
        return None
    masks = result_dict.get('masks', {})
    if masks:
        mask_retro = masks.get('retro', np.ones_like(S_R_raw))
        mask_pro = masks.get('pro', np.ones_like(S_R_raw))
        combined_mask = np.maximum(mask_retro, mask_pro)
        S_R = combined_mask * S_R_raw
        S_Z = combined_mask * S_Z_raw
        vprint('\nOK: Using polarization-masked Stockwell transforms for detection')
        vprint(f'   Retro mask strength: {np.mean(mask_retro):.2%}')
        vprint(f'   Pro mask strength: {np.mean(mask_pro):.2%}')
    else:
        vprint('\nWarning: No polarization masks found, using raw transforms')
        S_R = S_R_raw
        S_Z = S_Z_raw
        mask_retro = np.ones_like(S_R_raw)
        mask_pro = np.ones_like(S_R_raw)
        combined_mask = np.ones_like(S_R_raw)
    result_dict['mask_retro'] = mask_retro
    result_dict['mask_pro'] = mask_pro
    result_dict['mask_combined'] = combined_mask
    S_energy = np.abs(S_R) ** 2 + np.abs(S_Z) ** 2
    freq_mask = (freqs >= fmin_hz) & (freqs <= fmax_hz)
    freqs_band = freqs[freq_mask]
    S_energy_band = S_energy[freq_mask, :]
    time_min = (time_array + event_offset_sec) / 60
    energy_time_integrated = np.sum(S_energy_band, axis=0)
    if expected_distance_deg is not None:
        vprint(f'\nUsing distance-guided mode (expected: {expected_distance_deg:.1f} deg)')
        guided_extractions = distance_guided_extraction(S_energy_band, freqs_band, time_min, energy_time_integrated, expected_distance_deg, velocity_uncertainty=velocity_uncertainty, distance_uncertainty_deg=distance_uncertainty_deg, orbits_to_search=['R1', 'R2', 'R3'], S_R_raw=S_R_raw, S_Z_raw=S_Z_raw, masks=masks, freq_mask=freq_mask, verbose=verbose)
        peak_dispersions = []
        for extraction in guided_extractions:
            peak_item = {'peak_time': extraction['peak_time'], 'peak_energy': extraction['peak_energy'], 'peak_idx': extraction['peak_idx'], 'periods': extraction['periods'], 'arrivals': extraction['arrivals'], 'amps': extraction['amps'], 'orbit_hint': extraction['orbit'], 'is_distance_guided': True, 'extraction_method': extraction.get('extraction_method'), 'best_distance_deg': extraction.get('best_distance_deg'), 'model_score': extraction.get('model_score'), 'rms_to_model_sec': extraction.get('rms_to_model_sec'), 'raw_pick_rms_to_model_sec': extraction.get('raw_pick_rms_to_model_sec'), 'window_peak_time': extraction.get('window_peak_time'), 'window_peak_energy': extraction.get('window_peak_energy'), 'reference_period_sec': extraction.get('reference_period_sec'), 'reference_arrival_model_min': extraction.get('reference_arrival_model_min'), 'reference_arrival_picked_min': extraction.get('reference_arrival_picked_min'), 'mean_pick_score': extraction.get('mean_pick_score'), 'median_abs_pick_offset_min': extraction.get('median_abs_pick_offset_min')}
            for key in (
                'arrivals_picked_raw',
                'candidate_periods',
                'candidate_arrivals_model',
                'candidate_arrivals_picked_raw',
                'candidate_amps',
                'candidate_norm_energy',
                'candidate_proximity',
                'candidate_score',
                'candidate_time_offset_min',
                'candidate_accepted',
                'candidate_reason',
                'candidate_match_threshold',
                'candidate_pick_tolerance_min',
            ):
                if key in extraction:
                    peak_item[key] = extraction[key]
            peak_dispersions.append(peak_item)
        vprint(f'\nDistance-guided mode: {len(peak_dispersions)} orbit candidates extracted')
    else:
        vprint('\nUsing automatic peak detection mode')
        prominence_threshold = 0.05 * np.max(energy_time_integrated)
        min_distance_samples = int(len(time_min) * 0.03)
        peaks_idx, peak_props = find_peaks(energy_time_integrated, prominence=prominence_threshold, distance=min_distance_samples, height=0.03 * np.max(energy_time_integrated))
        peak_times = time_min[peaks_idx]
        peak_energies = energy_time_integrated[peaks_idx]
        vprint(f'\nDetected {len(peak_times)} energy peaks in Stockwell transform:')
        for i, (t, e) in enumerate(zip(peak_times, peak_energies)):
            vprint(f'   Peak {i + 1}: {t:.1f} min (energy: {e:.2e})')
        peak_dispersions = []
        for idx, (peak_time, peak_energy, peak_idx_orig) in enumerate(zip(peak_times, peak_energies, peaks_idx)):
            disp_result = extract_dispersion_curve_robust(S_energy_band, freqs_band, time_min, peak_time, orbit_hint=None)
            if disp_result is not None:
                peak_dispersions.append({'peak_time': peak_time, 'peak_energy': peak_energy, 'peak_idx': peak_idx_orig, 'periods': disp_result['periods'], 'arrivals': disp_result['arrivals'], 'amps': disp_result['amps'], 'is_distance_guided': False})
                if verbose:
                    corr_r1, _ = calculate_dispersion_shape_correlation(disp_result['periods'], disp_result['arrivals'], 95.0, 'R1')
                    vprint(f"   Peak {idx + 1} @ {peak_time:.1f}min: {len(disp_result['periods'])} points, shape_corr(R1)={corr_r1:.3f}")
        vprint(f'\nExtracted {len(peak_dispersions)} dispersion curves')
    detected_orbits = {}
    metadata = {}
    if use_joint_fitting and len(peak_dispersions) >= 2:
        vprint('\nUsing joint fitting with shared distance constraint.')
        if expected_distance_deg is not None and all(('orbit_hint' in p for p in peak_dispersions)):
            vprint('   (Distance-guided mode: estimating shared distance on a grid)')
            guided_distance_range = (
                max(1.0, float(expected_distance_deg) - float(distance_uncertainty_deg)),
                min(179.0, float(expected_distance_deg) + float(distance_uncertainty_deg)),
            )
            prior_sigma = distance_prior_sigma_deg
            if prior_sigma is None:
                prior_sigma = float(distance_uncertainty_deg)
            joint_solution = joint_fit_guided_orbits(
                peak_dispersions,
                guided_distance_range,
                guide_distance_deg=expected_distance_deg,
                prior_sigma_deg=prior_sigma,
                grid_step_deg=distance_grid_step_deg,
            )
        else:
            joint_solution = joint_fit_orbits(peak_dispersions, distance_range_deg)
        if joint_solution is not None:
            shared_distance = joint_solution['distance_deg']
            assignments = joint_solution['assignments']
            vprint(f'\nOK: Joint solution found:')
            vprint(f'   Shared distance: {shared_distance:.1f} deg')
            vprint(f"   Orbits detected: {joint_solution['n_orbits']}")
            for orbit, assignment_info in assignments.items():
                peak_idx = assignment_info['peak_idx']
                disp_data = _dispersion_with_distance_branch(
                    peak_dispersions[peak_idx], orbit, shared_distance
                )
                arrivals_for_score = disp_data.get('arrivals_picked_raw', disp_data['arrivals'])
                confidence, components = calculate_confidence_score(disp_data['periods'], arrivals_for_score, disp_data['amps'], shared_distance, orbit, energy_time_integrated, disp_data['peak_idx'])
                confidence, components = apply_model_guided_confidence_penalties(
                    confidence, components, disp_data, shared_distance, orbit, distance_uncertainty_deg
                )
                adaptive_threshold = get_adaptive_confidence_threshold(orbit, components['snr'], min_confidence)
                if confidence >= adaptive_threshold:
                    best_match = {'distance_deg': shared_distance, 'dispersion': disp_data, 'confidence': confidence, 'confidence_components': components, 'rms_residual_sec': components['rms_residual_sec']}
                    if best_match['rms_residual_sec'] > 600:
                        vprint(f"   REJECTED {orbit} REJECTED: RMS too large ({best_match['rms_residual_sec']:.1f}s)")
                        continue
                    detected_orbits[orbit] = best_match
                    extra = ''
                    if 'raw_pick_rms_to_model_sec' in components:
                        extra += f" | rawRMS={components['raw_pick_rms_to_model_sec']:.1f}s"
                    if 'model_guided_distance_mismatch_deg' in components:
                        extra += f" | dmis={components['model_guided_distance_mismatch_deg']:.1f} deg"
                    vprint(f'   OK: {orbit} ACCEPTED: confidence={confidence:.3f} (threshold={adaptive_threshold:.3f})' + extra)
                else:
                    vprint(f'   {orbit}: confidence={confidence:.3f} < {adaptive_threshold} (rejected)')
            metadata = {
                'method': 'joint',
                'shared_distance_deg': shared_distance,
                'map_distance_deg': float(joint_solution.get('map_distance_deg', shared_distance)),
                'mean_distance_deg': joint_solution.get('mean_distance_deg'),
                'distance_std_deg': joint_solution.get('distance_std_deg'),
                'guide_distance_deg': expected_distance_deg,
                'distance_prior_center_deg': joint_solution.get('distance_prior_center_deg'),
                'distance_prior_sigma_deg': joint_solution.get('distance_prior_sigma_deg'),
                'distance_grid_step_deg': joint_solution.get('distance_grid_step_deg'),
                'n_orbits_detected': len(detected_orbits),
                'joint_score': joint_solution['total_score'],
            }
            for key in (
                'distance_grid_deg',
                'distance_log_likelihood',
                'distance_log_prior',
                'distance_log_posterior',
                'distance_posterior_pdf',
            ):
                if key in joint_solution:
                    metadata[key] = joint_solution[key]
            if 'R2' not in detected_orbits and 'R1' in detected_orbits:
                vprint('\nR2 missing from joint fit; starting targeted search.')
                R1_data = detected_orbits['R1']
                R1_arrival = R1_data['dispersion']['peak_time']
                R1_distance = R1_data['distance_deg']
                R2_candidate = targeted_R2_search(peak_dispersions=peak_dispersions, S_energy_band=S_energy_band, freqs_band=freqs_band, time_min=time_min, energy_time_integrated=energy_time_integrated, R1_arrival_time=R1_arrival, distance_est=R1_distance, verbose=verbose)
                if R2_candidate is not None:
                    vprint(f"   Found R2 candidate at {R2_candidate['peak_time']:.1f} min")
                    confidence, components = calculate_confidence_score(R2_candidate['periods'], R2_candidate['arrivals'], R2_candidate['amps'], shared_distance, 'R2', energy_time_integrated, R2_candidate['peak_idx'])
                    adaptive_threshold = min_confidence * 0.7
                    if components['snr'] > 2.0:
                        adaptive_threshold *= 0.85
                    vprint(f'   Confidence: {confidence:.3f}, Threshold: {adaptive_threshold:.3f}')
                    vprint(f"   Shape corr: {components['shape_correlation']:.3f}, RMS: {components['rms_residual_sec']:.1f}s, SNR: {components['snr']:.1f}")
                    if confidence >= adaptive_threshold:
                        detected_orbits['R2'] = {'distance_deg': shared_distance, 'dispersion': R2_candidate, 'confidence': confidence, 'confidence_components': components, 'rms_residual_sec': components['rms_residual_sec'], 'detection_method': 'targeted_search'}
                        metadata['n_orbits_detected'] = len(detected_orbits)
                        metadata['R2_via_targeted_search'] = True
                        vprint(f'   R2 accepted via targeted search.')
                    else:
                        vprint(f'   Failed: R2 candidate rejected: confidence {confidence:.3f} < {adaptive_threshold:.3f}')
                else:
                    vprint('   No viable R2 candidate found in targeted search')
        else:
            vprint('Warning: Joint fitting failed, falling back to independent fitting')
            use_joint_fitting = False
    if not use_joint_fitting or len(peak_dispersions) < 2:
        vprint('\nUsing independent fitting for each orbit.')
        for orbit in ['R1', 'R2', 'R3']:
            vprint(f'\n   Testing {orbit} orbit:')
            best_confidence = 0
            best_match = None
            for peak_idx, disp_data in enumerate(peak_dispersions):
                peak_time = disp_data["peak_time"]

                disp_reextracted = fit_dispersion_curve_model(
                    S_energy_band,
                    freqs_band,
                    time_min,
                    peak_time,
                    orbit=orbit,
                    distance_range=distance_range_deg,
                )

                if disp_reextracted is not None:
                    # Store and plot the same curve we are about to score.
                    disp_for_match = dict(disp_data)
                    disp_for_match.update(disp_reextracted)
                    disp_for_match["is_model_reextracted"] = True

                    periods_obs = np.asarray(disp_for_match["periods"], dtype=float)

                    # Score the measured local maxima, not the model branch.
                    arrivals_obs = np.asarray(
                        disp_for_match.get(
                            "arrivals_picked_raw",
                            disp_for_match["arrivals"],
                        ),
                        dtype=float,
                    )
                    amps_obs = np.asarray(disp_for_match["amps"], dtype=float)

                    peak_idx_for_snr = int(
                        disp_for_match.get(
                            "peak_idx",
                            np.argmin(
                                np.abs(
                                    time_min
                                    - float(
                                        np.median(
                                            disp_for_match.get(
                                                "arrivals_picked_raw",
                                                disp_for_match["arrivals"],
                                            )
                                        )
                                    )
                                )
                            ),
                        )
                    )
                else:
                    # Fall back to the original robust extraction.
                    disp_for_match = dict(disp_data)
                    disp_for_match["is_model_reextracted"] = False

                    periods_obs = np.asarray(disp_data["periods"], dtype=float)
                    arrivals_obs = np.asarray(disp_data["arrivals"], dtype=float)
                    amps_obs = np.asarray(disp_data["amps"], dtype=float)
                    peak_idx_for_snr = int(disp_data["peak_idx"])

                def objective(d):
                    _, rms = calculate_dispersion_shape_correlation(
                        periods_obs,
                        arrivals_obs,
                        d,
                        orbit,
                    )
                    return rms

                result = minimize_scalar(
                    objective,
                    bounds=(distance_range_deg[0], distance_range_deg[1]),
                    method="bounded",
                )
                optimal_distance = float(result.x)

                # If this is a model-reextracted curve, update the model branch
                # to the final RMS-optimal distance so the plotted dotted branch
                # matches the distance used for confidence.
                if disp_for_match.get("is_model_reextracted", False):
                    R_earth = 6371.0
                    if orbit == "R1":
                        path_km = optimal_distance * R_earth * np.pi / 180.0
                    elif orbit == "R2":
                        path_km = (360.0 - optimal_distance) * R_earth * np.pi / 180.0
                    elif orbit == "R3":
                        path_km = (360.0 + optimal_distance) * R_earth * np.pi / 180.0
                    else:
                        path_km = optimal_distance * R_earth * np.pi / 180.0

                    U_pred = rayleigh_group_velocity_prem(periods_obs)
                    arrivals_model_final = path_km / U_pred / 60.0
                    disp_for_match["arrivals"] = arrivals_model_final
                    disp_for_match["distance"] = optimal_distance
                    disp_for_match["best_distance_deg"] = optimal_distance

                    if "arrivals_picked_raw" in disp_for_match:
                        raw_pick_rms_to_model_sec = float(
                            np.sqrt(
                                np.mean(
                                    (
                                        (
                                            np.asarray(
                                                disp_for_match["arrivals_picked_raw"],
                                                dtype=float,
                                            )
                                            - arrivals_model_final
                                        )
                                        * 60.0
                                    )
                                    ** 2
                                )
                            )
                        )
                        disp_for_match[
                            "raw_pick_rms_to_model_sec"
                        ] = raw_pick_rms_to_model_sec
                        disp_for_match[
                            "raw_pick_median_abs_to_model_sec"
                        ] = float(
                            np.median(
                                np.abs(
                                    (
                                        np.asarray(
                                            disp_for_match["arrivals_picked_raw"],
                                            dtype=float,
                                        )
                                        - arrivals_model_final
                                    )
                                    * 60.0
                                )
                            )
                        )

                confidence, components = calculate_confidence_score(
                    periods_obs,
                    arrivals_obs,
                    amps_obs,
                    optimal_distance,
                    orbit,
                    energy_time_integrated,
                    peak_idx_for_snr,
                )

                # Penalize disagreement between the measured local maxima and
                # the model branch when that diagnostic exists.
                if disp_for_match.get("raw_pick_rms_to_model_sec") is not None:
                    raw_rms = float(disp_for_match["raw_pick_rms_to_model_sec"])
                    tol = {"R1": 150.0, "R2": 240.0, "R3": 300.0}.get(orbit, 220.0)
                    raw_quality = float(np.exp(-raw_rms / tol))

                    components["raw_pick_rms_to_model_sec"] = raw_rms
                    components["raw_pick_rms_tolerance_sec"] = tol
                    components["raw_pick_quality"] = raw_quality

                    confidence *= 0.75 + 0.25 * raw_quality
                    components["overall_confidence"] = float(confidence)

                vprint(
                    f"      Peak {peak_idx + 1} @ {disp_data['peak_time']:.1f}min: "
                    f"dist={optimal_distance:.1f} deg, "
                    f"conf={confidence:.3f}, "
                    f"RMS={components['rms_residual_sec']:.1f}s"
                )

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = {
                        "distance_deg": optimal_distance,
                        "dispersion": disp_for_match,
                        "confidence": confidence,
                        "confidence_components": components,
                        "rms_residual_sec": components["rms_residual_sec"],
                    }

            if best_match is not None and best_confidence >= min_confidence:
                if best_match['rms_residual_sec'] > 600:
                    vprint(f"   REJECTED {orbit} REJECTED: RMS too large ({best_match['rms_residual_sec']:.1f}s > 600s)")
                    vprint(f"      (This peak doesn't match {orbit} dispersion)")
                else:
                    detected_orbits[orbit] = best_match
                    vprint(f'   OK: {orbit} ACCEPTED: confidence={best_confidence:.3f}')
            else:
                vprint(f'   REJECTED {orbit} REJECTED: best confidence={best_confidence:.3f} < {min_confidence}')
        metadata = {'method': 'independent', 'n_orbits_detected': len(detected_orbits)}
    if output_dir is not None:
        from .plotting import (
            plot_no_detection_summary,
            plot_orbit_detection_diagnostics,
        )

        if len(detected_orbits) > 0:
            orbit_energies = build_particleman_orbit_energies(result_dict=result_dict, detected_orbits=detected_orbits, time_min=time_min, fmin_hz=fmin_hz, fmax_hz=fmax_hz)
            plot_orbit_detection_diagnostics(detected_orbits, peak_dispersions, S_energy_band, freqs_band, time_min, energy_time_integrated, metadata, output_dir, orbit_energies=orbit_energies)
        else:
            plot_no_detection_summary(peak_dispersions, S_energy_band, freqs_band, time_min, energy_time_integrated, metadata, output_dir, min_confidence)
    detected_orbits['metadata'] = metadata
    vprint(f"\nAccepted: Detection complete: {metadata['n_orbits_detected']} orbit(s) detected")
    return detected_orbits



def plot_no_detection_summary(*args, **kwargs):
    """Create a no-detection diagnostic figure.

    This compatibility wrapper delegates to :mod:`rayleigh_wave_tools.plotting`.
    New code should import the function from ``rayleigh_wave_tools.plotting``.
    """
    from .plotting import plot_no_detection_summary as _plot

    return _plot(*args, **kwargs)


def plot_orbit_detection_diagnostics(*args, **kwargs):
    """Create orbit-detection diagnostic figures.

    This compatibility wrapper delegates to :mod:`rayleigh_wave_tools.plotting`.
    New code should import the function from ``rayleigh_wave_tools.plotting``.
    """
    from .plotting import plot_orbit_detection_diagnostics as _plot

    return _plot(*args, **kwargs)

__all__ = [
    "detect_rayleigh_orbits_from_stockwell",
    "distance_guided_extraction",
    "extract_dispersion_curve_model_guided",
    "extract_dispersion_curve_robust",
    "calculate_confidence_score",
    "apply_model_guided_confidence_penalties",
    "calculate_dispersion_shape_correlation",
    "joint_fit_guided_orbits",
    "rayleigh_group_velocity_prem",
]
