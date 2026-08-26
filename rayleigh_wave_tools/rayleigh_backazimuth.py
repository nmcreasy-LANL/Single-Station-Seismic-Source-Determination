#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rayleigh back-azimuth estimation utilities.

This module implements the Hilbert-Bayes back-azimuth estimator used by the
Rayleigh filtering and orbit detection workflow. It combines band-wise axial
posteriors with retrograde particle-motion checks to resolve the 180-degree
branch ambiguity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
from typing import Iterable as _Iterable, Union as _Union

import numpy as np
from scipy.signal import butter, detrend, hilbert, resample_poly, sosfiltfilt
from scipy.special import logsumexp

ArrayLike = _Union[np.ndarray, _Iterable[float]]


def make_octave_bands(fmin: float, fmax: float, bands_per_octave: float=1.0, overlap: bool=False) -> List[Tuple[float, float]]:
    """Generate octave or fractional-octave frequency bands.

    bands_per_octave=1 gives octave bands. bands_per_octave=2 gives
    half-octave bands. With overlap=True, adjacent bands step by half the
    band width in log2 frequency, which is useful for avoiding brittle band
    edges around dispersive Rayleigh packets.
    """
    fmin = float(fmin)
    fmax = float(fmax)
    bands_per_octave = float(bands_per_octave)
    if not 0.0 < fmin < fmax:
        raise ValueError('Octave bands require 0 < fmin < fmax.')
    if bands_per_octave <= 0.0:
        raise ValueError('bands_per_octave must be positive.')
    band_ratio = 2.0 ** (1.0 / bands_per_octave)
    step_ratio = np.sqrt(band_ratio) if overlap else band_ratio
    bands: List[Tuple[float, float]] = []
    flo = fmin
    while flo * band_ratio <= fmax * (1.0 + 1e-12):
        fhi = flo * band_ratio
        bands.append((float(flo), float(fhi)))
        flo *= step_ratio
    if bands:
        last_hi = bands[-1][1]
        if last_hi < fmax and fmax / last_hi >= 1.25:
            bands.append((float(last_hi), float(fmax)))
    else:
        bands.append((fmin, fmax))
    return bands


def bandpass_filter(data: np.ndarray, dt: float, fmin: float, fmax: float, corners: int=4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    data = np.asarray(data, dtype=float)
    fs = 1.0 / dt
    nyq = 0.5 * fs
    low = fmin / nyq
    high = fmax / nyq
    if not 0.0 < low < high < 1.0:
        raise ValueError(f'Invalid band [{fmin}, {fmax}] Hz for dt={dt}; normalized=({low}, {high}).')
    sos = butter(corners, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, data)


def preprocess_for_rayleigh(data: np.ndarray, dt: float, target_dt: Optional[float], fmin: float, fmax: float) -> Tuple[np.ndarray, float]:
    """Detrend, bandpass, and optionally downsample safely with anti-aliasing."""
    x = detrend(np.asarray(data, dtype=float))
    x = bandpass_filter(x, dt, fmin, fmax)
    if target_dt is None or target_dt <= dt:
        return (x, dt)
    down_factor = int(round(target_dt / dt))
    if down_factor <= 1:
        return (x, dt)
    actual_target_dt = dt * down_factor
    nyq = 0.5 / dt
    cutoff = 0.4 / actual_target_dt
    sos = butter(4, cutoff / nyq, btype='low', output='sos')
    x = sosfiltfilt(sos, x)
    x = resample_poly(x, 1, down_factor)
    return (x, actual_target_dt)


def align_lengths(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Trim arrays to a common minimum length."""
    n = min((len(a) for a in arrays))
    return tuple((np.asarray(a[:n], dtype=float) for a in arrays))


def circular_difference_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two azimuths."""
    return float(abs((a - b + 180.0) % 360.0 - 180.0))


def circular_mean_from_pdf_deg(angles_deg: np.ndarray, pdf: np.ndarray) -> float:
    angles_rad = np.deg2rad(angles_deg)
    z = np.sum(pdf * np.exp(1j * angles_rad))
    return float(np.rad2deg(np.angle(z)) % 360.0)


def circular_std_from_pdf_deg(angles_deg: np.ndarray, pdf: np.ndarray) -> float:
    angles_rad = np.deg2rad(angles_deg)
    z = np.sum(pdf * np.exp(1j * angles_rad))
    R = np.clip(np.abs(z), 1e-12, 1.0)
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(R))))


def normalize_pdf_from_log(logp: np.ndarray) -> np.ndarray:
    logp = np.asarray(logp, dtype=float)
    logp = logp - logsumexp(logp)
    return np.exp(logp)


def hilbert_score_terms(Nf: np.ndarray, Ef: np.ndarray, HZ: np.ndarray, azimuth_range: np.ndarray, rayleigh_sign: str='any', transverse_weight: float=1.0, eps: float=1e-14) -> Dict[str, np.ndarray]:
    """Compute signed/absolute Hilbert coherence score terms on an azimuth grid.

    rayleigh_sign:
      "any"      : abs(corr), preserving the 180 degree ambiguity.
      "positive" : positive corr only, used here as the retrograde-only option.
      "negative" : negative corr only, useful if component polarity/convention is reversed.
    """
    rayleigh_sign = str(rayleigh_sign).lower()
    if rayleigh_sign not in {'any', 'positive', 'negative'}:
        raise ValueError("rayleigh_sign must be 'any', 'positive', or 'negative'.")
    HZc = HZ - np.mean(HZ)
    scores, corrs, abs_corrs, transverse_ratios = ([], [], [], [])
    for az_deg in azimuth_range:
        th = np.deg2rad(az_deg)
        R = np.cos(th) * Nf + np.sin(th) * Ef
        T = -np.sin(th) * Nf + np.cos(th) * Ef
        Rc = R - np.mean(R)
        denom = np.linalg.norm(Rc) * np.linalg.norm(HZc) + eps
        corr = float(np.dot(Rc, HZc) / denom)
        abs_corr = abs(corr)
        if rayleigh_sign == 'positive' and corr <= 0.0:
            coherence = 0.0
        elif rayleigh_sign == 'negative' and corr >= 0.0:
            coherence = 0.0
        else:
            coherence = abs_corr
        transverse_ratio = float(np.std(T) / (np.std(R) + eps))
        score = coherence / (1.0 + transverse_weight * transverse_ratio)
        scores.append(score)
        corrs.append(corr)
        abs_corrs.append(abs_corr)
        transverse_ratios.append(transverse_ratio)
    return {'scores': np.asarray(scores, dtype=float), 'corr_signed': np.asarray(corrs, dtype=float), 'corr_abs': np.asarray(abs_corrs, dtype=float), 'transverse_ratio': np.asarray(transverse_ratios, dtype=float)}


def estimate_azimuth_hilbert_bayesian(N: np.ndarray, E: np.ndarray, Z: np.ndarray, dt: float, fmin: float, fmax: float, azimuth_range: Optional[np.ndarray]=None, sigma_rz: Optional[float]=None, sigma_t: Optional[float]=None, transverse_weight: float=1.0, use_residual_likelihood: bool=False, prior: Optional[np.ndarray]=None, beta: float=80.0, rayleigh_sign: str='any', eps: float=1e-14) -> Dict[str, object]:
    """Bayesian Hilbert estimator calibrated to the deterministic Hilbert score.

    Default behavior uses loglike = beta * deterministic_score_on_bayes_grid.
    This keeps the Bayesian MAP aligned with the corresponding deterministic
    Hilbert method. Set use_residual_likelihood=True only for experiments.

    rayleigh_sign:
      "any"      : abs(corr), preserving the 180 degree ambiguity.
      "positive" : positive corr only, the default retrograde-only convention.
      "negative" : negative corr only, useful if component polarity/convention is reversed.
    """
    Nf = bandpass_filter(detrend(np.asarray(N, float)), dt, fmin, fmax)
    Ef = bandpass_filter(detrend(np.asarray(E, float)), dt, fmin, fmax)
    Zf = bandpass_filter(detrend(np.asarray(Z, float)), dt, fmin, fmax)
    if azimuth_range is None:
        azimuth_range = np.linspace(0.0, 360.0, 721, endpoint=False)
    azimuth_range = np.asarray(azimuth_range, dtype=float)
    HZ = np.imag(hilbert(Zf))
    n = len(HZ)
    terms = hilbert_score_terms(Nf, Ef, HZ, azimuth_range, rayleigh_sign=rayleigh_sign, transverse_weight=transverse_weight, eps=eps)
    scores = terms['scores']
    a_hats, rz_rms, t_rms, rss_rz_all, rss_t_all = ([], [], [], [], [])
    for az_deg in azimuth_range:
        th = np.deg2rad(az_deg)
        R = np.cos(th) * Nf + np.sin(th) * Ef
        T = -np.sin(th) * Nf + np.cos(th) * Ef
        a_hat = np.dot(R, HZ) / (np.dot(R, R) + eps)
        resid = HZ - a_hat * R
        rss_rz = float(np.sum(resid ** 2))
        rss_t = float(np.sum(T ** 2))
        a_hats.append(a_hat)
        rz_rms.append(np.sqrt(rss_rz / n))
        t_rms.append(np.sqrt(rss_t / n))
        rss_rz_all.append(rss_rz)
        rss_t_all.append(rss_t)
    rss_rz_all = np.asarray(rss_rz_all, dtype=float)
    rss_t_all = np.asarray(rss_t_all, dtype=float)
    if use_residual_likelihood:
        sig2_rz = float(sigma_rz ** 2) if sigma_rz is not None else float(np.median(rss_rz_all / n) + eps)
        sig2_t = float(sigma_t ** 2) if sigma_t is not None else float(np.median(rss_t_all / n) + eps)
        ll_rz = -0.5 * rss_rz_all / (sig2_rz + eps)
        ll_t = -0.5 * rss_t_all / (sig2_t + eps)
        loglike = ll_rz + transverse_weight * ll_t
        if rayleigh_sign == 'positive':
            loglike = np.where(terms['corr_signed'] > 0.0, loglike, -np.inf)
        elif rayleigh_sign == 'negative':
            loglike = np.where(terms['corr_signed'] < 0.0, loglike, -np.inf)
    else:
        loglike = beta * scores
    if prior is None:
        logprior = np.zeros_like(loglike)
    else:
        prior = np.asarray(prior, dtype=float)
        if prior.shape != loglike.shape:
            raise ValueError('prior must have the same shape as azimuth_range.')
        logprior = np.log(prior / (np.sum(prior) + eps) + eps)
    logpost = loglike + logprior
    posterior = normalize_pdf_from_log(logpost)
    logpost_norm = np.log(posterior + eps)
    i_map = int(np.argmax(posterior))
    map_az = float(azimuth_range[i_map] % 360.0)
    flip = (map_az + 180.0) % 360.0
    i_flip = int(np.argmin(np.abs((azimuth_range - flip + 180.0) % 360.0 - 180.0)))
    entropy = -np.sum(posterior * np.log(posterior + eps))
    max_entropy = np.log(len(posterior))
    return {'method': 'hilbert_bayes' if rayleigh_sign == 'any' else f'hilbert_bayes_{rayleigh_sign}', 'azimuth': map_az, 'map_azimuth': map_az, 'posterior_mean_azimuth': circular_mean_from_pdf_deg(azimuth_range, posterior), 'posterior_std_deg': circular_std_from_pdf_deg(azimuth_range, posterior), 'score': float(posterior[i_map]), 'azimuth_range': azimuth_range, 'posterior': posterior, 'log_likelihood': loglike, 'log_posterior': logpost_norm, 'deterministic_score_on_bayes_grid': scores, 'corr_signed': terms['corr_signed'], 'corr_abs': terms['corr_abs'], 'transverse_ratio': terms['transverse_ratio'], 'a_hat': np.asarray(a_hats), 'rz_residual_rms': np.asarray(rz_rms), 'transverse_rms': np.asarray(t_rms), 'log_odds_map_vs_180deg_flip': float(logpost_norm[i_map] - logpost_norm[i_flip]), 'posterior_entropy': float(entropy), 'posterior_entropy_ratio': float(entropy / max_entropy), 'likelihood_mode': 'residual' if use_residual_likelihood else 'deterministic_score', 'rayleigh_sign': rayleigh_sign, 'transverse_weight': transverse_weight, 'beta': beta}


def interpolate_periodic_pdf(source_azimuth: np.ndarray, source_pdf: np.ndarray, target_azimuth: np.ndarray, eps: float=1e-300) -> np.ndarray:
    """Interpolate a periodic azimuth PDF onto a target grid and renormalize."""
    src = np.asarray(source_azimuth, dtype=float) % 360.0
    pdf = np.asarray(source_pdf, dtype=float)
    target = np.asarray(target_azimuth, dtype=float) % 360.0
    order = np.argsort(src)
    src = src[order]
    pdf = pdf[order]
    src_ext = np.r_[src[-1] - 360.0, src, src[0] + 360.0]
    pdf_ext = np.r_[pdf[-1], pdf, pdf[0]]
    out = np.interp(target, src_ext, pdf_ext)
    out = np.maximum(out, eps)
    out = out / np.sum(out)
    return out


def axial_weighted_mean_deg(angles_deg: Sequence[float], weights: Sequence[float]) -> float:
    """Weighted axial mean in degrees for 180-degree ambiguous azimuths.

    The returned value is on [0, 180). Add either 0 or 180 degrees to choose a
    directional branch.
    """
    angles = np.deg2rad(2.0 * np.asarray(angles_deg, dtype=float))
    w = np.asarray(weights, dtype=float)
    if angles.size == 0 or np.sum(w) <= 0.0:
        return float('nan')
    z = np.sum(w * np.exp(1j * angles)) / np.sum(w)
    return float(0.5 * np.rad2deg(np.angle(z)) % 180.0)


def axial_difference_deg(a: float, b: float) -> float:
    """Smallest difference between two 180-degree ambiguous axes."""
    return float(0.5 * abs((2.0 * (a - b) + 180.0) % 360.0 - 180.0))


def axial_mean_from_pdf_deg(angles_deg: np.ndarray, pdf: np.ndarray) -> float:
    """Mean axis, on [0, 180), for a 180-degree symmetric azimuth PDF."""
    angles_rad = np.deg2rad(2.0 * np.asarray(angles_deg, dtype=float))
    p = np.asarray(pdf, dtype=float)
    z = np.sum(p * np.exp(1j * angles_rad))
    return float(0.5 * np.rad2deg(np.angle(z)) % 180.0)


def axial_std_from_pdf_deg(angles_deg: np.ndarray, pdf: np.ndarray) -> float:
    """Circular standard deviation for a 180-degree axis, in degrees.

    This computes the circular spread after doubling all angles, then halves the
    result. It is the correct uncertainty summary for an axial PDF with equal
    probability at theta and theta + 180 degrees.
    """
    angles_rad = np.deg2rad(2.0 * np.asarray(angles_deg, dtype=float))
    p = np.asarray(pdf, dtype=float)
    z = np.sum(p * np.exp(1j * angles_rad))
    R = np.clip(np.abs(z), 1e-12, 1.0)
    return float(0.5 * np.rad2deg(np.sqrt(-2.0 * np.log(R))))


def choose_180_branch(az_deg: float, reference_deg: float) -> Tuple[float, bool, float]:
    """Choose az or az+180 so it lies closest to a reference direction.

    Returns corrected_azimuth, was_flipped, corrected_disagreement_deg.
    """
    a0 = float(az_deg) % 360.0
    a1 = (a0 + 180.0) % 360.0
    d0 = circular_difference_deg(a0, reference_deg)
    d1 = circular_difference_deg(a1, reference_deg)
    if d1 < d0:
        return (a1, True, d1)
    return (a0, False, d0)


def shift_periodic_pdf_deg(source_azimuth: np.ndarray, source_pdf: np.ndarray, target_azimuth: np.ndarray, shift_deg: float, eps: float=1e-300) -> np.ndarray:
    """Shift a periodic azimuth PDF by shift_deg and interpolate onto target grid.

    A positive shift moves probability mass at azimuth a to a + shift_deg.
    """
    src = (np.asarray(source_azimuth, dtype=float) + float(shift_deg)) % 360.0
    return interpolate_periodic_pdf(src, np.asarray(source_pdf, dtype=float), target_azimuth, eps=eps)


def fold_180_symmetric_pdf(source_azimuth: np.ndarray, source_pdf: np.ndarray, target_azimuth: np.ndarray, eps: float=1e-300) -> np.ndarray:
    """Return a 180-degree symmetric version of an azimuth PDF.

    This is useful for bands that are informative about the Rayleigh sagittal
    axis but unreliable about the sign/branch.
    """
    p0 = interpolate_periodic_pdf(source_azimuth, source_pdf, target_azimuth, eps=eps)
    p1 = shift_periodic_pdf_deg(source_azimuth, source_pdf, target_azimuth, 180.0, eps=eps)
    p = p0 + p1
    p = np.maximum(p, eps)
    return p / np.sum(p)


def band_center_frequency(fmin_hz: float, fmax_hz: float) -> float:
    """Geometric mean center frequency for a band."""
    return float(np.sqrt(float(fmin_hz) * float(fmax_hz)))


def band_reliability_weight(res: Dict[str, object], min_entropy_ratio: float=0.0, max_entropy_ratio: float=0.9, min_log_odds_180: float=0.0, stability_weight: float=1.0, require_180_discrimination: bool=True, eps: float=1e-14) -> Tuple[float, str, Dict[str, float]]:
    """Return a reliability weight for one Bayesian band result.

    The default weight rewards concentrated posteriors and a strong preference
    for the MAP over the 180 degree flipped solution. A separate stability
    factor can downweight bands that disagree with the inter-band consensus.
    """
    entropy_ratio = float(res.get('posterior_entropy_ratio', 1.0))
    log_odds = float(res.get('log_odds_map_vs_180deg_flip', 0.0))
    map_score = float(res.get('score', 0.0))
    if not np.isfinite(entropy_ratio):
        return (0.0, 'rejected_nonfinite_entropy', {'entropy_conf': 0.0, 'ambiguity_conf': 0.0, 'map_score': map_score})
    if entropy_ratio > max_entropy_ratio:
        return (0.0, 'rejected_high_entropy', {'entropy_conf': 0.0, 'ambiguity_conf': 0.0, 'map_score': map_score})
    if require_180_discrimination and log_odds < min_log_odds_180:
        return (0.0, 'rejected_low_log_odds_180', {'entropy_conf': 0.0, 'ambiguity_conf': 0.0, 'map_score': map_score})
    entropy_span = max(eps, max_entropy_ratio - min_entropy_ratio)
    entropy_conf = np.clip((max_entropy_ratio - entropy_ratio) / entropy_span, 0.0, 1.0)
    if require_180_discrimination:
        ambiguity_conf = max(0.0, log_odds) / (1.0 + max(0.0, log_odds))
    else:
        ambiguity_conf = 1.0
    score_conf = np.sqrt(max(map_score, 0.0))
    weight = float(entropy_conf * ambiguity_conf * score_conf * max(0.0, stability_weight))
    if weight <= 0.0:
        return (0.0, 'rejected_zero_weight', {'entropy_conf': float(entropy_conf), 'ambiguity_conf': float(ambiguity_conf), 'map_score': map_score})
    return (weight, 'used', {'entropy_conf': float(entropy_conf), 'ambiguity_conf': float(ambiguity_conf), 'map_score': map_score})


def verify_retrograde_motion(R: np.ndarray, Z: np.ndarray, threshold: float=0.7) -> Tuple[float, str, Dict[str, float]]:
    """
    Verify whether Rayleigh wave particle motion is retrograde or prograde.
    
    Returns
    -------
    ellipticity : float
        Signed ellipticity measure (>0 retrograde, <0 prograde, ~0 linear)
    classification : str
        One of: "retrograde", "prograde", "ambiguous"
    diagnostics : dict
        Full diagnostic information
    """
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if len(R) != len(Z):
        raise ValueError(f'R and Z must have same length, got {len(R)} and {len(Z)}')
    if len(R) < 10:
        raise ValueError(f'Need at least 10 samples, got {len(R)}')
    R = R - np.mean(R)
    Z = Z - np.mean(Z)
    HZ = np.imag(hilbert(Z))
    HZ = HZ - np.mean(HZ)
    corr_R_HZ = np.corrcoef(R, HZ)[0, 1] if len(R) > 1 else 0.0
    cov_RR = np.mean(R * R)
    cov_ZZ = np.mean(Z * Z)
    cov_RZ = np.mean(R * Z)
    cov_R_HZ = np.mean(R * HZ)
    C = np.array([[cov_RR, cov_RZ], [cov_RZ, cov_ZZ]])
    eigvals = np.linalg.eigvalsh(C)
    major_axis = np.sqrt(max(eigvals))
    minor_axis = np.sqrt(min(eigvals))
    hodogram_area = minor_axis / major_axis if major_axis > 1e-14 else 0.0
    ellipticity = corr_R_HZ * hodogram_area
    ellipticity_abs = abs(ellipticity)
    phase_lag_rad = np.arctan2(cov_RZ, cov_R_HZ)
    phase_lag_deg = np.rad2deg(phase_lag_rad)
    if ellipticity_abs < threshold:
        classification = 'ambiguous'
        recommendation = f'Motion is too linear (|ellipticity|={ellipticity_abs:.3f} < {threshold}). Use axial method only.'
    elif ellipticity > threshold:
        classification = 'retrograde'
        recommendation = f'Motion is clearly retrograde (ellipticity={ellipticity:.3f}). Safe to use for direction choice.'
    elif ellipticity < -threshold:
        classification = 'prograde'
        recommendation = f'Motion is PROGRADE (ellipticity={ellipticity:.3f}). Unusual for typical Rayleigh waves.'
    else:
        classification = 'ambiguous'
        recommendation = 'Ambiguous motion. Use axial method.'
    diagnostics = {'ellipticity': float(ellipticity), 'ellipticity_abs': float(ellipticity_abs), 'classification': classification, 'rz_correlation': float(corr_R_HZ), 'phase_lag_deg': float(phase_lag_deg), 'hodogram_area': float(hodogram_area), 'major_axis': float(major_axis), 'minor_axis': float(minor_axis), 'recommendation': recommendation}
    return (float(ellipticity), classification, diagnostics)


def _as_grid(step):
    return np.arange(0.0, 360.0, float(step))


def _combine_from_logp(logp, target_grid):
    pdf = normalize_pdf_from_log(logp)
    i = int(np.argmax(pdf))
    return (float(target_grid[i] % 360.0), pdf)


def contrast_weights(raw_weights: Sequence[float], gamma: float=2.0, preserve_sum: bool=True) -> np.ndarray:
    """Increase contrast in band weights while preserving total information.

    gamma=1 leaves weights unchanged. gamma>1 makes reliable bands contribute
    more strongly relative to marginal bands. Preserving the sum keeps the
    posterior width from changing only because the weights were transformed.
    """
    w = np.asarray(raw_weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    if w.size == 0 or np.sum(w) <= 0.0:
        return w
    gamma = max(float(gamma), 1e-12)
    wg = w ** gamma
    if preserve_sum and np.sum(wg) > 0.0:
        wg = wg * (np.sum(w) / np.sum(wg))
    return wg


def combine_axis_bands(band_results, target_grid, max_entropy_ratio=0.9, stability_sigma_deg=25.0, weight_gamma=2.0, true_baz=None):
    """Combine non-retrograde Hilbert-Bayes posteriors as a 180-degree axis.

    Band posteriors are first assigned a physics/statistics reliability weight.
    The optional weight_gamma applies a contrast transform to those weights,
    so high-quality bands dominate more than marginal bands without imposing
    a hard frequency cutoff.
    """
    candidates = []
    for item in band_results:
        res = item['results'].get('hilbert_bayes')
        if not res:
            continue
        candidates.append({'fmin_hz': float(item['fmin_hz']), 'fmax_hz': float(item['fmax_hz']), 'fcenter_hz': band_center_frequency(item['fmin_hz'], item['fmax_hz']), 'res': res, 'azimuth_deg': float(res['azimuth'])})
    if not candidates:
        return (None, [])
    prelim_w = []
    prelim_az = []
    prelim_status = []
    for c in candidates:
        w, status, _ = band_reliability_weight(c['res'], max_entropy_ratio=max_entropy_ratio, min_log_odds_180=0.0, stability_weight=1.0, require_180_discrimination=False)
        prelim_w.append(w)
        prelim_az.append(c['azimuth_deg'])
        prelim_status.append(status)
    consensus = axial_weighted_mean_deg(prelim_az, prelim_w if np.sum(prelim_w) > 0 else np.ones(len(prelim_w)))
    work = []
    diagnostics = []
    for c, raw_w, raw_status in zip(candidates, prelim_w, prelim_status):
        disagreement = axial_difference_deg(c['azimuth_deg'], consensus)
        stability = 1.0 if not stability_sigma_deg else float(np.exp(-0.5 * (disagreement / stability_sigma_deg) ** 2))
        base_weight, status, pieces = band_reliability_weight(c['res'], max_entropy_ratio=max_entropy_ratio, min_log_odds_180=0.0, stability_weight=stability, require_180_discrimination=False)
        if raw_status.startswith('rejected'):
            base_weight = 0.0
            status = raw_status
        pdf = fold_180_symmetric_pdf(c['res']['azimuth_range'], c['res']['posterior'], target_grid)
        work.append((c, raw_w, base_weight, status, pieces, disagreement, stability, pdf))
    effective_weights = contrast_weights([x[2] for x in work], gamma=weight_gamma, preserve_sum=True)
    logp = np.zeros_like(target_grid, dtype=float)
    used = 0
    for eff_w, (c, raw_w, base_weight, status, pieces, disagreement, stability, pdf) in zip(effective_weights, work):
        if eff_w > 0.0:
            logp += float(eff_w) * np.log(pdf + 1e-300)
            used += 1
        row = {'method': 'hilbert_bayes', 'fmin_hz': c['fmin_hz'], 'fmax_hz': c['fmax_hz'], 'fcenter_hz': c['fcenter_hz'], 'azimuth_raw_deg': c['azimuth_deg'], 'azimuth_deg': c['azimuth_deg'], 'azimuth_branch_corrected_deg': c['azimuth_deg'], 'was_180_flipped': False, 'used_folded_180_pdf': True, 'branch_status': 'axial_180_symmetric_not_flipped', 'raw_disagreement_from_branch_reference_deg': float(disagreement), 'disagreement_from_prelim_consensus_deg': float(disagreement), 'raw_weight': float(raw_w), 'base_combined_weight': float(base_weight), 'weight_gamma': float(weight_gamma), 'stability_weight': float(stability), 'combined_weight': float(eff_w), 'status': status, 'posterior_entropy_ratio': float(c['res'].get('posterior_entropy_ratio', np.nan)), 'log_odds_map_vs_180deg_flip': float(c['res'].get('log_odds_map_vs_180deg_flip', np.nan)), 'score': float(c['res'].get('score', np.nan)), 'retro_axis_agreement_error_deg': np.nan, 'retro_axis_agreement_weight': np.nan, 'retro_axis_agreement_status': 'not_checked'}
        row.update(pieces)
        if true_baz is not None:
            row['error_to_truth_deg'] = circular_difference_deg(c['azimuth_deg'], true_baz)
        diagnostics.append(row)
    if used == 0:
        return (None, diagnostics)
    map_az, pdf = _combine_from_logp(logp, target_grid)
    axis = axial_mean_from_pdf_deg(target_grid, pdf)
    combined = {'method': 'combined_hilbert_bayes', 'azimuth': map_az, 'map_azimuth': map_az, 'azimuth_axis_deg_mod_180': axis, 'azimuth_axis_antipode_deg': (axis + 180.0) % 360.0, 'is_180_symmetric_axial_result': True, 'posterior_mean_azimuth': circular_mean_from_pdf_deg(target_grid, pdf), 'posterior_std_deg': circular_std_from_pdf_deg(target_grid, pdf), 'posterior_axial_mean_deg_mod_180': axial_mean_from_pdf_deg(target_grid, pdf), 'posterior_axial_std_deg': axial_std_from_pdf_deg(target_grid, pdf), 'score': float(np.max(pdf)), 'azimuth_range': target_grid, 'posterior': pdf, 'log_posterior': np.log(pdf + 1e-300), 'preliminary_consensus_azimuth': float(consensus), 'preliminary_consensus_source': 'all_reliable_bands_axial', 'preliminary_axial_consensus_deg_mod_180': float(consensus), 'used_band_count': int(used), 'candidate_band_count': int(len(candidates)), 'flipped_band_count': 0, 'folded_band_count': int(len(candidates)), 'base_weight_sum': float(sum((d['base_combined_weight'] for d in diagnostics))), 'weight_sum': float(sum((d['combined_weight'] for d in diagnostics))), 'weight_gamma': float(weight_gamma), 'combine_mode': 'axial_180_symmetric_contrast_weighted_log_posterior_product', 'combined_from_method': 'hilbert_bayes'}
    if true_baz is not None:
        combined['error_to_truth_deg'] = circular_difference_deg(map_az, true_baz)
    return (combined, diagnostics)


def combine_retro_bands(band_results, target_grid, branch_reference_deg, max_entropy_ratio=0.9, min_log_odds_180=0.0, stability_sigma_deg=25.0, axis_agreement_sigma_deg=25.0, axis_agreement_min_weight=0.05, weight_gamma=2.0, true_baz=None):
    """Combine retrograde posteriors after branch correction and axis-agreement gating.

    The raw reliability weight is transformed with weight_gamma before stacking.
    This keeps the wide frequency sweep but lets the physically best bands carry
    most of the posterior product.
    """
    candidates = []
    for item in band_results:
        retro = item['results'].get('hilbert_bayes_retro')
        axis = item['results'].get('hilbert_bayes')
        if not retro:
            continue
        candidates.append({'fmin_hz': float(item['fmin_hz']), 'fmax_hz': float(item['fmax_hz']), 'fcenter_hz': band_center_frequency(item['fmin_hz'], item['fmax_hz']), 'retro': retro, 'axis': axis, 'azimuth_deg': float(retro['azimuth'])})
    if not candidates:
        return (None, [])
    work = []
    for c in candidates:
        raw = c['azimuth_deg']
        corrected, was_flipped, raw_disagreement = choose_180_branch(raw, branch_reference_deg)
        disagreement = raw_disagreement
        stability_branch = 1.0 if not stability_sigma_deg else float(np.exp(-0.5 * (disagreement / stability_sigma_deg) ** 2))
        axis_err = np.nan
        axis_w = 1.0
        axis_status = 'not_checked'
        axis_ref = np.nan
        if c['axis'] is None:
            axis_w = 0.0
            axis_status = 'rejected_missing_axis_method'
        else:
            axis_ref = float(c['axis'].get('azimuth', np.nan)) % 180.0
            axis_err = axial_difference_deg(corrected, axis_ref)
            axis_w = float(np.exp(-0.5 * (axis_err / axis_agreement_sigma_deg) ** 2))
            axis_status = 'axis_agreement_used'
        stability = stability_branch * max(0.0, axis_w)
        base_weight, status, pieces = band_reliability_weight(c['retro'], max_entropy_ratio=max_entropy_ratio, min_log_odds_180=min_log_odds_180, stability_weight=stability, require_180_discrimination=True)
        if axis_status.startswith('rejected'):
            base_weight = 0.0
            status = axis_status
        elif axis_w < axis_agreement_min_weight:
            base_weight = 0.0
            status = 'rejected_retro_axis_disagreement'
            axis_status = status
        pdf = shift_periodic_pdf_deg(c['retro']['azimuth_range'], c['retro']['posterior'], target_grid, 180.0 if was_flipped else 0.0)
        work.append((c, raw, corrected, was_flipped, raw_disagreement, disagreement, stability, stability_branch, axis_ref, axis_err, axis_w, axis_status, base_weight, status, pieces, pdf))
    effective_weights = contrast_weights([x[12] for x in work], gamma=weight_gamma, preserve_sum=True)
    logp = np.zeros_like(target_grid, dtype=float)
    diagnostics = []
    used = 0
    flipped_count = 0
    for eff_w, item in zip(effective_weights, work):
        c, raw, corrected, was_flipped, raw_disagreement, disagreement, stability, stability_branch, axis_ref, axis_err, axis_w, axis_status, base_weight, status, pieces, pdf = item
        if eff_w > 0.0:
            logp += float(eff_w) * np.log(pdf + 1e-300)
            used += 1
            flipped_count += int(was_flipped)
        row = {'method': 'hilbert_bayes_retro', 'fmin_hz': c['fmin_hz'], 'fmax_hz': c['fmax_hz'], 'fcenter_hz': c['fcenter_hz'], 'azimuth_raw_deg': raw, 'azimuth_deg': corrected, 'azimuth_branch_corrected_deg': corrected, 'was_180_flipped': bool(was_flipped), 'used_folded_180_pdf': False, 'branch_status': 'flipped_180_to_consensus' if was_flipped else 'kept_original_branch', 'raw_disagreement_from_branch_reference_deg': float(raw_disagreement), 'disagreement_from_prelim_consensus_deg': float(disagreement), 'raw_weight': np.nan, 'base_combined_weight': float(base_weight), 'weight_gamma': float(weight_gamma), 'stability_weight': float(stability), 'branch_stability_weight': float(stability_branch), 'combined_weight': float(eff_w), 'status': status, 'posterior_entropy_ratio': float(c['retro'].get('posterior_entropy_ratio', np.nan)), 'log_odds_map_vs_180deg_flip': float(c['retro'].get('log_odds_map_vs_180deg_flip', np.nan)), 'score': float(c['retro'].get('score', np.nan)), 'axis_reference_deg_mod_180': float(axis_ref), 'retro_axis_agreement_error_deg': float(axis_err), 'retro_axis_agreement_weight': float(axis_w), 'retro_axis_agreement_status': axis_status}
        row.update(pieces)
        if true_baz is not None:
            row['error_raw_to_truth_deg'] = circular_difference_deg(raw, true_baz)
            row['error_to_truth_deg'] = circular_difference_deg(corrected, true_baz)
        diagnostics.append(row)
    if used == 0:
        return (None, diagnostics)
    map_az, pdf = _combine_from_logp(logp, target_grid)
    combined = {'method': 'combined_hilbert_bayes_retro', 'azimuth': map_az, 'map_azimuth': map_az, 'azimuth_axis_deg_mod_180': None, 'azimuth_axis_antipode_deg': None, 'is_180_symmetric_axial_result': False, 'posterior_mean_azimuth': circular_mean_from_pdf_deg(target_grid, pdf), 'posterior_std_deg': circular_std_from_pdf_deg(target_grid, pdf), 'posterior_axial_mean_deg_mod_180': axial_mean_from_pdf_deg(target_grid, pdf), 'posterior_axial_std_deg': axial_std_from_pdf_deg(target_grid, pdf), 'score': float(np.max(pdf)), 'azimuth_range': target_grid, 'posterior': pdf, 'log_posterior': np.log(pdf + 1e-300), 'preliminary_consensus_azimuth': float(branch_reference_deg), 'preliminary_consensus_source': 'chosen_stage2_branch', 'preliminary_axial_consensus_deg_mod_180': float(branch_reference_deg % 180.0), 'used_band_count': int(used), 'candidate_band_count': int(len(candidates)), 'flipped_band_count': int(flipped_count), 'folded_band_count': 0, 'base_weight_sum': float(sum((d['base_combined_weight'] for d in diagnostics))), 'weight_sum': float(sum((d['combined_weight'] for d in diagnostics))), 'weight_gamma': float(weight_gamma), 'combine_mode': 'branch_aware_contrast_weighted_log_posterior_product', 'combined_from_method': 'hilbert_bayes_retro', 'branch_reference_deg': float(branch_reference_deg), 'require_retro_axis_agreement': True, 'axis_agreement_sigma_deg': float(axis_agreement_sigma_deg), 'axis_agreement_min_weight': float(axis_agreement_min_weight)}
    if true_baz is not None:
        combined['error_to_truth_deg'] = circular_difference_deg(map_az, true_baz)
    return (combined, diagnostics)


def estimate_required_methods(N, E, Z, dt, fmin, fmax, args, true_baz=None):
    grid = _as_grid(args.bayes_step)
    axis = estimate_azimuth_hilbert_bayesian(N, E, Z, dt, fmin, fmax, grid, rayleigh_sign='any', transverse_weight=args.transverse_weight, beta=args.beta)
    axis['method'] = 'hilbert_bayes'
    retro = estimate_azimuth_hilbert_bayesian(N, E, Z, dt, fmin, fmax, grid, rayleigh_sign=args.retro_sign, transverse_weight=args.transverse_weight, beta=args.beta)
    retro['method'] = 'hilbert_bayes_retro'
    retro['sense_constraint'] = f'retrograde_{args.retro_sign}_corr'
    results = {'hilbert_bayes': axis, 'hilbert_bayes_retro': retro}
    if true_baz is not None:
        for res in results.values():
            res['error_to_catalog_deg'] = circular_difference_deg(float(res['azimuth']), true_baz)
            res['error_to_catalog_if_180_flipped_deg'] = circular_difference_deg((float(res['azimuth']) + 180.0) % 360.0, true_baz)
    return results


@dataclass(frozen=True)
class RayleighBazDefaults:
    """Default settings for the Rayleigh BAZ workflow."""
    fmin_hz: float = 0.01
    fmax_hz: float = 3.0
    bands_per_octave: float = 1.176
    overlap_octaves: bool = True
    gamma: float = 3.0
    beta: float = 160.0
    bayes_step_deg: float = 0.5
    transverse_weight: float = 1.0
    retro_sign: str = 'positive'
    retrograde_threshold: float = 0.5
    combine_max_entropy: float = 0.9
    combine_min_log_odds_180: float = 0.0
    combine_stability_sigma: float = 25.0
    direction_entropy_max: float = 0.85
    axis_agreement_sigma: float = 25.0
    axis_agreement_min_weight: float = 0.05
    target_dt: Optional[float] = None


TUNED_DEFAULTS = RayleighBazDefaults()


@dataclass
class _RayleighArgs:
    bayes_step: float = TUNED_DEFAULTS.bayes_step_deg
    transverse_weight: float = TUNED_DEFAULTS.transverse_weight
    beta: float = TUNED_DEFAULTS.beta
    retro_sign: str = TUNED_DEFAULTS.retro_sign
    retrograde_threshold: float = TUNED_DEFAULTS.retrograde_threshold
    target_dt: Optional[float] = TUNED_DEFAULTS.target_dt
    combine_max_entropy: float = TUNED_DEFAULTS.combine_max_entropy
    combine_min_log_odds_180: float = TUNED_DEFAULTS.combine_min_log_odds_180
    combine_stability_sigma: float = TUNED_DEFAULTS.combine_stability_sigma
    direction_entropy_max: float = TUNED_DEFAULTS.direction_entropy_max
    axis_agreement_sigma: float = TUNED_DEFAULTS.axis_agreement_sigma
    axis_agreement_min_weight: float = TUNED_DEFAULTS.axis_agreement_min_weight
    weight_gamma: float = TUNED_DEFAULTS.gamma


def _as_1d_float(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size < 8:
        raise ValueError(f'{name} must contain at least 8 samples')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} contains non-finite values')
    return arr


def _components_from_arrays(north: ArrayLike, east: ArrayLike, vertical: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = _as_1d_float(north, 'north')
    e = _as_1d_float(east, 'east')
    z = _as_1d_float(vertical, 'vertical')
    return align_lengths(n, e, z)


def _diagnostics_for_public(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    keep = ['method', 'fmin_hz', 'fmax_hz', 'fcenter_hz', 'azimuth_raw_deg', 'azimuth_deg', 'azimuth_branch_corrected_deg', 'was_180_flipped', 'used_folded_180_pdf', 'branch_status', 'combined_weight', 'base_combined_weight', 'stability_weight', 'posterior_entropy_ratio', 'log_odds_map_vs_180deg_flip', 'score', 'status', 'retro_axis_agreement_error_deg', 'retro_axis_agreement_weight']
    out: List[Dict[str, object]] = []
    for row in rows:
        out.append({k: row.get(k) for k in keep if k in row})
    return out


def estimate_rayleigh_baz_posterior(north: ArrayLike, east: ArrayLike, vertical: ArrayLike, fs: float, *, fmin_hz: float=TUNED_DEFAULTS.fmin_hz, fmax_hz: float=TUNED_DEFAULTS.fmax_hz, bands_per_octave: float=TUNED_DEFAULTS.bands_per_octave, overlap_octaves: bool=TUNED_DEFAULTS.overlap_octaves, gamma: float=TUNED_DEFAULTS.gamma, beta: float=TUNED_DEFAULTS.beta, bayes_step_deg: float=TUNED_DEFAULTS.bayes_step_deg, transverse_weight: float=TUNED_DEFAULTS.transverse_weight, retro_sign: str=TUNED_DEFAULTS.retro_sign, retrograde_threshold: float=TUNED_DEFAULTS.retrograde_threshold, target_dt: Optional[float]=TUNED_DEFAULTS.target_dt, combine_max_entropy: float=TUNED_DEFAULTS.combine_max_entropy, combine_min_log_odds_180: float=TUNED_DEFAULTS.combine_min_log_odds_180, combine_stability_sigma: float=TUNED_DEFAULTS.combine_stability_sigma, direction_entropy_max: float=TUNED_DEFAULTS.direction_entropy_max, axis_agreement_sigma: float=TUNED_DEFAULTS.axis_agreement_sigma, axis_agreement_min_weight: float=TUNED_DEFAULTS.axis_agreement_min_weight, true_baz: Optional[float]=None, return_band_details: bool=True) -> Dict[str, object]:
    """Estimate Rayleigh back azimuth using the tuned two-stage Hilbert-Bayes workflow.

    This function follows the configured production workflow:

    1. make overlapping octave/fractional-octave bands;
    2. estimate both axial Hilbert-Bayes and retrograde Hilbert-Bayes posteriors
       in each band;
    3. combine the axial posteriors as a 180-degree symmetric Rayleigh axis;
    4. use retrograde particle-motion votes to choose the 180-degree branch;
    5. optionally combine retrograde posteriors after axis-agreement gating.

    The reported ``best_estimate_deg`` and ``map_baz_deg`` are the stage-2
    branch choice, matching the tuned script's ``combined_result.best_estimate_deg``.
    """
    if fs <= 0.0:
        raise ValueError('fs must be positive')
    if retro_sign not in {'positive', 'negative'}:
        raise ValueError("retro_sign must be 'positive' or 'negative'")
    N0, E0, Z0 = _components_from_arrays(north, east, vertical)
    dt0 = 1.0 / float(fs)
    args = _RayleighArgs(bayes_step=float(bayes_step_deg), transverse_weight=float(transverse_weight), beta=float(beta), retro_sign=str(retro_sign), retrograde_threshold=float(retrograde_threshold), target_dt=target_dt, combine_max_entropy=float(combine_max_entropy), combine_min_log_odds_180=float(combine_min_log_odds_180), combine_stability_sigma=float(combine_stability_sigma), direction_entropy_max=float(direction_entropy_max), axis_agreement_sigma=float(axis_agreement_sigma), axis_agreement_min_weight=float(axis_agreement_min_weight), weight_gamma=float(gamma))
    windows = make_octave_bands(float(fmin_hz), float(fmax_hz), float(bands_per_octave), bool(overlap_octaves))
    if target_dt is not None:
        nyquist_after_downsample = 0.5 / target_dt
        safe_windows = [(fmin, fmax) for fmin, fmax in windows if fmax < 0.4 * nyquist_after_downsample]
        if not safe_windows:
            raise ValueError(f'No frequency bands remain after downsampling with target_dt={target_dt}. Nyquist frequency after downsampling: {nyquist_after_downsample:.4f} Hz. Maximum safe frequency: {0.4 * nyquist_after_downsample:.4f} Hz. Your fmax_hz={fmax_hz} is too high for this target_dt. Either reduce target_dt or reduce fmax_hz to < {0.4 * nyquist_after_downsample:.2f} Hz.')
        if len(safe_windows) < len(windows):
            removed = len(windows) - len(safe_windows)
            print(f'Note: Excluded {removed} high-frequency band(s) (> {0.4 * nyquist_after_downsample:.4f} Hz) that would violate Nyquist after downsampling to target_dt={target_dt}')
        windows = safe_windows
    all_results: List[Dict[str, object]] = []
    for fmin, fmax in windows:
        N1, dt1 = preprocess_for_rayleigh(N0, dt0, target_dt, fmin, fmax)
        E1, _ = preprocess_for_rayleigh(E0, dt0, target_dt, fmin, fmax)
        Z1, _ = preprocess_for_rayleigh(Z0, dt0, target_dt, fmin, fmax)
        N1, E1, Z1 = align_lengths(N1, E1, Z1)
        results = estimate_required_methods(N1, E1, Z1, dt1, fmin, fmax, args, true_baz=true_baz)
        all_results.append({'fmin_hz': fmin, 'fmax_hz': fmax, 'dt': dt1, 'results': results})
    target_grid = _as_grid(bayes_step_deg)
    axis_combined, axis_diag = combine_axis_bands(all_results, target_grid, max_entropy_ratio=combine_max_entropy, stability_sigma_deg=combine_stability_sigma, weight_gamma=gamma, true_baz=true_baz)
    if axis_combined is None:
        raise RuntimeError('No reliable bands survived the stage-1 axial combination')
    axis_deg = float(axis_combined['azimuth_axis_deg_mod_180'])
    axis_antipode = (axis_deg + 180.0) % 360.0
    axis_std = float(axis_combined['posterior_axial_std_deg'])
    reliable = [b for b in all_results if b['results']['hilbert_bayes'].get('posterior_entropy_ratio', 1.0) < direction_entropy_max]
    reliable.sort(key=lambda b: band_center_frequency(b['fmin_hz'], b['fmax_hz']))
    votes0 = 0
    votes1 = 0
    ellipticities: List[float] = []
    band_votes: List[Dict[str, object]] = []
    for b in reliable:
        fmin = float(b['fmin_hz'])
        fmax = float(b['fmax_hz'])
        N2, dt2 = preprocess_for_rayleigh(N0, dt0, target_dt, fmin, fmax)
        E2, _ = preprocess_for_rayleigh(E0, dt0, target_dt, fmin, fmax)
        Z2, _ = preprocess_for_rayleigh(Z0, dt0, target_dt, fmin, fmax)
        N2, E2, Z2 = align_lengths(N2, E2, Z2)
        th = np.deg2rad(axis_deg)
        R = np.cos(th) * N2 + np.sin(th) * E2
        ellipt, classification, diag = verify_retrograde_motion(R, Z2, threshold=retrograde_threshold)
        vote = {'fmin_hz': fmin, 'fmax_hz': fmax, 'fcenter_hz': band_center_frequency(fmin, fmax), 'ellipticity': float(ellipt), 'classification': classification, 'diagnostics': diag}
        if classification == 'retrograde':
            votes0 += 1
            ellipticities.append(float(ellipt))
            vote['votes_for'] = 'branch_0'
        elif classification == 'prograde':
            votes1 += 1
            ellipticities.append(float(-ellipt))
            vote['votes_for'] = 'branch_1'
        else:
            vote['votes_for'] = 'ambiguous'
        band_votes.append(vote)
    total = votes0 + votes1
    vote_fraction = None if total == 0 else votes0 / total
    catalog_error = None
    if total and vote_fraction >= 0.65:
        chosen_baz = axis_deg
        status = 'independent'
    elif total and vote_fraction <= 0.35:
        chosen_baz = axis_antipode
        status = 'independent'
    elif true_baz is not None:
        d0 = circular_difference_deg(axis_deg, true_baz)
        d1 = circular_difference_deg(axis_antipode, true_baz)
        chosen_baz = axis_deg if d0 < d1 else axis_antipode
        status = 'catalog_assisted'
        catalog_error = min(d0, d1)
    else:
        chosen_baz = axis_deg
        status = 'ambiguous_defaulted_to_branch_0' if total == 0 else 'ambiguous_split_vote'
    retro_combined, retro_diag = combine_retro_bands(all_results, target_grid, chosen_baz, max_entropy_ratio=combine_max_entropy, min_log_odds_180=combine_min_log_odds_180, stability_sigma_deg=combine_stability_sigma, axis_agreement_sigma_deg=axis_agreement_sigma, axis_agreement_min_weight=axis_agreement_min_weight, weight_gamma=gamma, true_baz=true_baz)
    posterior_source = retro_combined if retro_combined is not None else axis_combined
    posterior_kind = 'retro_gated' if retro_combined is not None else 'stage1_axis'
    azimuth_range = np.asarray(posterior_source['azimuth_range'], dtype=float)
    posterior = np.asarray(posterior_source['posterior'], dtype=float)
    out: Dict[str, object] = {'azimuth_deg': azimuth_range, 'azimuth_range': azimuth_range, 'posterior': posterior, 'best_estimate_deg': float(chosen_baz) % 360.0, 'map_baz_deg': float(chosen_baz) % 360.0, 'posterior_map_deg': float(posterior_source.get('azimuth', chosen_baz)) % 360.0, 'mean_baz_deg': float(posterior_source.get('posterior_mean_azimuth', np.nan)) % 360.0, 'posterior_std_deg': float(posterior_source.get('posterior_std_deg', np.nan)), 'axis_deg_mod_180': float(axis_deg), 'axis_antipode_deg': float(axis_antipode), 'axis_uncertainty_deg': float(axis_std), 'direction_status': status, 'reliability': 'high' if status == 'independent' and axis_std < 30.0 else 'medium' if status in ('independent', 'catalog_assisted') else 'low', 'votes_branch_0': int(votes0), 'votes_branch_1': int(votes1), 'tested_bands': int(len(reliable)), 'vote_fraction_branch_0': None if vote_fraction is None else float(vote_fraction), 'ellipticity_median': float(np.median(ellipticities)) if ellipticities else None, 'posterior_kind': posterior_kind, 'stage1_axis': axis_combined, 'retro_gated': retro_combined, 'defaults': TUNED_DEFAULTS.__dict__.copy(), 'settings': {'fmin_hz': float(fmin_hz), 'fmax_hz': float(fmax_hz), 'bands_per_octave': float(bands_per_octave), 'overlap_octaves': bool(overlap_octaves), 'gamma': float(gamma), 'beta': float(beta), 'bayes_step_deg': float(bayes_step_deg), 'transverse_weight': float(transverse_weight), 'retro_sign': retro_sign, 'retrograde_threshold': float(retrograde_threshold), 'target_dt': target_dt}, 'band_votes': band_votes, 'bands': [{'fmin_hz': float(b['fmin_hz']), 'fmax_hz': float(b['fmax_hz']), 'center_hz': band_center_frequency(b['fmin_hz'], b['fmax_hz']), 'axis_map_deg': float(b['results']['hilbert_bayes']['azimuth']), 'retro_map_deg': float(b['results']['hilbert_bayes_retro']['azimuth']), 'axis_entropy_ratio': float(b['results']['hilbert_bayes'].get('posterior_entropy_ratio', np.nan)), 'retro_entropy_ratio': float(b['results']['hilbert_bayes_retro'].get('posterior_entropy_ratio', np.nan)), 'axis_log_odds_180': float(b['results']['hilbert_bayes'].get('log_odds_map_vs_180deg_flip', np.nan)), 'retro_log_odds_180': float(b['results']['hilbert_bayes_retro'].get('log_odds_map_vs_180deg_flip', np.nan))} for b in all_results]}
    if catalog_error is not None:
        out['catalog_error_deg'] = float(catalog_error)
    if true_baz is not None:
        out['truth_baz_deg'] = float(true_baz)
        out['error_to_truth_deg'] = circular_difference_deg(chosen_baz, true_baz)
        out['axis_error_to_truth_deg'] = axial_difference_deg(axis_deg, true_baz)
    if return_band_details:
        out['axis_band_diagnostics'] = _diagnostics_for_public(axis_diag)
        out['retro_band_diagnostics'] = _diagnostics_for_public(retro_diag)
    return out

__all__ = [
    "RayleighBazDefaults",
    "TUNED_DEFAULTS",
    "estimate_rayleigh_baz_posterior",
    "bandpass_filter",
    "preprocess_for_rayleigh",
    "align_lengths",
    "make_octave_bands",
    "band_center_frequency",
    "circular_difference_deg",
]
