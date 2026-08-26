#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rayleigh filtering utilities.

This module provides time-domain and Stockwell-domain Rayleigh filters,
Particleman-style normalized inner product masks used by the orbit detection workflow.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import detrend
from scipy.signal import hilbert as _hilbert

from .rayleigh_backazimuth import (
    ArrayLike,
    _components_from_arrays,
    align_lengths,
    bandpass_filter,
    estimate_rayleigh_baz_posterior,
    preprocess_for_rayleigh,
)


def _resolve_baz_deg(north: np.ndarray, east: np.ndarray, vertical: np.ndarray, fs: float, baz_deg: Optional[float], posterior_result: Optional[Dict[str, object]]) -> float:
    """Return an explicit BAZ, using a posterior result only as a convenience."""
    if baz_deg is None:
        if posterior_result is None:
            posterior_result = estimate_rayleigh_baz_posterior(north, east, vertical, fs)
        baz_deg = posterior_result.get('best_estimate_deg', posterior_result.get('map_baz_deg'))
        if baz_deg is None:
            raise ValueError('posterior_result must contain best_estimate_deg or map_baz_deg')
    return float(baz_deg) % 360.0


def _validate_filter_args(fs: float, sense: str, envelope_component: str) -> None:
    if fs <= 0.0:
        raise ValueError('fs must be positive')
    if sense not in {'retro', 'pro', 'both'}:
        raise ValueError("sense must be 'retro', 'pro', or 'both'")
    if envelope_component not in {'rz', 'z', 'r'}:
        raise ValueError("envelope_component must be 'rz', 'z', or 'r'")


def _rotate_ne_to_rt(N: np.ndarray, E: np.ndarray, baz_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    th = np.deg2rad(float(baz_deg) % 360.0)
    R = np.cos(th) * N + np.sin(th) * E
    T = -np.sin(th) * N + np.cos(th) * E
    return (R, T)


def _rotate_rt_to_ne(R: np.ndarray, T: np.ndarray, baz_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    th = np.deg2rad(float(baz_deg) % 360.0)
    N = np.cos(th) * R - np.sin(th) * T
    E = np.sin(th) * R + np.cos(th) * T
    return (N, E)


def _component_envelope(R: np.ndarray, Z: np.ndarray, envelope_component: str) -> np.ndarray:
    if envelope_component == 'rz':
        return np.sqrt(np.abs(_hilbert(R)) ** 2 + np.abs(_hilbert(Z)) ** 2)
    if envelope_component == 'z':
        return np.abs(_hilbert(Z))
    if envelope_component == 'r':
        return np.abs(_hilbert(R))
    raise ValueError("envelope_component must be 'rz', 'z', or 'r'")


def _safe_corr(a: np.ndarray, b: np.ndarray, eps: float=1e-14) -> float:
    ac = np.asarray(a, dtype=float) - np.mean(a)
    bc = np.asarray(b, dtype=float) - np.mean(b)
    return float(np.dot(ac, bc) / (np.linalg.norm(ac) * np.linalg.norm(bc) + eps))


def simple_elliptic_filter_rayleigh(north: ArrayLike, east: ArrayLike, vertical: ArrayLike, fs: float, *, baz_deg: Optional[float]=None, posterior_result: Optional[Dict[str, object]]=None, fmin_hz: float=0.01, fmax_hz: float=0.1, sense: str='retro', corners: int=4, envelope_component: str='rz', eps: float=1e-14) -> Dict[str, object]:
    """Simple time-domain elliptic Rayleigh filter.

    This filter bandpasses the three components, rotates
    N/E to radial/transverse with the supplied BAZ, projects the R-Z pair onto
    the requested quadrature sense, and suppresses the transverse component.

    Convention
    ----------
    ``sense='retro'`` assumes the same convention as the BAZ estimator: radial
    motion is positively correlated with ``Hilbert(Z)``. ``sense='pro'`` uses
    the opposite sign. ``sense='both'`` returns the bandpassed R-Z sagittal-plane
    motion with transverse set to zero.
    """
    _validate_filter_args(float(fs), sense, envelope_component)
    N0, E0, Z0 = _components_from_arrays(north, east, vertical)
    dt = 1.0 / float(fs)
    baz = _resolve_baz_deg(N0, E0, Z0, fs, baz_deg, posterior_result)
    Nf = bandpass_filter(detrend(N0), dt, fmin_hz, fmax_hz, corners=corners)
    Ef = bandpass_filter(detrend(E0), dt, fmin_hz, fmax_hz, corners=corners)
    Zf = bandpass_filter(detrend(Z0), dt, fmin_hz, fmax_hz, corners=corners)
    Nf, Ef, Zf = align_lengths(Nf, Ef, Zf)
    Rf, Tf = _rotate_ne_to_rt(Nf, Ef, baz)

    def project(sgn: float) -> Tuple[np.ndarray, np.ndarray]:
        HR = np.imag(_hilbert(Rf))
        HZ = np.imag(_hilbert(Zf))
        Rq = 0.5 * (Rf + sgn * HZ)
        Zq = 0.5 * (Zf - sgn * HR)
        return (Rq, Zq)
    if sense == 'retro':
        Rr, Zr = project(+1.0)
    elif sense == 'pro':
        Rr, Zr = project(-1.0)
    else:
        Rretro, Zretro = project(+1.0)
        Rpro, Zpro = project(-1.0)
        Rr = Rretro + Rpro
        Zr = Zretro + Zpro
    Tr = np.zeros_like(Rr)
    Nr, Er = _rotate_rt_to_ne(Rr, Tr, baz)
    envelope = _component_envelope(Rr, Zr, envelope_component)
    HZf = np.imag(_hilbert(Zf))
    corr_r_hz = _safe_corr(Rf, HZf, eps=eps)
    transverse_ratio = float(np.std(Tf) / (np.sqrt(np.std(Rf) ** 2 + np.std(Zf) ** 2) + eps))
    return {'method': 'simple_elliptic', 'baz_deg': baz, 'sense': sense, 'fmin_hz': float(fmin_hz), 'fmax_hz': float(fmax_hz), 'north_rayleigh': Nr, 'east_rayleigh': Er, 'vertical_rayleigh': Zr, 'radial_rayleigh': Rr, 'transverse_rayleigh': Tr, 'rayleigh': (Nr, Er, Zr), 'rayleigh_envelope': envelope, 'input_bandpassed': {'north': Nf, 'east': Ef, 'vertical': Zf, 'radial': Rf, 'transverse': Tf}, 'diag': {'rz_hilbert_corr': corr_r_hz, 'transverse_ratio_before_filter': transverse_ratio, 'quadrature_convention': 'retro: R positively correlated with Hilbert(Z)', 'zero_transverse': True}}


def _discrete_stockwell(x: np.ndarray, dt: float, fmin: float, fmax: float, *, remove_mean: bool=True, width: float=1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discrete Stockwell transform on positive FFT bins in [fmin, fmax].

    The implementation follows the efficient frequency-domain construction used by the Stockwell filtering workflow: each row is an inverse FFT of a shifted spectrum
    multiplied by a frequency-dependent Gaussian. ``width`` scales the Gaussian
    width. The default value reproduces the earlier implementation.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        raise ValueError('Stockwell transform requires at least 4 samples')
    if width <= 0.0:
        raise ValueError('width must be positive')
    mean = float(np.mean(x)) if remove_mean else 0.0
    X = fft(x - mean)
    df = 1.0 / (n * dt)
    k_all = np.arange(1, n // 2) if n % 2 == 0 else np.arange(1, (n + 1) // 2)
    f_all = k_all * df
    keep = (f_all >= fmin) & (f_all <= fmax)
    k_pos = k_all[keep]
    freqs = f_all[keep]
    if len(k_pos) == 0:
        raise ValueError('No FFT bins inside requested frequency band')
    alpha = (np.fft.fftfreq(n) * n).astype(int)
    alpha_f = alpha.astype(float)
    S = np.empty((len(k_pos), n), dtype=np.complex128)
    for i, k in enumerate(k_pos):
        idx = (k + alpha) % n
        kw = max(float(k) * float(width), 1e-14)
        G = np.exp(-2.0 * np.pi ** 2 * alpha_f ** 2 / kw ** 2)
        S[i, :] = ifft(X[idx] * G)
    return (S, freqs, k_pos)


def _inverse_discrete_stockwell(S: np.ndarray, k_pos: np.ndarray, n: int) -> np.ndarray:
    """Reconstruct a real time series from positive-frequency S rows."""
    Xf = np.zeros(n, dtype=complex)
    for row, k in enumerate(k_pos):
        Xk = np.sum(S[row, :])
        Xf[k] = Xk
        Xf[-k % n] = np.conj(Xk)
    return np.real(ifft(Xf))


def _cdot(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.real(A) * np.real(B) + np.imag(A) * np.imag(B)


def _cnorm(A: np.ndarray) -> np.ndarray:
    return np.sqrt(np.real(A) ** 2 + np.imag(A) ** 2)


def _stockwell_nezrt(north: ArrayLike, east: ArrayLike, vertical: ArrayLike, fs: float, baz_deg: Optional[float], posterior_result: Optional[Dict[str, object]], fmin_hz: float, fmax_hz: float, stockwell_width: float, target_dt: Optional[float]=None) -> Dict[str, object]:
    N0, E0, Z0 = _components_from_arrays(north, east, vertical)
    dt = 1.0 / float(fs)
    baz = _resolve_baz_deg(N0, E0, Z0, fs, baz_deg, posterior_result)
    if target_dt is not None and target_dt > dt:
        N, dt_used = preprocess_for_rayleigh(N0, dt, target_dt, fmin_hz, fmax_hz)
        E, _ = preprocess_for_rayleigh(E0, dt, target_dt, fmin_hz, fmax_hz)
        Z, _ = preprocess_for_rayleigh(Z0, dt, target_dt, fmin_hz, fmax_hz)
        N, E, Z = align_lengths(N, E, Z)
    else:
        dt_used = dt
        N = detrend(N0)
        E = detrend(E0)
        Z = detrend(Z0)
    SN, freqs, k_pos = _discrete_stockwell(N, dt_used, fmin_hz, fmax_hz, width=stockwell_width)
    SE, _, _ = _discrete_stockwell(E, dt_used, fmin_hz, fmax_hz, width=stockwell_width)
    SZ, _, _ = _discrete_stockwell(Z, dt_used, fmin_hz, fmax_hz, width=stockwell_width)
    th = np.deg2rad(baz)
    R = np.cos(th) * N + np.sin(th) * E
    T = -np.sin(th) * N + np.cos(th) * E
    SR = np.cos(th) * SN + np.sin(th) * SE
    ST = -np.sin(th) * SN + np.cos(th) * SE
    speedup_factor = dt_used / dt if target_dt and dt_used > dt else 1.0
    return {
        'N': N, 'E': E, 'Z': Z, 'R': R, 'T': T,
        'SN': SN, 'SE': SE, 'SZ': SZ, 'SR': SR, 'ST': ST,
        'freqs': freqs, 'k_pos': k_pos, 'n': len(N), 'dt': dt_used,
        'baz_deg': baz, 'original_dt': dt,
        'downsampled': target_dt is not None and target_dt > dt,
        'speedup_factor': speedup_factor,
        'effective_sample_rate_hz': 1.0 / dt_used,
        'time_array': np.arange(len(N), dtype=float) * dt_used,
    }


def _stockwell_reconstruct_from_rt(SR: np.ndarray, ST: np.ndarray, SZ: np.ndarray, k_pos: np.ndarray, n: int, baz_deg: float, zero_transverse: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    R = _inverse_discrete_stockwell(SR, k_pos, n)
    if zero_transverse:
        T = np.zeros_like(R)
    else:
        T = _inverse_discrete_stockwell(ST, k_pos, n)
    Z = _inverse_discrete_stockwell(SZ, k_pos, n)
    N, E = _rotate_rt_to_ne(R, T, baz_deg)
    return (N, E, Z, R, T)


def _sense_masks_from_nip(SR: np.ndarray, SZ: np.ndarray, eps: float) -> Dict[str, np.ndarray]:
    """
    Compute normalized inner product for retrograde/prograde discrimination.

    Particleman convention:
      retro: +1j * SZ
      pro:   -1j * SZ

    Return nonnegative NIP gates for the Stockwell filter.
    """
    masks: Dict[str, np.ndarray] = {}

    for name, phase in {'retro': 1j, 'pro': -1j}.items():
        SZs = phase * SZ
        nip_raw = _cdot(SR, SZs) / (_cnorm(SR) * _cnorm(SZs) + eps)
        masks[name] = np.maximum(np.clip(nip_raw, -1.0, 1.0), 0.0)

    return masks


def particleman_style_nip(S_R, S_Z, eps=0.04):
    """
    Particleman-style normalized inner product for R-Z quadrature.

    eps is relative to max vertical Stockwell amplitude, not an absolute floor.
    Returns signed NIP arrays in [-1, 1].
    """
    nips = {}
    phases = {'retro': 1j, 'pro': -1j}
    for sense, phase in phases.items():
        S_Z_shifted = phase * S_Z
        amp_R = np.abs(S_R)
        amp_Z = np.abs(S_Z_shifted).copy()
        zmax = np.nanmax(amp_Z)
        if eps is not None and zmax > 0:
            low_amp = amp_Z / zmax < eps
            amp_Z[low_amp] += eps * zmax
        numerator = np.real(S_R) * np.real(S_Z_shifted) + np.imag(S_R) * np.imag(S_Z_shifted)
        denom = amp_R * amp_Z
        nip = np.zeros_like(numerator, dtype=float)
        good = denom > 0
        nip[good] = numerator[good] / denom[good]
        nips[sense] = np.clip(nip, -1.0, 1.0)
    return nips


def particleman_filter_from_nip(nip, threshold=0.8, width=0.1):
    """
    Smooth Particleman-style NIP filter.

    0 below threshold-width,
    cosine taper from threshold-width to threshold,
    1 above threshold.
    """
    filt = np.zeros_like(nip, dtype=float)
    mid = (nip > threshold - width) & (nip < threshold)
    high = nip >= threshold
    filt[mid] = 0.5 * np.cos(np.pi * (nip[mid] - threshold) / width) + 0.5
    filt[high] = 1.0
    return filt


def _smooth_gate_above(x, threshold, width):
    """Return a smooth gate that is one above threshold and zero below threshold-width."""
    x = np.asarray(x, dtype=float)
    if threshold is None or float(threshold) <= 0.0:
        return np.ones_like(x, dtype=float)
    threshold = float(threshold)
    width = max(float(width), 1e-12)
    gate = np.zeros_like(x, dtype=float)
    high = x >= threshold
    mid = (x > threshold - width) & (x < threshold)
    gate[high] = 1.0
    gate[mid] = 0.5 - 0.5 * np.cos(np.pi * (x[mid] - (threshold - width)) / width)
    return gate


def _smooth_gate_below(x, threshold, width):
    """Return a smooth gate that is one below threshold and zero above threshold+width."""
    x = np.asarray(x, dtype=float)
    if threshold is None:
        return np.ones_like(x, dtype=float)
    threshold = float(threshold)
    width = max(float(width), 1e-12)
    gate = np.zeros_like(x, dtype=float)
    low = x <= threshold
    mid = (x > threshold) & (x < threshold + width)
    gate[low] = 1.0
    gate[mid] = 0.5 + 0.5 * np.cos(np.pi * (x[mid] - threshold) / width)
    return gate


def _particleman_energy_gate(raw_energy, floor=0.0, width=0.02, mode='global'):
    """Smooth amplitude/energy gate for Particleman masks.

    mode='global' normalizes by the global maximum.
    mode='per_frequency' normalizes each frequency row by its own maximum so
    weak but coherent long-period energy can survive while low-amplitude random
    high-NIP pixels are suppressed.
    """
    if floor is None or float(floor) <= 0.0:
        return np.ones_like(raw_energy, dtype=float), np.ones_like(raw_energy, dtype=float)
    raw_energy = np.asarray(raw_energy, dtype=float)
    mode = str(mode).lower()
    if mode in {'per_frequency', 'per_freq', 'frequency'}:
        denom = np.nanmax(raw_energy, axis=1, keepdims=True)
    elif mode == 'global':
        denom = np.nanmax(raw_energy)
    else:
        raise ValueError("energy_floor_mode must be 'global' or 'per_frequency'")
    energy_norm = raw_energy / (denom + 1e-300)
    return _smooth_gate_above(energy_norm, float(floor), float(width)), energy_norm


def add_particleman_nip_energy_masks(
    result_dict,
    threshold=0.8,
    width=0.1,
    eps=0.04,
    *,
    energy_floor=0.0,
    energy_width=0.02,
    energy_floor_mode='global',
    transverse_max=None,
    transverse_width=0.15,
    polarization_mode='elliptic',
):
    """
    Replace any existing masks with Particleman-style NIP masks.

    The original Particleman gate used only R-Z quadrature/NIP.  That is useful
    but often too permissive because low-amplitude noise can have apparently
    high NIP.  The optional energy and transverse gates make the filter more
    selective while preserving the same downstream convention: masks are stored
    as sqrt(filter), because the orbit detector multiplies complex coefficients
    by mask and then squares.

    Parameters
    ----------
    threshold, width
        NIP threshold and taper width.
    energy_floor, energy_width, energy_floor_mode
        Optional smooth energy gate.  For real data, ``energy_floor_mode='per_frequency'``
        is usually the safest aggressive setting because each period is compared
        with itself rather than with the global R1 maximum.
    transverse_max, transverse_width
        Optional smooth gate on |S_T|^2 / (|S_R|^2 + |S_Z|^2).  Use this only
        when the BAZ is reliable enough that Rayleigh energy should be mostly
        radial-vertical.
    """
    S_R = result_dict.get('S_R_raw', result_dict.get('S_R'))
    S_Z = result_dict.get('S_Z_raw', result_dict.get('S_Z'))
    S_T = result_dict.get('S_T_raw', result_dict.get('S_T'))
    if S_R is None or S_Z is None:
        raise ValueError('result_dict must contain S_R/S_Z or S_R_raw/S_Z_raw')

    nips = particleman_style_nip(S_R, S_Z, eps=eps)
    retro_nip_filter = particleman_filter_from_nip(nips['retro'], threshold=threshold, width=width)
    pro_nip_filter = particleman_filter_from_nip(nips['pro'], threshold=threshold, width=width)

    raw_energy = np.abs(S_R) ** 2 + np.abs(S_Z) ** 2
    energy_gate, energy_norm = _particleman_energy_gate(
        raw_energy,
        floor=energy_floor,
        width=energy_width,
        mode=energy_floor_mode,
    )

    if transverse_max is not None and S_T is not None:
        transverse_ratio = np.abs(S_T) ** 2 / (raw_energy + 1e-300)
        transverse_gate = _smooth_gate_below(transverse_ratio, transverse_max, transverse_width)
    else:
        transverse_ratio = None
        transverse_gate = np.ones_like(raw_energy, dtype=float)

    shared_gate = energy_gate * transverse_gate
    retro_filter = retro_nip_filter * shared_gate
    pro_filter = pro_nip_filter * shared_gate

    # Any elliptical-polarization gate:
    # pass a pixel if either retrograde-like or prograde-like R-Z quadrature passes.
    elliptic_filter = np.maximum(retro_filter, pro_filter)

    polarization_mode = str(polarization_mode).lower()
    if polarization_mode in {'elliptic', 'either', 'both', 'any'}:
        selected_filter = elliptic_filter
        selected_mask = np.sqrt(np.clip(selected_filter, 0.0, 1.0))

        # Store retro/pro as the same elliptic mask so orbit labels remain
        # independent of the polarization-sense split.
        result_dict['masks'] = {
            'retro': selected_mask,
            'pro': selected_mask,
            'elliptic': selected_mask,
        }
        result_dict['mask'] = selected_mask

        # Keep separated products for diagnostics, while the default orbit
        # energy products use the elliptic gate.
        result_dict['S_energy_retro_nip_sense'] = raw_energy * retro_filter
        result_dict['S_energy_pro_nip_sense'] = raw_energy * pro_filter

        result_dict['S_energy_retro_nip'] = raw_energy * selected_filter
        result_dict['S_energy_pro_nip'] = raw_energy * selected_filter
        result_dict['S_energy_elliptic_nip'] = raw_energy * selected_filter

    elif polarization_mode in {'separate', 'sense', 'sense_separated'}:
        result_dict['masks'] = {
            'retro': np.sqrt(np.clip(retro_filter, 0.0, 1.0)),
            'pro': np.sqrt(np.clip(pro_filter, 0.0, 1.0)),
            'elliptic': np.sqrt(np.clip(elliptic_filter, 0.0, 1.0)),
        }
        result_dict['mask'] = np.maximum(result_dict['masks']['retro'], result_dict['masks']['pro'])

        result_dict['S_energy_retro_nip'] = raw_energy * retro_filter
        result_dict['S_energy_pro_nip'] = raw_energy * pro_filter
        result_dict['S_energy_elliptic_nip'] = raw_energy * elliptic_filter

    else:
        raise ValueError(
            "polarization_mode must be 'elliptic'/'either'/'both'/'any' "
            "or 'separate'/'sense'/'sense_separated'"
        )

    result_dict['S_energy_raw'] = raw_energy
    result_dict['S_R_filtered'] = result_dict['mask'] * S_R
    result_dict['S_Z_filtered'] = result_dict['mask'] * S_Z
    if S_T is not None:
        result_dict['S_T_filtered'] = result_dict['mask'] * S_T
    result_dict['S_energy_filtered'] = (
        np.abs(result_dict['S_R_filtered']) ** 2
        + np.abs(result_dict['S_Z_filtered']) ** 2
    )
    result_dict['polarization_mode'] = polarization_mode

    # Keep time-domain diagnostic outputs consistent with the selected mask.
    if 'k_pos' in result_dict and 'baz_deg' in result_dict:
        try:
            k_pos = np.asarray(result_dict['k_pos'])
            n = int(S_R.shape[1])
            baz = float(result_dict['baz_deg'])
            selected_mask = result_dict['mask']

            if S_T is None:
                S_T_for_recon = np.zeros_like(S_R)
            else:
                S_T_for_recon = S_T

            Nf, Ef, Zf, Rf, Tf = _stockwell_reconstruct_from_rt(
                selected_mask * S_R,
                selected_mask * S_T_for_recon,
                selected_mask * S_Z,
                k_pos,
                n,
                baz,
                zero_transverse=False,
            )

            result_dict['north_rayleigh'] = Nf
            result_dict['east_rayleigh'] = Ef
            result_dict['vertical_rayleigh'] = Zf
            result_dict['radial_rayleigh'] = Rf
            result_dict['transverse_rayleigh'] = Tf
            result_dict['rayleigh'] = (Nf, Ef, Zf)
            result_dict['rayleigh_envelope'] = _component_envelope(Rf, Zf, 'rz')
        except Exception as exc:
            result_dict.setdefault('diag', {})
            result_dict['diag']['particleman_reconstruction_error'] = repr(exc)

    diag_update = {
        'particleman_polarization_mode': polarization_mode,
        'particleman_retro_nip': nips['retro'],
        'particleman_pro_nip': nips['pro'],
        'particleman_retro_nip_filter': retro_nip_filter,
        'particleman_pro_nip_filter': pro_nip_filter,
        'particleman_energy_gate': energy_gate,
        'particleman_energy_norm': energy_norm,
        'particleman_transverse_gate': transverse_gate,
        'particleman_retro_filter': retro_filter,
        'particleman_pro_filter': pro_filter,
        'particleman_elliptic_filter': elliptic_filter,
        'particleman_threshold': threshold,
        'particleman_width': width,
        'particleman_eps': eps,
        'particleman_energy_floor': energy_floor,
        'particleman_energy_width': energy_width,
        'particleman_energy_floor_mode': energy_floor_mode,
        'particleman_transverse_max': transverse_max,
        'particleman_transverse_width': transverse_width,
    }
    if transverse_ratio is not None:
        diag_update['particleman_transverse_ratio'] = transverse_ratio
    result_dict['diag'] = {**result_dict.get('diag', {}), **diag_update}
    return result_dict



def _sum_sense_outputs(outputs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], sense: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if sense == 'both':
        return tuple((outputs['retro'][i] + outputs['pro'][i] for i in range(5)))
    return outputs[sense]


def _pack_filter_output(method: str, baz: float, sense: str, fmin_hz: float, fmax_hz: float, freqs: np.ndarray, k_pos: np.ndarray, rayleigh_rt_nez: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], envelope_component: str, masks: Dict[str, np.ndarray], diag: Dict[str, object], extra: Optional[Dict[str, object]]=None, stockwell_transforms: Optional[Dict[str, np.ndarray]]=None) -> Dict[str, object]:
    Nr, Er, Zr, Rr, Tr = rayleigh_rt_nez
    selected_mask = np.maximum(masks['retro'], masks['pro']) if sense == 'both' else masks[sense]
    selected_mask = np.clip(selected_mask, 0.0, 1.0)
    out: Dict[str, object] = {'method': method, 'baz_deg': baz, 'sense': sense, 'fmin_hz': float(fmin_hz), 'fmax_hz': float(fmax_hz), 'freqs': freqs, 'k_pos': k_pos, 'north_rayleigh': Nr, 'east_rayleigh': Er, 'vertical_rayleigh': Zr, 'radial_rayleigh': Rr, 'transverse_rayleigh': Tr, 'rayleigh': (Nr, Er, Zr), 'rayleigh_envelope': _component_envelope(Rr, Zr, envelope_component), 'mask': selected_mask, 'masks': masks, 'diag': diag}
    if stockwell_transforms is not None:
        raw_SR = stockwell_transforms['SR']
        raw_SZ = stockwell_transforms['SZ']
        raw_SN = stockwell_transforms.get('SN')
        raw_SE = stockwell_transforms.get('SE')
        raw_ST = stockwell_transforms.get('ST')
        out['S_R'] = raw_SR
        out['S_Z'] = raw_SZ
        out['S_N'] = raw_SN
        out['S_E'] = raw_SE
        out['S_T'] = raw_ST
        out['S_R_raw'] = raw_SR
        out['S_Z_raw'] = raw_SZ
        out['S_N_raw'] = raw_SN
        out['S_E_raw'] = raw_SE
        out['S_T_raw'] = raw_ST
        out['S_R_filtered'] = selected_mask * raw_SR
        out['S_Z_filtered'] = selected_mask * raw_SZ
        out['S_N_filtered'] = selected_mask * raw_SN if raw_SN is not None else None
        out['S_E_filtered'] = selected_mask * raw_SE if raw_SE is not None else None
        out['S_T_filtered'] = selected_mask * raw_ST if raw_ST is not None else None
        out['S_energy_raw'] = np.abs(raw_SR) ** 2 + np.abs(raw_SZ) ** 2
        out['S_energy_filtered'] = np.abs(out['S_R_filtered']) ** 2 + np.abs(out['S_Z_filtered']) ** 2

        # Time-domain inputs used to build the transform.  These are useful for
        # diagnostics and are on the Stockwell output sample grid, after any
        # anti-aliased downsampling requested by target_dt.
        out['input_north'] = stockwell_transforms.get('N')
        out['input_east'] = stockwell_transforms.get('E')
        out['input_vertical'] = stockwell_transforms.get('Z')
        out['input_radial'] = stockwell_transforms.get('R')
        out['input_transverse'] = stockwell_transforms.get('T')
        out['output_dt'] = float(stockwell_transforms.get('dt'))
        out['dt'] = float(stockwell_transforms.get('dt'))
        out['original_dt'] = float(stockwell_transforms.get('original_dt'))
        out['downsampled'] = bool(stockwell_transforms.get('downsampled', False))
        out['speedup_factor'] = float(stockwell_transforms.get('speedup_factor', 1.0))
        out['effective_sample_rate_hz'] = float(stockwell_transforms.get('effective_sample_rate_hz'))
        out['time_array'] = np.asarray(stockwell_transforms.get('time_array'), dtype=float)
    if extra:
        out.update(extra)
    return out


def simple_stockwell_filter_rayleigh(north: ArrayLike, east: ArrayLike, vertical: ArrayLike, 
                                     fs: float, *, baz_deg: Optional[float]=None, 
                                     posterior_result: Optional[Dict[str, object]]=None, 
                                     fmin_hz: float=0.01, fmax_hz: float=0.1, sense: str='retro', 
                                     quadrature_min: float=0.0, hard_mask: bool=False, 
                                     amp_min: float=0.0, zero_transverse: bool=True, 
                                     stockwell_width: float=1.0, envelope_component: str='rz', 
                                     target_dt: Optional[float]=None, eps: float=1e-14) -> Dict[str, object]:
    """Simple Stockwell Rayleigh filter.

    This Stockwell-domain filter uses local R-Z
    quadrature, plus an optional normalized amplitude floor. It omits reciprocal ellipticity and transverse rejection so the mask remains
    straightforward to interpret and tune.
    
    Parameters
    ----------
    target_dt : float, optional
        Target sampling interval for downsampling before Stockwell transform.
        For example, if your data is at 20 Hz (dt=0.05s) but you're filtering
        0.01-0.10 Hz, you could use target_dt=1.0 (1 Hz) for ~20x speedup.
        Rule of thumb: target_dt <= 1/(4*fmax_hz) to satisfy Nyquist.
    """
    _validate_filter_args(float(fs), sense, envelope_component)
    st = _stockwell_nezrt(north, east, vertical, fs, baz_deg, posterior_result, fmin_hz, fmax_hz, stockwell_width, target_dt)
    SR = st['SR']
    ST = st['ST']
    SZ = st['SZ']
    k_pos = st['k_pos']
    n = int(st['n'])
    baz = float(st['baz_deg'])
    nips = _sense_masks_from_nip(SR, SZ, eps)
    energy = np.abs(SR) ** 2 + np.abs(SZ) ** 2
    energy_norm = energy / (np.max(energy) + eps)
    amp_gate = (energy_norm >= float(amp_min)).astype(float) if amp_min > 0.0 else np.ones_like(energy_norm)
    masks: Dict[str, np.ndarray] = {}
    outputs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in ('retro', 'pro'):
        nip = nips[name]
        if hard_mask:
            M = (nip >= quadrature_min).astype(float)
        else:
            denom = max(1.0 - float(quadrature_min), eps)
            M = np.clip((nip - float(quadrature_min)) / denom, 0.0, 1.0)
        M = M * amp_gate
        masks[name] = M
        outputs[name] = _stockwell_reconstruct_from_rt(M * SR, M * ST, M * SZ, k_pos, n, baz, zero_transverse)
    rayleigh = _sum_sense_outputs(outputs, sense)
    diag = {'retro_nip': nips['retro'], 'pro_nip': nips['pro'], 'energy_norm': energy_norm, 'quadrature_min': float(quadrature_min), 'hard_mask': bool(hard_mask), 'amp_min': float(amp_min), 'zero_transverse': bool(zero_transverse), 'stockwell_width': float(stockwell_width)}
    return _pack_filter_output('simple_stockwell', baz, sense, fmin_hz, fmax_hz, st['freqs'], k_pos, rayleigh, envelope_component, masks, diag, extra={'retro': outputs['retro'][:3], 'pro': outputs['pro'][:3]}, stockwell_transforms=st)



def plot_rayleigh_filter_diagnostics(*args, **kwargs):
    """Create Rayleigh filter diagnostic plots.

    This compatibility wrapper delegates to :mod:`rayleigh_wave_tools.plotting`.
    New code should import the function from ``rayleigh_wave_tools.plotting``.
    """
    from .plotting import plot_rayleigh_filter_diagnostics as _plot

    return _plot(*args, **kwargs)

elliptical_filter_rayleigh = simple_elliptic_filter_rayleigh

__all__ = [
    "simple_elliptic_filter_rayleigh",
    "elliptical_filter_rayleigh",
    "simple_stockwell_filter_rayleigh",
    "particleman_style_nip",
    "particleman_filter_from_nip",
    "add_particleman_nip_energy_masks",
    "plot_rayleigh_filter_diagnostics",
]
