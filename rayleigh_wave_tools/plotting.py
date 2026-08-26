#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plotting utilities for Rayleigh-wave package diagnostics.

The computational modules keep plotting out of their core implementation.
This module centralizes filter diagnostics, orbit-detection diagnostics, and
batch summary figures used by the example workflow.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .prem_dispersion import rayleigh_group_velocity_prem


def _circular_difference_deg(estimate_deg, truth_deg):
    if estimate_deg is None or truth_deg is None:
        return np.nan
    return ((float(estimate_deg) - float(truth_deg) + 180.0) % 360.0) - 180.0


def _diagnostic_time_axis(result_dict: Dict[str, object], event_offset_sec: float = 0.0) -> np.ndarray:
    """Return a minutes-after-event axis for Stockwell output diagnostics."""
    if 'time_array' in result_dict:
        t = np.asarray(result_dict['time_array'], dtype=float).reshape(-1)
    else:
        n = None
        for key in ('S_R_raw', 'S_R', 'rayleigh_envelope'):
            if key in result_dict and result_dict[key] is not None:
                arr = np.asarray(result_dict[key])
                n = arr.shape[-1] if arr.ndim >= 2 else arr.size
                break
        if n is None:
            raise ValueError('Cannot infer diagnostic time axis from result_dict.')
        dt = float(result_dict.get('output_dt', result_dict.get('dt', 1.0)))
        t = np.arange(int(n), dtype=float) * dt
    return (t + float(event_offset_sec)) / 60.0



def _thin_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).astype(int))



def _stockwell_image(ax, data: np.ndarray, freqs: np.ndarray, time_min: np.ndarray,
                     title: str, *, log_power: bool = True, vmin_pct: float = 5.0,
                     vmax_pct: float = 99.0):
    """Plot a Stockwell time-frequency image with period on the vertical axis."""
    data = np.asarray(data)
    freqs = np.asarray(freqs, dtype=float)
    if data.ndim != 2 or data.shape[0] != freqs.size:
        raise ValueError(f'{title}: data must be (n_freq, n_time) and match freqs')
    if np.iscomplexobj(data):
        z = np.abs(data) ** 2
    else:
        z = np.asarray(data, dtype=float)
    if log_power:
        finite_positive = z[np.isfinite(z) & (z > 0)]
        floor = np.nanpercentile(finite_positive, 1.0) if finite_positive.size else 1e-30
        zplot = 10.0 * np.log10(np.maximum(z, floor))
        label = 'Energy (dB)'
    else:
        zplot = z
        label = 'Value'
    periods = 1.0 / freqs
    order = np.argsort(periods)
    periods_plot = periods[order]
    zplot = zplot[order, :]
    vmin = np.nanpercentile(zplot, vmin_pct)
    vmax = np.nanpercentile(zplot, vmax_pct)
    im = ax.imshow(
        zplot,
        aspect='auto',
        origin='lower',
        extent=[float(time_min[0]), float(time_min[-1]), float(periods_plot[0]), float(periods_plot[-1])],
        interpolation='nearest',
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_ylabel('Period (s)')
    ax.grid(True, alpha=0.2)
    return im, label



def _add_orbit_markers(ax, detected_orbits: Optional[Dict[str, object]], ymin: float, ymax: float) -> None:
    if not detected_orbits:
        return
    for orbit in ('R1', 'R2', 'R3'):
        if orbit not in detected_orbits:
            continue
        try:
            t = float(detected_orbits[orbit]['dispersion']['peak_time'])
        except Exception:
            continue
        ax.axvline(t, linestyle='--', linewidth=1.2, alpha=0.7)
        ax.text(t, ymax, orbit, ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))



def plot_rayleigh_filter_diagnostics(
    result_dict: Dict[str, object],
    *,
    output_dir: str = '.',
    prefix: str = 'rayleigh_filter',
    event_offset_sec: float = 0.0,
    detected_orbits: Optional[Dict[str, object]] = None,
    max_time_samples: int = 8000,
) -> Dict[str, str]:
    """Create waveform, Stockwell, mask, and energy diagnostics for the filter.

    The plots use the Stockwell output sample grid, after any downsampling inside
    ``simple_stockwell_filter_rayleigh``.  ``event_offset_sec`` shifts the x-axis
    to minutes after the event origin, matching the orbit detector.

    Returns a dictionary mapping diagnostic names to saved PNG paths.
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    saved: Dict[str, str] = {}
    time_min = _diagnostic_time_axis(result_dict, event_offset_sec=event_offset_sec)
    idx = _thin_indices(time_min.size, max_time_samples)
    t = time_min[idx]

    # ------------------------------------------------------------
    # 1. Time-domain components before and after filtering.
    # ------------------------------------------------------------
    component_rows = [
        ('N input', result_dict.get('input_north')),
        ('E input', result_dict.get('input_east')),
        ('Z input', result_dict.get('input_vertical')),
        ('R input', result_dict.get('input_radial')),
        ('T input', result_dict.get('input_transverse')),
        ('R filtered', result_dict.get('radial_rayleigh')),
        ('Z filtered', result_dict.get('vertical_rayleigh')),
        ('T filtered', result_dict.get('transverse_rayleigh')),
    ]
    component_rows = [(name, np.asarray(x, dtype=float)) for name, x in component_rows if x is not None]
    if component_rows:
        fig, axes = plt.subplots(len(component_rows), 1, figsize=(14, 1.6 * len(component_rows)), sharex=True)
        if len(component_rows) == 1:
            axes = [axes]
        for ax, (name, y) in zip(axes, component_rows):
            yplot = y[idx]
            scale = np.nanmax(np.abs(yplot))
            if np.isfinite(scale) and scale > 0:
                yplot = yplot / scale
                ylabel = f'{name}\n(norm.)'
            else:
                ylabel = name
            ax.plot(t, yplot, linewidth=0.8)
            ax.axhline(0.0, linewidth=0.6, alpha=0.5)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, alpha=0.25)
            if detected_orbits:
                ymin, ymax = ax.get_ylim()
                _add_orbit_markers(ax, detected_orbits, ymin, ymax)
        axes[-1].set_xlabel('Time (min after event)')
        fig.suptitle('Rayleigh filter component waveforms', fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        path = os.path.join(output_dir, f'{prefix}_components.png')
        fig.savefig(path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        saved['components'] = path

    # ------------------------------------------------------------
    # 2. Raw and filtered Stockwell energy images.
    # ------------------------------------------------------------
    freqs = np.asarray(result_dict.get('freqs'), dtype=float)
    if freqs.size and result_dict.get('S_R_raw') is not None and result_dict.get('S_Z_raw') is not None:
        panels = [
            ('Raw radial |S_R|^2', np.abs(result_dict['S_R_raw']) ** 2),
            ('Raw vertical |S_Z|^2', np.abs(result_dict['S_Z_raw']) ** 2),
        ]
        if 'S_energy_elliptic_nip' in result_dict:
            panels.append(('Particleman elliptic energy', result_dict['S_energy_elliptic_nip']))
        elif 'S_energy_filtered' in result_dict:
            panels.append(('Filtered R-Z energy', result_dict['S_energy_filtered']))
        fig, axes = plt.subplots(len(panels), 1, figsize=(14, 3.2 * len(panels)), sharex=True)
        if len(panels) == 1:
            axes = [axes]
        for ax, (title, arr) in zip(axes, panels):
            im, label = _stockwell_image(ax, arr, freqs, time_min, title)
            if detected_orbits:
                _add_orbit_markers(ax, detected_orbits, 1.0 / np.nanmax(freqs), 1.0 / np.nanmin(freqs))
            cbar = fig.colorbar(im, ax=ax, pad=0.01)
            cbar.set_label(label)
        axes[-1].set_xlabel('Time (min after event)')
        fig.suptitle('Stockwell energy diagnostics', fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        path = os.path.join(output_dir, f'{prefix}_stockwell_energy.png')
        fig.savefig(path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        saved['stockwell_energy'] = path

    # ------------------------------------------------------------
    # 3. NIP and mask images.
    # ------------------------------------------------------------
    diag = result_dict.get('diag', {})
    mask_panels = []
    for key, title in (
        ('particleman_retro_nip', 'Particleman retro NIP'),
        ('particleman_pro_nip', 'Particleman pro NIP'),
        ('particleman_energy_gate', 'Particleman energy gate'),
        ('particleman_transverse_gate', 'Particleman transverse gate'),
        ('particleman_retro_filter', 'Particleman retro filter'),
        ('particleman_pro_filter', 'Particleman pro filter'),
    ):
        if key in diag:
            mask_panels.append((title, np.asarray(diag[key], dtype=float), False))
    if not mask_panels:
        for key, title in (('retro_nip', 'Retro NIP'), ('pro_nip', 'Pro NIP')):
            if key in diag:
                mask_panels.append((title, np.asarray(diag[key], dtype=float), False))
        masks = result_dict.get('masks', {})
        for key, title in (('retro', 'Retro mask'), ('pro', 'Pro mask')):
            if key in masks:
                mask_panels.append((title, np.asarray(masks[key], dtype=float), False))
    if freqs.size and mask_panels:
        fig, axes = plt.subplots(len(mask_panels), 1, figsize=(14, 2.8 * len(mask_panels)), sharex=True)
        if len(mask_panels) == 1:
            axes = [axes]
        for ax, (title, arr, log_power) in zip(axes, mask_panels):
            im, label = _stockwell_image(ax, arr, freqs, time_min, title, log_power=log_power, vmin_pct=1, vmax_pct=99)
            if detected_orbits:
                _add_orbit_markers(ax, detected_orbits, 1.0 / np.nanmax(freqs), 1.0 / np.nanmin(freqs))
            cbar = fig.colorbar(im, ax=ax, pad=0.01)
            cbar.set_label(label)
        axes[-1].set_xlabel('Time (min after event)')
        fig.suptitle('NIP and mask diagnostics', fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        path = os.path.join(output_dir, f'{prefix}_masks.png')
        fig.savefig(path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        saved['masks'] = path

    # ------------------------------------------------------------
    # 4. Frequency-integrated energy traces by period band.
    # ------------------------------------------------------------
    def _get_energy_array(key):
        arr = result_dict.get(key)
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2:
            return None
        if arr.shape[0] != freqs.size:
            return None
        if arr.shape[1] != time_min.size:
            return None
        return arr

    raw_energy = _get_energy_array('S_energy_raw')
    if raw_energy is None and result_dict.get('S_R_raw') is not None and result_dict.get('S_Z_raw') is not None:
        raw_energy = (
            np.abs(result_dict['S_R_raw']) ** 2
            + np.abs(result_dict['S_Z_raw']) ** 2
        )

    filtered_energy = None
    filtered_label = None
    for key, label in (
        ('S_energy_elliptic_nip', 'filtered elliptic'),
        ('S_energy_filtered', 'filtered'),
        ('S_energy_retro_nip', 'filtered retro'),
        ('S_energy_pro_nip', 'filtered pro'),
    ):
        candidate = _get_energy_array(key)
        if candidate is not None:
            filtered_energy = candidate
            filtered_label = label
            break

    if freqs.size and raw_energy is not None:
        periods = 1.0 / freqs
        finite_periods = periods[np.isfinite(periods)]

        if finite_periods.size:
            period_min = float(np.nanmin(finite_periods))
            period_max = float(np.nanmax(finite_periods))
            dp = period_max - period_min

            # Equal-width period bands over the actual Stockwell period range.
            # For a 50-150 s band this gives roughly:
            # short: 50-83 s, middle: 83-117 s, long/low-frequency: 117-150 s.
            period_bands = [
                ('short period', period_min, period_min + dp / 3.0),
                ('middle period', period_min + dp / 3.0, period_min + 2.0 * dp / 3.0),
                ('long period / low freq', period_min + 2.0 * dp / 3.0, period_max),
            ]

            def _band_mask(lo, hi, is_last=False):
                if is_last:
                    return np.isfinite(periods) & (periods >= lo) & (periods <= hi)
                return np.isfinite(periods) & (periods >= lo) & (periods < hi)

            def _sum_band(arr, mask):
                if arr is None or not np.any(mask):
                    return None
                return np.nansum(arr[mask, :], axis=0)

            def _safe_for_log(y, floor):
                y = np.asarray(y, dtype=float)
                return np.maximum(y, floor)

            traces_for_floor = []

            raw_total = np.nansum(raw_energy, axis=0)
            traces_for_floor.append(raw_total)

            if filtered_energy is not None:
                filtered_total = np.nansum(filtered_energy, axis=0)
                traces_for_floor.append(filtered_total)
            else:
                filtered_total = None

            raw_band_traces = []
            filtered_band_traces = []

            for i, (band_name, lo, hi) in enumerate(period_bands):
                mask = _band_mask(lo, hi, is_last=(i == len(period_bands) - 1))

                y_raw = _sum_band(raw_energy, mask)
                if y_raw is not None:
                    traces_for_floor.append(y_raw)

                y_filt = _sum_band(filtered_energy, mask)
                if y_filt is not None:
                    traces_for_floor.append(y_filt)

                raw_band_traces.append((band_name, lo, hi, y_raw))
                filtered_band_traces.append((band_name, lo, hi, y_filt))

            positive = np.concatenate([
                np.asarray(y, dtype=float).ravel()
                for y in traces_for_floor
                if y is not None
            ])
            positive = positive[np.isfinite(positive) & (positive > 0.0)]

            if positive.size:
                floor = max(float(np.nanmax(positive)) * 1e-12, float(np.nanmin(positive)) * 0.1)
            else:
                floor = 1e-30

            fig, ax = plt.subplots(figsize=(14, 5.2))

            # Total energy: raw solid, filtered dashed.
            line_total, = ax.plot(
                time_min,
                _safe_for_log(raw_total, floor),
                linewidth=2.2,
                alpha=0.95,
                label='raw total',
            )

            if filtered_total is not None:
                ax.plot(
                    time_min,
                    _safe_for_log(filtered_total, floor),
                    linestyle='--',
                    linewidth=2.2,
                    alpha=0.95,
                    color=line_total.get_color(),
                    label=f'{filtered_label} total',
                )

            # After plotting the band traces, add body wave phases FIRST (before dispersion lines)
            if detected_orbits:
                try:
                    from obspy.taup import TauPyModel
                    model = TauPyModel(model="iasp91")
                    
                    # Get catalog distance from any detected orbit
                    catalog_distance = None
                    for orbit in ['R1', 'R2', 'R3']:
                        if orbit in detected_orbits:
                            catalog_distance = detected_orbits[orbit].get('distance_deg')
                            break
                    
                    if catalog_distance is not None:
                        # Get body wave arrivals
                        arrivals = model.get_travel_times(
                            source_depth_in_km=10.0,
                            distance_in_degree=catalog_distance,
                            phase_list=['P', 'S', 'PP', 'SS']
                        )
                        
                        # Plot body wave phases as black vertical lines with labels
                        phase_styles = {
                            'P': ('-', 2.0),
                            'S': ('--', 2.0),
                            'PP': ('-.', 1.5),
                            'SS': (':', 1.5)
                        }
                        
                        plotted_phases = set()
                        for arrival in arrivals:
                            arrival_min = arrival.time / 60.0
                            
                            if time_min[0] <= arrival_min <= time_min[-1]:
                                phase_name = arrival.name
                                if phase_name in phase_styles and phase_name not in plotted_phases:
                                    linestyle, linewidth = phase_styles[phase_name]
                                    ax.axvline(arrival_min, color='black', linestyle=linestyle,
                                              linewidth=linewidth, alpha=0.6, zorder=18, 
                                              label=phase_name)
                                    plotted_phases.add(phase_name)
            
                except ImportError:
                    pass
                except Exception as e:
                    print(f"Warning: Could not calculate body wave times: {e}")

            # Now add dispersion curve lines - ONE PER BAND PER ORBIT
            # Store band colors from the plotting loop
            band_colors = {}
            for idx, ((band_name, lo, hi, y_raw), (_, _, _, y_filt)) in enumerate(zip(raw_band_traces, filtered_band_traces)):
                if y_raw is None:
                    continue
                
                band_label = f'{band_name} ({lo:.0f}-{hi:.0f} s)'
                line_band, = ax.plot(
                    time_min,
                    _safe_for_log(y_raw, floor),
                    linewidth=1.35,
                    alpha=0.85,
                    label=f'raw {band_label}',
                )
                
                # Store color for this band
                band_colors[idx] = line_band.get_color()
                
                if y_filt is not None:
                    ax.plot(
                        time_min,
                        _safe_for_log(y_filt, floor),
                        linestyle='--',
                        linewidth=1.35,
                        alpha=0.9,
                        color=line_band.get_color(),
                        label=f'{filtered_label} {band_label}',
                    )
                    
                    # Add transparent fill under the dashed line
                    ax.fill_between(
                        time_min,
                        floor,
                        _safe_for_log(y_filt, floor),
                        color=line_band.get_color(),
                        alpha=0.15,  # Light transparent fill
                        linewidth=0,
                    )

            # Plot ONE vertical line per period band per orbit
            if detected_orbits:
                for orbit in ['R1', 'R2', 'R3']:
                    if orbit not in detected_orbits:
                        continue
                    
                    disp = detected_orbits[orbit]['dispersion']
                    periods = np.asarray(disp['periods'], dtype=float)
                    
                    # Use picked arrivals if available, otherwise model
                    arrivals = np.asarray(
                        disp.get('arrivals_picked_raw', disp['arrivals']),
                        dtype=float
                    )
                    
                    # For each period band, find periods in that band and plot ONE representative line
                    for band_idx, (band_name, lo, hi) in enumerate(period_bands):
                        # Find which periods fall in this band
                        if band_idx == len(period_bands) - 1:  # last band
                            in_band = (periods >= lo) & (periods <= hi)
                        else:
                            in_band = (periods >= lo) & (periods < hi)
                        
                        if not np.any(in_band):
                            continue
                        
                        # Get representative arrival time (median of arrivals in this band)
                        band_arrivals = arrivals[in_band]
                        representative_arrival = float(np.median(band_arrivals))
                        
                        # Get the band color
                        if band_idx in band_colors:
                            color = band_colors[band_idx]
                            
                            # Plot ONE vertical line per band
                            ax.axvline(representative_arrival, color=color, linestyle=':', 
                                      linewidth=2.0, alpha=0.7, zorder=17)

            ax.set_yscale('log')
            ax.set_xlim(time_min[0], time_min[-1])

            # Set reasonable y-axis limits for log scale
            # Collect all plotted traces to determine sensible limits
            all_plotted = []
            if filtered_total is not None:
                all_plotted.append(filtered_total)
            all_plotted.append(raw_total)
            for (_, _, _, y_raw), (_, _, _, y_filt) in zip(raw_band_traces, filtered_band_traces):
                if y_raw is not None:
                    all_plotted.append(y_raw)
                if y_filt is not None:
                    all_plotted.append(y_filt)

            if all_plotted:
                all_values = np.concatenate([y.ravel() for y in all_plotted])
                all_values = all_values[np.isfinite(all_values) & (all_values > floor)]
                
                if all_values.size:
                    y_min = max(float(np.percentile(all_values, 0.1)), floor)
                    y_max = float(np.percentile(all_values, 99.9)) * 2.0  # Add some headroom
                    ax.set_ylim(y_min, y_max)

            if detected_orbits:
                ymin, ymax = ax.get_ylim()
                _add_orbit_markers(ax, detected_orbits, ymin, ymax)

            ax.set_xlabel('Time (min after event)')
            ax.set_ylabel('Frequency-integrated energy')
            ax.set_title(
                'Integrated Stockwell energy envelopes by period band\n'
                'raw = solid, filtered = dashed'
            )
            ax.grid(True, alpha=0.3)

            # Make legend handle duplicates
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=8, ncol=2)

            fig.tight_layout()
            path = os.path.join(output_dir, f'{prefix}_energy_traces.png')
            fig.savefig(path, dpi=180, bbox_inches='tight')
            plt.close(fig)
            saved['energy_traces'] = path

    for name, path in saved.items():
        print(f'Saved {name} diagnostic: {path}')
    return saved


def plot_no_detection_summary(peak_dispersions, S_energy_band, freqs_band, time_min, energy_time_integrated, metadata, output_dir, min_confidence):
    """
    Create diagnostic plot when no orbits were detected, showing why.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])
    S_energy_db = 10 * np.log10(S_energy_band + 1e-12)
    periods_band = 1.0 / freqs_band
    period_order = np.argsort(periods_band)
    periods_plot = periods_band[period_order]
    S_energy_db_plot = S_energy_db[period_order, :]
    extent = tuple([time_min[0], time_min[-1], periods_plot[0], periods_plot[-1]])
    im = ax1.imshow(S_energy_db_plot, aspect='auto', cmap='hot', extent=extent, origin='lower', interpolation='bilinear')
    for i, disp in enumerate(peak_dispersions):
        ax1.scatter(disp['arrivals'], disp['periods'], c='cyan', s=30, alpha=0.6, label=f'Peak {i + 1}' if i < 3 else '', zorder=5)
    ax1.set_xlabel('Time (min after event)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Period (s)', fontweight='bold', fontsize=11)
    ax1.set_title('NO ORBITS DETECTED - All Peaks Shown', fontweight='bold', fontsize=13, color='red')
    if len(peak_dispersions) <= 3:
        ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, color='white', linewidth=0.5)
    plt.colorbar(im, ax=ax1, pad=0.01, label='Energy (dB)')
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_min, energy_time_integrated, 'k-', linewidth=1.5)
    ax2.fill_between(time_min, 0, energy_time_integrated, alpha=0.3, color='gray')
    for disp in peak_dispersions:
        ax2.axvline(disp['peak_time'], color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (min)', fontweight='bold')
    ax2.set_ylabel('Integrated Energy', fontweight='bold')
    ax2.set_title('Energy Time Series', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    explanation = f"\nNO ORBITS DETECTED\n\nPossible reasons:\n- All confidence scores < {min_confidence:.2f} (threshold)\n- Weak signal (low SNR)\n- Poor dispersion curve match to PREM\n- Inconsistent arrival times across periods\n\nDetected {len(peak_dispersions)} energy peaks, but none\npassed the confidence threshold.\n\nSuggestions:\n1. Lower min_confidence (try 0.2-0.3)\n2. Expand distance_range_deg\n3. Check if event distance is correct\n4. Verify frequency band is appropriate\n5. Try use_joint_fitting=False\n\nMethod used: {metadata.get('method', 'unknown').upper()}\n"
    ax3.text(0.1, 0.9, explanation, transform=ax3.transAxes, fontsize=11, verticalalignment='top', family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.suptitle('Orbit Detection Failed - Diagnostic Summary', fontsize=14, fontweight='bold', color='red')
    fig_path = os.path.join(output_dir, 'no_detection_summary.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved no-detection diagnostic: {fig_path}')
    plt.close(fig)



def plot_orbit_detection_diagnostics(detected_orbits, all_peak_dispersions, S_energy_band, freqs_band, time_min, energy_time_integrated, metadata, output_dir, orbit_energies=None):
    """
    Create comprehensive diagnostic plots - SHOWS RAW PICKS + WIDER GATES!
    """
    n_orbits = len([k for k in detected_orbits.keys() if k != 'metadata'])
    if n_orbits == 0:
        return
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :])
    S_energy_db = 10 * np.log10(S_energy_band + 1e-12)
    periods_band = 1.0 / freqs_band
    period_order = np.argsort(periods_band)
    periods_plot = periods_band[period_order]
    S_energy_db_plot = S_energy_db[period_order, :]
    extent = [time_min[0], time_min[-1], periods_plot[0], periods_plot[-1]]
    im = ax1.imshow(S_energy_db_plot, aspect='auto', cmap='viridis', extent=extent, origin='lower', interpolation='bilinear', vmin=np.percentile(S_energy_db_plot, 5), vmax=np.percentile(S_energy_db_plot, 95))
    for disp in all_peak_dispersions:
        if 'candidate_periods' in disp and 'candidate_arrivals_picked_raw' in disp:
            cand_periods = np.asarray(disp['candidate_periods'], dtype=float)
            cand_arrivals = np.asarray(disp['candidate_arrivals_picked_raw'], dtype=float)
            cand_accepted = np.asarray(disp.get('candidate_accepted', np.ones_like(cand_periods, dtype=bool)), dtype=bool)
            if np.any(~cand_accepted):
                ax1.scatter(cand_arrivals[~cand_accepted], cand_periods[~cand_accepted], c='white', marker='x', s=28, alpha=0.7, linewidths=0.8, zorder=4)
            if np.any(cand_accepted):
                ax1.scatter(cand_arrivals[cand_accepted], cand_periods[cand_accepted], c='lightgray', s=20, alpha=0.5, edgecolors='black', linewidths=0.5, zorder=5)
        else:
            arrivals_all = disp.get('arrivals_picked_raw', disp['arrivals'])
            ax1.scatter(arrivals_all, disp['periods'], c='lightgray', s=20, alpha=0.5, edgecolors='black', linewidths=0.5, zorder=5)
    colors = {'R1': '#00FFFF', 'R2': '#00FF00', 'R3': '#FFFF00'}
    for orbit, data in detected_orbits.items():
        if orbit == 'metadata':
            continue
        disp = data['dispersion']
        color = colors.get(orbit, 'white')
        if 'arrivals_picked_raw' in disp:
            arrivals_plot = disp['arrivals_picked_raw']
            periods_plot = disp['periods']
            label_suffix = '(raw picks)'
        else:
            arrivals_plot = disp['arrivals']
            periods_plot = disp['periods']
            label_suffix = '(fitted)'
        ax1.plot(arrivals_plot, periods_plot, color='black', linewidth=5, alpha=0.8, zorder=9)
        ax1.plot(arrivals_plot, periods_plot, color=color, linewidth=3, alpha=1.0, label=f"{orbit} {label_suffix} (conf={data['confidence']:.2f})", zorder=10)
        ax1.scatter(arrivals_plot, periods_plot, c=color, s=150, edgecolors='black', linewidths=2.5, zorder=11, alpha=0.9)
        distance = data['distance_deg']
        R_earth = 6371
        if orbit == 'R1':
            path_km = distance * R_earth * np.pi / 180
        elif orbit == 'R2':
            path_km = (360 - distance) * R_earth * np.pi / 180
        else:
            path_km = (360 + distance) * R_earth * np.pi / 180
        T_theory = np.linspace(50, 250, 100)
        U_theory = rayleigh_group_velocity_prem(T_theory)
        arrivals_theory = path_km / U_theory / 60
        ax1.plot(arrivals_theory, T_theory, ':', color=color, linewidth=1.5, alpha=0.5, zorder=7)
    ax1.set_xlabel('Time (min after event)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Period (s)', fontweight='bold', fontsize=12)
    
    title_str = f"Orbit Detection: {metadata['method'].upper()} fitting"
    if metadata['method'] == 'joint':
        title_str += f" | Shared distance: {metadata['shared_distance_deg']:.1f} deg"
    title_str += ' | Polarization-masked Stockwell'
    ax1.set_title(title_str, fontweight='bold', fontsize=14, pad=10)
    leg = ax1.legend(loc='upper right', fontsize=10, framealpha=0.95, edgecolor='black', fancybox=True)
    ax1.grid(True, alpha=0.4, color='gray', linewidth=0.5, linestyle=':')
    cbar = plt.colorbar(im, ax=ax1, pad=0.01)
    cbar.set_label('Energy (dB)', fontweight='bold', fontsize=11)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(time_min, energy_time_integrated, 'k-', linewidth=1.5, alpha=0.25, label='Total (unmasked)', zorder=1)
    ax2.fill_between(time_min, 0, energy_time_integrated, alpha=0.15, color='gray')
    if orbit_energies is not None:
        for orbit in ['R1', 'R2', 'R3']:
            if orbit not in orbit_energies:
                continue
            orbit_data = orbit_energies[orbit]
            energy = orbit_data['energy']
            sense = orbit_data['sense']
            peak_time = orbit_data['peak_time']
            color = colors[orbit]
            if np.max(energy) > 0:
                ax2.plot(time_min, energy, color=color, linewidth=2.5, label=f'{orbit} ({sense})', zorder=10, alpha=0.9)
                ax2.fill_between(time_min, 0, energy, color=color, alpha=0.15, zorder=5)
                peak_idx = np.argmax(energy)
                peak_energy_val = energy[peak_idx]
                ax2.axvline(peak_time, color=color, linestyle='--', linewidth=2.5, alpha=0.7, zorder=8)
                ax2.annotate(f'{orbit}\n{sense}', xy=(peak_time, peak_energy_val), xytext=(10, 10), textcoords='offset points', bbox=dict(boxstyle='round', facecolor=color, alpha=0.4, edgecolor='black', linewidth=1.5), fontsize=10, fontweight='bold', ha='left', va='bottom')
    else:
        for orbit in ['R1', 'R2', 'R3']:
            if orbit not in detected_orbits:
                continue
            data = detected_orbits[orbit]
            disp = data['dispersion']
            color = colors[orbit]
            if orbit == 'R1':
                time_window = 5.0
                freq_window = 3
            elif orbit == 'R2':
                time_window = 10.0
                freq_window = 5
            else:
                time_window = 15.0
                freq_window = 7
            gated_energy = np.zeros_like(time_min)
            if 'arrivals_picked_raw' in disp:
                arrival_times = disp['arrivals_picked_raw']
            else:
                arrival_times = disp['arrivals']
            if len(arrival_times) > 0:
                arrival_min = np.min(arrival_times)
                arrival_max = np.max(arrival_times)
                gate_start = arrival_min - time_window
                gate_end = arrival_max + time_window
                time_mask = (time_min >= gate_start) & (time_min <= gate_end)
                if len(disp['periods']) > 0:
                    period_min = np.min(disp['periods'])
                    period_max = np.max(disp['periods'])
                    freq_max = 1.0 / period_min
                    freq_min = 1.0 / period_max
                    freq_idx_min = np.argmin(np.abs(freqs_band - freq_min))
                    freq_idx_max = np.argmin(np.abs(freqs_band - freq_max))
                    freq_idx_start = max(0, freq_idx_min - freq_window)
                    freq_idx_end = min(len(freqs_band), freq_idx_max + freq_window)
                    if np.any(time_mask):
                        gated_energy[time_mask] = np.sum(S_energy_band[freq_idx_start:freq_idx_end, :][:, time_mask], axis=0)
            if np.max(gated_energy) > 0:
                ax2.plot(time_min, gated_energy, color=color, linewidth=2.5, label=f'{orbit} (time-gated)', zorder=10, alpha=0.9)
                ax2.fill_between(time_min, 0, gated_energy, color=color, alpha=0.15, zorder=5)
                peak_idx = np.argmax(gated_energy)
                peak_time = time_min[peak_idx]
                peak_energy = gated_energy[peak_idx]
                ax2.axvline(peak_time, color=color, linestyle='--', linewidth=2.5, alpha=0.7, zorder=8)
    ax2.set_xlabel('Time (min after event)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Energy (Polarization-Masked)', fontweight='bold', fontsize=12)
    ax2.set_title('Orbit-Specific Energy (Physically Correct Polarization Masks)', fontweight='bold', fontsize=13, pad=10)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.95, ncol=2)
    ax2.grid(True, alpha=0.4, linestyle=':')
    ax2.set_yscale('log')
    ax2.set_xlim(time_min[0], time_min[-1])
    orbit_list = [k for k in ['R1', 'R2', 'R3'] if k in detected_orbits]
    for idx, orbit in enumerate(orbit_list[:3]):
        data = detected_orbits[orbit]
        disp = data['dispersion']
        comp = data['confidence_components']
        ax = fig.add_subplot(gs[2, idx])
        distance = data['distance_deg']
        R_earth = 6371
        if orbit == 'R1':
            path_km = distance * R_earth * np.pi / 180
        elif orbit == 'R2':
            path_km = (360 - distance) * R_earth * np.pi / 180
        else:
            path_km = (360 + distance) * R_earth * np.pi / 180
        T_theory = np.linspace(50, 250, 100)
        U_theory = rayleigh_group_velocity_prem(T_theory)
        arrivals_theory = path_km / U_theory / 60
        ax.plot(T_theory, arrivals_theory, '--', color='gray', linewidth=3, alpha=0.7, label='PREM theory', zorder=8)
        if 'arrivals_picked_raw' in disp:
            # Plot all model-window local maxima first. Rejected maxima mark
            # candidate picks that failed the energy-proximity gate.
            if 'candidate_periods' in disp and 'candidate_arrivals_picked_raw' in disp:
                cand_periods = np.asarray(disp['candidate_periods'], dtype=float)
                cand_arrivals = np.asarray(disp['candidate_arrivals_picked_raw'], dtype=float)
                cand_accepted = np.asarray(disp.get('candidate_accepted', np.ones_like(cand_periods, dtype=bool)), dtype=bool)
                if np.any(~cand_accepted):
                    ax.scatter(cand_periods[~cand_accepted], cand_arrivals[~cand_accepted], s=35, c='0.65', marker='x', linewidths=1.0, alpha=0.8, label='Rejected local maxima', zorder=5)
                if np.any(cand_accepted):
                    ax.scatter(cand_periods[cand_accepted], cand_arrivals[cand_accepted], s=35, facecolors='none', edgecolors='0.35', linewidths=0.8, alpha=0.7, label='Accepted candidate set', zorder=6)
            # Plot accepted local Stockwell-energy maxima as the primary data.
            # The model branch used for scoring is shown separately.
            ax.plot(disp['periods'], disp['arrivals_picked_raw'], '-', color=colors[orbit], linewidth=2.0, alpha=0.75, label='Picked-maxima track', zorder=9)
            ax.scatter(disp['periods'], disp['arrivals_picked_raw'], s=150, c=colors[orbit], marker='o', edgecolors='black', linewidths=2.5, label='Picked maxima', zorder=10, alpha=0.9)
            if 'arrivals' in disp and len(disp['arrivals']) == len(disp['periods']):
                ax.plot(disp['periods'], disp['arrivals'], ':', color='black', linewidth=1.6, alpha=0.65, label='Model branch used for scoring', zorder=7)
            rms_to_raw = disp.get('raw_pick_rms_to_model_sec', comp['rms_residual_sec'])
            rms_label = f'RMS(picks)={rms_to_raw:.1f}s'
        else:
            ax.plot(disp['periods'], disp['arrivals'], '-', color=colors[orbit], linewidth=2.0, alpha=0.75, label='Arrival track', zorder=9)
            ax.scatter(disp['periods'], disp['arrivals'], s=150, c=colors[orbit], marker='o', edgecolors='black', linewidths=2.5, label='Arrivals', zorder=10, alpha=0.9)
            rms_label = f"RMS={comp['rms_residual_sec']:.1f}s"
        ax.set_xlabel('Period (s)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Arrival (min)', fontweight='bold', fontsize=11)
        title_lines = [f'{orbit} Dispersion', f"Dist={distance:.1f} deg | Conf={data['confidence']:.3f}", f"Corr={comp['shape_correlation']:.3f} | {rms_label}"]
        if 'candidate_accepted' in disp:
            n_cand = len(disp['candidate_accepted'])
            n_acc = int(np.sum(disp['candidate_accepted']))
            title_lines.append(f'Accepted {n_acc}/{n_cand} model-window maxima')
        ax.set_title('\n'.join(title_lines), fontweight='bold', fontsize=9, pad=8)
        ax.legend(fontsize=8, framealpha=0.95)
        ax.grid(True, alpha=0.4, linestyle=':')
    plt.suptitle(f"Rayleigh Orbit Detection Diagnostics\n{n_orbits} orbit(s) detected using {metadata['method']} fitting", fontsize=15, fontweight='bold', y=0.997)
    fig_path = os.path.join(output_dir, 'orbit_detection_diagnostics.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'\nSaved diagnostic plot: {fig_path}')
    plt.show(block=False)
    plt.pause(0.1)
    plt.close(fig)


def _truth_map_handles():
    return [
        Line2D([0], [0], color="black", lw=2.0, linestyle="-", label="true/catalog"),
        Line2D([0], [0], color="black", lw=2.0, linestyle="--", label="MAP/final"),
    ]



def plot_batch_baz_linear(rows, output_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    event_handles = []
    for row in rows:
        grid = row["baz_grid_deg"]
        pdf = row["baz_pdf"]
        line, = ax.plot(grid, pdf, lw=2.0, label=row["event_label"])
        ax.fill_between(grid, 0.0, pdf, alpha=0.18, color=line.get_color())
        event_handles.append(line)
        if row["catalog_baz_deg"] is not None:
            ax.axvline(row["catalog_baz_deg"], color=line.get_color(), lw=2.0, linestyle="-")
        ax.axvline(row["map_baz_deg"], color=line.get_color(), lw=2.0, linestyle="--")
    ax.set_title("Batch Rayleigh BAZ posterior summary")
    ax.set_xlabel("Back azimuth (degrees)")
    ax.set_ylabel("Normalized posterior PDF")
    ax.set_xlim(0, 360)
    ax.set_ylim(bottom=-0.03)
    ax.grid(True, alpha=0.3)
    handles = event_handles + _truth_map_handles()
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)



def plot_batch_baz_radial(rows, output_path):
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_title("Batch Rayleigh BAZ posterior radial summary", pad=20)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    max_radius = len(rows) + 1.0
    for idx, row in enumerate(rows, start=1):
        theta = np.deg2rad(row["baz_grid_deg"])
        base = float(idx)
        radius = base + 0.55 * row["baz_pdf"]
        line, = ax.plot(theta, radius, lw=2.0, label=row["event_label"])
        ax.fill_between(theta, base, radius, alpha=0.15, color=line.get_color())
        if row["catalog_baz_deg"] is not None:
            true_theta = np.deg2rad(row["catalog_baz_deg"])
            ax.plot([true_theta, true_theta], [base, base + 0.65], color=line.get_color(), lw=2.0, linestyle="-")
        map_theta = np.deg2rad(row["map_baz_deg"])
        ax.plot([map_theta, map_theta], [base, base + 0.65], color=line.get_color(), lw=2.0, linestyle="--")
        label_theta = np.deg2rad((row["map_baz_deg"] + 10.0) % 360.0)
        ax.text(label_theta, base + 0.72, row["event_name"], color=line.get_color(), fontsize=9)
    ax.set_rlim(0.5, max_radius)
    ax.set_yticklabels([])
    handles = ax.get_legend_handles_labels()[0] + _truth_map_handles()
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def plot_batch_distance_linear(rows, output_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    event_handles = []
    finite_truth = []
    finite_map = []
    for row in rows:
        grid = row["distance_grid_deg"]
        pdf = row["distance_pdf"]
        line, = ax.plot(grid, pdf, lw=2.0, label=row["event_label"])
        ax.fill_between(grid, 0.0, pdf, alpha=0.18, color=line.get_color())
        event_handles.append(line)
        finite_truth.append(row["catalog_distance_deg"])
        if row["map_distance_deg"] is not None and np.isfinite(row["map_distance_deg"]):
            finite_map.append(row["map_distance_deg"])
            ax.axvline(row["map_distance_deg"], color=line.get_color(), lw=2.0, linestyle="--")
        ax.axvline(row["catalog_distance_deg"], color=line.get_color(), lw=2.0, linestyle="-")
    all_x = [x for x in finite_truth + finite_map if x is not None and np.isfinite(x)]
    if all_x:
        pad = max(5.0, 0.15 * (max(all_x) - min(all_x) + 1.0))
        ax.set_xlim(max(0.0, min(all_x) - pad), min(180.0, max(all_x) + pad))
    ax.set_title("Batch Rayleigh distance posterior summary")
    ax.set_xlabel("Epicentral distance (degrees)")
    ax.set_ylabel("Normalized posterior PDF")
    ax.set_ylim(bottom=-0.03)
    ax.grid(True, alpha=0.3)
    handles = event_handles + _truth_map_handles()
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)



def plot_truth_vs_map_error_summary(rows, output_path):
    names = [row["event_name"] for row in rows]
    x = np.arange(len(rows), dtype=float)

    baz_abs_error = []
    distance_abs_error = []

    for row in rows:
        baz_abs_error.append(
            abs(_circular_difference_deg(row["map_baz_deg"], row["catalog_baz_deg"]))
        )
        distance_abs_error.append(
            abs(float(row["map_distance_deg"]) - float(row["catalog_distance_deg"]))
        )

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        x,
        baz_abs_error,
        marker="o",
        linestyle="-",
        label="BAZ absolute error",
    )
    ax.plot(
        x,
        distance_abs_error,
        marker="s",
        linestyle="--",
        label="Distance absolute error",
    )

    ax.set_ylabel("Absolute MAP minus catalog error (degrees)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_title("Truth vs MAP absolute errors across batch")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)




__all__ = [
    "plot_rayleigh_filter_diagnostics",
    "plot_no_detection_summary",
    "plot_orbit_detection_diagnostics",
    "plot_batch_baz_linear",
    "plot_batch_baz_radial",
    "plot_batch_distance_linear",
    "plot_truth_vs_map_error_summary",
]
