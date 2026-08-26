#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common Utilities for Seismic Analysis
Shared functions for phase picking and backazimuth calculation modules

This module provides:
- Logging functions (tee_print, open_log_file, close_log_file)
- Time conversion utilities (UTC <-> relative time)
- SAC header extraction
- Plotting utilities
- File/directory management
- Configuration helpers
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from obspy.core import UTCDateTime

#═════════════════════════════════════════════════════════════════════════════════
# GLOBAL LOGGING VARIABLES
#═════════════════════════════════════════════════════════════════════════════════

_log_file = None

def tee_print(*args, **kwargs):
    """
    Print to both console and log file simultaneously.
    Works exactly like built-in print() function.
    """
    global _log_file
    
    # Print to console
    print(*args, **kwargs)
    
    # Also write to log file if it's open
    if _log_file is not None:
        try:
            # Convert args to string just like print does
            output = ' '.join(str(arg) for arg in args)
            end = kwargs.get('end', '\n')
            _log_file.write(output + end)
            _log_file.flush()  # Ensure immediate write
        except Exception as e:
            # If logging fails, don't crash - just print to console
            print(f"[Log write error: {e}]")

def open_log_file(output_dir, station_name):
    """Open the log file for writing."""
    global _log_file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f'analysis_log_{station_name}_{timestamp}.txt')
    try:
        _log_file = open(log_path, 'w', encoding='utf-8')
        tee_print(f"Log file created: {log_path}")
        tee_print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        tee_print("="*80 + "\n")
        return log_path
    except Exception as e:
        print(f"Warning: Could not create log file: {e}")
        print("Continuing with console output only...")
        _log_file = None
        return None

def close_log_file():
    """Close the log file."""
    global _log_file
    if _log_file is not None:
        try:
            tee_print("\n" + "="*80)
            tee_print(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            tee_print("="*80)
            _log_file.close()
            _log_file = None
        except Exception as e:
            print(f"Warning: Error closing log file: {e}")

#═════════════════════════════════════════════════════════════════════════════════
# FILE AND DIRECTORY MANAGEMENT
#═════════════════════════════════════════════════════════════════════════════════

def create_output_directory(output_dir, station_name):
    """Create output directory for saving figures and initialize log file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        tee_print(f"\nCreated output directory: {output_dir}")
    else:
        tee_print(f"\nUsing output directory: {output_dir}")
    
    # Initialize log file
    open_log_file(output_dir, station_name)
    
    return output_dir

def check_files_exist(files, data_dir):
    """Check if data files exist and are accessible."""
    missing_files = []
    for file in files:
        filepath = os.path.join(data_dir, file)
        if not os.path.exists(filepath):
            missing_files.append(filepath)
    
    if missing_files:
        tee_print("ERROR: The following data files are missing:")
        for f in missing_files:
            tee_print(f"  - {f}")
        tee_print(f"\nPlease ensure data files are in: {data_dir}")
        return False
    return True

#═════════════════════════════════════════════════════════════════════════════════
# SAC HEADER EXTRACTION
#═════════════════════════════════════════════════════════════════════════════════

def _is_set(value):
    """Check if a SAC header value is set (not -12345.0)."""
    if value is None:
        return False
    try:
        return abs(float(value) + 12345.0) > 1e-6
    except (ValueError, TypeError):
        return False

def timing_from_sac_origin(trace):
    """
    Extract event timing information from SAC header.
    
    Parameters:
        trace: ObsPy trace with SAC header
    
    Returns:
        dict: {
            'sac_reference_time': reference time from SAC header,
            'trace_start_time': trace start time,
            'event_time': event origin time,
            'b': SAC b value (start offset),
            'o': SAC o value (origin offset),
            'event_offset_sec': offset from trace start to event
        }
    """
    if not hasattr(trace.stats, 'sac'):
        raise ValueError("Trace has no SAC header. Cannot derive origin time.")
    sac = trace.stats.sac
    if not _is_set(getattr(sac, 'o', None)):
        raise ValueError("SAC header does not contain a valid origin offset 'o'.")
    
    # Get reference time from SAC header
    ref_time = UTCDateTime(
        year=int(sac.nzyear),
        julday=int(sac.nzjday),
        hour=int(sac.nzhour),
        minute=int(sac.nzmin),
        second=int(sac.nzsec),
        microsecond=int(sac.nzmsec) * 1000,
    )
    
    b = float(sac.b) if _is_set(getattr(sac, 'b', None)) else 0.0
    o = float(sac.o)
    event_time = ref_time + o
    event_offset_sec = float(trace.stats.starttime - event_time)
    
    return {
        'sac_reference_time': ref_time,
        'trace_start_time': trace.stats.starttime,
        'event_time': event_time,
        'b': b,
        'o': o,
        'event_offset_sec': event_offset_sec,
    }

def get_sac_distance_deg(trace, fallback_distance_deg=None):
    """Extract distance from SAC header."""
    sac = getattr(trace.stats, 'sac', None)
    if sac is not None and _is_set(getattr(sac, 'gcarc', None)):
        return float(sac.gcarc)
    if fallback_distance_deg is not None:
        return float(fallback_distance_deg)
    return None

def get_sac_baz_deg(trace):
    """Extract backazimuth from SAC header."""
    sac = getattr(trace.stats, 'sac', None)
    if sac is not None and _is_set(getattr(sac, 'baz', None)):
        return float(sac.baz)
    return None

#═════════════════════════════════════════════════════════════════════════════════
# TIME REFERENCE MANAGEMENT
#═════════════════════════════════════════════════════════════════════════════════

# Global time reference (set by phase picking module)
P_ARRIVAL_TIME = None

def utc_to_relative(utc_time, p_arrival_time):
    """
    Convert UTC timestamp to seconds relative to P-arrival.
    
    Parameters:
        utc_time: UTCDateTime object
        p_arrival_time: UTCDateTime object of P-arrival reference time
    
    Returns:
        float: seconds after P-arrival (negative if before P-arrival)
    """
    if p_arrival_time is None:
        raise ValueError("P_ARRIVAL_TIME not provided as parameter.")
    
    return utc_time - p_arrival_time

def relative_to_utc(seconds_after_p, p_arrival_time):
    """
    Convert seconds relative to P-arrival back to UTC timestamp.
    
    Parameters:
        seconds_after_p: float, seconds after P-arrival
        p_arrival_time: UTCDateTime object of P-arrival reference time
    
    Returns:
        UTCDateTime object
    """
    if p_arrival_time is None:
        raise ValueError("P_ARRIVAL_TIME not provided as parameter.")
    
    return p_arrival_time + seconds_after_p

def format_time_with_uncertainty(utc_time, uncertainty_sec):
    """
    Format UTC time with error bars in seconds.
    
    Parameters:
        utc_time: UTCDateTime object
        uncertainty_sec: uncertainty in seconds (±)
    
    Returns:
        str: formatted string like "2015-06-29T09:09:21.5 UTC (±12.3s)"
    """
    time_str = utc_time.strftime("%Y-%m-%dT%H:%M:%S")
    
    # Add fractional seconds if present
    if utc_time.microsecond > 0:
        time_str += f".{int(utc_time.microsecond/100000)}"
    
    return f"{time_str} UTC (±{uncertainty_sec:.1f}s)"

#═════════════════════════════════════════════════════════════════════════════════
# STOCKWELL TIME AXIS HELPER
#═════════════════════════════════════════════════════════════════════════════════

def stockwell_time_axis_seconds(result_dict, reference_trace):
    """
    Extract time axis from Stockwell result dictionary.
    
    Parameters:
        result_dict: Output from simple_stockwell_filter_rayleigh
        reference_trace: Original trace for duration reference
    
    Returns:
        np.array: Time array in seconds from trace start
    """
    # Determine number of time samples
    S_R = result_dict.get('S_R')
    if S_R is not None and np.ndim(S_R) == 2:
        n_times = int(S_R.shape[1])
    elif 'rayleigh_envelope' in result_dict:
        n_times = len(result_dict['rayleigh_envelope'])
    else:
        raise ValueError("Cannot determine Stockwell time dimension.")
    
    trace_duration_sec = (reference_trace.stats.npts - 1) * reference_trace.stats.delta
    
    # Try to find time array in result_dict
    for key in ('time_array', 'time_seconds', 'times_sec', 'times', 't'):
        if key in result_dict:
            arr = np.asarray(result_dict[key], dtype=float).squeeze()
            if arr.ndim == 1 and arr.size == n_times:
                arr0 = arr - arr[0]
                span = float(arr0[-1] - arr0[0])
                if np.isfinite(span) and span > 0:
                    # Check if span matches trace duration
                    if abs(span - trace_duration_sec) / trace_duration_sec <= 0.05:
                        return arr0
                    # Check if in minutes
                    if abs(span * 60.0 - trace_duration_sec) / trace_duration_sec <= 0.05:
                        return arr0 * 60.0
    
    # Try to get dt from result_dict
    reported_dt = None
    for key in ('output_dt', 'target_dt', 'dt_out', 'dt'):
        if key in result_dict:
            try:
                reported_dt = float(result_dict[key])
            except Exception:
                pass
            if reported_dt is not None and np.isfinite(reported_dt) and reported_dt > 0:
                break
    
    if reported_dt is None:
        reported_dt = float(reference_trace.stats.delta)
    
    reported_duration = (n_times - 1) * reported_dt
    if abs(reported_duration - trace_duration_sec) / trace_duration_sec > 0.05:
        reported_dt = trace_duration_sec / (n_times - 1)
    
    return np.arange(n_times, dtype=float) * reported_dt

#═════════════════════════════════════════════════════════════════════════════════
# PLOTTING UTILITIES
#═════════════════════════════════════════════════════════════════════════════════

def plot_three_component_data(t, dataE, dataN, dataZ, arrivals, title, xlim=(0, 1800), 
                               stage1_picks=None, use_utc=False, start_time=None, reference_time=None):
    """
    Plot three-component seismic data with phase arrivals.
    
    Parameters:
        t: time array in seconds from start of trace
        dataE, dataN, dataZ: waveform data arrays
        arrivals: TauP arrivals for phase predictions
        title: plot title
        xlim: x-axis limits (seconds if use_utc=False, ignored if use_utc=True)
        stage1_picks: dict of Stage 1 picks to show as reference (optional)
        use_utc: if True, convert x-axis to UTC time (default: False)
        start_time: ObsPy UTCDateTime object (required if use_utc=True)
        reference_time: ObsPy UTCDateTime for converting picks (required if stage1_picks provided)
    """
    from matplotlib.dates import DateFormatter, SecondLocator
    import matplotlib.dates as mdates
    
    fig, ax = plt.subplots(figsize=(6, 3))
    
    # Create time axis (UTC if requested)
    if use_utc and start_time is not None:
        # Convert to matplotlib datetime
        t_utc = [start_time + ti for ti in t]
        t_plot = mdates.date2num(t_utc)
    else:
        t_plot = t
    
    # Normalize and plot data
    MM = 1
    dataZ_norm = MM * dataZ / np.max(np.abs(dataZ))
    dataE_norm = MM * dataE / np.max(np.abs(dataZ))
    dataN_norm = MM * dataN / np.max(np.abs(dataZ))
    
    ax.plot(t_plot, dataZ_norm, 'k', linewidth=0.8)
    ax.plot(t_plot, dataE_norm + 1, 'k', linewidth=0.8)
    ax.plot(t_plot, dataN_norm + 2, 'k', linewidth=0.8)
    
    # Add component labels
    if use_utc:
        label_x = t_plot[len(t_plot)//4] if len(t_plot) > 0 else t_plot[0]
    else:
        label_x = 750
    ax.text(label_x, 0.1, 'Z', fontsize=11)
    ax.text(label_x, 1.1, 'E', fontsize=11)
    ax.text(label_x, 2.1, 'N', fontsize=11)
    
    # Plot predicted phase arrival lines
    phase_colors = {'P': 'r', 'PP': 'pink', 'S': 'darkred', 'PS': 'tomato'}
    phase_labels = ['P', 'PP', 'S', 'PS']
    
    for i, phase in enumerate(phase_labels[:len(arrivals)]):
        time_sec = arrivals[i].time
        if use_utc and start_time is not None:
            time_plot = mdates.date2num(start_time + time_sec)
        else:
            time_plot = time_sec
        color = phase_colors[phase]
        ax.plot([time_plot, time_plot], [-0.5, 2.5], color=color, linewidth=1.0, 
                linestyle='-', alpha=0.6)
        if use_utc:
            text_offset = (10/86400.0)
        else:
            text_offset = 10
        ax.text(time_plot + text_offset, 1.5, phase + ' (pred)', fontsize=9, color=color)
    
    # Plot Stage 1 picks if provided
    if stage1_picks is not None and reference_time is not None:
        for phase, times in stage1_picks.items():
            if times and phase in phase_colors:
                for pick_time in times:
                    # pick_time is UTCDateTime, plot it appropriately
                    if use_utc and start_time is not None:
                        # Convert UTCDateTime directly to matplotlib date format
                        pick_plot = mdates.date2num(pick_time.datetime)
                    else:
                        # For relative time display, convert to seconds from trace start
                        pick_trace_time = float(pick_time - start_time)
                        pick_plot = pick_trace_time
                    
                    ax.plot([pick_plot, pick_plot], [-0.5, 2.5], 
                           color=phase_colors[phase], linewidth=0.7, 
                           linestyle='-', alpha=0.8)
                    if use_utc:
                        text_offset = (5/86400.0)
                    else:
                        text_offset = 5
                    ax.text(pick_plot + text_offset, 2.2, phase + ' (S1)', fontsize=8, 
                           color=phase_colors[phase], weight='bold')
    
    # Set x-axis limits and formatting
    if use_utc:
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        data_duration_sec = len(t) / (len(t) / (t[-1] - t[0])) if len(t) > 1 else 3600
        if data_duration_sec > 3000:
            interval = 600
        elif data_duration_sec > 1800:
            interval = 300
        else:
            interval = 120
        ax.xaxis.set_major_locator(SecondLocator(interval=interval))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.set_xlabel('UTC Time (HH:MM:SS)', fontweight='bold')
    else:
        ax.set_xlim(xlim)
        ax.set_xlabel('Time (s)', fontweight='bold')
    
    ax.set_ylim(-0.5, 2.5)
    ax.set_ylabel('Normalized Amplitude', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

def plot_pdf(x, y, true_value, xlabel, ylabel='Probability Density', title='PDF'):
    """Plot probability density function with improved styling."""
    fig, ax = plt.subplots(figsize=(5, 2.5))
    
    ax.fill_between(x, y, alpha=0.3, color='#1f77b4')
    ax.plot(x, y, '-', linewidth=2, color='#1f77b4', label='PDF')
    ax.axvline(true_value, color='#d62728', linestyle='-',
               linewidth=0.8, label=f'True value: {true_value:.2f}', alpha=0.8)
    
    # Mark maximum
    max_idx = np.argmax(y)
    ax.plot(x[max_idx], y[max_idx], 'ko', markersize=8, 
            label=f'Estimated: {x[max_idx]:.2f}', zorder=5)
    
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

#═════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION HELPERS
#═════════════════════════════════════════════════════════════════════════════════

def generate_frequency_bands(use_auto_bands, num_bands, period_min, period_max, overlap_percent):
    """Generate frequency bands based on configuration."""
    if use_auto_bands:
        # Calculate band spacing with overlap
        # For n bands with overlap, we need: (n-1)*step + width = total_range
        # where step = width * (1 - overlap/100)
        total_range = period_max - period_min
        
        # Solve for band_width: total_range = (num_bands - 1) * band_width * (1 - overlap/100) + band_width
        # Simplifies to: total_range = band_width * [1 + (num_bands - 1) * (1 - overlap/100)]
        band_width = total_range / (1 + (num_bands - 1) * (1 - overlap_percent / 100))
        step_size = band_width * (1 - overlap_percent / 100)

        band_high = np.zeros(num_bands)
        band_low = np.zeros(num_bands)

        for i in range(num_bands):
            # Start from period_min and step upward
            band_low[i] = period_min + i * step_size
            band_high[i] = band_low[i] + band_width

        # Ensure we stay within bounds
        band_high = np.clip(band_high, period_min, period_max)
        band_low = np.clip(band_low, period_min, period_max)
    else:
        raise NotImplementedError("Manual band configuration not implemented")

    return band_high, band_low

def print_frequency_band_info(use_auto_bands, num_bands, period_min, period_max, 
                               overlap_percent, band_high, band_low):
    """Print information about the generated frequency bands."""
    tee_print("\nFrequency Band Configuration:")
    tee_print("-" * 60)
    if use_auto_bands:
        tee_print(f"  Method: Automatic spacing")
        tee_print(f"  Number of bands: {num_bands}")
        tee_print(f"  Period range: {period_min}-{period_max} seconds")
        tee_print(f"  Overlap: {overlap_percent}%")
    
    tee_print("\n  Generated Bands:")
    for i in range(len(band_high)):
        freq_high = 1 / band_low[i]
        freq_low = 1 / band_high[i]
        bandwidth = band_high[i] - band_low[i]
        tee_print(f"    Band {i+1}: {band_low[i]:.1f}-{band_high[i]:.1f}s "
              f"({freq_low:.4f}-{freq_high:.4f} Hz, width: {bandwidth:.1f}s)")
    tee_print("-" * 60)
