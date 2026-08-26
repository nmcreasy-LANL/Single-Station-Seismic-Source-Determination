#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 17:20:49 2026
functions for body wave depth and distance
@author: nmcreasy
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, SecondLocator
from datetime import datetime
from obspy.core import UTCDateTime
from scipy.signal import find_peaks
import obspy.signal

# Add parent directory to path for common_utils import
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS_DIR = os.path.join(PARENT_DIR, 'utils')
sys.path.insert(0, UTILS_DIR)
sys.path.insert(0, PARENT_DIR)  # Also add parent to find rayleigh_wave_tools

# Import tee_print from common_utils
from common_utils import tee_print

# Import Rayleigh wave tools
from rayleigh_wave_tools import rayleigh_group_velocity_prem

def combine_distance_pdfs(x1, pdf1, x2, pdf2, true_distance, label1="PDF 1", label2="PDF 2"):
    """
    Combine two distance PDFs using Bayesian multiplication approach.
    
    Parameters:
        x1: x-axis values for first PDF (e.g., distances in degrees)
        pdf1: probability density values for first PDF
        x2: x-axis values for second PDF
        pdf2: probability density values for second PDF
        true_distance: true distance value for plotting
        label1: label for first PDF
        label2: label for second PDF
    
    Returns:
        tuple: (x_combined, pdf_combined, estimated_distance, std_combined)
    """
    tee_print(f"\n{'='*80}")
    tee_print(f"COMBINING DISTANCE PDFs: {label1} + {label2}")
    tee_print(f"{'='*80}")
    
    # Create common grid spanning both PDFs
    x_min = min(np.min(x1), np.min(x2))
    x_max = max(np.max(x1), np.max(x2))
    x_common = np.linspace(x_min, x_max, 1000)
    
    # Interpolate both PDFs onto common grid
    pdf1_interp = np.interp(x_common, x1, pdf1, left=0, right=0)
    pdf2_interp = np.interp(x_common, x2, pdf2, left=0, right=0)
    
    # Normalize individual PDFs
    if np.trapz(pdf1_interp, x_common) > 0:
        pdf1_interp /= np.trapz(pdf1_interp, x_common)
    if np.trapz(pdf2_interp, x_common) > 0:
        pdf2_interp /= np.trapz(pdf2_interp, x_common)
    
    # Combine PDFs (Bayesian multiplication)
    pdf_combined = pdf1_interp * pdf2_interp
    
    # Normalize combined PDF
    if np.trapz(pdf_combined, x_common) > 0:
        pdf_combined /= np.trapz(pdf_combined, x_common)
    else:
        tee_print("WARNING: Combined PDF integration is zero!")
        return x_common, pdf_combined, None, None
    
    # Calculate statistics
    mean_combined = np.trapz(x_common * pdf_combined, x_common)
    variance_combined = np.trapz((x_common - mean_combined)**2 * pdf_combined, x_common)
    std_combined = np.sqrt(variance_combined)
    
    # Find maximum (most probable value)
    max_idx = np.argmax(pdf_combined)
    estimated_distance = x_common[max_idx]
    
    # Calculate individual estimates for comparison
    max_idx1 = np.argmax(pdf1_interp)
    estimate1 = x_common[max_idx1]
    max_idx2 = np.argmax(pdf2_interp)
    estimate2 = x_common[max_idx2]
    
    tee_print(f"\nIndividual Estimates:")
    tee_print(f"  {label1}: {estimate1:.2f}°")
    tee_print(f"  {label2}: {estimate2:.2f}°")
    tee_print(f"\nCombined Estimate:")
    tee_print(f"  Mean: {mean_combined:.2f}°")
    tee_print(f"  Std:  {std_combined:.2f}°")
    tee_print(f"  Max (most probable): {estimated_distance:.2f}°")
    tee_print(f"\nTrue distance: {true_distance:.2f}°")
    tee_print(f"Combined error: {abs(estimated_distance - true_distance):.2f}°")
    tee_print(f"{'='*80}\n")
    
    return x_common, pdf_combined, estimated_distance, std_combined


def calculate_95_confidence_interval(x, pdf):
    """
    Calculate 95% confidence interval from a PDF using cumulative distribution.
    
    Parameters:
        x: x-axis values
        pdf: probability density values (must be normalized)
    
    Returns:
        tuple: (lower_bound, upper_bound) or (None, None) if calculation fails
    """
    # Calculate cumulative distribution function
    cdf = np.cumsum(pdf) * (x[1] - x[0]) if len(x) > 1 else np.cumsum(pdf)
    
    # Normalize CDF (in case PDF wasn't perfectly normalized)
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    else:
        return None, None
    
    # Find indices where CDF crosses 2.5% and 97.5%
    idx_lower = np.argmin(np.abs(cdf - 0.025))
    idx_upper = np.argmin(np.abs(cdf - 0.975))
    
    lower_bound = x[idx_lower]
    upper_bound = x[idx_upper]
    
    return lower_bound, upper_bound


def analyze_rayleigh_waves(st_raw, reference_time, estimated_distance=None,
                           p_arrival_time=None, event_time=None,
                           use_known_origin=False, distance_deg=None,
                           band_high=None, band_low=None, output_dir=None):
    """
    Perform Rayleigh wave group velocity analysis.
    
    Parameters:
        st_raw: raw ObsPy stream (unfiltered)
        reference_time: reference time for analysis (EVENT_TIME or P_ARRIVAL_TIME)
        estimated_distance: estimated distance from Stage 1 (degrees), optional
        p_arrival_time: P-wave arrival time (for time reference)
        event_time: event origin time (can be None for blind analysis)
        use_known_origin: bool, True if event_time is known
        distance_deg: true distance for validation (degrees)
        band_high: array of high-period band edges (seconds)
        band_low: array of low-period band edges (seconds)
        output_dir: output directory for plots
    
    Returns:
        tuple: (estimated_distance_rayleigh, distance_pdf_x, distance_pdf_y,
                timing_pdf_x, timing_pdf_y, tr1, band_high, band_low)
    """
    tee_print(f"\n{'='*80}")
    tee_print(f"RAYLEIGH WAVE ANALYSIS")
    tee_print(f"{'='*80}")
    
    # Check if p_arrival_time has been set
    if p_arrival_time is None:
        tee_print("ERROR: p_arrival_time not set. Cannot perform Rayleigh wave analysis.")
        tee_print("       P-wave must be picked first to establish time reference.")
        return None, None, None, None, None, None, None, None
    
    # Determine reference time for analysis
    if event_time is not None:
        time_mode = "EVENT_TIME"
        tee_print(f"Using EVENT_TIME as reference: {event_time}")
    else:
        time_mode = "P_ARRIVAL"
        tee_print(f"Using P_ARRIVAL_TIME as reference: {p_arrival_time}")
    
    # Use estimated distance if available, otherwise use true distance for initial guess
    if estimated_distance is None:
        if use_known_origin and distance_deg is not None:
            estimated_distance = distance_deg
            tee_print(f"Using true distance for initial estimate: {distance_deg}°")
        else:
            tee_print("WARNING: No distance estimate available for Rayleigh wave analysis.")
            tee_print("         Using default estimate of 90° for initial calculation.")
            estimated_distance = 90.0
    else:
        tee_print(f"Using Stage 1 estimated distance: {estimated_distance:.2f}°")
    
    # Calculate distances for R1, R2, R3
    dist_km = 6371 * estimated_distance * math.pi / 180
    r2_dist = 6371 * (360 - estimated_distance) * math.pi / 180
    r3_dist = 6371 * (360 + estimated_distance) * math.pi / 180
    
    tee_print(f"\nRayleigh wave path distances:")
    tee_print(f"  R1 (minor arc): {dist_km:.1f} km")
    tee_print(f"  R2 (major arc): {r2_dist:.1f} km")
    tee_print(f"  R3 (around globe): {r3_dist:.1f} km")
    
    # Initialize arrays for storing results
    num_bands = len(band_high)
    tr1 = np.zeros((1, num_bands))
    tr2 = np.zeros((1, num_bands))
    tr3 = np.zeros((1, num_bands))
    Ar1 = np.zeros((1, num_bands))
    Ar2 = np.zeros((1, num_bands))
    Ar3 = np.zeros((1, num_bands))
    delta = np.zeros((1, num_bands))
    t0i = np.zeros((1, num_bands))
    Iu = np.zeros((1, num_bands))
    
    tee_print(f"\nProcessing {num_bands} frequency bands...")
    
    # Use full waveform for Rayleigh wave analysis (needed for R2 and R3 arrivals)
    st_rayleigh = st_raw.copy()
    # Note: Not trimming to allow R2 (~2.8 hours) and R3 (~3.3 hours) arrivals
    
    npts = st_rayleigh[0].stats.npts
    samprate = st_rayleigh[0].stats.sampling_rate
    t = np.arange(0, npts / samprate, 1 / samprate)
    start = st_rayleigh[0].stats.starttime
    diff1 = start - reference_time
    
    # Process each frequency band
    for i in range(num_bands):
        tee_print(f"\n  Band {i+1}/{num_bands}: {band_low[i]:.1f}-{band_high[i]:.1f}s")
        
        # Filter data
        st_filt = st_rayleigh.copy()
        st_filt.taper(0.1, type='hann', side='both')
        st_filt.filter('bandpass', freqmin=1/band_high[i], freqmax=1/band_low[i],
                      corners=2, zerophase=False)
        
        # Calculate envelope for Z component
        data_envelope = obspy.signal.filter.envelope(st_filt[1].data)
        
        # Calculate noise level
        stnoise = st_raw.copy()
        if event_time is not None:
            stnoise = stnoise.trim(st_raw[0].stats.starttime, event_time)
        else:
            # Use first 10 minutes as noise if no event time
            stnoise = stnoise.trim(st_raw[0].stats.starttime, st_raw[0].stats.starttime + 600)
        
        if len(stnoise) > 0:
            stnoise.taper(0.1, type='hann', side='both')
            stnoise.filter('bandpass', freqmin=1/band_high[i], freqmax=1/band_low[i],
                          corners=2, zerophase=False)
            noise_envelope = obspy.signal.filter.envelope(stnoise[1].data)
            noise_threshold = 1.5 * np.max(noise_envelope)
        else:
            noise_threshold = 0
        
        time = (t + diff1) / 60  # time in minutes
        data = data_envelope
        
        # Pick R1 (minor arc)
        start_time = (dist_km/3.7)/60 - 10
        end_time = (dist_km/3.7)/60 + 10
        time_in_window = time[(time >= start_time) & (time <= end_time)]
        data_in_window = data[(time >= start_time) & (time <= end_time)]
        
        peaks_indices, _ = find_peaks(data_in_window, height=noise_threshold)
        
        if len(peaks_indices) > 0:
            peak_values = data_in_window[peaks_indices]
            max_peak_index = peaks_indices[np.argmax(peak_values)]
            tr1[0, i] = time_in_window[max_peak_index]
            Ar1[0, i] = peak_values[np.argmax(peak_values)]
            tee_print(f"    R1 peak: {tr1[0,i]:.2f} min, amplitude: {Ar1[0,i]:.2e}")
        else:
            tee_print(f"    R1: No peaks above noise threshold")
            tr1[0, i] = (dist_km/3.7)/60
            Ar1[0, i] = 0
        
        # Pick R2 (major arc)
        start_time = (r2_dist/3.7)/60 - 10
        end_time = (r2_dist/3.7)/60 + 10
        time_in_window = time[(time >= start_time) & (time <= end_time)]
        data_in_window = data[(time >= start_time) & (time <= end_time)]
        
        peaks_indices, _ = find_peaks(data_in_window, height=noise_threshold)
        
        if len(peaks_indices) > 0:
            peak_values = data_in_window[peaks_indices]
            max_peak_index = peaks_indices[np.argmax(peak_values)]
            tr2[0, i] = time_in_window[max_peak_index]
            Ar2[0, i] = peak_values[np.argmax(peak_values)]
            tee_print(f"    R2 peak: {tr2[0,i]:.2f} min, amplitude: {Ar2[0,i]:.2e}")
        else:
            tee_print(f"    R2: No peaks above noise threshold")
            tr2[0, i] = (r2_dist/3.7)/60
            Ar2[0, i] = 0
        
        # Pick R3 (around globe)
        start_time = (r3_dist/3.7)/60 - 5
        end_time = (r3_dist/3.7)/60 + 5
        time_in_window = time[(time >= start_time) & (time <= end_time)]
        data_in_window = data[(time >= start_time) & (time <= end_time)]
        
        peaks_indices, _ = find_peaks(data_in_window)
        
        if len(peaks_indices) > 0:
            peak_values = data_in_window[peaks_indices]
            max_peak_index = peaks_indices[np.argmax(peak_values)]
            tr3[0, i] = time_in_window[max_peak_index]
            Ar3[0, i] = peak_values[np.argmax(peak_values)]
            tee_print(f"    R3 peak: {tr3[0,i]:.2f} min, amplitude: {Ar3[0,i]:.2e}")
        else:
            tee_print(f"    R3: No peaks above noise threshold")
            tr3[0, i] = (r3_dist/3.7)/60
            Ar3[0, i] = 0
        
        # Plot envelope for middle frequency band (diagnostic plot)
        if i == num_bands // 2 and output_dir is not None:
            tee_print(f"    Creating diagnostic envelope plot for Band {i+1}")
            
            # Convert to UTC time for x-axis
            t_utc = [start + ti for ti in t]
            t_plot = mdates.date2num(t_utc)
            
            # Convert R1, R2, R3 arrival times to UTC (tr1/tr2/tr3 are in minutes)
            r1_utc = mdates.date2num(reference_time + tr1[0,i]*60)
            r2_utc = mdates.date2num(reference_time + tr2[0,i]*60)
            r3_utc = mdates.date2num(reference_time + tr3[0,i]*60)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Plot filtered waveform and envelope
            ax.plot(t_plot, st_filt[1].data, 'k', linewidth=0.5, alpha=0.7, label='Filtered waveform')
            ax.plot(t_plot, data_envelope, 'm-', linewidth=1.5, label='Envelope')
            
            # Plot R1, R2, R3 arrival times
            ax.axvline(r1_utc, color='b', linestyle='-', linewidth=0.8,
                      alpha=0.7, label='R1 (minor arc)')
            ax.axvline(r2_utc, color='c', linestyle='-', linewidth=0.8,
                      alpha=0.7, label='R2 (major arc)')
            ax.axvline(r3_utc, color='g', linestyle='-', linewidth=0.8,
                      alpha=0.7, label='R3 (around globe)')
            
            # Format x-axis with UTC time
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
            # Set tick interval based on data length
            data_duration_sec = len(t) / samprate if len(t) > 1 else 3600
            if data_duration_sec > 10800:  # > 3 hours
                interval = 1800  # 30 minutes
            elif data_duration_sec > 7200:  # > 2 hours
                interval = 900  # 15 minutes
            elif data_duration_sec > 3600:  # > 1 hour
                interval = 600  # 10 minutes
            else:
                interval = 300  # 5 minutes
            ax.xaxis.set_major_locator(SecondLocator(interval=interval))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            ax.set_xlabel('UTC Time (HH:MM:SS)', fontweight='bold', fontsize=10)
            ax.set_ylabel('Amplitude', fontweight='bold', fontsize=10)
            ax.set_title(f'Rayleigh Wave Envelope - Band {i+1}: {band_low[i]:.1f}-{band_high[i]:.1f}s', 
                        fontweight='bold', fontsize=11)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            fig_path = os.path.join(output_dir, f'rayleigh_envelope_band{i+1}.png')
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            tee_print(f"    Saved: {fig_path}")
            
            plt.show(block=False)
            plt.pause(0.1)
        
        # Calculate group velocity and distance using R1/R2/R3 arrival times
        if tr3[0, i] > tr1[0, i]:
            # Calculate angular velocity from R1 and R3 time difference
            Ub = 2*math.pi / (tr3[0, i]*60 - tr1[0, i]*60)
            
            # Convert to linear group velocity
            Ul = Ub * 6371
            
            # Calculate epicentral distance from R2 arrival time
            delta[0, i] = math.pi - 0.5 * Ub * (tr2[0, i]*60 - tr1[0, i]*60)
            
            # Back-calculate origin time from R1 arrival
            t0i[0, i] = tr1[0, i]*60 - delta[0, i] / Ub
            
            # Calculate quality weight based on velocity
            Umean = 3.6
            sigmaU = 0.75
            if abs(Ul - Umean) < sigmaU:
                Iu[0, i] = math.exp(-0.5 * ((Ul - Umean)/sigmaU)**2)
            else:
                Iu[0, i] = 0.1
            
            tee_print(f"    Group velocity: {Ul:.2f} km/s")
            tee_print(f"    Estimated distance: {delta[0,i]*180/math.pi:.2f}°")
        else:
            tee_print(f"    Cannot calculate velocity (R3 <= R1)")
            Iu[0, i] = 0
    
    # Calculate weights
    wbi = (Ar1 + Ar2 + Ar3) / 3 * Iu
    wbmean = wbi / np.max(wbi) if np.max(wbi) > 0 else wbi
    
    # Calculate distance and timing PDFs
    deltaDeg = delta * 180 / math.pi
    mean_delta = np.sum(wbmean * deltaDeg) / np.sum(wbmean) if np.sum(wbmean) > 0 else estimated_distance
    mean_t0 = np.sum(wbmean * t0i) / np.sum(wbmean) if np.sum(wbmean) > 0 else 0
    std_delta = np.sqrt(np.sum(wbmean * (deltaDeg - mean_delta)**2) / np.sum(wbmean)) if np.sum(wbmean) > 0 else 1
    std_t0 = np.sqrt(np.sum(wbmean * (t0i - mean_t0)**2) / np.sum(wbmean)) if np.sum(wbmean) > 0 else 1
    
    # Create PDFs
    al_delts = np.arange(max(0, mean_delta - 20), mean_delta + 20, 0.1)
    al_times = np.arange(mean_t0 - 500, mean_t0 + 500, 1)
    
    X1 = 1/(std_delta*math.sqrt(2*math.pi)) * np.exp(-0.5*((deltaDeg - mean_delta)/std_delta)**2)
    Xall = 1/(std_delta*math.sqrt(2*math.pi)) * np.exp(-0.5*((al_delts - mean_delta)/std_delta)**2)
    X2 = 1/(std_t0*math.sqrt(2*math.pi)) * np.exp(-0.5*((t0i - mean_t0)/std_t0)**2)
    X2all = 1/(std_t0*math.sqrt(2*math.pi)) * np.exp(-0.5*((al_times - mean_t0)/std_t0)**2)
    
    # Plot distance PDF
    if output_dir is not None:
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.plot(deltaDeg[0], X1[0], 'o', markersize=6, label='Individual bands', alpha=0.6)
        ax.plot(al_delts, Xall, 'b-', linewidth=1.5, label='Combined PDF')
        if distance_deg is not None:
            ax.axvline(distance_deg, color='r', linestyle='-', linewidth=0.8,
                      label=f'True distance: {distance_deg}°', alpha=0.8)
        ax.plot(mean_delta, np.max(Xall), 'k*', markersize=12,
               label=f'Estimated: {mean_delta:.2f}°', zorder=10)
        ax.set_xlabel('Distance (degrees)', fontweight='bold', fontsize=9)
        ax.set_ylabel('Probability Density', fontweight='bold', fontsize=9)
        ax.set_title('Rayleigh Wave Distance Estimation', fontweight='bold', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(output_dir, 'rayleigh_distance_pdf.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        tee_print(f"  Saved: {fig_path}")
        
        plt.show(block=False)
        plt.pause(0.1)
    
    # Plot timing PDF
    if output_dir is not None:
        fig, ax = plt.subplots(figsize=(5, 2.5))
        
        # Convert time arrays to UTC
        if reference_time is not None:
            # Convert x-axis array to UTC
            al_times_utc = [reference_time + t for t in al_times]
            al_times_plot = mdates.date2num(al_times_utc)
            
            # Convert individual band estimates to UTC
            t0i_utc = [reference_time + t for t in t0i[0]]
            t0i_plot = mdates.date2num(t0i_utc)
            
            # Convert mean estimate to UTC
            mean_t0_utc = reference_time + mean_t0
            mean_t0_plot = mdates.date2num(mean_t0_utc)
            
            # Plot individual bands and combined PDF
            ax.plot(t0i_plot, X2[0], 'o', markersize=6, label='Individual bands', alpha=0.6)
            ax.plot(al_times_plot, X2all, 'b-', linewidth=1.5, label='Combined PDF')
            
            # Plot true origin time if event_time is known
            if use_known_origin and event_time is not None:
                true_origin_plot = mdates.date2num(event_time)
                ax.axvline(true_origin_plot, color='r', linestyle='-', linewidth=0.8,
                          label=f'True origin: {event_time.strftime("%H:%M:%S")}', alpha=0.8)
            
            # Mark estimated origin time
            ax.plot(mean_t0_plot, np.max(X2all), 'k*', markersize=12,
                   label=f'Estimated: {mean_t0_utc.strftime("%H:%M:%S")}', zorder=10)
            
            # Format x-axis with UTC time
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(SecondLocator(interval=120))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            ax.set_xlabel('Origin Time (UTC, HH:MM:SS)', fontweight='bold', fontsize=9)
        else:
            # Fallback if reference_time not set
            ax.plot(t0i[0], X2[0], 'o', markersize=6, label='Individual bands', alpha=0.6)
            ax.plot(al_times, X2all, 'b-', linewidth=1.5, label='Combined PDF')
            ax.plot(mean_t0, np.max(X2all), 'k*', markersize=12,
                   label=f'Estimated: {mean_t0:.1f}s after reference', zorder=10)
            ax.set_xlabel('Time after reference (s)', fontweight='bold', fontsize=9)
        
        ax.set_ylabel('Probability Density', fontweight='bold', fontsize=9)
        ax.set_title('Rayleigh Wave Origin Time PDF', fontweight='bold', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(output_dir, 'rayleigh_timing_pdf.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        tee_print(f"  Saved: {fig_path}")
        
        plt.show(block=False)
        plt.pause(0.1)
    
    tee_print(f"\n{'='*80}")
    tee_print(f"RAYLEIGH WAVE ANALYSIS COMPLETE")
    tee_print(f"{'='*80}")
    tee_print(f"Estimated distance: {mean_delta:.2f}° (±{std_delta:.2f}°)")
    if distance_deg is not None:
        tee_print(f"True distance: {distance_deg}°")
        tee_print(f"Error: {abs(mean_delta - distance_deg):.2f}°")
    tee_print(f"Estimated origin time offset: {mean_t0:.1f}s (±{std_t0:.1f}s)")
    tee_print(f"{'='*80}\n")
    
    # Return PDF arrays for potential combination with body wave PDFs
    # Also return tr1, band_high, band_low for R1 polarization analysis
    return mean_delta, al_delts, Xall, al_times, X2all, tr1, band_high, band_low


def extract_orbit_pdfs(detected_orbits, catalog_distance_deg, event_offset_sec,
                       output_dir, event_name, use_known_origin=False, distance_deg=None):
    """
    Extract and plot distance/timing PDFs from Stockwell orbit detection results.
    
    This function processes the output from rayleigh_orbit_detector to create
    probability density functions for both distance and arrival timing estimates.
    
    Parameters:
        detected_orbits: dict with keys 'R1', 'R2', 'R3' containing orbit detection results
        catalog_distance_deg: reference distance from body wave analysis (degrees)
        event_offset_sec: time offset between data start and event origin (seconds)
        output_dir: output directory for plots
        event_name: event name for plot titles and filenames
        use_known_origin: bool, True if true distance is known (for validation plots)
        distance_deg: true distance for validation (degrees), optional
    
    Returns:
        tuple: (distance_pdf_dict, timing_pdf_dict)
            distance_pdf_dict: dict with keys 'x', 'pdf', 'estimate', 'std'
            timing_pdf_dict: dict with keys 'x', 'pdf', 'estimate', 'std' or None
    """
    tee_print(f"\n{'='*80}")
    tee_print(f"EXTRACTING ORBIT PDFs FROM STOCKWELL RESULTS")
    tee_print(f"{'='*80}")
    
    # ==================================================================================
    # STEP 1: Extract distance estimates from orbits
    # ==================================================================================
    
    orbit_distances = []
    orbit_confidences = []
    
    for orbit_name in ['R1', 'R2', 'R3']:
        if orbit_name in detected_orbits and detected_orbits[orbit_name] is not None:
            orbit_data = detected_orbits[orbit_name]
            if 'detected' in orbit_data and orbit_data['detected']:
                orbit_distance_deg = orbit_data['distance_deg']
                confidence = orbit_data['confidence']
                
                orbit_distances.append(orbit_distance_deg)
                orbit_confidences.append(confidence)
                
                tee_print(f"\n{orbit_name} detected:")
                tee_print(f"  Distance: {orbit_distance_deg:.2f}°")
                tee_print(f"  Confidence: {confidence:.3f}")
    
    if len(orbit_distances) == 0:
        tee_print("\nWARNING: No orbits detected, cannot create distance PDF")
        return None, None
    
    # Calculate weighted mean and standard deviation
    weights = np.array(orbit_confidences)
    weights = weights / np.sum(weights)  # Normalize
    
    mean_distance = np.average(orbit_distances, weights=weights)
    variance = np.average((np.array(orbit_distances) - mean_distance)**2, weights=weights)
    std_distance = np.sqrt(variance)
    
    # Add uncertainty floor (minimum ~2° from velocity model uncertainty)
    if std_distance < 2.0:
        std_distance = 2.0
    
    tee_print(f"\nWeighted Distance Estimate:")
    tee_print(f"  Mean: {mean_distance:.2f}°")
    tee_print(f"  Std:  {std_distance:.2f}°")
    tee_print(f"  Orbits used: {len(orbit_distances)}")
    
    # Create distance PDF (Gaussian)
    distance_x = np.arange(max(0, mean_distance - 30), mean_distance + 30, 0.1)
    distance_pdf = np.exp(-0.5 * ((distance_x - mean_distance) / std_distance)**2)
    distance_pdf = distance_pdf / np.trapz(distance_pdf, distance_x)  # Normalize
    
    distance_estimate = mean_distance
    
    # ==================================================================================
    # STEP 2: Extract timing estimates from orbits (if dispersion data available)
    # ==================================================================================
    
    orbit_origin_times = []
    orbit_arrival_times = []
    orbit_confidences_timing = []
    
    for orbit_name in ['R1', 'R2', 'R3']:
        if orbit_name in detected_orbits and detected_orbits[orbit_name] is not None:
            orbit_data = detected_orbits[orbit_name]
            if 'detected' in orbit_data and orbit_data['detected']:
                if 'dispersion' in orbit_data and orbit_data['dispersion'] is not None:
                    distance_deg = orbit_data['distance_deg']
                    disp_data = orbit_data['dispersion']
                    
                    # Peak time is arrival time (minutes after event origin from dispersion analysis)
                    arrival_time_min = disp_data['peak_time']
                    arrival_time_sec = arrival_time_min * 60.0  # Convert to seconds
                    
                    # Get periods from dispersion curve to calculate representative group velocity
                    periods_sec = np.array(disp_data['periods'])
                    arrivals_min = np.array(disp_data['arrivals'])
                    
                    # Use median period from the dispersion curve
                    median_period_sec = np.median(periods_sec)
                    
                    # Calculate group velocity using PREM for this period
                    group_velocity_km_s = rayleigh_group_velocity_prem(median_period_sec)
                    
                    # Calculate travel time = distance / velocity
                    distance_km = distance_deg * (np.pi / 180.0) * 6371.0
                    travel_time_sec = distance_km / group_velocity_km_s
                    
                    # Back-calculate origin time
                    # arrival_time is relative to event origin (should equal travel_time if consistent)
                    # origin_time = arrival_time - travel_time
                    # If perfectly consistent, this should be ~0
                    origin_time_sec = arrival_time_sec - travel_time_sec
                    
                    orbit_origin_times.append(origin_time_sec)
                    orbit_arrival_times.append(arrival_time_sec)
                    orbit_confidences_timing.append(orbit_data['confidence'])
                    
                    tee_print(f"\n{orbit_name}:")
                    tee_print(f"  Distance: {distance_deg:.1f} deg ({distance_km:.0f} km)")
                    tee_print(f"  Median period: {median_period_sec:.1f} s")
                    tee_print(f"  Group velocity: {group_velocity_km_s:.3f} km/s")
                    tee_print(f"  Arrival time: {arrival_time_min:.2f} min = {arrival_time_sec:.1f} s")
                    tee_print(f"  Travel time: {travel_time_sec/60.0:.2f} min = {travel_time_sec:.1f} s")
                    tee_print(f"  Back-calculated origin: {origin_time_sec:.1f} s (relative to event)")
    
    if len(orbit_origin_times) > 0:
        # Calculate weighted mean origin time
        weights_timing = np.array(orbit_confidences_timing)
        weights_timing = weights_timing / np.sum(weights_timing)
        mean_origin_sec = np.average(orbit_origin_times, weights=weights_timing)
        std_origin_sec = np.sqrt(np.average((np.array(orbit_origin_times) - mean_origin_sec)**2, 
                                            weights=weights_timing))
        
        # Add uncertainty from distance/velocity uncertainties (typically ~50-100s)
        uncertainty_from_model = 100.0  # seconds (accounts for velocity model uncertainty)
        total_std = np.sqrt(std_origin_sec**2 + uncertainty_from_model**2)
        
        # Create timing PDF (centered on mean origin time)
        timing_x = np.arange(mean_origin_sec - 300, mean_origin_sec + 300, 1.0)
        timing_pdf = np.exp(-0.5 * ((timing_x - mean_origin_sec) / total_std)**2)
        timing_pdf = timing_pdf / np.trapz(timing_pdf, timing_x)
        
        tee_print(f"\nOrigin Time PDF (back-calculated from Rayleigh arrivals):")
        tee_print(f"  Mean origin estimate: {mean_origin_sec:.1f} s (relative to event)")
        tee_print(f"  Measurement scatter: ±{std_origin_sec:.1f} s")
        tee_print(f"  Model uncertainty: ±{uncertainty_from_model:.1f} s")
        tee_print(f"  Total uncertainty: ±{total_std:.1f} s")
        tee_print(f"  Orbits used: {len(orbit_origin_times)}")
        
        # Calculate mean and std for arrival times
        mean_arrival_sec = np.average(orbit_arrival_times, weights=weights_timing)
        std_arrival_sec = np.sqrt(np.average((np.array(orbit_arrival_times) - mean_arrival_sec)**2, 
                                             weights=weights_timing))
    else:
        tee_print("\nWARNING: No orbit timing data, cannot create timing PDF")
        timing_x = None
        timing_pdf = None
        mean_origin_sec = None
        mean_arrival_sec = None
        std_arrival_sec = None
    
    # ==================================================================================
    # STEP 3: Plot distance PDF
    # ==================================================================================
    
    fig, ax = plt.subplots(figsize=(5, 2.5))
    
    ax.fill_between(distance_x, distance_pdf, alpha=0.3, color='purple')
    ax.plot(distance_x, distance_pdf, 'purple', linewidth=2.5, label='Stockwell Distance PDF')
    
    # Mark estimate
    ax.axvline(distance_estimate, color='darkviolet', linestyle='-', linewidth=2,
              label=f'Stockwell: {distance_estimate:.1f}° ± {std_distance:.1f}°')
    
    # Mark catalog distance
    if catalog_distance_deg is not None:
        ax.axvline(catalog_distance_deg, color='green', linestyle='--', linewidth=2,
                  label=f'Body Wave: {catalog_distance_deg:.1f}°', alpha=0.7)
        
        error = abs(distance_estimate - catalog_distance_deg)
        tee_print(f"\nDistance comparison:")
        tee_print(f"  Stockwell: {distance_estimate:.1f}°")
        tee_print(f"  Body wave: {catalog_distance_deg:.1f}°")
        tee_print(f"  Difference: {error:.1f}°")
    
    # Mark true distance if known
    if use_known_origin and distance_deg is not None:
        ax.axvline(distance_deg, color='red', linestyle='-', linewidth=1.5,
                  label=f'True: {distance_deg:.1f}°', alpha=0.8)
    
    ax.set_xlabel('Distance (°)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Probability Density', fontweight='bold', fontsize=12)
    ax.set_title('Stockwell Orbit Detection - Distance PDF', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, f'{event_name}_stockwell_distance_pdf.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    tee_print(f"\nSaved distance PDF: {fig_path}")
    plt.show(block=False)
    plt.pause(0.1)
    
    # ==================================================================================
    # STEP 4: Plot timing PDF if available
    # ==================================================================================
    
    if timing_x is not None and timing_pdf is not None:
        fig, ax = plt.subplots(figsize=(5, 2.5))
        
        ax.fill_between(timing_x, timing_pdf, alpha=0.3, color='orange')
        ax.plot(timing_x, timing_pdf, 'orange', linewidth=2.5, label='Stockwell Timing PDF')
        
        # Mark estimate
        ax.axvline(mean_arrival_sec, color='darkorange', linestyle='-', linewidth=2,
                  label=f'Mean: {mean_arrival_sec:.1f} s ± {std_arrival_sec:.1f} s')
        
        # Mark individual orbit arrivals
        colors_orbits = {'R1': 'blue', 'R2': 'cyan', 'R3': 'green'}
        for i, (orbit_name, orbit_time) in enumerate(zip(['R1', 'R2', 'R3'], orbit_arrival_times)):
            if i < len(orbit_arrival_times):
                ax.axvline(orbit_time, color=colors_orbits.get(orbit_name, 'gray'), 
                          linestyle=':', linewidth=1.5, alpha=0.6,
                          label=f'{orbit_name}: {orbit_time:.1f} s')
        
        ax.set_xlabel('Time after origin (s)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Probability Density', fontweight='bold', fontsize=12)
        ax.set_title('Stockwell Orbit Detection - Arrival Time PDF', fontweight='bold', fontsize=13)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig_path = os.path.join(output_dir, f'{event_name}_stockwell_timing_pdf.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        tee_print(f"Saved timing PDF: {fig_path}")
        plt.show(block=False)
        plt.pause(0.1)
    
    tee_print(f"{'='*80}\n")
    
    # Package results
    distance_pdf_dict = {
        'x': distance_x,
        'pdf': distance_pdf,
        'estimate': distance_estimate,
        'std': std_distance
    }
    
    timing_pdf_dict = {
        'x': timing_x,
        'pdf': timing_pdf,
        'estimate': mean_arrival_sec,
        'std': std_arrival_sec
    } if timing_x is not None else None
    
    return distance_pdf_dict, timing_pdf_dict


# ============================================================================
# STOCKWELL TRANSFORM PDF EXTRACTION FUNCTIONS
# ============================================================================

def extract_stockwell_distance_pdf(detected_orbits, catalog_distance_deg=None, n_points=721):
    """
    Extract distance PDF from Stockwell orbit detection results.
    
    Uses the Gaussian approximation method from run_example_events.py:
    - Collects distances and confidences from detected R1, R2, R3 orbits
    - Calculates weighted mean and variance
    - Generates Gaussian PDF centered on weighted mean
    
    Parameters
    ----------
    detected_orbits : dict
        Output from detect_rayleigh_orbits_from_stockwell()
        Expected structure: {
            'R1': {'distance_deg': float, 'confidence': float, ...},
            'R2': {...},
            'R3': {...},
            'metadata': {...}
        }
    catalog_distance_deg : float, optional
        Catalog distance for fallback/validation
    n_points : int, optional
        Number of points in output PDF grid (default: 721)
    
    Returns
    -------
    distance_pdf_dict : dict or None
        {
            'x': np.ndarray,          # Distance grid in degrees
            'pdf': np.ndarray,        # Normalized probability density
            'estimate': float,        # MAP estimate (weighted mean)
            'std': float,             # Standard deviation
            'n_orbits': int,          # Number of orbits used
            'orbit_distances': list,  # Individual orbit distances
            'orbit_confidences': list # Individual orbit confidences
        }
        Returns None if no orbits detected.
    
    Notes
    -----
    - PDF is normalized so max(pdf) = 1.0
    - If only 1 orbit detected, uses default sigma = 5.0 degrees
    - If 2+ orbits detected, sigma from weighted variance
    - Grid extends ±5*sigma around weighted mean
    
    Examples
    --------
    >>> detected_orbits = detect_rayleigh_orbits_from_stockwell(...)
    >>> dist_pdf = extract_stockwell_distance_pdf(detected_orbits)
    >>> print(f"Distance: {dist_pdf['estimate']:.1f} ± {dist_pdf['std']:.1f} deg")
    >>> plt.plot(dist_pdf['x'], dist_pdf['pdf'])
    """
    if not isinstance(detected_orbits, dict):
        return None
    
    # Extract distances and confidences from detected orbits
    orbit_distances = []
    orbit_confidences = []
    
    for orbit in ['R1', 'R2', 'R3']:
        if orbit in detected_orbits and isinstance(detected_orbits[orbit], dict):
            orbit_data = detected_orbits[orbit]
            if 'distance_deg' in orbit_data:
                dist = float(orbit_data['distance_deg'])
                conf = float(orbit_data.get('confidence', 1.0))
                orbit_distances.append(dist)
                orbit_confidences.append(conf)
    
    n_orbits = len(orbit_distances)
    
    if n_orbits == 0:
        print("Warning: No orbits detected in Stockwell results - cannot extract distance PDF")
        return None
    
    # Convert to arrays
    orbit_distances = np.array(orbit_distances, dtype=float)
    orbit_confidences = np.array(orbit_confidences, dtype=float)
    
    # Normalize confidences to use as weights
    if np.sum(orbit_confidences) <= 0:
        orbit_confidences = np.ones_like(orbit_confidences)
    weights = orbit_confidences / np.sum(orbit_confidences)
    
    # Calculate weighted mean distance
    mean_distance = np.sum(orbit_distances * weights)
    
    # Calculate weighted standard deviation
    if n_orbits > 1:
        # Weighted variance
        variance = np.sum(weights * (orbit_distances - mean_distance)**2)
        sigma = max(np.sqrt(variance), 2.0)  # Minimum 2 degrees
    else:
        # Single orbit: use default uncertainty
        sigma = 5.0  # degrees
    
    # Check metadata for explicit uncertainty
    metadata = detected_orbits.get('metadata', {})
    if 'distance_std_deg' in metadata:
        try:
            sigma = max(float(metadata['distance_std_deg']), 1.0)
        except:
            pass
    
    # Define grid extent
    if catalog_distance_deg is not None and np.isfinite(catalog_distance_deg):
        center = mean_distance
        lo = max(0.0, min(catalog_distance_deg, center) - 5.0 * sigma - 10.0)
        hi = min(180.0, max(catalog_distance_deg, center) + 5.0 * sigma + 10.0)
    else:
        center = mean_distance
        lo = max(0.0, center - 5.0 * sigma - 10.0)
        hi = min(180.0, center + 5.0 * sigma + 10.0)
    
    if hi <= lo:
        lo = max(0.0, center - 30.0)
        hi = min(180.0, center + 30.0)
    
    # Create grid and Gaussian PDF
    distance_grid = np.linspace(lo, hi, n_points)
    pdf = np.exp(-0.5 * ((distance_grid - mean_distance) / sigma)**2)
    
    # Normalize PDF (max = 1.0 for plotting)
    pdf = pdf / np.max(pdf)
    
    return {
        'x': distance_grid,
        'pdf': pdf,
        'estimate': mean_distance,
        'std': sigma,
        'n_orbits': n_orbits,
        'orbit_distances': orbit_distances.tolist(),
        'orbit_confidences': orbit_confidences.tolist()
    }


def extract_stockwell_timing_pdf(detected_orbits, event_time, n_points=721):
    """
    Extract origin time PDF from Stockwell orbit detection results.
    
    Uses arrival times from R1, R2, R3 orbits to back-calculate origin time estimates.
    Creates Gaussian PDF from weighted mean and variance of origin time estimates.
    
    Parameters
    ----------
    detected_orbits : dict
        Output from detect_rayleigh_orbits_from_stockwell()
    event_time : UTCDateTime or float
        Reference event time (for calculating relative timing)
        If float, interpreted as seconds from some reference
    n_points : int, optional
        Number of points in output PDF grid (default: 721)
    
    Returns
    -------
    timing_pdf_dict : dict or None
        {
            'x': np.ndarray,          # Time grid (seconds relative to event_time)
            'pdf': np.ndarray,        # Normalized probability density
            'estimate': float,        # MAP estimate (weighted mean offset)
            'std': float,             # Standard deviation (seconds)
            'n_orbits': int,          # Number of orbits used
            'orbit_arrivals': list,   # Arrival times (minutes)
            'orbit_confidences': list # Orbit confidences
        }
        Returns None if no orbits with arrival times detected.
    
    Notes
    -----
    - Arrival times from dispersion curves are in minutes after origin
    - Back-calculates origin time: origin = arrival_time - predicted_travel_time
    - PDF is normalized so max(pdf) = 1.0
    - If only 1 orbit, uses default sigma = 10 seconds
    
    Examples
    --------
    >>> from obspy import UTCDateTime
    >>> event_time = UTCDateTime("2015-06-29T09:09:21")
    >>> timing_pdf = extract_stockwell_timing_pdf(detected_orbits, event_time)
    >>> print(f"Origin time offset: {timing_pdf['estimate']:.1f} ± {timing_pdf['std']:.1f} s")
    """
    if not isinstance(detected_orbits, dict):
        return None
    
    # Extract arrival times and confidences
    orbit_arrivals_min = []
    orbit_confidences = []
    orbit_names = []
    
    for orbit in ['R1', 'R2', 'R3']:
        if orbit in detected_orbits and isinstance(detected_orbits[orbit], dict):
            orbit_data = detected_orbits[orbit]
            
            # Get arrival time from dispersion data
            dispersion = orbit_data.get('dispersion', {})
            if 'peak_time' in dispersion:
                arrival_min = float(dispersion['peak_time'])
                conf = float(orbit_data.get('confidence', 1.0))
                
                orbit_arrivals_min.append(arrival_min)
                orbit_confidences.append(conf)
                orbit_names.append(orbit)
    
    n_orbits = len(orbit_arrivals_min)
    
    if n_orbits == 0:
        print("Warning: No orbit arrival times found in Stockwell results")
        return None
    
    # Convert to arrays
    orbit_arrivals_min = np.array(orbit_arrivals_min, dtype=float)
    orbit_confidences = np.array(orbit_confidences, dtype=float)
    
    # Normalize confidences
    if np.sum(orbit_confidences) <= 0:
        orbit_confidences = np.ones_like(orbit_confidences)
    weights = orbit_confidences / np.sum(orbit_confidences)
    
    # Calculate weighted mean arrival time
    mean_arrival_min = np.sum(orbit_arrivals_min * weights)
    mean_arrival_sec = mean_arrival_min * 60.0
    
    # Calculate weighted standard deviation of arrival times
    if n_orbits > 1:
        variance_min = np.sum(weights * (orbit_arrivals_min - mean_arrival_min)**2)
        sigma_sec = max(np.sqrt(variance_min) * 60.0, 5.0)  # Minimum 5 seconds
    else:
        # Single orbit: use default uncertainty
        sigma_sec = 10.0  # seconds
    
    # Check metadata for explicit timing uncertainty
    metadata = detected_orbits.get('metadata', {})
    if 'timing_std_sec' in metadata:
        try:
            sigma_sec = max(float(metadata['timing_std_sec']), 1.0)
        except:
            pass
    
    # Create grid (seconds relative to event_time)
    # For Rayleigh waves, we're looking at the arrival time, not origin time correction
    # So the PDF is centered on the mean arrival time
    lo = mean_arrival_sec - 5.0 * sigma_sec - 60.0
    hi = mean_arrival_sec + 5.0 * sigma_sec + 60.0
    
    timing_grid = np.linspace(lo, hi, n_points)
    pdf = np.exp(-0.5 * ((timing_grid - mean_arrival_sec) / sigma_sec)**2)
    
    # Normalize PDF
    pdf = pdf / np.max(pdf)
    
    return {
        'x': timing_grid,
        'pdf': pdf,
        'estimate': mean_arrival_sec,
        'std': sigma_sec,
        'n_orbits': n_orbits,
        'orbit_arrivals': orbit_arrivals_min.tolist(),
        'orbit_confidences': orbit_confidences.tolist()
    }


def plot_stockwell_pdfs(distance_pdf, timing_pdf, event_name, output_dir, catalog_distance=None):
    """
    Plot distance and timing PDFs extracted from Stockwell results.
    
    Parameters
    ----------
    distance_pdf : dict
        Output from extract_stockwell_distance_pdf()
    timing_pdf : dict or None
        Output from extract_stockwell_timing_pdf()
    event_name : str
        Event identifier for plot titles and filenames
    output_dir : str
        Directory to save plots
    catalog_distance : float, optional
        True/catalog distance to show on plot
    
    Returns
    -------
    fig_paths : list
        Paths to saved figure files
    """
    import matplotlib.pyplot as plt
    
    fig_paths = []
    
    # Plot distance PDF
    if distance_pdf is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(distance_pdf['x'], distance_pdf['pdf'], 'b-', linewidth=2, label='Stockwell PDF')
        ax.axvline(distance_pdf['estimate'], color='b', linestyle='--', linewidth=1.5,
                   label=f"Estimate: {distance_pdf['estimate']:.1f}° ± {distance_pdf['std']:.1f}°")
        
        # Mark individual orbit distances
        for i, (dist, conf) in enumerate(zip(distance_pdf['orbit_distances'], 
                                             distance_pdf['orbit_confidences'])):
            orbit_name = ['R1', 'R2', 'R3'][i] if i < 3 else f'Orbit{i+1}'
            ax.axvline(dist, color='gray', linestyle=':', alpha=0.5)
            ax.text(dist, 0.9 - i*0.1, f'{orbit_name}: {dist:.1f}° (conf={conf:.2f})',
                   rotation=90, va='top', fontsize=9, alpha=0.7)
        
        if catalog_distance is not None and np.isfinite(catalog_distance):
            ax.axvline(catalog_distance, color='r', linestyle='--', linewidth=1.5,
                      label=f'Catalog: {catalog_distance:.1f}°')
        
        ax.set_xlabel('Distance (degrees)', fontsize=12)
        ax.set_ylabel('Normalized Probability Density', fontsize=12)
        ax.set_title(f'{event_name} - Stockwell Distance PDF ({distance_pdf["n_orbits"]} orbits)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig_path = os.path.join(output_dir, f'{event_name}_stockwell_distance_pdf.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        fig_paths.append(fig_path)
        print(f"Saved Stockwell distance PDF: {fig_path}")
        plt.close()
    
    # Plot timing PDF
    if timing_pdf is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(timing_pdf['x'] / 60.0, timing_pdf['pdf'], 'g-', linewidth=2, label='Stockwell PDF')
        ax.axvline(timing_pdf['estimate'] / 60.0, color='g', linestyle='--', linewidth=1.5,
                   label=f"Estimate: {timing_pdf['estimate']/60:.1f} min ± {timing_pdf['std']/60:.1f} min")
        
        # Mark individual orbit arrivals
        for i, (arrival, conf) in enumerate(zip(timing_pdf['orbit_arrivals'],
                                                timing_pdf['orbit_confidences'])):
            orbit_name = ['R1', 'R2', 'R3'][i] if i < 3 else f'Orbit{i+1}'
            ax.axvline(arrival, color='gray', linestyle=':', alpha=0.5)
            ax.text(arrival, 0.9 - i*0.1, f'{orbit_name}: {arrival:.1f} min (conf={conf:.2f})',
                   rotation=90, va='top', fontsize=9, alpha=0.7)
        
        ax.set_xlabel('Arrival Time (minutes after origin)', fontsize=12)
        ax.set_ylabel('Normalized Probability Density', fontsize=12)
        ax.set_title(f'{event_name} - Stockwell Timing PDF ({timing_pdf["n_orbits"]} orbits)',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig_path = os.path.join(output_dir, f'{event_name}_stockwell_timing_pdf.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        fig_paths.append(fig_path)
        print(f"Saved Stockwell timing PDF: {fig_path}")
        plt.close()
    
    return fig_paths
