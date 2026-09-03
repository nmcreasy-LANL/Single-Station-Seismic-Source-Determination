#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backazimuth Analysis Module

This module contains functions for estimating backazimuth from seismic data using:
- Rayleigh wave polarization analysis
- Stockwell transform with orbit detection
- Multi-band Hilbert-Bayes algorithm

Created on Thu Jul 23 08:48:55 2026
@author: nmcreasy
"""

import os
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from obspy.core import UTCDateTime

# Add parent directory to path for rayleigh_wave_tools import
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)

# Import Rayleigh wave tools
from rayleigh_wave_tools import estimate_rayleigh_baz_posterior
from rayleigh_wave_tools.rayleigh_filter import (
    simple_stockwell_filter_rayleigh,
    add_particleman_nip_energy_masks,
)
from rayleigh_wave_tools.rayleigh_orbit_detector import detect_rayleigh_orbits_from_stockwell
from rayleigh_wave_tools.plotting import plot_rayleigh_filter_diagnostics

# Import common utilities
UTILS_DIR = os.path.join(PARENT_DIR, 'utils')
sys.path.insert(0, UTILS_DIR)
import common_utils
from common_utils import tee_print

# Global variables (set by main script)
EVENT_TIME = None
P_ARRIVAL_TIME = None
DISTANCE_DEG = None

#=================================================================================================
# Backazimuth Extraction Helper Functions
#=================================================================================================

def calculate_baz_from_kde(kde_data, kde_weights):
    """
    Calculate backazimuth and error from KDE data.
    
    Parameters:
        kde_data: Array of azimuth values
        kde_weights: Array of weights for KDE
    
    Returns:
        baz: Peak backazimuth in degrees
        error: [left_error, right_error] in degrees
        xs: x-values for plotting
        ys: y-values for plotting
    """
    from scipy import stats
    
    # Convert to numpy array and wrap negative angles to 0-360 range
    kde_data = np.array(kde_data)
    kde_data = np.where(kde_data < 0, kde_data + 360, kde_data)
    
    # Create KDE with tight bandwidth to show more structure
    kernel = stats.gaussian_kde(kde_data, weights=kde_weights)
    kernel.covariance_factor = lambda: .08  # Tighter fit - shows more detail
    kernel._compute_covariance()
    
    # Extend to positive and negative spaces for wrapping
    xs = np.linspace(-50, 360+50, 1000)
    ys = kernel(xs)
    
    # Find maximum
    index = np.argmax(ys)
    baz = xs[index]
    
    # Wrap BAZ to 0-360 range
    if baz < 0:
        baz = baz + 360
    elif baz > 360:
        baz = baz - 360
    
    # Calculate FWHM error (half-width at half-maximum)
    half_max = ys[index] / 2.0
    
    # Find left bound
    left_indices = np.where(xs < baz)[0]
    if len(left_indices) > 0:
        left_crossings = np.where(ys[left_indices] >= half_max)[0]
        if len(left_crossings) > 0:
            left_error = baz - xs[left_indices[left_crossings[0]]]
        else:
            left_error = 20.0  # Default
    else:
        left_error = 20.0
    
    # Find right bound
    right_indices = np.where(xs > baz)[0]
    if len(right_indices) > 0:
        right_crossings = np.where(ys[right_indices] >= half_max)[0]
        if len(right_crossings) > 0:
            right_error = xs[right_indices[right_crossings[-1]]] - baz
        else:
            right_error = 20.0  # Default
    else:
        right_error = 20.0
    
    error = [baz - left_error, baz + right_error]
    
    return baz, error, xs, ys


def extract_baz_from_pickle(pickle_path, wave_type, output_dir, event_name):
    """
    Extract backazimuth PDF from polarization pickle file.
    
    Parameters:
        pickle_path: Path to pickle file
        wave_type: 'P' or 'P-from-S'
        output_dir: Output directory for plots
        event_name: Event name for file naming
    
    Returns:
        tuple: (baz, error, pdf_x, pdf_y) or (None, None, None, None) if error
    """
    import pickle as pkl
    
    output_file = os.path.join(output_dir, f'{event_name}_BAZ_{wave_type.replace("-", "_")}.png')
    
    tee_print(f"  Extracting {wave_type} backazimuth...")
    
    # Check if pickle file exists
    if not os.path.exists(pickle_path):
        tee_print(f"  ERROR: Pickle file not found: {pickle_path}")
        return None, None, None, None
    
    try:
        # Load pickle file directly
        with open(pickle_path, 'rb') as f:
            kde_dataframe = pkl.load(f)
        
        if wave_type == 'P':
            # Extract P-wave data
            if isinstance(kde_dataframe, dict) and 'P' in kde_dataframe:
                p_wave_list = kde_dataframe['P']
                p_wave_data = p_wave_list[1]  # Row 1 = azimuth
                
                if 'P' in p_wave_data and 'weights' in p_wave_data:
                    # Calculate BAZ and get full PDF
                    baz, error, xs, ys = calculate_baz_from_kde(
                        p_wave_data['P'],
                        p_wave_data['weights']
                    )
                    
                    # Create plot
                    fig, ax = plt.subplots(figsize=(5, 2.5))
                    valid_mask = (xs >= 0) & (xs <= 360)
                    ax.plot(xs[valid_mask], ys[valid_mask], 'C0', linewidth=2.5, label='P-wave')
                    ax.fill_between(xs[valid_mask], ys[valid_mask], alpha=0.3, color='C0')
                    ax.axvline(baz, color='C0', linestyle='--', linewidth=1.5, alpha=0.7)
                    ax.set_xlabel('Backazimuth (°)', fontsize=13)
                    ax.set_ylabel('Probability Density', fontsize=13)
                    ax.set_title('P-wave Backazimuth PDF', fontsize=14, fontweight='bold')
                    ax.set_xlim(0, 360)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    plt.tight_layout()
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    if baz is not None:
                        tee_print(f"  {wave_type} BAZ: {baz:.1f}° [{error[0]:.1f}°, {error[1]:.1f}°]")
                        tee_print(f"  Plot saved: {output_file}")
                    
                    return baz, error, xs, ys
                else:
                    tee_print(f"  ERROR: Expected data structure not found")
                    return None, None, None, None
            else:
                tee_print(f"  ERROR: Unexpected pickle structure")
                return None, None, None, None
                
        else:  # P-from-S
            # For P-from-S, use cross-product method (more complex)
            tee_print(f"  P-from-S extraction not yet implemented in standalone mode")
            tee_print(f"  Use plot_pwave_baz_from_pickle.py for P-from-S analysis")
            return None, None, None, None
        
    except Exception as e:
        tee_print(f"ERROR extracting BAZ: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

#=================================================================================================
# Advanced Rayleigh Wave Analysis with Stockwell Transform and Orbit Detection
#=================================================================================================

def analyze_rayleigh_waves_stockwell(st_raw, output_dir, event_name,
                                     catalog_distance_deg=None, use_sac_headers=True):
    """
    Advanced Rayleigh wave analysis using Stockwell transform and orbit detection.
    
    This function implements the complete workflow from run_example_events.py:
    1. Extract event timing and distance from SAC headers
    2. Estimate backazimuth using multi-band Hilbert-Bayes algorithm
    3. Apply Stockwell filter with NIP energy masks
    4. Detect R1/R2/R3 orbits with joint distance fitting
    5. Generate comprehensive diagnostic plots
    6. Save distance and timing PDFs
    
    Parameters:
        st_raw: Raw ObsPy stream (3-component: E, Z, N order)
        output_dir: Output directory for saving results
        event_name: Event name for file naming
        catalog_distance_deg: Fallback distance if SAC header unavailable (degrees)
        use_sac_headers: If True, extract timing/distance from SAC headers
    
    Returns:
        dict: {
            'distance_deg': estimated distance,
            'distance_pdf': distance PDF arrays,
            'timing_pdf': origin time PDF arrays,
            'baz_deg': estimated backazimuth,
            'baz_pdf': backazimuth PDF arrays,
            'detected_orbits': orbit detection results,
            'reliability': analysis reliability metrics
        }
    """
    tee_print(f"\n{'='*80}")
    tee_print(f"ADVANCED RAYLEIGH WAVE ANALYSIS - STOCKWELL + ORBIT DETECTION")
    tee_print(f"{'='*80}")
    
    # ==================================================================================
    # STEP 1: Extract event information from SAC headers
    # ==================================================================================
    
    trZ = st_raw.select(channel="*Z")[0]
    trN = st_raw.select(channel="*N")[0]
    trE = st_raw.select(channel="*E")[0]
    
    dataN, dataE, dataZ = trN.data, trE.data, trZ.data
    fs = trZ.stats.sampling_rate
    
    tee_print(f"\nLoaded data: {len(dataZ)} samples @ {fs} Hz")
    tee_print(f"  Duration: {(len(dataZ) - 1) / fs / 60.0:.2f} minutes")
    
    if use_sac_headers:
        try:
            timing = common_utils.timing_from_sac_origin(trZ)
            catalog_distance = common_utils.get_sac_distance_deg(trZ, fallback_distance_deg=catalog_distance_deg)
            header_baz = common_utils.get_sac_baz_deg(trZ)
            
            event_time = timing['event_time']
            event_offset_sec = timing['event_offset_sec']
            
            tee_print(f"\nEvent information from SAC headers:")
            tee_print(f"  Event origin time: {event_time}")
            tee_print(f"  Trace start offset: {event_offset_sec:.6f} s after origin")
            tee_print(f"  Distance (gcarc): {catalog_distance:.3f}°")
            if header_baz is not None:
                tee_print(f"  Header backazimuth: {header_baz:.3f}°")
        except Exception as e:
            tee_print(f"\nWARNING: Could not extract SAC headers: {e}")
            tee_print("Falling back to manual parameters...")
            use_sac_headers = False
    
    if not use_sac_headers:
        # Use global EVENT_TIME and P_ARRIVAL_TIME if available
        if EVENT_TIME is not None:
            event_time = EVENT_TIME
            event_offset_sec = float(trZ.stats.starttime - EVENT_TIME)
        elif P_ARRIVAL_TIME is not None:
            # Use P-arrival as proxy for event time
            event_time = P_ARRIVAL_TIME
            event_offset_sec = 0.0
            tee_print("WARNING: Using P-arrival time as event reference")
        else:
            tee_print("ERROR: No timing reference available")
            return None
        
        catalog_distance = catalog_distance_deg if catalog_distance_deg else DISTANCE_DEG
        header_baz = None
        
        tee_print(f"\nUsing manual parameters:")
        tee_print(f"  Event/reference time: {event_time}")
        tee_print(f"  Estimated distance: {catalog_distance:.3f}°")
    
    # ==================================================================================
    # STEP 2: Estimate backazimuth using multi-band algorithm
    # ==================================================================================
    
    tee_print(f"\n{'='*80}")
    tee_print(f"ESTIMATING BACK-AZIMUTH")
    tee_print(f"{'='*80}")
    
    posterior = estimate_rayleigh_baz_posterior(
        north=dataN,
        east=dataE,
        vertical=dataZ,
        fs=fs,
        target_dt=0.2,
        fmin_hz=0.01,
        fmax_hz=3.0,
        return_band_details=True
    )
    
    baz_deg = float(posterior['best_estimate_deg'])
    baz_std = float(posterior.get('posterior_std_deg', 0))
    
    # Convert reliability string to numeric value
    reliability_map = {'high': 1.0, 'medium': 0.5, 'low': 0.25}
    baz_reliability_str = posterior.get('reliability', 'low')
    baz_reliability = reliability_map.get(baz_reliability_str, 0.0) if isinstance(baz_reliability_str, str) else float(baz_reliability_str)
    
    tee_print(f"\nEstimated BAZ: {baz_deg:.2f}° (±{baz_std:.1f}°)")
    tee_print(f"Reliability: {baz_reliability:.3f}")
    if header_baz is not None:
        tee_print(f"Header BAZ: {header_baz:.2f}°")
        tee_print(f"Difference: {abs(baz_deg - header_baz):.2f}°")
    
    # Save BAZ plot
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.fill_between(posterior['azimuth_range'], posterior['posterior'], alpha=0.3, color='blue')
    ax.plot(posterior['azimuth_range'], posterior['posterior'], 'b-', linewidth=2)
    ax.axvline(baz_deg, color='red', linestyle='-', linewidth=2,
              label=f'Estimated: {baz_deg:.1f}° ± {baz_std:.1f}°')
    if header_baz is not None:
        ax.axvline(header_baz, color='green', linestyle='--', linewidth=2,
                  label=f'Header: {header_baz:.1f}°', alpha=0.7)
    ax.set_xlabel('Backazimuth (°)', fontweight='bold')
    ax.set_ylabel('Posterior Probability', fontweight='bold')
    ax.set_title(f'Rayleigh Wave Backazimuth\nReliability: {baz_reliability:.3f}',
                fontweight='bold')
    ax.set_xlim(0, 360)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, f'{event_name}_stockwell_baz.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    tee_print(f"Saved: {fig_path}")
    plt.show(block=False)
    plt.pause(0.1)
    
    # ==================================================================================
    # STEP 3: Apply Stockwell filter with NIP energy masks
    # ==================================================================================
    
    tee_print(f"\n{'='*80}")
    tee_print(f"APPLYING STOCKWELL FILTER WITH NIP ENERGY MASKS")
    tee_print(f"{'='*80}")
    
    fmin_hz = 1.0 / 150.0
    fmax_hz = 1.0 / 50.0
    
    tee_print(f"Frequency range: {fmin_hz:.6f} - {fmax_hz:.6f} Hz")
    tee_print(f"Period range: {1/fmax_hz:.1f} - {1/fmin_hz:.1f} s")
    
    result_dict = simple_stockwell_filter_rayleigh(
        dataN,
        dataE,
        dataZ,
        fs,
        baz_deg=baz_deg,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
        sense='retro',
        quadrature_min=0.0,
        hard_mask=False,
        amp_min=0.0,
        target_dt=0.5,
        zero_transverse=False,
        eps=1e-30,
    )
    
    add_particleman_nip_energy_masks(
        result_dict,
        threshold=0.8,
        width=0.05,
        eps=0.04,
        energy_floor=1.e-19,
        energy_width=0.02,
        energy_floor_mode='per_frequency',
        transverse_max=None,
        polarization_mode='elliptic',
    )
    
    tee_print(f"\nStockwell transform complete:")
    tee_print(f"  Envelope samples: {len(result_dict['rayleigh_envelope'])}")
    
    # Extract time axis
    time_array = common_utils.stockwell_time_axis_seconds(result_dict, trZ)
    tee_print(f"  Time array: {len(time_array)} points, {time_array[-1]/60:.2f} minutes")
    
    # Plot pre-detection diagnostics - DISABLED (only keeping BAZ and distance/timing PDFs)
    # plot_rayleigh_filter_diagnostics(
    #     result_dict,
    #     output_dir=str(output_dir),
    #     prefix=f'{event_name}_filter_pre_detection',
    #     event_offset_sec=event_offset_sec,
    # )
    # tee_print(f"\nSaved pre-detection diagnostic plots to: {output_dir}")
    
    # ==================================================================================
    # STEP 4: Detect R1/R2/R3 orbits with joint fitting
    # ==================================================================================
    
    tee_print(f"\n{'='*80}")
    tee_print(f"DETECTING R1/R2/R3 ORBITS")
    tee_print(f"{'='*80}")
    
    # Print time window sanity check
    time_min = (time_array + event_offset_sec) / 60.0
    tee_print(f"\nTime window check:")
    tee_print(f"  Detector window: {time_min[0]:.2f} to {time_min[-1]:.2f} min after origin")
    tee_print(f"  Duration: {(time_array[-1] - time_array[0]) / 60.0:.2f} min")
    
    R_earth_km = 6371.0
    U100_km_s = 3.77
    for orbit, path_deg in {'R1': catalog_distance, 'R2': 360.0 - catalog_distance, 
                            'R3': 360.0 + catalog_distance}.items():
        arrival_min = (path_deg * np.pi / 180.0 * R_earth_km / U100_km_s) / 60.0
        marker = 'inside' if time_min[0] <= arrival_min <= time_min[-1] else 'OUTSIDE'
        tee_print(f"  {orbit}: {arrival_min:.1f} min ({marker} window)")
    
    detected_orbits = detect_rayleigh_orbits_from_stockwell(
        result_dict=result_dict,
        time_array=time_array,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
        distance_range_deg=(1., 175),
        event_offset_sec=event_offset_sec,
        use_joint_fitting=True,
        min_confidence=0.20,
        expected_distance_deg=catalog_distance,
        velocity_uncertainty=0.25,
        distance_uncertainty_deg=20.0,
        distance_prior_sigma_deg=10.0,
        distance_grid_step_deg=0.05,
        output_dir=str(output_dir),
        verbose=True,
    )
    
    # Plot post-detection diagnostics - DISABLED (only keeping BAZ and distance/timing PDFs)
    # plot_rayleigh_filter_diagnostics(
    #     result_dict,
    #     output_dir=str(output_dir),
    #     prefix=f'{event_name}_filter_post_detection',
    #     event_offset_sec=event_offset_sec,
    #     detected_orbits=detected_orbits,
    # )
    # tee_print(f"\nSaved post-detection diagnostic plots to: {output_dir}")
    
    # ==================================================================================
    # STEP 5: Print detection summary
    # ==================================================================================
    
    metadata = detected_orbits.get('metadata', {})
    n_detected = metadata.get('n_orbits_detected', 0)
    
    tee_print(f"\n{'='*80}")
    tee_print(f"DETECTION RESULTS")
    tee_print(f"{'='*80}")
    tee_print(f"\nDetected {n_detected}/3 orbits")
    
    if metadata.get('method') == 'joint' and 'shared_distance_deg' in metadata:
        shared_dist = metadata['shared_distance_deg']
        error = abs(shared_dist - catalog_distance)
        tee_print(f"\nJoint fit distance: {shared_dist:.1f}°")
        tee_print(f"   Catalog distance: {catalog_distance:.1f}°")
        tee_print(f"   Error: {error:.1f}° ({error / catalog_distance * 100:.1f}%)")
    
    for orbit in ['R1', 'R2', 'R3']:
        if orbit in detected_orbits:
            data = detected_orbits[orbit]
            comp = data.get('confidence_components', {})
            tee_print(f"\n{orbit} DETECTED:")
            tee_print(f"   Arrival: {data['dispersion']['peak_time']:.1f} min")
            tee_print(f"   Distance: {data['distance_deg']:.1f}°")
            tee_print(f"   Confidence: {data['confidence']:.3f}")
            tee_print(f"   Shape correlation: {comp.get('shape_correlation', np.nan):.3f}")
            tee_print(f"   SNR: {comp.get('snr', np.nan):.1f}")
            tee_print(f"   RMS: {comp.get('rms_residual_sec', np.nan):.1f} s")
        else:
            tee_print(f"\n{orbit} NOT DETECTED")
    
    tee_print(f"\n{'='*80}")
    tee_print(f"STOCKWELL ANALYSIS COMPLETE")
    tee_print(f"{'='*80}\n")
    
    # ==================================================================================
    # STEP 6: Package and return results
    # ==================================================================================
    
    results = {
        'distance_deg': metadata.get('shared_distance_deg'),
        'distance_pdf': None,  # TODO: Extract from orbit detection results
        'timing_pdf': None,     # TODO: Extract from orbit detection results
        'baz_deg': baz_deg,
        'baz_std': baz_std,
        'baz_posterior': posterior['posterior'],
        'baz_azimuth_range': posterior['azimuth_range'],
        'baz_reliability': baz_reliability,
        'detected_orbits': detected_orbits,
        'reliability': {
            'baz_reliability': baz_reliability,
            'n_orbits_detected': n_detected,
            'joint_fit_used': metadata.get('method') == 'joint'
        }
    }
    
    return results



#=================================================================================================
# Polarization Analysis Functions (for Backazimuth Extraction)
#=================================================================================================

def combine_baz_pdfs(baz_x1, pdf1, baz_x2, pdf2, label1="Rayleigh", label2="P-wave"):
    """
    Combine two backazimuth PDFs using Bayesian multiplication.
    
    Parameters:
        baz_x1: azimuth array for first PDF (degrees, 0-360)
        pdf1: probability density values for first PDF
        baz_x2: azimuth array for second PDF (degrees, typically -50 to 410 from KDE)
        pdf2: probability density values for second PDF
        label1: label for first PDF
        label2: label for second PDF
    
    Returns:
        tuple: (baz_common, pdf_combined, baz_estimate, circular_std)
    """
    tee_print(f"\n{'='*80}")
    tee_print(f"COMBINING BACKAZIMUTH PDFs: {label1} + {label2}")
    tee_print(f"{'='*80}")
    
    # Create common grid 0-360° with 1° spacing
    baz_common = np.arange(0, 360, 1)
    
    # Interpolate both PDFs onto common grid
    # Handle wraparound for azimuth (circular variable)
    pdf1_interp = np.interp(baz_common, baz_x1, pdf1, left=0, right=0)
    pdf2_interp = np.interp(baz_common, baz_x2, pdf2, left=0, right=0)
    
    # Normalize individually
    if np.trapz(pdf1_interp, baz_common) > 0:
        pdf1_interp /= np.trapz(pdf1_interp, baz_common)
    if np.trapz(pdf2_interp, baz_common) > 0:
        pdf2_interp /= np.trapz(pdf2_interp, baz_common)
    
    # Combine (Bayesian multiplication)
    pdf_combined = pdf1_interp * pdf2_interp
    
    # Normalize combined
    if np.trapz(pdf_combined, baz_common) > 0:
        pdf_combined /= np.trapz(pdf_combined, baz_common)
    else:
        tee_print("WARNING: Combined PDF integration is zero!")
        return baz_common, pdf_combined, None, None
    
    # Find peak (most probable value)
    max_idx = np.argmax(pdf_combined)
    baz_estimate = baz_common[max_idx]
    
    # Calculate uncertainty using circular statistics
    angles_rad = np.deg2rad(baz_common)
    x_mean = np.trapz(pdf_combined * np.cos(angles_rad), baz_common)
    y_mean = np.trapz(pdf_combined * np.sin(angles_rad), baz_common)
    
    # Circular mean
    circular_mean = np.rad2deg(np.arctan2(y_mean, x_mean)) % 360
    
    # Circular standard deviation
    R = np.sqrt(x_mean**2 + y_mean**2)
    circular_std = np.rad2deg(np.sqrt(-2 * np.log(R))) if R > 0 else 90
    
    # Calculate individual estimates for comparison
    max_idx1 = np.argmax(pdf1_interp)
    estimate1 = baz_common[max_idx1]
    max_idx2 = np.argmax(pdf2_interp)
    estimate2 = baz_common[max_idx2]
    
    tee_print(f"\nIndividual Estimates:")
    tee_print(f"  {label1}: {estimate1:.1f}°")
    tee_print(f"  {label2}: {estimate2:.1f}°")
    tee_print(f"\nCombined Estimate:")
    tee_print(f"  Peak (most probable): {baz_estimate:.1f}°")
    tee_print(f"  Circular mean: {circular_mean:.1f}°")
    tee_print(f"  Circular std: {circular_std:.1f}°")
    tee_print(f"{'='*80}\n")
    
    return baz_common, pdf_combined, baz_estimate, circular_std


def run_polarization_analysis(st_raw, event_time, p_pick, s_pick, polarization_params, 
                               output_dir, event_name):
    """
    Run polarization analysis using picked P and S times.
    
    Parameters:
        st_raw: Raw ObsPy stream
        event_time: Event origin time (UTCDateTime)
        p_pick: P-wave pick time (can be seconds after event OR UTCDateTime)
        s_pick: S-wave pick time (can be seconds after event OR UTCDateTime)
        polarization_params: Dictionary of polarization parameters
        output_dir: Output directory
        event_name: Event name for file naming
    
    Returns:
        pickle_path: Path to generated pickle file
    """
    try:
        from polarization_picker_analysis import analyze_polarization_from_picks
    except ImportError:
        tee_print("ERROR: Could not import polarization_picker_analysis module")
        tee_print("Please ensure polarization_picker_analysis.py is in the same directory")
        return None
    
    # Convert picks to seconds relative to event if they're in UTC
    if isinstance(p_pick, UTCDateTime):
        p_pick_sec = p_pick - event_time
        tee_print(f"  P pick: {p_pick} UTC = {p_pick_sec:.2f}s (relative to event)")
    else:
        p_pick_sec = p_pick
        tee_print(f"  P pick: {p_pick:.2f}s (relative to event)")
    
    if isinstance(s_pick, UTCDateTime):
        s_pick_sec = s_pick - event_time
        tee_print(f"  S pick: {s_pick} UTC = {s_pick_sec:.2f}s (relative to event)")
    else:
        s_pick_sec = s_pick
        tee_print(f"  S pick: {s_pick:.2f}s (relative to event)")
    
    # Create picks dictionary for polarization analysis
    # picks_dict expects times in seconds relative to event
    picks_dict = {
        'P': [p_pick_sec],
        'S': [s_pick_sec]
    }
    
    tee_print(f"  Time window P: {polarization_params['t_window_P']}")
    tee_print(f"  Time window S: {polarization_params['t_window_S']}")
    
    # Define output filename (just the base name, function will add path and extension)
    # The analyze_polarization_from_picks function expects fname without directory
    # and will create: f"{fname}_Polarizationout.pkl" in the current directory
    # So we need to pass the full path as fname
    fname = os.path.join(output_dir, event_name)
    
    tee_print(f"\n  Running polarization analysis...")
    tee_print(f"  Output file base: {fname}")
    tee_print(f"  This may take a few minutes...")
    
    # Run polarization analysis
    try:
        analyze_polarization_from_picks(
            st=st_raw,
            event_time=event_time,
            picks_dict=picks_dict,
            fname=fname,
            **polarization_params
        )
        
        # Check for pickle file with multiple possible naming conventions
        # The polarization module may create either:
        # 1. {fname}_Polarizationout.pkl (expected format)
        # 2. {fname}out.pkl (actual format created by some versions)
        pickle_path_new = f'{fname}_Polarizationout.pkl'
        pickle_path_old = f'{fname}out.pkl'
        
        if os.path.exists(pickle_path_new):
            pickle_path = pickle_path_new
            tee_print(f"  Polarization analysis complete!")
            tee_print(f"  Pickle saved: {pickle_path}")
            return pickle_path
        elif os.path.exists(pickle_path_old):
            pickle_path = pickle_path_old
            tee_print(f"  Polarization analysis complete!")
            tee_print(f"  Pickle saved: {pickle_path}")
            tee_print(f"  Note: Using alternate pickle naming format")
            return pickle_path
        else:
            tee_print(f"ERROR: Polarization analysis completed but pickle file not found!")
            tee_print(f"  Looked for: {pickle_path_new}")
            tee_print(f"  Looked for: {pickle_path_old}")
            return None
    except Exception as e:
        tee_print(f"ERROR during polarization analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_polarization_baz_analysis(st_raw, stage2_picks, EVENT_TIME, OUTPUT_DIR, EVENT_NAME,
                                   stockwell_results=None, POLARIZATION_PARAMS=None):
    """
    Run polarization analysis and combine with Stockwell BAZ (Phase 6 extraction).
    
    Parameters:
        st_raw: Raw ObsPy stream
        stage2_picks: Stage 2 pick dictionary with UTC times
        EVENT_TIME: Event origin time (UTCDateTime)
        OUTPUT_DIR: Output directory path
        EVENT_NAME: Event name for file naming
        stockwell_results: Stockwell results dict (optional, for BAZ combination)
        POLARIZATION_PARAMS: Dictionary of polarization parameters
    
    Returns:
        dict or None: {
            'p_baz': float,
            'p_baz_error': [float, float],
            'p_baz_x': array,
            'p_baz_pdf': array,
            'pfroms_baz': float (optional),
            'pfroms_error': [float, float] (optional),
            'combined_baz': float (optional),
            'combined_baz_std': float (optional),
            'combined_baz_x': array (optional),
            'combined_baz_pdf': array (optional)
        }
    """
    # Check if polarization analysis is enabled (when called from main workflow)
    # If ENABLE_POLARIZATION_ANALYSIS is not defined, assume enabled (standalone mode)
    try:
        if not ENABLE_POLARIZATION_ANALYSIS:
            return None
    except NameError:
        # Variable not defined - running in standalone mode (backazimuth_analysis.py)
        # Polarization analysis is implicitly enabled
        pass
    
    # Extract pick times (they are UTCDateTime objects)
    p_pick_time = stage2_picks['P'][0] if stage2_picks.get('P') and len(stage2_picks['P']) > 0 else None
    s_pick_time = stage2_picks['S'][0] if stage2_picks.get('S') and len(stage2_picks['S']) > 0 else None
    
    if not (p_pick_time and s_pick_time):
        tee_print("\nWARNING: Polarization analysis skipped - need both P and S picks")
        tee_print(f"  P pick: {'available' if p_pick_time else 'missing'}")
        tee_print(f"  S pick: {'available' if s_pick_time else 'missing'}\n")
        return None
    
    tee_print("\n" + "="*80)
    tee_print("POLARIZATION ANALYSIS - BACKAZIMUTH EXTRACTION")
    tee_print("="*80)
    tee_print("Running polarization analysis with P and S picks...")
    
    # Picks are already UTCDateTime objects, use them directly
    p_pick_utc = p_pick_time
    s_pick_utc = s_pick_time
    
    # Calculate relative times for display
    p_pick_rel = float(p_pick_utc - EVENT_TIME) if EVENT_TIME else None
    s_pick_rel = float(s_pick_utc - EVENT_TIME) if EVENT_TIME else None
    
    if EVENT_TIME and p_pick_rel is not None and s_pick_rel is not None:
        tee_print(f"  P pick: {p_pick_rel:.2f}s relative = {p_pick_utc} UTC")
        tee_print(f"  S pick: {s_pick_rel:.2f}s relative = {s_pick_utc} UTC")
    else:
        tee_print(f"  P pick: {p_pick_utc} UTC")
        tee_print(f"  S pick: {s_pick_utc} UTC")
    
    # Run polarization analysis with UTC picks
    pickle_path = run_polarization_analysis(
        st_raw=st_raw,
        event_time=EVENT_TIME,
        p_pick=p_pick_utc,
        s_pick=s_pick_utc,
        polarization_params=POLARIZATION_PARAMS,
        output_dir=OUTPUT_DIR,
        event_name=EVENT_NAME
    )
    
    if not pickle_path:
        tee_print("\nWARNING: Polarization analysis failed. Skipping BAZ extraction.\n")
        return None
    
    # Extract P-wave BAZ with full PDF
    tee_print("\nExtracting P-wave backazimuth PDF (automatic)...")
    baz_p, error_p, baz_p_x, baz_p_pdf = extract_baz_from_pickle(
        pickle_path,
        wave_type='P',
        output_dir=OUTPUT_DIR,
        event_name=EVENT_NAME
    )
    
    # Optional: P-from-S BAZ (user prompt)
    pfroms_response = input("\nAlso extract P-from-S backazimuth? (y/n): ")
    baz_pfroms = None
    error_pfroms = None
    if pfroms_response.lower() in ['y', 'yes']:
        tee_print("Extracting P-from-S backazimuth PDF...")
        baz_pfroms, error_pfroms, _, _ = extract_baz_from_pickle(
            pickle_path,
            wave_type='P-from-S',
            output_dir=OUTPUT_DIR,
            event_name=EVENT_NAME
        )
    
    # Initialize combined BAZ variables
    baz_combined = None
    baz_combined_std = None
    baz_combined_x = None
    baz_combined_pdf = None
    
    # Check if we have Stockwell BAZ results to combine
    if (stockwell_results and baz_p is not None and 
        baz_p_x is not None and baz_p_pdf is not None):
        
        # Ask user if they want to combine BAZ PDFs
        combine_response = input("\nCombine Stockwell and P-wave backazimuth PDFs? (y/n): ")
        
        if combine_response.lower() in ['y', 'yes']:
            tee_print("\n" + "="*80)
            tee_print("COMBINING BACKAZIMUTH PDFs")
            tee_print("="*80)
            
            # Extract Stockwell BAZ PDF
            stockwell_baz = stockwell_results['baz_deg']
            stockwell_baz_std = stockwell_results['baz_std']
            stockwell_baz_posterior = stockwell_results['baz_posterior']
            stockwell_baz_x = stockwell_results['baz_azimuth_range']
            
            tee_print(f"Stockwell BAZ: {stockwell_baz:.1f}° ± {stockwell_baz_std:.1f}°")
            tee_print(f"P-wave BAZ: {baz_p:.1f}° [{error_p[0]:.1f}°, {error_p[1]:.1f}°]")
            
            # Get SAC header BAZ for comparison
            sac_header_baz = common_utils.get_sac_baz_deg(st_raw[0])
            if sac_header_baz is not None:
                tee_print(f"SAC header BAZ: {sac_header_baz:.1f}°")
            
            # Combine PDFs using Bayesian multiplication
            baz_combined_x, baz_combined_pdf, baz_combined, baz_combined_std = combine_baz_pdfs(
                stockwell_baz_x, stockwell_baz_posterior,
                baz_p_x, baz_p_pdf,
                label1="Rayleigh (Stockwell)",
                label2="P-wave (Polarization)"
            )
            
            if baz_combined is not None:
                # Plot combined BAZ PDF
                fig, ax = plt.subplots(figsize=(8, 5))
                
                # Plot individual PDFs
                ax.plot(stockwell_baz_x, stockwell_baz_posterior, 'b-',
                       linewidth=1.2, label='Rayleigh (Stockwell)', alpha=0.7)
                
                # For P-wave PDF, only plot valid range (0-360°)
                valid_mask = (baz_p_x >= 0) & (baz_p_x <= 360)
                ax.plot(baz_p_x[valid_mask], baz_p_pdf[valid_mask], 'g-',
                       linewidth=1.2, label='P-wave (Polarization)', alpha=0.7)
                
                # Plot combined PDF
                ax.fill_between(baz_combined_x, baz_combined_pdf,
                               alpha=0.3, color='red')
                ax.plot(baz_combined_x, baz_combined_pdf, 'r-',
                       linewidth=1.5, label='Combined PDF', zorder=10)
                
                # Mark individual estimates
                ax.axvline(stockwell_baz, color='b', linestyle='--',
                          linewidth=0.8, alpha=0.6)
                ax.axvline(baz_p, color='g', linestyle='--',
                          linewidth=0.8, alpha=0.6)
                
                # Mark SAC header BAZ if available
                if sac_header_baz is not None:
                    ax.axvline(sac_header_baz, color='black', linestyle='-',
                              linewidth=0.8, label=f'SAC header: {sac_header_baz:.1f}°',
                              alpha=0.8, zorder=5)
                
                # Mark combined estimate
                max_idx = np.argmax(baz_combined_pdf)
                ax.plot(baz_combined, baz_combined_pdf[max_idx], 'r*',
                       markersize=15,
                       label=f'Combined: {baz_combined:.1f}° ± {baz_combined_std:.1f}°',
                       zorder=20)
                
                ax.set_xlabel('Backazimuth (°)', fontweight='bold', fontsize=10)
                ax.set_ylabel('Probability Density', fontweight='bold', fontsize=10)
                ax.set_title('Combined Backazimuth: Rayleigh + P-wave Polarization',
                            fontweight='bold', fontsize=11)
                ax.set_xlim(0, 360)
                ax.legend(fontsize=9, loc='best')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save figure
                fig_path = os.path.join(OUTPUT_DIR, 'combined_baz_pdf.png')
                plt.savefig(fig_path, dpi=150, bbox_inches='tight')
                tee_print(f"\nSaved: {fig_path}")
                
                plt.show(block=False)
                plt.pause(0.1)
                
                # Export combined BAZ PDF for Monte Carlo sampling
                tee_print("\nExporting combined BAZ PDF for Monte Carlo sampling...")
                import pickle
                
                baz_pdf_data = {
                    'baz_x': baz_combined_x,
                    'baz_pdf': baz_combined_pdf,
                    'estimated_baz': baz_combined,
                    'circular_std': baz_combined_std,
                    'source': 'combined_stockwell_polarization',
                    'stockwell_baz': stockwell_baz,
                    'stockwell_baz_std': stockwell_baz_std,
                    'pwave_baz': baz_p,
                    'pwave_baz_error': error_p,
                    'sac_header_baz': sac_header_baz,
                    'true_baz': None,
                    'analysis_timestamp': datetime.now(),
                    'code_version': '4.0',
                    'notes': 'Combined BAZ using Bayesian multiplication'
                }
                
                baz_pdf_path = os.path.join(OUTPUT_DIR, f'{EVENT_NAME}_baz_pdf.pkl')
                with open(baz_pdf_path, 'wb') as f:
                    pickle.dump(baz_pdf_data, f)
                tee_print(f"  BAZ PDF saved: {baz_pdf_path}")
                
                # Print comparison summary
                tee_print("\n" + "="*80)
                tee_print("BACKAZIMUTH COMPARISON")
                tee_print("="*80)
                tee_print(f"Stockwell estimate: {stockwell_baz:.1f}° ± {stockwell_baz_std:.1f}°")
                tee_print(f"P-wave estimate: {baz_p:.1f}° [{error_p[0]:.1f}°, {error_p[1]:.1f}°]")
                tee_print(f"COMBINED estimate: {baz_combined:.1f}° ± {baz_combined_std:.1f}°")
                
                if sac_header_baz is not None:
                    error_stockwell = abs(stockwell_baz - sac_header_baz)
                    error_pwave = abs(baz_p - sac_header_baz)
                    error_combined = abs(baz_combined - sac_header_baz)
                    # Handle wraparound for circular difference
                    if error_stockwell > 180:
                        error_stockwell = 360 - error_stockwell
                    if error_pwave > 180:
                        error_pwave = 360 - error_pwave
                    if error_combined > 180:
                        error_combined = 360 - error_combined
                    
                    tee_print(f"\nSAC header BAZ: {sac_header_baz:.1f}°")
                    tee_print(f"  Stockwell error: {error_stockwell:.1f}°")
                    tee_print(f"  P-wave error: {error_pwave:.1f}°")
                    tee_print(f"  Combined error: {error_combined:.1f}°")
                
                tee_print("="*80 + "\n")
    
    tee_print("="*80 + "\n")
    
    # Return results dictionary
    return {
        'p_baz': baz_p,
        'p_baz_error': error_p,
        'p_baz_x': baz_p_x,
        'p_baz_pdf': baz_p_pdf,
        'pfroms_baz': baz_pfroms,
        'pfroms_error': error_pfroms,
        'combined_baz': baz_combined,
        'combined_baz_std': baz_combined_std,
        'combined_baz_x': baz_combined_x,
        'combined_baz_pdf': baz_combined_pdf
    }



