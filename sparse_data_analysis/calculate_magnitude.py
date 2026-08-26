#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate Seismic Magnitude PDF from Distance PDF and Seismic Amplitudes

Loads distance PDF from rayleighwave_picker_v4.py output and combines with
measured seismic amplitudes to produce a magnitude probability distribution.

Supports multiple magnitude types:
- ML: Local magnitude (regional scales: UK, CAL)
  * Each scale has its own default component (e.g., 'H' for horizontal maximum)
  * Component options: N, E, Z, or H (horizontal max of N and E)
- mb: Body-wave magnitude (P-waves)
- Ms: Surface-wave magnitude (Rayleigh waves)

@author: nmcreasy
Created: 2026-07-03
Updated: 2026-07-06 - Added scale-specific component selection for ML
                    - Added 'H' (horizontal maximum) component support
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
from obspy import read
from obspy.signal.invsim import cosine_taper
import argparse
from math import log10, sqrt

# Magnitude type configurations
MAGNITUDE_TYPES = {
    'ML': {
        'name': 'Local Magnitude',
        'default_window': [50, 150],
        'unit': 'nm',
        'distance_unit': 'km'
    },
    'mb': {
        'name': 'Body-Wave Magnitude',
        'default_component': 'Z',
        'default_window': [5, 15],
        'unit': 'um',
        'distance_unit': 'degrees',
        'period': 1.0,
        'formula': 'mb = log10(A/T) + 1.66*log10(Δ) + 3.3'
    },
    'Ms': {
        'name': 'Surface-Wave Magnitude',
        'default_component': 'Z',
        'default_window': [100, 300],
        'unit': 'um',
        'distance_unit': 'degrees',
        'period': 20.0,
        'min_distance': 20.0,
        'formula': 'Ms = log10(A/T) + 1.66*log10(Δ) + 3.3'
    }
}

# Regional ML scale parameters
# Formula: ML = log10(ampl) + a*log10(hypo_dist) + b*hypo_dist + c
# where ampl is in nm and hypo_dist is in km

ML_SCALES = {
    'UK': {
        'name': 'UK (Ottemöller and Sargeant, 2013)',
        'component': 'H',
        'a': 0.95,
        'b': 0.00183,
        'c': -1.76,
        'reference': 'Ottemöller and Sargeant (2013), BSSA, doi:10.1785/0120130085'
    },
    'CAL': {
        'name': 'Southern California (IASPEI, 2005)',
        'component': 'H',
        'a': 1.11,
        'b': 0.00189,
        'c': -2.09,
        'reference': 'IASPEI (2005), www.iaspei.org/commissions/CSOI/summary_of_WG_recommendations_2005.pdf'
    },
    'RUS': {
        'name': 'Arctic Russia (Morozov, 2020)',
        'component': 'Z',
        'a': 1.5,
        'b': 0.0001,
        'c': 3.0,
        'ref_distance': 100,
        'reference': 'Morozov, A.N., Vaganova, N.V., Asming, V.E., and Evtyugina, Z.A., The ML scale for western Eurasian Arctic, Ros. Seismol. Zh., 2020, vol. 2, no. 4, pp. 63–68.'
    },
    'KOR': {
        'name': 'Arctic Russia (Morozov, 2020)',
        'component': 'Z',
        'a': 0.5107,
        'b': 0.001699,
        'c': 3.0,
        'ref_distance': 100,
        'reference': 'Dong‐Hoon Sheen, Tae‐Seob Kang, Junkee Rhie; A Local Magnitude Scale for South Korea. Bulletin of the Seismological Society of America 2018;; 108 (5A): 2748–2755. doi: https://doi.org/10.1785/0120180112.'
    }
    
}


def load_distance_pdf(pkl_path):
    """
    Load distance PDF from rayleighwave_picker_v4.py output.
    
    Parameters:
        pkl_path: Path to distance PDF pickle file
    
    Returns:
        dict: Distance PDF data
    """
    print(f"\nLoading distance PDF from: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"  Distance estimate: {data['estimated_distance']:.2f}° ± {data['std_distance']:.2f}°")
    print(f"  Source: {data['source']}")
    print(f"  P-arrival time: {data.get('P_ARRIVAL_TIME', 'N/A')}")
    
    return data


def extract_amplitude_from_sac(sac_files, event_time, amplitude_window, 
                               remove_response=True, component='N'):
    """
    Extract peak ground displacement amplitude from SAC files.
    
    Parameters:
        sac_files: List of SAC file paths
        event_time: ObsPy UTCDateTime of event origin
        amplitude_window: [start, end] in seconds relative to event time
        remove_response: Whether to remove instrument response
        component: Component to use ('N', 'E', 'Z', or 'H' for horizontal maximum)
    
    Returns:
        tuple: (amplitude_nm, amplitude_std_nm, quality_factor)
    """
    print(f"\n{'='*80}")
    print("EXTRACTING AMPLITUDE FROM WAVEFORMS")
    print(f"{'='*80}")
    print(f"Component: {component}")
    if component == 'H':
        print("  Using horizontal maximum (max of N and E components)")
    print(f"Measurement window: {amplitude_window[0]:.1f} to {amplitude_window[1]:.1f} s")
    
    amplitudes = []
    
    # Special handling for 'H' (horizontal maximum)
    if component == 'H':
        # Group files by station to find matching N/E pairs
        station_files = {}
        for sac_file in sac_files:
            # Extract station identifier (everything before component)
            basename = os.path.basename(sac_file)
            # Assume format like: NET.STA.LOC.CHN.M.YEAR.DAY.TIME.SAC
            # Component is typically the last character before extension or .M
            if basename.upper().endswith('N.SAC') or basename.upper().endswith('N.M'):
                station_key = basename[:-5] if basename.upper().endswith('.SAC') else basename[:-3]
                if station_key not in station_files:
                    station_files[station_key] = {}
                station_files[station_key]['N'] = sac_file
            elif basename.upper().endswith('E.SAC') or basename.upper().endswith('E.M'):
                station_key = basename[:-5] if basename.upper().endswith('.SAC') else basename[:-3]
                if station_key not in station_files:
                    station_files[station_key] = {}
                station_files[station_key]['E'] = sac_file
        
        # Process each station's N/E pair
        for station_key, components in station_files.items():
            if 'N' not in components or 'E' not in components:
                print(f"\nWARNING: Station {station_key} missing N or E component, skipping")
                continue
            
            print(f"\nProcessing station: {station_key}")
            amplitudes_ne = {}
            
            for comp in ['N', 'E']:
                sac_file = components[comp]
                print(f"  Component {comp}: {os.path.basename(sac_file)}")
                
                try:
                    st = read(sac_file)
                    tr = st[0]
                    
                    # Get sampling rate
                    sr = tr.stats.sampling_rate
                    
                    # Remove instrument response if requested
                    if remove_response:
                        pre_filt = [0.001, 0.005, 45, 50]
                        try:
                            tr.remove_response(output='DISP', pre_filt=pre_filt, water_level=60)
                        except Exception as e:
                            print(f"    WARNING: Could not remove response: {e}")
                    
                    # Extract measurement window
                    window_start = event_time + amplitude_window[0]
                    window_end = event_time + amplitude_window[1]
                    tr_window = tr.slice(window_start, window_end)
                    
                    if len(tr_window.data) == 0:
                        print(f"    WARNING: No data in measurement window")
                        continue
                    
                    # Calculate peak amplitude
                    data = tr_window.data
                    peak_ampl_m = np.max(np.abs(data))
                    peak_ampl_nm = peak_ampl_m * 1e9
                    
                    print(f"    Peak amplitude: {peak_ampl_nm:.2f} nm")
                    amplitudes_ne[comp] = peak_ampl_nm
                    
                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue
            
            # Take maximum of N and E components
            if len(amplitudes_ne) == 2:
                horiz_max = max(amplitudes_ne['N'], amplitudes_ne['E'])
                print(f"  Horizontal maximum: {horiz_max:.2f} nm")
                amplitudes.append(horiz_max)
            elif len(amplitudes_ne) == 1:
                # If only one component available, use it
                comp_val = list(amplitudes_ne.values())[0]
                print(f"  Using single component: {comp_val:.2f} nm")
                amplitudes.append(comp_val)
    
    else:
        # Standard single-component processing
        for sac_file in sac_files:
            # Check if this is the desired component
            if not sac_file.endswith(f'{component}.SAC') and not sac_file.endswith(f'{component}.M'):
                # Try case-insensitive match
                if not sac_file.upper().endswith(f'{component}.SAC') and not sac_file.upper().endswith(f'{component}.M'):
                    continue
            
            print(f"\nProcessing: {os.path.basename(sac_file)}")
            
            try:
                st = read(sac_file)
                tr = st[0]
                
                # Get sampling rate
                sr = tr.stats.sampling_rate
                print(f"  Sampling rate: {sr} Hz")
                
                # Remove instrument response if requested
                if remove_response:
                    print(f"  Removing instrument response...")
                    
                    # Create response dictionary from SAC headers
                    # This is a simplified approach - adjust based on your data
                    pre_filt = [0.001, 0.005, 45, 50]  # Frequency taper
                    
                    try:
                        # Try to remove response to displacement (output='DISP')
                        tr.remove_response(output='DISP', pre_filt=pre_filt, water_level=60)
                        print(f"  Response removed → ground displacement (m)")
                    except Exception as e:
                        print(f"  WARNING: Could not remove response: {e}")
                        print(f"  Using raw counts (may not be accurate!)")
                
                # Extract measurement window
                window_start = event_time + amplitude_window[0]
                window_end = event_time + amplitude_window[1]
                
                tr_window = tr.slice(window_start, window_end)
                
                if len(tr_window.data) == 0:
                    print(f"  WARNING: No data in measurement window")
                    continue
                
                # Calculate peak amplitude
                data = tr_window.data
                peak_ampl_m = np.max(np.abs(data))
                peak_ampl_nm = peak_ampl_m * 1e9  # Convert m to nm
                
                print(f"  Peak amplitude: {peak_ampl_nm:.2f} nm")
                
                # Quality check
                if peak_ampl_nm > 1e6:  # Suspiciously large
                    print(f"  WARNING: Amplitude may indicate clipping or bad response removal")
                
                amplitudes.append(peak_ampl_nm)
                
            except Exception as e:
                print(f"  ERROR processing file: {e}")
                continue
    
    if len(amplitudes) == 0:
        raise ValueError("No valid amplitude measurements obtained!")
    
    # Calculate statistics
    amplitude_mean = np.mean(amplitudes)
    amplitude_std = np.std(amplitudes) if len(amplitudes) > 1 else amplitude_mean * 0.2
    
    print(f"\n{'─'*80}")
    print(f"AMPLITUDE SUMMARY")
    print(f"{'─'*80}")
    print(f"Number of measurements: {len(amplitudes)}")
    print(f"Mean amplitude: {amplitude_mean:.2f} nm")
    print(f"Std deviation: {amplitude_std:.2f} nm")
    
    if component == 'H':
        quality = len(amplitudes) / (len(sac_files) / 2) if sac_files else 0  # Divided by 2 since we need pairs
    else:
        quality = len(amplitudes) / len(sac_files) if sac_files else 0
    print(f"Quality factor: {quality:.2f} ({len(amplitudes)} measurements)")
    print(f"{'='*80}\n")
    
    return amplitude_mean, amplitude_std, quality


def calculate_ml(amplitude_nm, distance_km, scale='CAL'):
    """
    Calculate local magnitude using regional scale.
    
    Parameters:
        amplitude_nm: Peak ground displacement (nanometers)
        distance_km: Hypocentral distance (kilometers)
        scale: 'UK', 'CAL', or 'RUS'
    
    Returns:
        float: Local magnitude ML
    """
    params = ML_SCALES[scale]
    a = params['a']
    b = params['b']
    c = params['c']
    ref_distance = params.get('ref_distance', None)  # Optional reference distance
    
    if ref_distance and ref_distance > 0:
        # Formula with reference distance: ML = log10(A) + a*log10(R/R_ref) + b*(R - R_ref) + c
        # Used by scales like Russian Arctic (R_ref = 100 km)
        ml = log10(amplitude_nm) + a*log10(distance_km/ref_distance) + b*(distance_km - ref_distance) + c
    else:
        # Standard formula: ML = log10(A) + a*log10(R) + b*R + c
        # Used by UK and CAL scales
        ml = log10(amplitude_nm) + a*log10(distance_km) + b*distance_km + c
    
    return ml


def calculate_mb(amplitude_um, distance_deg, period=1.0):
    """
    Calculate body-wave magnitude using standard formula.
    
    Parameters:
        amplitude_um: P-wave peak amplitude (micrometers)
        distance_deg: Epicentral distance (degrees)
        period: Wave period in seconds (default: 1.0)
    
    Returns:
        float: Body-wave magnitude mb
    """
    # Standard formula: mb = log10(A/T) + Q(Δ, h)
    # Q(Δ, h) ≈ 1.66*log10(Δ) + 3.3 (Gutenberg-Richter)
    
    if distance_deg < 5:
        print(f"  WARNING: mb formula may not be reliable for Δ < 5° (current: {distance_deg:.1f}°)")
    
    mb = log10(amplitude_um / period) + 1.66 * log10(distance_deg) + 3.3
    
    return mb


def calculate_ms(amplitude_um, distance_deg, period=20.0):
    """
    Calculate surface-wave magnitude using standard formula.
    
    Parameters:
        amplitude_um: Rayleigh wave peak amplitude (micrometers)
        distance_deg: Epicentral distance (degrees)
        period: Wave period in seconds (default: 20.0)
    
    Returns:
        float: Surface-wave magnitude Ms
    """
    # Prague formula: Ms = log10(A/T) + 1.66*log10(Δ) + 3.3
    
    if distance_deg < 20:
        print(f"  WARNING: Ms formula not recommended for Δ < 20° (current: {distance_deg:.1f}°)")
    
    ms = log10(amplitude_um / period) + 1.66 * log10(distance_deg) + 3.3
    
    return ms


def propagate_uncertainty(distance_x, distance_pdf, amplitude_mean, amplitude_std,
                         magnitude_type='ML', scale='CAL', n_samples=1000, depth_km=10.0):
    """
    Propagate distance and amplitude uncertainty to magnitude PDF.
    
    Parameters:
        distance_x: Distance PDF x-values (degrees)
        distance_pdf: Distance PDF y-values (normalized)
        amplitude_mean: Mean amplitude (nm)
        amplitude_std: Amplitude standard deviation (nm)
        magnitude_type: 'ML', 'mb', or 'Ms'
        scale: Regional scale ('UK' or 'CAL') - only for ML
        n_samples: Number of Monte Carlo samples
        depth_km: Assumed event depth (km) for hypocentral distance
    
    Returns:
        tuple: (magnitude_samples, magnitude_x, magnitude_pdf)
    """
    print(f"\n{'='*80}")
    print("PROPAGATING UNCERTAINTY TO MAGNITUDE")
    print(f"{'='*80}")
    
    mag_config = MAGNITUDE_TYPES[magnitude_type]
    
    if magnitude_type == 'ML':
        print(f"Magnitude type: {mag_config['name']} ({ML_SCALES[scale]['name']})")
    else:
        print(f"Magnitude type: {mag_config['name']}")
        print(f"Formula: {mag_config['formula']}")
    
    print(f"Monte Carlo samples: {n_samples}")
    print(f"Assumed depth: {depth_km} km")
    
    # Normalize distance PDF
    distance_pdf_norm = distance_pdf / np.trapz(distance_pdf, distance_x)
    
    # Sample from distance PDF
    distance_samples_deg = np.random.choice(distance_x, size=n_samples, 
                                           p=distance_pdf_norm)
    
    # Convert to km (epicentral distance)
    distance_samples_epi_km = distance_samples_deg * 111.195
    
    # Calculate hypocentral distance
    distance_samples_hypo_km = np.sqrt(distance_samples_epi_km**2 + depth_km**2)
    
    print(f"\nDistance statistics:")
    print(f"  Epicentral: {np.mean(distance_samples_epi_km):.1f} ± {np.std(distance_samples_epi_km):.1f} km")
    print(f"  Epicentral (deg): {np.mean(distance_samples_deg):.2f} ± {np.std(distance_samples_deg):.2f}°")
    print(f"  Hypocentral: {np.mean(distance_samples_hypo_km):.1f} ± {np.std(distance_samples_hypo_km):.1f} km")
    
    # Check distance validity for Ms
    if magnitude_type == 'Ms':
        min_dist = mag_config['min_distance']
        if np.mean(distance_samples_deg) < min_dist:
            print(f"\n  ⚠️  WARNING: Mean distance {np.mean(distance_samples_deg):.1f}° < {min_dist}°")
            print(f"      Ms formula is not recommended for close distances!")
    
    # Sample from amplitude distribution (log-normal)
    if amplitude_std > 0:
        # Use log-normal distribution for amplitudes (more realistic)
        log_mean = np.log(amplitude_mean)
        log_std = np.log(1 + amplitude_std / amplitude_mean)
        amplitude_samples = np.random.lognormal(log_mean, log_std, n_samples)
    else:
        amplitude_samples = np.ones(n_samples) * amplitude_mean
    
    # Convert amplitude to appropriate units
    if magnitude_type in ['mb', 'Ms']:
        # Convert nm to μm for mb/Ms
        amplitude_samples_converted = amplitude_samples / 1000.0
        unit_str = "μm"
    else:
        # Keep as nm for ML
        amplitude_samples_converted = amplitude_samples
        unit_str = "nm"
    
    print(f"\nAmplitude statistics:")
    print(f"  Mean: {np.mean(amplitude_samples_converted):.1f} {unit_str}")
    print(f"  Std: {np.std(amplitude_samples_converted):.1f} {unit_str}")
    
    # Calculate magnitude for each sample
    magnitude_samples = np.zeros(n_samples)
    
    for i in range(n_samples):
        if magnitude_type == 'ML':
            magnitude_samples[i] = calculate_ml(amplitude_samples[i], 
                                                distance_samples_hypo_km[i], 
                                                scale)
        elif magnitude_type == 'mb':
            period = mag_config['period']
            magnitude_samples[i] = calculate_mb(amplitude_samples_converted[i], 
                                                distance_samples_deg[i], 
                                                period)
        elif magnitude_type == 'Ms':
            period = mag_config['period']
            magnitude_samples[i] = calculate_ms(amplitude_samples_converted[i], 
                                                distance_samples_deg[i], 
                                                period)
    
    # Create magnitude PDF using KDE
    kde = stats.gaussian_kde(magnitude_samples, bw_method=0.1)
    magnitude_x = np.linspace(magnitude_samples.min(), magnitude_samples.max(), 200)
    magnitude_pdf = kde(magnitude_x)
    
    print(f"\nMagnitude statistics:")
    print(f"  Mean: {np.mean(magnitude_samples):.2f}")
    print(f"  Median: {np.median(magnitude_samples):.2f}")
    print(f"  Std: {np.std(magnitude_samples):.2f}")
    print(f"  Range: {magnitude_samples.min():.2f} to {magnitude_samples.max():.2f}")
    print(f"{'='*80}\n")
    
    return magnitude_samples, magnitude_x, magnitude_pdf


def plot_magnitude_pdf(magnitude_samples, magnitude_x, magnitude_pdf, 
                      magnitude_type, scale, output_path, true_magnitude=None):
    """
    Create visualization of magnitude PDF.
    
    Parameters:
        magnitude_samples: Raw magnitude samples
        magnitude_x: Magnitude PDF x-values
        magnitude_pdf: Magnitude PDF y-values
        magnitude_type: 'ML', 'mb', or 'Ms'
        scale: Regional scale used (for ML only)
        output_path: Path to save figure
        true_magnitude: True/catalog magnitude for comparison (optional)
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Calculate statistics
    mag_mean = np.mean(magnitude_samples)
    mag_median = np.median(magnitude_samples)
    mag_std = np.std(magnitude_samples)
    mag_peak = magnitude_x[np.argmax(magnitude_pdf)]
    mag_ci_lower = np.percentile(magnitude_samples, 2.5)
    mag_ci_upper = np.percentile(magnitude_samples, 97.5)
    
    # Subplot 1: Histogram + KDE
    ax1 = axes[0]
    
    ax1.hist(magnitude_samples, bins=30, density=True, alpha=0.6, 
            color='steelblue', edgecolor='black', label='Samples')
    ax1.plot(magnitude_x, magnitude_pdf, 'darkblue', linewidth=2.5, label='KDE')
    
    # Reference lines
    ax1.axvline(mag_mean, color='green', linestyle='-', linewidth=2.5,
               label=f'Mean: {mag_mean:.2f}', zorder=5)
    ax1.axvline(mag_median, color='blue', linestyle='--', linewidth=2,
               label=f'Median: {mag_median:.2f}', zorder=5)
    ax1.axvline(mag_peak, color='purple', linestyle='--', linewidth=2,
               label=f'Peak: {mag_peak:.2f}', zorder=5)
    
    if true_magnitude is not None:
        ax1.axvline(true_magnitude, color='gold', linestyle='-', linewidth=3,
                   label=f'True: {true_magnitude:.2f}', zorder=6)
    
    # Confidence interval shading
    ax1.axvspan(mag_ci_lower, mag_ci_upper, alpha=0.2, color='green',
               label='95% CI')
    
    # Statistics text box
    stats_text = f'Mean: {mag_mean:.2f}\n'
    stats_text += f'Median: {mag_median:.2f}\n'
    stats_text += f'Std: {mag_std:.2f}\n'
    stats_text += f'95% CI: [{mag_ci_lower:.2f}, {mag_ci_upper:.2f}]'
    if true_magnitude is not None:
        error = mag_mean - true_magnitude
        stats_text += f'\nError: {error:+.2f}'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Set labels based on magnitude type
    if magnitude_type == 'ML':
        xlabel = 'Local Magnitude (ML)'
        title = f'Magnitude PDF - {ML_SCALES[scale]["name"]}'
    elif magnitude_type == 'mb':
        xlabel = 'Body-Wave Magnitude (mb)'
        title = 'Magnitude PDF - Body-Wave (Gutenberg-Richter)'
    elif magnitude_type == 'Ms':
        xlabel = 'Surface-Wave Magnitude (Ms)'
        title = 'Magnitude PDF - Surface-Wave (Prague Formula)'
    
    ax1.set_xlabel(xlabel, fontweight='bold', fontsize=11)
    ax1.set_ylabel('Probability Density', fontweight='bold', fontsize=11)
    ax1.set_title(title, fontweight='bold', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Cumulative distribution
    ax2 = axes[1]
    
    sorted_samples = np.sort(magnitude_samples)
    cumulative = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
    
    ax2.plot(sorted_samples, cumulative, 'darkblue', linewidth=2)
    ax2.axhline(0.5, color='blue', linestyle='--', alpha=0.5, label='Median')
    ax2.axhline(0.025, color='green', linestyle='--', alpha=0.5, label='2.5%')
    ax2.axhline(0.975, color='green', linestyle='--', alpha=0.5, label='97.5%')
    
    if true_magnitude is not None:
        ax2.axvline(true_magnitude, color='gold', linestyle='-', linewidth=3,
                   label=f'True: {true_magnitude:.2f}')
    
    ax2.set_xlabel(xlabel, fontweight='bold', fontsize=11)
    ax2.set_ylabel('Cumulative Probability', fontweight='bold', fontsize=11)
    ax2.set_title('Cumulative Distribution Function', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved magnitude PDF plot: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Calculate seismic magnitude PDF from distance PDF and seismic amplitudes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Local magnitude
  python calculate_magnitude.py \\
      --distance-pdf results/Peru_Event_distance_pdf.pkl \\
      --sac-files data/Peru/*.SAC \\
      --magnitude-type ML \\
      --scale CAL \\
      --output results/
  
  # Body-wave magnitude
  python calculate_magnitude.py \\
      --distance-pdf results/Peru_Event_distance_pdf.pkl \\
      --sac-files data/Peru/*.SAC \\
      --magnitude-type mb \\
      --output results/
  
  # Surface-wave magnitude  
  python calculate_magnitude.py \\
      --distance-pdf results/Peru_Event_distance_pdf.pkl \\
      --sac-files data/Peru/*.SAC \\
      --magnitude-type Ms \\
      --output results/
        """
    )
    
    parser.add_argument('--distance-pdf', type=str, required=True,
                       help='Path to distance PDF pickle file')
    parser.add_argument('--sac-files', type=str, nargs='+', required=True,
                       help='SAC files to extract amplitudes from')
    parser.add_argument('--magnitude-type', type=str, default='ML', choices=['ML', 'mb', 'Ms'],
                       help='Magnitude type: ML (local), mb (body-wave), Ms (surface-wave) (default: ML)')
    parser.add_argument('--component', type=str, default=None, choices=['N', 'E', 'Z', 'H'],
                       help='Seismic component: N, E, Z, or H (horizontal max) (default: auto-select based on magnitude type/scale)')
    parser.add_argument('--window', type=float, nargs=2, default=None,
                       help='Measurement window [start, end] seconds (default: auto-select based on magnitude type)')
    parser.add_argument('--scale', type=str, default='CAL', choices=['UK', 'CAL', 'RUS'],
                       help='Regional ML scale (default: CAL, only applies to ML)')
    parser.add_argument('--depth', type=float, default=10.0,
                       help='Assumed event depth in km (default: 10.0)')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of Monte Carlo samples (default: 1000)')
    parser.add_argument('--true-magnitude', type=float, default=None,
                       help='True/catalog magnitude for comparison (optional)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--no-remove-response', action='store_true',
                       help='Skip instrument response removal (use raw data)')
    
    args = parser.parse_args()
    
    # Auto-select component and window based on magnitude type
    mag_config = MAGNITUDE_TYPES[args.magnitude_type]
    
    if args.component is None:
        if args.magnitude_type == 'ML':
            # For ML, use scale-specific component
            args.component = ML_SCALES[args.scale]['component']
            print(f"Auto-selected component: {args.component} (default for {args.magnitude_type} {args.scale} scale)")
        else:
            # For mb/Ms, use magnitude type default
            args.component = mag_config['default_component']
            print(f"Auto-selected component: {args.component} (default for {args.magnitude_type})")
    
    if args.window is None:
        args.window = mag_config['default_window']
        print(f"Auto-selected window: {args.window[0]}-{args.window[1]}s (default for {args.magnitude_type})")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    mag_config = MAGNITUDE_TYPES[args.magnitude_type]
    
    print("="*80)
    print(f"{mag_config['name'].upper()} ({args.magnitude_type}) CALCULATION")
    print("="*80)
    print(f"Distance PDF: {args.distance_pdf}")
    print(f"SAC files: {len(args.sac_files)} files")
    print(f"Magnitude type: {mag_config['name']}")
    print(f"Component: {args.component}")
    print(f"Measurement window: {args.window[0]}-{args.window[1]} s")
    
    if args.magnitude_type == 'ML':
        print(f"Regional scale: {ML_SCALES[args.scale]['name']}")
    else:
        print(f"Formula: {mag_config['formula']}")
        print(f"Period: {mag_config['period']} s")
    
    print(f"Assumed depth: {args.depth} km")
    print(f"Monte Carlo samples: {args.samples}")
    if args.true_magnitude:
        print(f"True magnitude: {args.true_magnitude}")
    print("="*80)
    
    # Load distance PDF
    distance_data = load_distance_pdf(args.distance_pdf)
    distance_x = np.array(distance_data['distance_x'])
    distance_pdf = np.array(distance_data['distance_pdf'])
    
    # Get event time from distance PDF data
    event_time = distance_data.get('P_ARRIVAL_TIME')
    if event_time is None:
        event_time = distance_data.get('EVENT_TIME')
    
    if event_time is None:
        raise ValueError("No event time found in distance PDF file!")
    
    # Extract amplitude from SAC files
    amplitude_mean, amplitude_std, quality = extract_amplitude_from_sac(
        args.sac_files, event_time, args.window,
        remove_response=not args.no_remove_response,
        component=args.component
    )
    
    # Propagate uncertainty to magnitude
    magnitude_samples, magnitude_x, magnitude_pdf = propagate_uncertainty(
        distance_x, distance_pdf, amplitude_mean, amplitude_std,
        magnitude_type=args.magnitude_type, scale=args.scale, 
        n_samples=args.samples, depth_km=args.depth
    )
    
    # Plot results
    output_plot = os.path.join(args.output, f'{args.magnitude_type}_pdf.png')
    plot_magnitude_pdf(magnitude_samples, magnitude_x, magnitude_pdf,
                      args.magnitude_type, args.scale, output_plot, args.true_magnitude)
    
    # Export magnitude PDF
    magnitude_pdf_data = {
        'magnitude_type': args.magnitude_type,
        'magnitude_x': magnitude_x,
        'magnitude_pdf': magnitude_pdf,
        'magnitude_samples': magnitude_samples,
        'magnitude_mean': np.mean(magnitude_samples),
        'magnitude_median': np.median(magnitude_samples),
        'magnitude_std': np.std(magnitude_samples),
        'magnitude_ci_lower': np.percentile(magnitude_samples, 2.5),
        'magnitude_ci_upper': np.percentile(magnitude_samples, 97.5),
        'amplitude_mean_nm': amplitude_mean,
        'amplitude_std_nm': amplitude_std,
        'depth_assumed_km': args.depth,
        'component': args.component,
        'measurement_window': args.window,
        'true_magnitude': args.true_magnitude,
        'distance_pdf_source': args.distance_pdf
    }
    
    # Add scale info for ML
    if args.magnitude_type == 'ML':
        magnitude_pdf_data['scale'] = args.scale
        magnitude_pdf_data['scale_info'] = ML_SCALES[args.scale]
    
    output_pkl = os.path.join(args.output, f'{args.magnitude_type}_pdf.pkl')
    with open(output_pkl, 'wb') as f:
        pickle.dump(magnitude_pdf_data, f)
    
    print(f"\n{'='*80}")
    print("EXPORT COMPLETE")
    print(f"{'='*80}")
    print(f"  Magnitude PDF saved: {output_pkl}")
    print(f"  Plot saved: {output_plot}")
    print(f"{'='*80}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL MAGNITUDE ESTIMATE")
    print(f"{'='*80}")
    print(f"{args.magnitude_type} = {magnitude_pdf_data['magnitude_mean']:.2f} ± {magnitude_pdf_data['magnitude_std']:.2f}")
    print(f"95% CI: [{magnitude_pdf_data['magnitude_ci_lower']:.2f}, {magnitude_pdf_data['magnitude_ci_upper']:.2f}]")
    if args.true_magnitude:
        error = magnitude_pdf_data['magnitude_mean'] - args.true_magnitude
        print(f"Error: {error:+.2f} magnitude units")
    
    if args.magnitude_type == 'ML':
        print(f"Scale: {ML_SCALES[args.scale]['name']}")
        print(f"Reference: {ML_SCALES[args.scale]['reference']}")
    else:
        print(f"Formula: {mag_config['formula']}")
        print(f"Period: {mag_config['period']} s")
    
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
