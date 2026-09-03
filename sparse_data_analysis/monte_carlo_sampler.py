#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo Sampler for Seismic Event Location
Samples from distance and backazimuth PDFs to generate location estimates

@author: nmcreasy
Created: 2026-06-17
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from obspy import read
from pyproj import Geod
import argparse

def load_par_file(par_file_path):
    """
    Load parameter file dynamically.
    
    Returns:
        par_module: Parameter file module
    """
    import importlib.util
    
    if not os.path.exists(par_file_path):
        raise FileNotFoundError(f"Parameter file not found: {par_file_path}")
    
    spec = importlib.util.spec_from_file_location("par_config", par_file_path)
    par_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(par_module)
    return par_module


def load_baz_pdf(pickle_path):
    """
    Load backazimuth PDF from either combined BAZ PDF or polarization pickle file.
    
    Handles two file formats:
    1. Combined BAZ PDF (from sparse_data_location_analysis.py):
       - Simple dict with 'baz_x' and 'baz_pdf' keys
       - Already combined Rayleigh + Polarization data
    2. Polarization-only pickle (from polarization package):
       - Complex nested dict with 'P' and 'S' wave data
       - Only P-wave polarization information
    
    Returns:
        baz_values, baz_pdf (arrays), source_type (str)
    """
    print(f"Loading BAZ data from: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Check if this is a combined BAZ PDF (priority format)
    if isinstance(data, dict) and 'baz_x' in data and 'baz_pdf' in data:
        # Combined BAZ PDF format (Rayleigh + Polarization)
        baz_values = np.array(data['baz_x'])
        baz_pdf = np.array(data['baz_pdf'])
        
        print(f"  Format: Combined BAZ PDF (Rayleigh + Polarization)")
        print(f"  Loaded {len(baz_values)} BAZ points")
        print(f"  BAZ range: {np.min(baz_values):.1f}° to {np.max(baz_values):.1f}°")
        
        if 'estimated_baz' in data:
            print(f"  Estimated BAZ: {data['estimated_baz']:.1f}°")
        if 'estimated_std' in data:
            print(f"  Uncertainty: ±{data['estimated_std']:.1f}°")
        
        return baz_values, baz_pdf, 'combined'
    
    # Otherwise, try polarization-only format
    elif isinstance(data, dict) and 'P' in data:
        # Polarization pickle format (P-wave only)
        p_data = data['P']
        if len(p_data) > 1:
            azimuth_data = p_data[1]  # Row 1: azimuth (0-360 degrees)
            
            # Extract azimuth values and weights
            baz_values = np.array(azimuth_data['P'])
            weights = np.array(azimuth_data['weights'])
            
            print(f"  Format: Polarization-only (P-wave)")
            print(f"  Loaded {len(baz_values)} BAZ samples")
            print(f"  BAZ range: {np.min(baz_values):.1f}° to {np.max(baz_values):.1f}°")
            
            return baz_values, weights, 'polarization'
        else:
            raise ValueError("P-wave data structure incomplete")
    else:
        raise ValueError("Unexpected pickle file structure - not a recognized BAZ PDF format")


def load_distance_pdf(pickle_path):
    """
    Load distance PDF from either a standalone distance PDF pickle or a
    comprehensive Step 1 results pickle.
    
    Returns:
        distance_x, distance_pdf (arrays)
    """
    print(f"Loading distance data from: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    if 'distance_x' in data and 'distance_pdf' in data:
        distance_x = np.array(data['distance_x'])
        distance_pdf = np.array(data['distance_pdf'])
        estimated_distance = data.get('estimated_distance')
        source = 'standalone distance PDF'
    elif 'combined' in data:
        combined = data['combined']
        distance_x = np.array(combined['distance_pdf_x'])
        distance_pdf = np.array(combined['distance_pdf_y'])
        estimated_distance = combined.get('distance_estimate')
        source = 'combined distance PDF from comprehensive results'
    else:
        raise ValueError(
            'Unexpected pickle file structure - not a recognized distance PDF '
            'or comprehensive results file'
        )

    if distance_x.size == 0 or distance_pdf.size == 0:
        raise ValueError(f'The {source} contains an empty distance PDF')
    if distance_x.shape != distance_pdf.shape:
        raise ValueError(
            f'Distance PDF axis and values have incompatible shapes: '
            f'{distance_x.shape} and {distance_pdf.shape}'
        )
    
    print(f"  Format: {source}")
    print(f"  Loaded distance PDF with {len(distance_x)} points")
    print(f"  Distance range: {np.min(distance_x):.2f}° to {np.max(distance_x):.2f}°")
    if estimated_distance is not None:
        print(f"  Estimated distance: {estimated_distance:.2f}°")
    
    return distance_x, distance_pdf


def load_station_coords(sac_file):
    """
    Load station coordinates from SAC file header.
    
    Returns:
        station_lat, station_lon
    """
    print(f"Loading station coordinates from: {sac_file}")
    
    st = read(sac_file)
    tr = st[0]
    
    station_lat = tr.stats.sac.stla
    station_lon = tr.stats.sac.stlo
    
    print(f"  Station: {tr.stats.station}")
    print(f"  Latitude: {station_lat:.4f}°")
    print(f"  Longitude: {station_lon:.4f}°")
    
    return station_lat, station_lon


def sample_pdf(values, pdf, n_samples):
    """
    Sample from a probability distribution.
    
    Parameters:
        values: x-axis values
        pdf: probability density values (will be normalized)
        n_samples: number of samples to generate
    
    Returns:
        samples array
    """
    # Normalize PDF to sum to 1
    pdf_normalized = pdf / np.sum(pdf)
    
    # Sample using numpy random choice
    samples = np.random.choice(values, size=n_samples, p=pdf_normalized)
    
    return samples


def convert_to_latlon(station_lat, station_lon, distance_deg, backazimuth_deg):
    """
    Convert (distance, backazimuth) to (lat, lon) using geodesic calculations.
    
    Parameters:
        station_lat: station latitude (degrees)
        station_lon: station longitude (degrees)
        distance_deg: distance from station (degrees)
        backazimuth_deg: backazimuth from station (degrees, 0=North, clockwise)
    
    Returns:
        event_lat, event_lon
    """
    # Initialize geodesic calculator (WGS84 ellipsoid)
    g = Geod(ellps='WGS84')
    
    # Convert distance from degrees to meters (1° ≈ 111.195 km on Earth)
    distance_m = distance_deg * 111195.0
    
    # Calculate back azimuth from station to event
    # (backazimuth is from station to event)
    forward_azimuth = backazimuth_deg
    
    # Compute endpoint given starting point, azimuth, and distance
    event_lon, event_lat, back_az = g.fwd(station_lon, station_lat, forward_azimuth, distance_m)
    
    return event_lat, event_lon


def main():
    parser = argparse.ArgumentParser(description='Monte Carlo sampler for seismic event location')
    parser.add_argument('--results', type=str, required=True, 
                       help='Results directory containing pickle files')
    parser.add_argument('--samples', type=int, default=300,
                       help='Number of Monte Carlo samples (default: 300)')
    parser.add_argument('--sac-file', type=str, default=None,
                       help='SAC file for station coordinates (defaults to the first file in --par-file)')
    parser.add_argument('--par-file', type=str, required=True,
                       help='Parameter file (contains TRUE_EVENT_LAT/LON if known)')
    
    args = parser.parse_args()
    
    # Load true event location from PAR_FILE
    print(f"Loading parameters from: {args.par_file}")
    try:
        par = load_par_file(args.par_file)
        true_event_lat = getattr(par, 'TRUE_EVENT_LAT', None)
        true_event_lon = getattr(par, 'TRUE_EVENT_LON', None)
    except Exception as e:
        print(f"ERROR: Failed to load PAR_FILE: {e}")
        return 1

    if args.sac_file is None:
        try:
            args.sac_file = os.path.join(par.DATA_DIR, par.FILES[0])
            print(f"Using SAC file from parameter file: {args.sac_file}")
        except (AttributeError, IndexError) as e:
            print(f"ERROR: Could not determine a SAC file from {args.par_file}: {e}")
            print("Please provide one explicitly with --sac-file PATH")
            return 1

    if not os.path.isfile(args.sac_file):
        print(f"ERROR: SAC file not found: {args.sac_file}")
        print("Provide the correct path with --sac-file PATH")
        return 1
    
    print("="*80)
    print("MONTE CARLO SAMPLER FOR SEISMIC EVENT LOCATION")
    print("="*80)
    print(f"Results directory: {args.results}")
    print(f"Number of samples: {args.samples}")
    if true_event_lat is not None and true_event_lon is not None:
        print(f"True event location: {true_event_lat:.4f}°, {true_event_lon:.4f}°")
    else:
        print("True event location: Not specified (blind analysis mode)")
    print("="*80 + "\n")
    
    # Find pickle files in results directory
    baz_pickle = None
    distance_pickle = None
    
    # PRIORITY 1: Search for combined BAZ PDF (Rayleigh + Polarization)
    # This is the preferred format from sparse_data_location_analysis.py
    for file in os.listdir(args.results):
        if file.endswith('_baz_pdf.pkl'):
            baz_pickle = os.path.join(args.results, file)
            print(f"Found combined BAZ PDF (priority): {file}")
            break
    
    # PRIORITY 2: Search for polarization pickle if combined not found
    if baz_pickle is None:
        print("Combined BAZ PDF (*_baz_pdf.pkl) not found, searching for polarization pickle...")
        for file in os.listdir(args.results):
            # Look for _Polarizationout.pkl pattern (documented format)
            if file.endswith('_Polarizationout.pkl'):
                baz_pickle = os.path.join(args.results, file)
                print(f"  Found polarization pickle: {file}")
                break
    
    # PRIORITY 3: Try fallback pattern *out.pkl (but exclude distance_pdf.pkl)
    if baz_pickle is None:
        print("  *_Polarizationout.pkl not found, searching for *out.pkl pattern...")
        for file in os.listdir(args.results):
            if file.endswith('out.pkl') and not file.endswith('_distance_pdf.pkl'):
                # Make sure it's not other types of out.pkl files
                # We want files like "20150629_082243out.pkl"
                baz_pickle = os.path.join(args.results, file)
                print(f"  Found alternate pattern: {file}")
                break
    
    # Search for a standalone distance PDF (legacy format)
    for file in os.listdir(args.results):
        if file.endswith('_distance_pdf.pkl'):
            distance_pickle = os.path.join(args.results, file)
            print(f"Found standalone distance PDF: {file}")
            break

    # Current Step 1 output stores the distance PDF in the comprehensive pickle.
    if distance_pickle is None:
        print("Standalone distance PDF not found, searching for comprehensive results...")
        for file in os.listdir(args.results):
            if file.endswith('_comprehensive_results.pkl'):
                distance_pickle = os.path.join(args.results, file)
                print(f"Found comprehensive results for distance PDF: {file}")
                break
    
    # Error handling
    if baz_pickle is None:
        print("\nERROR: Could not find BAZ pickle file")
        print("  Searched for patterns (in priority order):")
        print("    1. *_baz_pdf.pkl (combined Rayleigh + Polarization - PREFERRED)")
        print("    2. *_Polarizationout.pkl (P-wave polarization only)")
        print("    3. *out.pkl (alternate polarization format)")
        print(f"  In directory: {args.results}")
        return
    
    if distance_pickle is None:
        print("ERROR: Could not find a distance PDF source")
        print("  Searched for: *_distance_pdf.pkl or *_comprehensive_results.pkl")
        return
    
    print("Found required files:")
    print(f"  BAZ pickle: {os.path.basename(baz_pickle)}")
    print(f"  Distance pickle: {os.path.basename(distance_pickle)}\n")
    
    # Load data
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    baz_values, baz_weights, baz_source = load_baz_pdf(baz_pickle)
    distance_x, distance_pdf = load_distance_pdf(distance_pickle)
    station_lat, station_lon = load_station_coords(args.sac_file)
    
    # Print BAZ source information
    if baz_source == 'combined':
        print(f"\n✓ Using COMBINED BAZ PDF (Rayleigh + Polarization)")
        print(f"  This provides the most constrained backazimuth estimate!")
    else:
        print(f"\n  Using polarization-only BAZ")
        print(f"  Note: For better results, run sparse_data_location_analysis.py")
        print(f"        to generate combined BAZ PDF (*_baz_pdf.pkl)")
    
    print("\n" + "="*80)
    print("MONTE CARLO SAMPLING")
    print("="*80)
    print(f"Generating {args.samples} samples...")
    
    # Sample from distance PDF
    distance_samples = sample_pdf(distance_x, distance_pdf, args.samples)
    print(f"  Distance samples: mean={np.mean(distance_samples):.2f}°, "
          f"std={np.std(distance_samples):.2f}°")
    
    # Sample from BAZ PDF (using weights as probabilities)
    baz_samples = sample_pdf(baz_values, baz_weights, args.samples)
    print(f"  BAZ samples: mean={np.mean(baz_samples):.2f}°, "
          f"std={np.std(baz_samples):.2f}°")
    
    # Convert to lat/lon
    print("\nConverting samples to lat/lon coordinates...")
    event_lats = np.zeros(args.samples)
    event_lons = np.zeros(args.samples)
    
    for i in range(args.samples):
        event_lats[i], event_lons[i] = convert_to_latlon(
            station_lat, station_lon, 
            distance_samples[i], baz_samples[i]
        )
    
    print(f"  Latitude range: {np.min(event_lats):.2f}° to {np.max(event_lats):.2f}°")
    print(f"  Longitude range: {np.min(event_lons):.2f}° to {np.max(event_lons):.2f}°")
    
    # Save samples to CSV
    output_csv = os.path.join(args.results, 'mc_samples.csv')
    print(f"\nSaving samples to: {output_csv}")
    
    with open(output_csv, 'w') as f:
        f.write("distance_deg,backazimuth_deg,latitude,longitude\n")
        for i in range(args.samples):
            f.write(f"{distance_samples[i]:.6f},{baz_samples[i]:.6f},"
                   f"{event_lats[i]:.6f},{event_lons[i]:.6f}\n")
    
    print("  Saved successfully!")
    
    # Create verification plots
    print("\n" + "="*80)
    print("CREATING VERIFICATION PLOTS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Distance histogram
    ax = axes[0, 0]
    ax.hist(distance_samples, bins=30, density=True, alpha=0.6, color='blue', 
            edgecolor='black', label='MC Samples')
    ax.plot(distance_x, distance_pdf, 'r-', linewidth=2, label='Original PDF')
    ax.axvline(np.mean(distance_samples), color='green', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(distance_samples):.2f}°')
    ax.set_xlabel('Distance (degrees)', fontweight='bold')
    ax.set_ylabel('Probability Density', fontweight='bold')
    ax.set_title('Distance Distribution Verification', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: BAZ KDE (matching plot_pwave_baz_from_pickle.py)
    ax = axes[0, 1]
    
    # Plot histogram of MC samples first
    ax.hist(baz_samples, bins=36, density=True, alpha=0.6, color='lightblue', 
            edgecolor='black', label='MC Samples')
    
    # Create KDE with tight bandwidth to show more structure
    from scipy import stats
    kernel = stats.gaussian_kde(baz_values, weights=baz_weights)
    kernel.covariance_factor = lambda: .08  # Tighter fit - shows more detail
    kernel._compute_covariance()
    
    # Evaluate KDE
    xs = np.linspace(0, 360, 1000)
    ys = kernel(xs)
    
    # Plot smooth KDE curve on top
    ax.plot(xs, ys, 'darkblue', linewidth=2.5, label='Original KDE')
    
    # Mark the peak and mean
    peak_baz = xs[np.argmax(ys)]
    ax.axvline(peak_baz, color='blue', linestyle='--', 
               linewidth=1.5, alpha=0.7, label=f'Peak: {peak_baz:.1f}°')
    ax.axvline(np.mean(baz_samples), color='green', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(baz_samples):.1f}°')
    
    ax.set_xlabel('Backazimuth (degrees)', fontweight='bold')
    ax.set_ylabel('Probability Density', fontweight='bold')
    ax.set_title('Backazimuth Distribution Verification', fontweight='bold')
    ax.set_xlim(0, 360)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: 2D scatter (Distance vs BAZ)
    ax = axes[1, 0]
    scatter = ax.scatter(distance_samples, baz_samples, c=range(args.samples), 
                        cmap='viridis', alpha=0.5, s=20)
    ax.set_xlabel('Distance (degrees)', fontweight='bold')
    ax.set_ylabel('Backazimuth (degrees)', fontweight='bold')
    ax.set_title('Distance vs Backazimuth Samples', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Sample Index')
    
    # Plot 4: Geographic scatter (Lat vs Lon)
    ax = axes[1, 1]
    ax.scatter(event_lons, event_lats, alpha=0.5, s=20, c='blue', 
              edgecolors='black', linewidths=0.5, label='MC Samples')
    ax.scatter(station_lon, station_lat, marker='^', s=200, c='red', 
              edgecolors='black', linewidths=2, label='Station', zorder=10)
    
    # Plot true event location if known
    if true_event_lat is not None and true_event_lon is not None:
        ax.scatter(true_event_lon, true_event_lat, marker='*', s=300, c='gold', 
                  edgecolors='black', linewidths=2, label='True Event', zorder=10)
    
    ax.set_xlabel('Longitude (degrees)', fontweight='bold')
    ax.set_ylabel('Latitude (degrees)', fontweight='bold')
    ax.set_title('Event Location Samples (Geographic)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save figure
    output_fig = os.path.join(args.results, 'mc_distribution_verification.png')
    plt.savefig(output_fig, dpi=150, bbox_inches='tight')
    print(f"Saved verification plot: {output_fig}")
    
    plt.show()
    
    # Create LAT/LON PDF plots
    print("\n" + "="*80)
    print("CREATING LAT/LON PDF PLOTS")
    print("="*80)
    
    # Calculate statistics
    lat_mean = np.mean(event_lats)
    lat_median = np.median(event_lats)
    lat_std = np.std(event_lats)
    lat_ci_lower = np.percentile(event_lats, 2.5)
    lat_ci_upper = np.percentile(event_lats, 97.5)
    
    lon_mean = np.mean(event_lons)
    lon_median = np.median(event_lons)
    lon_std = np.std(event_lons)
    lon_ci_lower = np.percentile(event_lons, 2.5)
    lon_ci_upper = np.percentile(event_lons, 97.5)
    
    # Calculate KDE peak location (needed for all subplots)
    print("\nCalculating 2D KDE peak location...")
    from scipy.stats import gaussian_kde
    xy = np.vstack([event_lons, event_lats])
    kde_2d = gaussian_kde(xy, bw_method=0.1)  # Match PyGMT bandwidth

    # Create grid for KDE evaluation
    lon_grid = np.linspace(event_lons.min(), event_lons.max(), 100)
    lat_grid = np.linspace(event_lats.min(), event_lats.max(), 100)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    positions = np.vstack([lon_mesh.ravel(), lat_mesh.ravel()])
    density = kde_2d(positions).reshape(lon_mesh.shape)

    # Find estimated location as peak of KDE (maximum density) - MATCHING PYGMT METHOD
    max_density_idx = np.argmax(density)
    kde_peak_lat = lat_mesh.ravel()[max_density_idx]
    kde_peak_lon = lon_mesh.ravel()[max_density_idx]

    print(f"  KDE Peak location: {kde_peak_lat:.4f}°, {kde_peak_lon:.4f}°")
    print(f"  Simple Mean location: {lat_mean:.4f}°, {lon_mean:.4f}°")

    # Calculate errors from the true location (if known).
    if true_event_lat is not None and true_event_lon is not None:
        g = Geod(ellps='WGS84')
        _, _, dist_error_m = g.inv(lon_mean, lat_mean, true_event_lon, true_event_lat)
        dist_error_km = dist_error_m / 1000.0
        lat_error = kde_peak_lat - true_event_lat
        lon_error = kde_peak_lon - true_event_lon
        _, _, dist_kde_error_m = g.inv(kde_peak_lon, kde_peak_lat, true_event_lon, true_event_lat)
        dist_kde_error_km = dist_kde_error_m / 1000.0
        print(f"  KDE Peak error: {dist_kde_error_km:.2f} km")
    else:
        dist_error_km = None
        lat_error = None
        lon_error = None

    # Print statistics
    print("\nLatitude Statistics:")
    if lat_error is not None:
        print(f"  Mean:   {lat_mean:.4f}° (KDE peak error: {lat_error:+.4f}°)")
    else:
        print(f"  Mean:   {lat_mean:.4f}°")
    print(f"  Median: {lat_median:.4f}°")
    print(f"  Std:    {lat_std:.4f}°")
    print(f"  95% CI: [{lat_ci_lower:.4f}°, {lat_ci_upper:.4f}°]")
    if true_event_lat is not None:
        print(f"  True:   {true_event_lat:.4f}°")

    print("\nLongitude Statistics:")
    if lon_error is not None:
        print(f"  Mean:   {lon_mean:.4f}° (KDE peak error: {lon_error:+.4f}°)")
    else:
        print(f"  Mean:   {lon_mean:.4f}°")
    print(f"  Median: {lon_median:.4f}°")
    print(f"  Std:    {lon_std:.4f}°")
    print(f"  95% CI: [{lon_ci_lower:.4f}°, {lon_ci_upper:.4f}°]")
    if true_event_lon is not None:
        print(f"  True:   {true_event_lon:.4f}°")

    if dist_error_km is not None:
        print(f"\nMean location error: {dist_error_km:.2f} km")
    
    # Create figure with 3 subplots
    fig2 = plt.figure(figsize=(14, 10))
    
    # Subplot 1: Latitude PDF
    ax1 = plt.subplot(2, 2, 1)
    
    # Histogram
    counts, bins, patches = ax1.hist(event_lats, bins=30, density=True, alpha=0.6, 
                                     color='steelblue', edgecolor='black', 
                                     label='MC Samples')
    
    # KDE curve
    from scipy import stats
    kde_lat = stats.gaussian_kde(event_lats, bw_method=0.1)
    lat_xs = np.linspace(event_lats.min(), event_lats.max(), 200)
    lat_kde_ys = kde_lat(lat_xs)
    ax1.plot(lat_xs, lat_kde_ys, 'darkblue', linewidth=2.5, label='KDE')
    
    # Reference lines
    ax1.axvline(lat_mean, color='green', linestyle='-', linewidth=2.5, 
                label=f'Mean: {lat_mean:.3f}°', zorder=5)
    ax1.axvline(lat_median, color='blue', linestyle='--', linewidth=2, 
                label=f'Median: {lat_median:.3f}°', zorder=5)
    ax1.axvline(kde_peak_lat, color='purple', linestyle='--', linewidth=2, 
                label=f'KDE Peak: {kde_peak_lat:.3f}°', zorder=5)
    
    # True event marker (if known)
    if true_event_lat is not None:
        ax1.axvline(true_event_lat, color='gold', linestyle='-', linewidth=3, 
                    label=f'True: {true_event_lat:.3f}°', zorder=6)
    
    # Confidence interval shading
    ax1.axvspan(lat_ci_lower, lat_ci_upper, alpha=0.2, color='green', 
                label='95% CI')
    
    # Statistics text box
    stats_text = f'Mean: {lat_mean:.4f}°\n'
    stats_text += f'Median: {lat_median:.4f}°\n'
    stats_text += f'Std: {lat_std:.4f}°\n'
    stats_text += f'95% CI: [{lat_ci_lower:.4f}°,\n          {lat_ci_upper:.4f}°]'
    if lat_error is not None:
        stats_text += f'\nError: {lat_error:+.4f}°'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlabel('Latitude (degrees)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Probability Density', fontweight='bold', fontsize=11)
    ax1.set_title('Latitude Distribution', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Longitude PDF
    ax2 = plt.subplot(2, 2, 2)
    
    # Histogram
    counts, bins, patches = ax2.hist(event_lons, bins=30, density=True, alpha=0.6, 
                                     color='coral', edgecolor='black', 
                                     label='MC Samples')
    
    # KDE curve
    kde_lon = stats.gaussian_kde(event_lons, bw_method=0.1)
    lon_xs = np.linspace(event_lons.min(), event_lons.max(), 200)
    lon_kde_ys = kde_lon(lon_xs)
    ax2.plot(lon_xs, lon_kde_ys, 'darkred', linewidth=2.5, label='KDE')
    
    # Reference lines
    ax2.axvline(lon_mean, color='green', linestyle='-', linewidth=2.5, 
                label=f'Mean: {lon_mean:.3f}°', zorder=5)
    ax2.axvline(lon_median, color='blue', linestyle='--', linewidth=2, 
                label=f'Median: {lon_median:.3f}°', zorder=5)
    ax2.axvline(kde_peak_lon, color='purple', linestyle='--', linewidth=2, 
                label=f'KDE Peak: {kde_peak_lon:.3f}°', zorder=5)
    
    # True event marker (if known)
    if true_event_lon is not None:
        ax2.axvline(true_event_lon, color='gold', linestyle='-', linewidth=3, 
                    label=f'True: {true_event_lon:.3f}°', zorder=6)
    
    # Confidence interval shading
    ax2.axvspan(lon_ci_lower, lon_ci_upper, alpha=0.2, color='green', 
                label='95% CI')
    
    # Statistics text box
    stats_text = f'Mean: {lon_mean:.4f}°\n'
    stats_text += f'Median: {lon_median:.4f}°\n'
    stats_text += f'Std: {lon_std:.4f}°\n'
    stats_text += f'95% CI: [{lon_ci_lower:.4f}°,\n          {lon_ci_upper:.4f}°]'
    if lon_error is not None:
        stats_text += f'\nError: {lon_error:+.4f}°'
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.set_xlabel('Longitude (degrees)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Probability Density', fontweight='bold', fontsize=11)
    ax2.set_title('Longitude Distribution', fontweight='bold', fontsize=12)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: 2D Density Plot
    ax3 = plt.subplot(2, 2, 3)
    
    # 2D histogram with hexbin for smoother appearance
    hb = ax3.hexbin(event_lons, event_lats, gridsize=25, cmap='YlOrRd', 
                    mincnt=1, alpha=0.8)
    
    # Plot contours (using density calculated earlier)
    contours = ax3.contour(lon_mesh, lat_mesh, density, colors='black', 
                           alpha=0.4, linewidths=1.5, levels=5)
    
    # Mark key locations
    ax3.scatter(station_lon, station_lat, marker='^', s=250, c='red', 
                edgecolors='black', linewidths=2, label='Station', zorder=10)
    
    # True event marker (if known)
    if true_event_lat is not None and true_event_lon is not None:
        ax3.scatter(true_event_lon, true_event_lat, marker='*', s=400, c='gold', 
                    edgecolors='black', linewidths=2.5, label='True Event', zorder=11)
    
    # Use KDE peak instead of simple mean (matching PyGMT method)
    ax3.scatter(kde_peak_lon, kde_peak_lat, marker='o', s=200, c='green', 
                edgecolors='black', linewidths=2, label='KDE Peak Estimate', zorder=10)
    ax3.scatter(kde_peak_lon, kde_peak_lat, marker='+', s=200, c='black', 
                linewidths=3, zorder=10)
    
    # Colorbar
    cb = plt.colorbar(hb, ax=ax3, label='Sample Density')
    
    ax3.set_xlabel('Longitude (degrees)', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Latitude (degrees)', fontweight='bold', fontsize=11)
    ax3.set_title('2D Event Location Density', fontweight='bold', fontsize=12)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    
    # Subplot 4: Summary statistics table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Create summary table
    summary_data = [
        ['Statistic', 'Latitude', 'Longitude'],
        ['Mean', f'{lat_mean:.4f}°', f'{lon_mean:.4f}°'],
        ['Median', f'{lat_median:.4f}°', f'{lon_median:.4f}°'],
        ['KDE Peak', f'{kde_peak_lat:.4f}°', f'{kde_peak_lon:.4f}°'],
        ['Std Dev', f'{lat_std:.4f}°', f'{lon_std:.4f}°'],
        ['95% CI Lower', f'{lat_ci_lower:.4f}°', f'{lon_ci_lower:.4f}°'],
        ['95% CI Upper', f'{lat_ci_upper:.4f}°', f'{lon_ci_upper:.4f}°'],
    ]
    
    # Add true value and error rows only if known
    if true_event_lat is not None and true_event_lon is not None:
        summary_data.append(['True Value', f'{true_event_lat:.4f}°', f'{true_event_lon:.4f}°'])
        summary_data.append(['Error', f'{lat_error:+.4f}°', f'{lon_error:+.4f}°'])
    
    table = ax4.table(cellText=summary_data, loc='center', cellLoc='center',
                     colWidths=[0.35, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.3)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(summary_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            else:
                table[(i, j)].set_facecolor('#F2F2F2')
    
    plt.tight_layout()
    
    # Save figure
    output_fig_latlon = os.path.join(args.results, 'mc_latlon_pdf.png')
    plt.savefig(output_fig_latlon, dpi=150, bbox_inches='tight')
    print(f"\nSaved lat/lon PDF plot: {output_fig_latlon}")
    
    plt.show()
    
    #=============================================================================
    # EXPORT LAT/LON PDFs FOR MULTI-STATION COMBINATION
    #=============================================================================
    print("\n" + "="*80)
    print("EXPORTING LAT/LON PDFs")
    print("="*80)
    
    latlon_pdf_data = {
        'lat_x': lat_xs,              # X-axis values (degrees)
        'lat_pdf': lat_kde_ys,        # PDF values (normalized)
        'lon_x': lon_xs,              # X-axis values (degrees)  
        'lon_pdf': lon_kde_ys,        # PDF values (normalized)
        'lat_mean': lat_mean,
        'lon_mean': lon_mean,
        'lat_median': lat_median,
        'lon_median': lon_median,
        'lat_std': lat_std,
        'lon_std': lon_std,
        'lat_ci_lower': lat_ci_lower,
        'lat_ci_upper': lat_ci_upper,
        'lon_ci_lower': lon_ci_lower,
        'lon_ci_upper': lon_ci_upper,
        'kde_peak_lat': kde_peak_lat,
        'kde_peak_lon': kde_peak_lon,
        'samples': args.samples,
        'station_lat': station_lat,
        'station_lon': station_lon,
        'event_lats': event_lats,     # Raw sample arrays
        'event_lons': event_lons,     # Raw sample arrays
        'kde_bandwidth': 0.1
    }
    
    latlon_pdf_path = os.path.join(args.results, 'mc_latlon_pdf.pkl')
    
    import pickle
    with open(latlon_pdf_path, 'wb') as f:
        pickle.dump(latlon_pdf_data, f)
    
    print(f"  Lat/Lon PDF data saved: {latlon_pdf_path}")
    print(f"  Includes: 1D PDFs, statistics, raw samples, and metadata")
    print("="*80)
    
    print("\n" + "="*80)
    print("MONTE CARLO SAMPLING COMPLETE")
    print("="*80)
    print(f"Generated {args.samples} samples")
    print(f"Output files:")
    print(f"  - {output_csv}")
    print(f"  - {output_fig}")
    print(f"  - {output_fig_latlon}")
    print("\nNext step: Run plot_location_map_pygmt.py in PyGMT environment")
    print("="*80)


if __name__ == "__main__":
    main()
