#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Station PDF Combination for Seismic Event Location
Combines location estimates from multiple stations using Bayesian multiplication

@author: nmcreasy
Created: 2026-07-02
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy import interpolate
from pyproj import Geod
import argparse

# True event location for comparison (can be overridden by command line)
TRUE_EVENT_LAT = -15.9902
TRUE_EVENT_LON = -74.1826

def load_station_data(pkl_path):
    """
    Load lat/lon PDF data from a station's Monte Carlo results.
    
    Parameters:
        pkl_path: Path to mc_latlon_pdf.pkl file
    
    Returns:
        dict: Station data containing PDFs, samples, and metadata
    """
    print(f"\nLoading station data from: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"  Station: ({data['station_lat']:.4f}°, {data['station_lon']:.4f}°)")
    print(f"  KDE Peak estimate: ({data['kde_peak_lat']:.4f}°, {data['kde_peak_lon']:.4f}°)")
    print(f"  Samples: {data['samples']}")
    print(f"  KDE bandwidth: {data.get('kde_bandwidth', 'N/A')}")
    
    return data


def create_2d_kde_from_samples(lats, lons, bandwidth=0.1):
    """
    Create 2D KDE from lat/lon samples.
    
    Parameters:
        lats: Array of latitude samples
        lons: Array of longitude samples
        bandwidth: KDE bandwidth parameter
    
    Returns:
        kde: Gaussian KDE object
    """
    xy = np.vstack([lons, lats])
    kde = gaussian_kde(xy, bw_method=bandwidth)
    return kde


def evaluate_kde_on_grid(kde, lon_range, lat_range, grid_points=150):
    """
    Evaluate KDE on a 2D grid.
    
    Parameters:
        kde: Gaussian KDE object
        lon_range: (min_lon, max_lon) tuple
        lat_range: (min_lat, max_lat) tuple
        grid_points: Number of grid points in each dimension
    
    Returns:
        lon_mesh, lat_mesh, density: Grid meshes and density values
    """
    lon_grid = np.linspace(lon_range[0], lon_range[1], grid_points)
    lat_grid = np.linspace(lat_range[0], lat_range[1], grid_points)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    positions = np.vstack([lon_mesh.ravel(), lat_mesh.ravel()])
    density = kde(positions).reshape(lon_mesh.shape)
    
    return lon_mesh, lat_mesh, density


def combine_station_pdfs(station_data_list, bandwidth=0.1):
    """
    Combine PDFs from multiple stations using Bayesian multiplication.
    
    Parameters:
        station_data_list: List of station data dictionaries
        bandwidth: KDE bandwidth for creating 2D PDFs
    
    Returns:
        dict: Combined results with grid, density, and statistics
    """
    print(f"\n{'='*80}")
    print(f"COMBINING PDFs FROM {len(station_data_list)} STATIONS")
    print(f"{'='*80}")
    
    # Determine common grid boundaries (union of all station ranges)
    all_lats = np.concatenate([data['event_lats'] for data in station_data_list])
    all_lons = np.concatenate([data['event_lons'] for data in station_data_list])
    
    # Add padding to ensure all data is captured
    lat_padding = (all_lats.max() - all_lats.min()) * 0.1
    lon_padding = (all_lons.max() - all_lons.min()) * 0.1
    
    lat_range = (all_lats.min() - lat_padding, all_lats.max() + lat_padding)
    lon_range = (all_lons.min() - lon_padding, all_lons.max() + lon_padding)
    
    print(f"\nCommon grid:")
    print(f"  Latitude: {lat_range[0]:.2f}° to {lat_range[1]:.2f}°")
    print(f"  Longitude: {lon_range[0]:.2f}° to {lon_range[1]:.2f}°")
    
    # Create 2D KDEs for each station and evaluate on common grid
    individual_densities = []
    station_kdes = []
    
    for i, data in enumerate(station_data_list, 1):
        print(f"\nProcessing Station {i}...")
        
        # Create 2D KDE from samples
        kde = create_2d_kde_from_samples(data['event_lats'], data['event_lons'], bandwidth)
        station_kdes.append(kde)
        
        # Evaluate on common grid
        lon_mesh, lat_mesh, density = evaluate_kde_on_grid(kde, lon_range, lat_range)
        
        # Normalize (integrate to 1)
        density_normalized = density / np.trapz(np.trapz(density, lon_mesh[0, :]), lat_mesh[:, 0])
        individual_densities.append(density_normalized)
        
        print(f"  Peak density: {density_normalized.max():.6f}")
        print(f"  Integral: {np.trapz(np.trapz(density_normalized, lon_mesh[0, :]), lat_mesh[:, 0]):.6f}")
    
    # Combine PDFs using Bayesian multiplication
    print(f"\n{'─'*80}")
    print("BAYESIAN MULTIPLICATION")
    print(f"{'─'*80}")
    
    combined_density = np.ones_like(individual_densities[0])
    
    for i, density in enumerate(individual_densities, 1):
        combined_density *= density
        print(f"  After multiplying station {i}: max = {combined_density.max():.6e}")
    
    # Normalize combined PDF
    integral = np.trapz(np.trapz(combined_density, lon_mesh[0, :]), lat_mesh[:, 0])
    print(f"\n  Combined PDF integral before normalization: {integral:.6e}")
    
    if integral > 0:
        combined_density /= integral
        print(f"  Combined PDF integral after normalization: {np.trapz(np.trapz(combined_density, lon_mesh[0, :]), lat_mesh[:, 0]):.6f}")
    else:
        print("  WARNING: Combined PDF integral is zero!")
    
    # Find peak location (MAP estimate)
    max_idx = np.argmax(combined_density)
    peak_lat = lat_mesh.ravel()[max_idx]
    peak_lon = lon_mesh.ravel()[max_idx]
    
    print(f"\n{'─'*80}")
    print("COMBINED ESTIMATE")
    print(f"{'─'*80}")
    print(f"  Peak location: ({peak_lat:.4f}°, {peak_lon:.4f}°)")
    print(f"  Peak density: {combined_density.max():.6f}")
    
    # Calculate statistics (mean, std)
    # Weight each grid point by its probability
    lon_flat = lon_mesh.ravel()
    lat_flat = lat_mesh.ravel()
    density_flat = combined_density.ravel()
    
    mean_lon = np.sum(lon_flat * density_flat) / np.sum(density_flat)
    mean_lat = np.sum(lat_flat * density_flat) / np.sum(density_flat)
    
    var_lon = np.sum((lon_flat - mean_lon)**2 * density_flat) / np.sum(density_flat)
    var_lat = np.sum((lat_flat - mean_lat)**2 * density_flat) / np.sum(density_flat)
    
    std_lon = np.sqrt(var_lon)
    std_lat = np.sqrt(var_lat)
    
    print(f"  Mean location: ({mean_lat:.4f}°, {mean_lon:.4f}°)")
    print(f"  Std deviation: (±{std_lat:.4f}°, ±{std_lon:.4f}°)")
    
    # Return results
    return {
        'lon_mesh': lon_mesh,
        'lat_mesh': lat_mesh,
        'combined_density': combined_density,
        'individual_densities': individual_densities,
        'peak_lat': peak_lat,
        'peak_lon': peak_lon,
        'mean_lat': mean_lat,
        'mean_lon': mean_lon,
        'std_lat': std_lat,
        'std_lon': std_lon,
        'lon_range': lon_range,
        'lat_range': lat_range
    }


def plot_combined_results(station_data_list, combined_results, output_dir, true_lat=None, true_lon=None):
    """
    Create visualization of individual and combined PDFs.
    
    Parameters:
        station_data_list: List of station data dictionaries
        combined_results: Dict from combine_station_pdfs()
        output_dir: Directory to save plots
        true_lat, true_lon: True event location (optional)
    """
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    n_stations = len(station_data_list)
    
    # Figure 1: Overview with all stations
    fig = plt.figure(figsize=(16, 10))
    
    # Individual station PDFs
    for i, (data, density) in enumerate(zip(station_data_list, combined_results['individual_densities']), 1):
        ax = plt.subplot(2, n_stations, i)
        
        # Filled contours
        levels = np.linspace(0, density.max(), 10)
        cf = ax.contourf(combined_results['lon_mesh'], combined_results['lat_mesh'], 
                        density, levels=levels, cmap='Blues', alpha=0.8)
        
        # Contour lines
        ax.contour(combined_results['lon_mesh'], combined_results['lat_mesh'], 
                  density, levels=5, colors='darkblue', linewidths=1, alpha=0.6)
        
        # Mark station
        ax.scatter(data['station_lon'], data['station_lat'], marker='^', s=200, 
                  c='red', edgecolors='black', linewidths=2, label='Station', zorder=10)
        
        # Mark station's estimate
        ax.scatter(data['kde_peak_lon'], data['kde_peak_lat'], marker='o', s=150, 
                  c='green', edgecolors='black', linewidths=2, label='Estimate', zorder=10)
        
        # Mark true event if provided
        if true_lat is not None and true_lon is not None:
            ax.scatter(true_lon, true_lat, marker='*', s=300, c='gold', 
                      edgecolors='black', linewidths=2, label='True Event', zorder=11)
        
        ax.set_xlabel('Longitude (°)', fontweight='bold')
        ax.set_ylabel('Latitude (°)', fontweight='bold')
        ax.set_title(f'Station {i} PDF', fontweight='bold', fontsize=12)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Colorbar
        plt.colorbar(cf, ax=ax, label='Probability Density')
    
    # Combined PDF (larger plot spanning bottom row)
    ax_combined = plt.subplot(2, 1, 2)
    
    # Filled contours with more levels for detail
    levels_combined = np.linspace(0, combined_results['combined_density'].max(), 20)
    cf_combined = ax_combined.contourf(combined_results['lon_mesh'], combined_results['lat_mesh'], 
                                       combined_results['combined_density'], 
                                       levels=levels_combined, cmap='YlOrRd', alpha=0.9)
    
    # Contour lines
    ax_combined.contour(combined_results['lon_mesh'], combined_results['lat_mesh'], 
                       combined_results['combined_density'], levels=8, 
                       colors='darkred', linewidths=1.5, alpha=0.7)
    
    # Mark all stations
    for i, data in enumerate(station_data_list, 1):
        ax_combined.scatter(data['station_lon'], data['station_lat'], marker='^', s=250, 
                           c='red', edgecolors='black', linewidths=2, zorder=10,
                           label='Station' if i == 1 else '')
    
    # Mark combined estimate
    ax_combined.scatter(combined_results['peak_lon'], combined_results['peak_lat'], 
                       marker='o', s=300, c='lime', edgecolors='black', linewidths=3, 
                       label='Combined Estimate', zorder=12)
    ax_combined.scatter(combined_results['peak_lon'], combined_results['peak_lat'], 
                       marker='+', s=300, c='black', linewidths=3, zorder=12)
    
    # Mark true event if provided
    if true_lat is not None and true_lon is not None:
        ax_combined.scatter(true_lon, true_lat, marker='*', s=500, c='gold', 
                           edgecolors='black', linewidths=3, label='True Event', zorder=11)
    
    ax_combined.set_xlabel('Longitude (degrees)', fontweight='bold', fontsize=12)
    ax_combined.set_ylabel('Latitude (degrees)', fontweight='bold', fontsize=12)
    ax_combined.set_title('Combined Multi-Station PDF (Bayesian Multiplication)', 
                         fontweight='bold', fontsize=14)
    ax_combined.legend(fontsize=10, loc='best')
    ax_combined.grid(True, alpha=0.3)
    ax_combined.set_aspect('equal', adjustable='box')
    
    # Colorbar
    plt.colorbar(cf_combined, ax=ax_combined, label='Combined Probability Density')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'multistation_combined_pdf.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved combined PDF plot: {output_path}")
    
    plt.show()
    
    # Figure 2: Statistics summary
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Calculate errors if true location provided
    if true_lat is not None and true_lon is not None:
        g = Geod(ellps='WGS84')
        
        # Combined estimate error
        _, _, dist_peak_m = g.inv(combined_results['peak_lon'], combined_results['peak_lat'], 
                                   true_lon, true_lat)
        dist_peak_km = dist_peak_m / 1000.0
        
        # Individual station errors
        station_errors = []
        for data in station_data_list:
            _, _, dist_m = g.inv(data['kde_peak_lon'], data['kde_peak_lat'], 
                                 true_lon, true_lat)
            station_errors.append(dist_m / 1000.0)
    
    # Create summary table
    summary_data = [
        ['Metric', 'Latitude', 'Longitude'],
        ['Combined Peak', f"{combined_results['peak_lat']:.4f}°", f"{combined_results['peak_lon']:.4f}°"],
        ['Combined Mean', f"{combined_results['mean_lat']:.4f}°", f"{combined_results['mean_lon']:.4f}°"],
        ['Std Deviation', f"±{combined_results['std_lat']:.4f}°", f"±{combined_results['std_lon']:.4f}°"],
    ]
    
    if true_lat is not None and true_lon is not None:
        summary_data.extend([
            ['True Location', f"{true_lat:.4f}°", f"{true_lon:.4f}°"],
            ['Peak Error', f"{combined_results['peak_lat'] - true_lat:+.4f}°", 
             f"{combined_results['peak_lon'] - true_lon:+.4f}°"],
            ['Distance Error', f"{dist_peak_km:.2f} km", '']
        ])
    
    # Add individual station summaries
    summary_data.append(['', '', ''])
    summary_data.append(['Individual Stations', '', ''])
    
    for i, data in enumerate(station_data_list, 1):
        error_str = f"{station_errors[i-1]:.2f} km" if true_lat is not None else "N/A"
        summary_data.append([
            f"Station {i}",
            f"({data['station_lat']:.2f}°, {data['station_lon']:.2f}°)",
            f"Error: {error_str}"
        ])
    
    table = ax.table(cellText=summary_data, loc='center', cellLoc='left',
                    colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(summary_data)):
        for j in range(3):
            if summary_data[i][0] == '' or summary_data[i][0] == 'Individual Stations':
                table[(i, j)].set_facecolor('#D0D0D0')
                table[(i, j)].set_text_props(weight='bold')
            elif i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            else:
                table[(i, j)].set_facecolor('#F2F2F2')
    
    plt.title('Multi-Station Combination Statistics', fontweight='bold', fontsize=14, pad=20)
    plt.tight_layout()
    
    # Save figure
    output_path2 = os.path.join(output_dir, 'multistation_statistics.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved statistics table: {output_path2}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Combine location estimates from multiple seismic stations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python combine_multistation_pdfs.py \\
      --station results/Station1/mc_latlon_pdf.pkl \\
      --station results/Station2/mc_latlon_pdf.pkl \\
      --station results/Station3/mc_latlon_pdf.pkl \\
      --output combined_results/
        """
    )
    
    parser.add_argument('--station', type=str, action='append', required=True,
                       help='Path to station mc_latlon_pdf.pkl file (use multiple times for multiple stations)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for combined results')
    parser.add_argument('--bandwidth', type=float, default=0.1,
                       help='KDE bandwidth for 2D PDFs (default: 0.1)')
    parser.add_argument('--true-lat', type=float, default=TRUE_EVENT_LAT,
                       help=f'True event latitude for comparison (default: {TRUE_EVENT_LAT}°)')
    parser.add_argument('--true-lon', type=float, default=TRUE_EVENT_LON,
                       help=f'True event longitude for comparison (default: {TRUE_EVENT_LON}°)')
    parser.add_argument('--no-true-event', action='store_true',
                       help='Do not plot true event location')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("="*80)
    print("MULTI-STATION PDF COMBINATION")
    print("="*80)
    print(f"Number of stations: {len(args.station)}")
    print(f"Output directory: {args.output}")
    print(f"KDE bandwidth: {args.bandwidth}")
    
    if not args.no_true_event:
        print(f"True event location: ({args.true_lat:.4f}°, {args.true_lon:.4f}°)")
    print("="*80)
    
    # Load all station data
    station_data_list = []
    for i, pkl_path in enumerate(args.station, 1):
        print(f"\n{'─'*80}")
        print(f"STATION {i}")
        print(f"{'─'*80}")
        
        if not os.path.exists(pkl_path):
            print(f"ERROR: File not found: {pkl_path}")
            return
        
        data = load_station_data(pkl_path)
        station_data_list.append(data)
    
    # Combine PDFs
    combined_results = combine_station_pdfs(station_data_list, bandwidth=args.bandwidth)
    
    # Calculate error from true location
    if not args.no_true_event:
        g = Geod(ellps='WGS84')
        _, _, dist_error_m = g.inv(combined_results['peak_lon'], combined_results['peak_lat'], 
                                    args.true_lon, args.true_lat)
        dist_error_km = dist_error_m / 1000.0
        
        print(f"\n{'='*80}")
        print("FINAL RESULTS")
        print(f"{'='*80}")
        print(f"Combined estimate: ({combined_results['peak_lat']:.4f}°, {combined_results['peak_lon']:.4f}°)")
        print(f"True location:     ({args.true_lat:.4f}°, {args.true_lon:.4f}°)")
        print(f"Distance error:    {dist_error_km:.2f} km")
        print(f"Uncertainty (std): ±{combined_results['std_lat']:.4f}° lat, ±{combined_results['std_lon']:.4f}° lon")
        print(f"{'='*80}")
    
    # Create visualizations
    true_lat = None if args.no_true_event else args.true_lat
    true_lon = None if args.no_true_event else args.true_lon
    
    plot_combined_results(station_data_list, combined_results, args.output, true_lat, true_lon)
    
    # Export combined results
    output_pkl = os.path.join(args.output, 'combined_results.pkl')
    with open(output_pkl, 'wb') as f:
        pickle.dump({
            'combined_results': combined_results,
            'station_data': station_data_list,
            'true_lat': true_lat,
            'true_lon': true_lon
        }, f)
    
    print(f"\n{'='*80}")
    print("EXPORT COMPLETE")
    print(f"{'='*80}")
    print(f"  Combined results saved: {output_pkl}")
    print(f"  Plots saved in: {args.output}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
