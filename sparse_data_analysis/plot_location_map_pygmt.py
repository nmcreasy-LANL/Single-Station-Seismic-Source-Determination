#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyGMT Location Map Plotter
Creates azimuthal equidistant projection map with Monte Carlo samples

@author: nmcreasy
Created: 2026-06-17
"""

import os
import numpy as np
import pandas as pd
import pygmt
import argparse
from scipy.stats import gaussian_kde

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


# Note: Station coordinates are extracted from SAC headers in monte_carlo_sampler.py
# This script focuses on visualizing the event location probability distribution


def main():
    parser = argparse.ArgumentParser(description='Plot location map with Monte Carlo samples using PyGMT')
    parser.add_argument('--results', type=str, required=True,
                       help='Results directory containing mc_samples.csv')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename (default: auto-generated based on mode)')
    parser.add_argument('--width', type=float, default=20,
                       help='Map width in cm (default: 20)')
    parser.add_argument('--mode', type=str, default='density',
                       choices=['density', 'contour', 'points'],
                       help='Visualization mode: density (colored points), contour (filled contours), points (simple dots)')
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
    
    print("="*80)
    print("PYGMT LOCATION MAP PLOTTER")
    print("="*80)
    print(f"Results directory: {args.results}")
    if true_event_lat is not None and true_event_lon is not None:
        print(f"True event location: {true_event_lat:.4f}°, {true_event_lon:.4f}°")
    else:
        print("True event location: Not specified (blind analysis mode)")
    print("="*80 + "\n")
    
    # Load Monte Carlo samples
    csv_file = os.path.join(args.results, 'mc_samples.csv')
    if not os.path.exists(csv_file):
        print(f"ERROR: Monte Carlo samples file not found: {csv_file}")
        print("Please run monte_carlo_sampler.py first!")
        return
    
    print(f"Loading samples from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Try to load pre-calculated KDE peak from monte_carlo_sampler.py output
    pkl_file = os.path.join(args.results, 'mc_latlon_pdf.pkl')
    kde_peak_lat_saved = None
    kde_peak_lon_saved = None
    
    if os.path.exists(pkl_file):
        print(f"Loading pre-calculated KDE peak from: {pkl_file}")
        import pickle
        with open(pkl_file, 'rb') as f:
            latlon_data = pickle.load(f)
            if 'kde_peak_lat' in latlon_data and 'kde_peak_lon' in latlon_data:
                kde_peak_lat_saved = latlon_data['kde_peak_lat']
                kde_peak_lon_saved = latlon_data['kde_peak_lon']
                print(f"  Using saved KDE peak: {kde_peak_lat_saved:.4f}°, {kde_peak_lon_saved:.4f}°")
    else:
        print(f"Note: {pkl_file} not found - will calculate KDE peak from samples if needed")
    
    n_samples = len(df)
    print(f"  Loaded {n_samples} samples")
    print(f"  Latitude range: {df['latitude'].min():.2f}° to {df['latitude'].max():.2f}°")
    print(f"  Longitude range: {df['longitude'].min():.2f}° to {df['longitude'].max():.2f}°")
    
    # Calculate map extent based on event samples (zoom on event region)
    # Add buffer around samples for better visualization
    lat_min = df['latitude'].min()
    lat_max = df['latitude'].max()
    lon_min = df['longitude'].min()
    lon_max = df['longitude'].max()
    
    # Calculate range
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    
    # Add 20% buffer on each side for better visualization
    buffer_pct = 0.20
    lat_buffer = lat_range * buffer_pct
    lon_buffer = lon_range * buffer_pct
    
    lat_min = lat_min - lat_buffer
    lat_max = lat_max + lat_buffer
    lon_min = lon_min - lon_buffer
    lon_max = lon_max + lon_buffer
    
    # Also ensure true event is included in the view (if known)
    if true_event_lat is not None and true_event_lon is not None:
        lat_min = min(lat_min, true_event_lat - lat_buffer)
        lat_max = max(lat_max, true_event_lat + lat_buffer)
        lon_min = min(lon_min, true_event_lon - lon_buffer)
        lon_max = max(lon_max, true_event_lon + lon_buffer)
    
    print(f"\nMap extent (zoomed on event region):")
    print(f"  Sample latitude range: {df['latitude'].min():.4f}° to {df['latitude'].max():.4f}°")
    print(f"  Sample longitude range: {df['longitude'].min():.4f}° to {df['longitude'].max():.4f}°")
    
    # Ensure lat/lon are within valid ranges (Mercator projection limit)
    lat_min = max(lat_min, -85)  # Mercator cannot include poles
    lat_max = min(lat_max, 85)   # Mercator cannot include poles
    
    # Handle longitude wrapping
    if lon_max - lon_min > 360:
        # Map spans more than 360 degrees, use global
        lon_min = -180
        lon_max = 180
    
    print(f"  Latitude: {lat_min:.1f}° to {lat_max:.1f}°")
    print(f"  Longitude: {lon_min:.1f}° to {lon_max:.1f}°")
    
    # Create PyGMT figure
    print("\n" + "="*80)
    print("CREATING PYGMT MAP")
    print("="*80)
    print("Projection: Mercator (zoomed on event region)")
    print(f"Map width: {args.width} cm\n")
    
    fig = pygmt.Figure()
    
    # Define projection: Mercator (regional view)
    # M = Mercator projection
    # Format: Mwidth
    projection = f"M{args.width}c"
    
    # Define region
    region = [lon_min, lon_max, lat_min, lat_max]
    
    # Create basemap with appropriate grid intervals for zoomed view
    # Calculate appropriate interval based on map extent
    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    
    # Use automatic fine intervals for small regions, or 2-5 degree intervals
    if lat_span < 10 and lon_span < 10:
        # Very zoomed in - use automatic fine grid
        frame_spec = ["afg", "WSnE"]
    elif lat_span < 20 and lon_span < 20:
        # Moderately zoomed - use 2 degree intervals
        frame_spec = ["a2g2", "WSnE"]
    else:
        # Larger region - use 5 degree intervals
        frame_spec = ["a5g5", "WSnE"]
    
    fig.basemap(
        region=region,
        projection=projection,
        frame=frame_spec
    )
    
    # Add coastlines and land BEFORE density plotting (bottom layer)
    print("  Adding high-detail coastlines and borders...")
    fig.coast(
        land="lightgray",
        water="white",  # White ocean for clean appearance
        shorelines="1.0p,black",  # Thicker shorelines for visibility
        resolution="h",  # high resolution for detailed coastlines
        borders="1/0.5p,darkgray"  # National borders (thicker)
    )
    
    # Plot MC samples based on mode
    print(f"\nVisualization mode: {args.mode}")
    
    # Determine estimated location (KDE peak) for ALL modes
    # This ensures consistency across all visualization types
    if kde_peak_lat_saved is not None and kde_peak_lon_saved is not None:
        mean_lat = kde_peak_lat_saved
        mean_lon = kde_peak_lon_saved
        print(f"  Using KDE peak as estimated location: {mean_lat:.4f}°, {mean_lon:.4f}°")
    else:
        # Fallback if pickle file not available
        mean_lat = df['latitude'].mean()
        mean_lon = df['longitude'].mean()
        print(f"  Using sample mean as estimated location: {mean_lat:.4f}°, {mean_lon:.4f}°")
    
    if args.mode == 'points':
        # Mode 3: Simple points only (no density calculation)
        print(f"  Plotting {n_samples} MC sample points (simple dots)...")
        fig.plot(
            x=df['longitude'],
            y=df['latitude'],
            style="c0.08c",  # small circles
            fill="blue",
            pen="0.2p,darkblue",
            transparency=40
        )
        
    elif args.mode in ['density', 'contour']:
        # Modes 1 & 2: Calculate KDE density
        print(f"  Creating 2D probability density from {n_samples} Monte Carlo samples...")
        
        # Prepare data for KDE
        lons = df['longitude'].values
        lats = df['latitude'].values
        values = np.vstack([lons, lats])
        
        # Create kernel density estimate with increased bandwidth for smoothness
        # Higher bandwidth = smoother, less streaky appearance
        kernel = gaussian_kde(values, bw_method=0.15)  # Increased from 0.08 for smoother result
        
        if args.mode == 'density':
            # Mode 1: Plot sample points colored by density (within 95% confidence)
            print("  Calculating density at each sample point...")
            
            # Evaluate KDE at each sample point location
            sample_positions = np.vstack([lons, lats])
            sample_densities = kernel(sample_positions)
            
            # Normalize densities to 0-1 range
            density_norm = (sample_densities - sample_densities.min()) / (sample_densities.max() - sample_densities.min())
            
            # Apply 95% confidence threshold
            threshold_95 = np.percentile(density_norm, 5)  # Keep top 95%
            print(f"  Applying 95% confidence threshold: {threshold_95:.4f}")
            
            # Filter samples: keep only those above threshold
            mask = density_norm >= threshold_95
            filtered_lons = lons[mask]
            filtered_lats = lats[mask]
            filtered_densities = density_norm[mask]
            
            n_filtered = len(filtered_lons)
            print(f"  Plotting {n_filtered} sample points (95% confidence region)...")
            
            # Create colormap for density
            pygmt.makecpt(cmap="jet", series=[threshold_95, 1.0, 0.05], reverse=False)
            
            # Plot only filtered sample points
            fig.plot(
                x=filtered_lons,
                y=filtered_lats,
                fill=filtered_densities,
                style="c0.08c",  # small circles
                cmap=True,
                transparency=30
            )
            
        elif args.mode == 'contour':
            # Mode 2: Plot as filled contours (needs grid)
            print("  Creating grid for contour plot...")
            
            # Create a higher-resolution grid for smoother contours
            grid_res = 200  # Higher resolution for smoother contours
            lon_grid = np.linspace(lon_min, lon_max, grid_res)
            lat_grid = np.linspace(lat_min, lat_max, grid_res)
            lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
            
            # Evaluate KDE on grid
            positions = np.vstack([lon_mesh.ravel(), lat_mesh.ravel()])
            density = kernel(positions).reshape(lon_mesh.shape)
            
            # Apply Gaussian smoothing to eliminate any remaining streaks
            from scipy.ndimage import gaussian_filter
            density_smoothed = gaussian_filter(density, sigma=2.0)
            
            # Normalize density to 0-1 range
            density_norm = (density_smoothed - density_smoothed.min()) / (density_smoothed.max() - density_smoothed.min())
            
            # Apply 95% confidence threshold
            valid_density = density_norm[~np.isnan(density_norm)]
            threshold_95 = np.percentile(valid_density, 5)
            print(f"  Applying 95% confidence threshold: {threshold_95:.4f}")
            
            # Save grid to temporary NetCDF file for PyGMT
            import xarray as xr
            import tempfile
            
            # Create xarray DataArray
            da = xr.DataArray(
                density_norm,
                coords={'lat': lat_grid, 'lon': lon_grid},
                dims=['lat', 'lon']
            )
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
                tmp_grid_file = tmp.name
                da.to_netcdf(tmp_grid_file)
            
            # Use grdclip to set everything below threshold to NaN at the GMT level
            # This ensures proper transparency handling by GMT
            print(f"  Clipping grid: setting values below {threshold_95:.3f} to NaN...")
            with tempfile.NamedTemporaryFile(suffix='_clipped.nc', delete=False) as tmp_clip:
                clipped_grid_file = tmp_clip.name
            
            clipped_grid = pygmt.grdclip(
                grid=tmp_grid_file,
                below=[threshold_95, "NaN"],  # Set everything below threshold to NaN
                outgrid=clipped_grid_file
            )
            
            # Plot contour lines (not filled) with labels
            print(f"  Drawing contour lines with labels...")
            
            fig.grdcontour(
                grid=clipped_grid_file,
                interval=0.2,       # Draw contour every 0.2 units (more lines)
                annotation=0.2,     # Label every 0.4 units (fewer labels for clarity)
                pen="1.5p,blue"
            )
            
            # Clean up temp files
            os.remove(tmp_grid_file)
            os.remove(clipped_grid_file)
    
    # Plot true event location (if known)
    if true_event_lat is not None and true_event_lon is not None:
        print("  Plotting true event location...")
        fig.plot(
            x=[true_event_lon],
            y=[true_event_lat],
            style="a0.6c",  # star, 0.6 cm
            fill="gold",
            pen="1.5p,black",
            label="True Event"
        )
    
    # Plot estimated event location (calculated earlier based on mode)
    if args.mode == 'points':
        print("  Plotting estimated event location (mean of samples)...")
    else:
        print("  Plotting estimated event location (KDE peak)...")
    fig.plot(
        x=[mean_lon],
        y=[mean_lat],
        style="a0.6c",  # star, 0.6 cm (matching true event)
        fill="white",
        pen="1.5p,black",
        label="Estimated Event"
    )
    
    # Calculate location error for display (if true location is known)
    if true_event_lat is not None and true_event_lon is not None:
        from pyproj import Geod
        g = Geod(ellps='WGS84')
        _, _, dist_m = g.inv(true_event_lon, true_event_lat, mean_lon, mean_lat)
        error_km = dist_m / 1000
        print(f"  Location error: {error_km:.1f} km")
    
    
    # Add colorbar for probability density (only for density mode)
    if args.mode == 'density':
        print("  Adding colorbar...")
        fig.colorbar(
            position="JMR+w8c/0.5c+v",  # Right side, vertical
            frame="x+lnormalized probability density"
        )
    
    # Add legend (bottom-left)
    print("  Adding legend...")
    fig.legend(
        position="JBL+w5c+o0.5c",  # Bottom-left, 3.5cm wide, 0.3cm offset
        box="+gwhite+p1p,black"       # White background, black border
    )
    
    
    # Add title centered at top of map
    center_lon = (lon_min + lon_max) / 2
    fig.text(
        x=center_lon,
        y=lat_max,
        text=f"Event Location Estimate ({n_samples} MC Samples)",
        font="14p,Helvetica-Bold,black",
        justify="TC",
        offset="0/0.5c"  # Offset above the map edge
    )
    
    # Save figure with mode-specific filename
    if args.output is None:
        if args.mode == 'density':
            filename = 'location_map_density.png'
        elif args.mode == 'contour':
            filename = 'location_map_contour.png'
        elif args.mode == 'points':
            filename = 'location_map_points_only.png'
        output_file = os.path.join(args.results, filename)
    else:
        output_file = args.output
    
    print(f"\nSaving figure to: {output_file}")
    fig.savefig(output_file, dpi=300)
    
    print("\n" + "="*80)
    print("MAP CREATION COMPLETE")
    print("="*80)
    print(f"Output: {output_file}")
    print("\nSummary Statistics:")
    
    if true_event_lat is not None and true_event_lon is not None:
        print(f"  True event:      {true_event_lat:.4f}°, {true_event_lon:.4f}°")
    
    print(f"  Estimated event: {mean_lat:.4f}°, {mean_lon:.4f}°")
    
    # Calculate error (if true location is known)
    if true_event_lat is not None and true_event_lon is not None:
        from pyproj import Geod
        g = Geod(ellps='WGS84')
        _, _, dist_m = g.inv(true_event_lon, true_event_lat, mean_lon, mean_lat)
        dist_km = dist_m / 1000
        print(f"  Location error:  {dist_km:.1f} km")
    
    print("="*80)
    
    # Show the figure
    print("\nDisplaying map...")
    fig.show()


if __name__ == "__main__":
    main()
