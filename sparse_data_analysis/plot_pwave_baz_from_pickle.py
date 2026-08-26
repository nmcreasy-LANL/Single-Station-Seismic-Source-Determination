#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract and Plot P-wave Backazimuth PDF from Polarization Analysis

This script loads the pickle file output from example_picker_with_polarization.py
and extracts/plots the P-wave backazimuth probability density function.

Supports two modes:
  - P-wave mode: Extract BAZ from P-wave window data (default)
  - P-from-S mode: Infer P-wave BAZ from S-wave polarization using cross-product

Usage:
    python plot_pwave_baz_from_pickle.py <pickle_file> [options]
    
Examples:
    python plot_pwave_baz_from_pickle.py Peru_Event_Polarizationout.pkl
    python plot_pwave_baz_from_pickle.py Peru_Event_Polarizationout.pkl --output my_baz_plot.png
    python plot_pwave_baz_from_pickle.py Peru_Event_Polarizationout.pkl --wave-type P-from-S

@author: nmcreasy
Created: 2026-06-16
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse
import os


def azi_inc_to_xyz_vector(azi, inc, r=1):
    """
    Convert azimuth and inclination to 3D Cartesian vector.
    
    Parameters:
        azi: Azimuth in radians
        inc: Inclination in radians (0 = horizontal, pi/2 = vertical down)
        r: Magnitude of vector
    
    Returns:
        vector: [x, y, z] numpy array
    """
    y = np.sin(np.pi/2 - inc) * np.sin(azi) * r
    x = np.sin(np.pi/2 - inc) * np.cos(azi) * r
    z = np.cos(np.pi/2 - inc) * r
    vector = np.array([x, y, z])
    return vector


def calculate_p_from_s_crossproduct(s_azi_array, s_inc_array, weights, 
                                    baz_step=5, inc_step=3):
    """
    Calculate P-wave BAZ from S-wave polarization using cross-product method.
    
    This implements the algorithm from polarisation_plot.py:
    - Creates a grid of all possible P-wave directions
    - For each possible P-wave, calculates cross product with all S-wave vectors
    - Cross product is maximized when P and S are perpendicular
    - Returns azimuth distribution weighted by likelihood
    
    Parameters:
        s_azi_array: Array of S-wave azimuths in degrees
        s_inc_array: Array of S-wave inclinations in degrees  
        weights: Weights for each measurement
        baz_step: Step size for azimuth grid (degrees)
        inc_step: Step size for inclination grid (degrees)
    
    Returns:
        p_azimuths: Weighted P-wave azimuths
        p_weights: Corresponding weights
    """
    print(f"  - Using cross-product method with {baz_step}° × {inc_step}° grid")
    
    # Convert S-wave data to radians and create 3D vectors (vectorized)
    s_azi_rad = np.deg2rad(s_azi_array)
    s_inc_rad = np.deg2rad(s_inc_array)
    
    # Create S-wave vectors in Cartesian coordinates (N x 3 array)
    s_x = np.sin(np.pi/2 - s_inc_rad) * np.cos(s_azi_rad) * weights
    s_y = np.sin(np.pi/2 - s_inc_rad) * np.sin(s_azi_rad) * weights
    s_z = np.cos(np.pi/2 - s_inc_rad) * weights
    s_vectors = np.column_stack([s_x, s_y, s_z])  # Shape: (N, 3)
    
    print(f"  - Processing {len(s_vectors)} S-wave vectors...")
    
    # Create grid of possible P-wave directions
    baz_grid = np.arange(0, 360, baz_step)
    inc_grid = np.arange(0, 90, inc_step)
    
    # Calculate likelihood for each possible P-wave direction
    azimuth_likelihood = np.zeros(len(baz_grid))
    
    # Process each azimuth (this is the main loop we care about)
    for i, baz_P in enumerate(baz_grid):
        baz_P_rad = np.deg2rad(baz_P)
        
        # For this azimuth, sum contributions from all inclinations
        inc_likelihood = 0
        for inc_P in inc_grid:
            inc_P_rad = np.deg2rad(inc_P)
            
            # Create P-wave unit vector for this (baz, inc)
            px = np.sin(np.pi/2 - inc_P_rad) * np.cos(baz_P_rad)
            py = np.sin(np.pi/2 - inc_P_rad) * np.sin(baz_P_rad)
            pz = np.cos(np.pi/2 - inc_P_rad)
            
            # Vectorized cross product: P × all S vectors
            # cross(P, S) = [py*sz - pz*sy, pz*sx - px*sz, px*sy - py*sx]
            cross_x = py * s_vectors[:, 2] - pz * s_vectors[:, 1]
            cross_y = pz * s_vectors[:, 0] - px * s_vectors[:, 2]
            cross_z = px * s_vectors[:, 1] - py * s_vectors[:, 0]
            
            # Magnitude of cross products (vectorized)
            cross_mag = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
            
            # Sum all magnitudes for this P-wave direction
            inc_likelihood += np.sum(cross_mag)
        
        azimuth_likelihood[i] = inc_likelihood
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"    Processed {i+1}/{len(baz_grid)} azimuths...")
    
    # Normalize to create weights
    total_likelihood = np.sum(azimuth_likelihood)
    if total_likelihood > 0:
        azimuth_weights = azimuth_likelihood / total_likelihood
    else:
        azimuth_weights = np.ones_like(azimuth_likelihood) / len(azimuth_likelihood)
    
    # Create samples based on likelihood (for KDE)
    n_samples = 10000
    p_azimuths = np.random.choice(baz_grid, size=n_samples, p=azimuth_weights)
    p_weights = np.ones(n_samples) / n_samples
    
    return p_azimuths, p_weights


def fwhm_error_from_kde(xs, ys, index):
    """
    Calculate the Full Width at Half Maximum (FWHM) error from KDE curve.
    
    Parameters:
        xs: x-values (backazimuth angles)
        ys: y-values (probability density)
        index: index of maximum value
    
    Returns:
        error: [left_error, right_error] in degrees
    """
    max_y = max(ys)
    indexes_ymax = [x for x in range(len(ys)) if ys[x] > max_y/2.0]
    
    # Handle edge case where peak is at boundary
    if len(indexes_ymax) == 0:
        print("  WARNING: Could not calculate FWHM - no values above half maximum")
        baz = xs[index]
        if baz < 0:
            baz = baz + 360
        elif baz > 360:
            baz = baz - 360
        return [baz - 10, baz + 10]  # Return ±10° as rough estimate
    
    if index not in indexes_ymax:
        print(f"  WARNING: Peak at boundary (index {index}), using nearby indices")
        # Find closest index in the list
        closest_idx = min(indexes_ymax, key=lambda x: abs(x - index))
        index_local = indexes_ymax.index(closest_idx)
    else:
        # Get correct FWHM in case there are several peaks above the halfway mark
        index_local = indexes_ymax.index(index)
    
    # Initialize with defaults
    index_high = indexes_ymax[-1]
    index_low = indexes_ymax[0]
    
    # Search forward through list
    for k in range(index_local, len(indexes_ymax)-1):
        if indexes_ymax[k+1] > indexes_ymax[k]+1:
            index_high = indexes_ymax[k]
            break
        elif k == len(indexes_ymax)-2:  # if there is only one peak
            index_high = indexes_ymax[-1]
    
    # Search backward through list
    for k in range(index_local, 0, -1):
        if indexes_ymax[k-1] < indexes_ymax[k]-1:
            index_low = indexes_ymax[k]
            break
        elif k == 1:  # if there is only one peak
            index_low = indexes_ymax[0]
    
    left_error = xs[index_low]
    right_error = xs[index_high]
    
    # Wrap the errors around 0
    if left_error < 0.:
        left_error = 360. + left_error
    if right_error > 360.:
        right_error = right_error - 360.
    
    error = [left_error, right_error]
    return error


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
    
    # Calculate FWHM error
    error = fwhm_error_from_kde(xs, ys, index)
    
    return baz, error, xs, ys


def plot_pwave_baz_pdf(pickle_file, output_file=None, show_plot=True, 
                       include_comparison=False, figsize=(10, 6)):
    """
    Load pickle file and plot P-wave backazimuth PDF.
    
    Parameters:
        pickle_file: Path to pickle file from polarization analysis
        output_file: Path to save output figure (optional)
        show_plot: Whether to display the plot
        include_comparison: Include S-wave and noise KDE curves
        figsize: Figure size (width, height) in inches
    
    Returns:
        baz: Peak backazimuth in degrees
        error: [left_error, right_error] in degrees
    """
    # Load the pickle file
    print(f"Loading pickle file: {pickle_file}")
    try:
        with open(pickle_file, 'rb') as f:
            kde_dataframe = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: Pickle file not found: {pickle_file}")
        return None, None
    except Exception as e:
        print(f"ERROR loading pickle file: {e}")
        return None, None
    
    print(f"Pickle file loaded successfully")
    print(f"Data structure: {type(kde_dataframe)}")
    
    # The pickle contains a dictionary with 'P' and 'S' keys
    # Each contains a list of dictionaries, one per row:
    # Row 0 = amplitude data (dB)
    # Row 1 = azimuth data (degrees)
    # Row 2 = inclination data (degrees)
    
    # Handle both old format (list only S) and new format (dict with 'P' and 'S' keys)
    if isinstance(kde_dataframe, dict) and 'P' in kde_dataframe:
        # New format: {'P': [...], 'S': [...]}
        p_wave_list = kde_dataframe['P']
        s_wave_list = kde_dataframe.get('S', None)
        print(f"  - Format: New (dict with P and S keys)")
        print(f"  - P-wave data rows: {len(p_wave_list)}")
        
        # Extract P-wave azimuth data (SECOND row, not first!)
        p_wave_data = p_wave_list[1]  # Row 1 = azimuth
        
        if 'P' in p_wave_data and 'weights' in p_wave_data:
            print(f"\nP-wave data found:")
            print(f"  - Number of samples: {len(p_wave_data['P'])}")
            print(f"  - Azimuth range: {np.min(p_wave_data['P']):.1f}° to {np.max(p_wave_data['P']):.1f}°")
            
            # Calculate BAZ and error
            baz, error, xs, ys = calculate_baz_from_kde(
                p_wave_data['P'], 
                p_wave_data['weights']
            )
            
            print(f"\nBackazimuth Analysis:")
            print(f"  - Peak BAZ: {baz:.1f}°")
            print(f"  - Error range (FWHM): [{error[0]:.1f}°, {error[1]:.1f}°]")
            
            # Create the plot
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot P-wave KDE (only 0-360 range)
            valid_mask = (xs >= 0) & (xs <= 360)
            ax.plot(xs[valid_mask], ys[valid_mask], 'C0', linewidth=2.5, label='P-wave')
            ax.fill_between(xs[valid_mask], ys[valid_mask], alpha=0.3, color='C0')
            
            # Mark the peak BAZ
            ax.axvline(baz, color='C0', linestyle='--', linewidth=1.5, alpha=0.7)
            
            # Mark the error range (FWHM)
            ax.axvline(error[0], color='C0', linestyle=':', linewidth=1, alpha=0.5)
            ax.axvline(error[1], color='C0', linestyle=':', linewidth=1, alpha=0.5)
            
            # Add text annotation for BAZ
            y_pos = np.max(ys[valid_mask]) * 0.9
            ax.text(baz, y_pos, f'BAZ = {baz:.1f}°\n±[{error[0]:.1f}°, {error[1]:.1f}°]',
                   ha='center', va='top', fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Optionally add S-wave and noise for comparison
            if include_comparison and 'S' in p_wave_data:
                if len(p_wave_data['S']) > 0:
                    _, _, xs_s, ys_s = calculate_baz_from_kde(
                        p_wave_data['S'], 
                        kde_dataframe[0]['weights']  # Use same weights structure
                    )
                    ax.plot(xs_s[valid_mask], ys_s[valid_mask], 'firebrick', 
                           linewidth=2, alpha=0.7, label='S-wave')
                
                if 'Noise' in kde_dataframe[0]:
                    if len(kde_dataframe[0]['Noise']) > 0:
                        noise_weights = kde_dataframe[0].get('weights', 
                                                             p_wave_data['weights'])
                        _, _, xs_n, ys_n = calculate_baz_from_kde(
                            kde_dataframe[0]['Noise'], 
                            noise_weights
                        )
                        ax.fill_between(xs_n[valid_mask], ys_n[valid_mask], 
                                       alpha=0.3, color='grey', label='Noise')
            
            # Styling
            ax.set_xlabel('Backazimuth (°)', fontsize=13)
            ax.set_ylabel('Probability Density', fontsize=13)
            ax.set_title('P-wave Backazimuth Probability Density Function', 
                        fontsize=14, fontweight='bold')
            ax.set_xlim(0, 360)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper right', fontsize=11)
            
            # Set x-ticks every 30 degrees
            ax.set_xticks(np.arange(0, 361, 30))
            
            plt.tight_layout()
            
            # Save if output file specified
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"\nPlot saved to: {output_file}")
            
            # Show plot if requested
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return baz, error
            
        else:
            print("ERROR: Expected data structure not found in pickle file")
            print(f"Available keys: {p_wave_data.keys()}")
            return None, None
    else:
        print(f"ERROR: Unexpected data structure in pickle file")
        return None, None


def plot_pwave_from_swave_pdf(pickle_file, output_file=None, show_plot=True, figsize=(10, 6)):
    """
    Load pickle file and plot P-wave BAZ inferred from S-wave polarization.
    
    Parameters:
        pickle_file: Path to pickle file from polarization analysis
        output_file: Path to save output figure (optional)
        show_plot: Whether to display the plot
        figsize: Figure size (width, height) in inches
    
    Returns:
        baz: Peak backazimuth in degrees
        error: [left_error, right_error] in degrees
    """
    print(f"Loading pickle file: {pickle_file}")
    try:
        with open(pickle_file, 'rb') as f:
            kde_dataframe = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: Pickle file not found: {pickle_file}")
        return None, None
    except Exception as e:
        print(f"ERROR loading pickle file: {e}")
        return None, None
    
    print(f"Pickle file loaded successfully")
    
    if isinstance(kde_dataframe, dict) and 'S' in kde_dataframe:
        s_wave_list = kde_dataframe['S']
        print(f"  - S-wave data rows: {len(s_wave_list)}")
        
        # Extract S-wave azimuth (Row 1) and inclination (Row 2)
        s_azi_data = s_wave_list[1]  # Row 1 = azimuth
        s_inc_data = s_wave_list[2]  # Row 2 = inclination
        
        if 'S' in s_azi_data and 'S' in s_inc_data and 'weights' in s_azi_data:
            print(f"\nS-wave data found:")
            print(f"  - Number of samples: {len(s_azi_data['S'])}")
            print(f"  - Azimuth range: {np.min(s_azi_data['S']):.1f}° to {np.max(s_azi_data['S']):.1f}°")
            print(f"  - Inclination range: {np.min(s_inc_data['S']):.1f}° to {np.max(s_inc_data['S']):.1f}°")
            
            # Calculate P-wave from S-wave using cross-product method
            print("\nCalculating P-wave BAZ from S-wave polarization...")
            p_azimuths, p_weights = calculate_p_from_s_crossproduct(
                s_azi_data['S'],
                s_inc_data['S'],
                s_azi_data['weights']
            )
            
            print(f"  - Generated {len(p_azimuths)} P-wave azimuth samples")
            
            # Calculate BAZ and error
            baz, error, xs, ys = calculate_baz_from_kde(p_azimuths, p_weights)
            
            print(f"\nP-wave Backazimuth (from S-wave):")
            print(f"  - Peak BAZ: {baz:.1f}°")
            print(f"  - Error range (FWHM): [{error[0]:.1f}°, {error[1]:.1f}°]")
            
            # Create the plot
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot P-wave KDE (only 0-360 range)
            valid_mask = (xs >= 0) & (xs <= 360)
            ax.plot(xs[valid_mask], ys[valid_mask], 'firebrick', linewidth=2.5, 
                   label='P from S-wave')
            ax.fill_between(xs[valid_mask], ys[valid_mask], alpha=0.3, color='firebrick')
            
            # Mark the peak BAZ
            ax.axvline(baz, color='firebrick', linestyle='--', linewidth=1.5, alpha=0.7)
            
            # Mark the error range (FWHM)
            ax.axvline(error[0], color='firebrick', linestyle=':', linewidth=1, alpha=0.5)
            ax.axvline(error[1], color='firebrick', linestyle=':', linewidth=1, alpha=0.5)
            
            # Add text annotation for BAZ
            y_pos = np.max(ys[valid_mask]) * 0.9
            ax.text(baz, y_pos, f'BAZ = {baz:.1f}°\n±[{error[0]:.1f}°, {error[1]:.1f}°]',
                   ha='center', va='top', fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Styling
            ax.set_xlabel('Backazimuth (°)', fontsize=13)
            ax.set_ylabel('Probability Density', fontsize=13)
            ax.set_title('P-wave Backazimuth (inferred from S-wave polarization)', 
                        fontsize=14, fontweight='bold')
            ax.set_xlim(0, 360)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper right', fontsize=11)
            ax.set_xticks(np.arange(0, 361, 30))
            
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"\nPlot saved to: {output_file}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return baz, error
            
        else:
            print("ERROR: Expected data structure not found in S-wave data")
            return None, None
    else:
        print(f"ERROR: No S-wave data found in pickle file")
        return None, None


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Extract and plot P-wave backazimuth PDF from polarization analysis pickle file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract P-wave BAZ from P-wave window (default)
  %(prog)s Peru_Event_Polarizationout.pkl
  %(prog)s Peru_Event_Polarizationout.pkl --output my_baz.png
  
  # Extract P-wave BAZ inferred from S-wave polarization
  %(prog)s Peru_Event_Polarizationout.pkl --wave-type P-from-S
  %(prog)s Peru_Event_Polarizationout.pkl --wave-type P-from-S --output p_from_s.png
        """
    )
    
    parser.add_argument('pickle_file', 
                       help='Path to pickle file from polarization analysis')
    parser.add_argument('-o', '--output', 
                       help='Output figure filename (e.g., baz_plot.png)')
    parser.add_argument('--wave-type', choices=['P', 'P-from-S'], default='P',
                       help='Wave type: "P" for P-wave window (default), "P-from-S" for P inferred from S-wave')
    parser.add_argument('--compare', action='store_true',
                       help='Include S-wave and noise KDE curves for comparison (P mode only)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display the plot')
    parser.add_argument('--figsize', nargs=2, type=float, default=[10, 6],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Figure size in inches (default: 10 6)')
    
    args = parser.parse_args()
    
    # Check if pickle file exists
    if not os.path.exists(args.pickle_file):
        print(f"ERROR: Pickle file not found: {args.pickle_file}")
        print("\nMake sure you've run example_picker_with_polarization.py first")
        print("to generate the polarization analysis output.")
        return
    
    # Generate default output filename if not specified
    if args.output is None and args.no_show:
        base_name = os.path.splitext(args.pickle_file)[0]
        if args.wave_type == 'P-from-S':
            args.output = f"{base_name}_P_from_S.png"
        else:
            args.output = f"{base_name}_BAZ_PDF.png"
    
    # Plot based on wave type
    if args.wave_type == 'P-from-S':
        print("Mode: P-wave BAZ from S-wave polarization")
        baz, error = plot_pwave_from_swave_pdf(
            args.pickle_file,
            output_file=args.output,
            show_plot=not args.no_show,
            figsize=tuple(args.figsize)
        )
        wave_type_label = "P-from-S"
    else:
        print("Mode: P-wave BAZ from P-wave window")
        baz, error = plot_pwave_baz_pdf(
            args.pickle_file,
            output_file=args.output,
            show_plot=not args.no_show,
            include_comparison=args.compare,
            figsize=tuple(args.figsize)
        )
        wave_type_label = "P-wave"
    
    if baz is not None:
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"Wave Type: {wave_type_label}")
        print(f"Peak Backazimuth: {baz:.2f}°")
        print(f"FWHM Error Range: [{error[0]:.2f}°, {error[1]:.2f}°]")
        print("="*60)


if __name__ == "__main__":
    main()
