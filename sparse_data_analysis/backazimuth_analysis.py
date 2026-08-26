#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backazimuth Analysis - Version 4.1 (Backazimuth Only)
Created on Mon Jul 28 14:44:47 2025
@author: nmcreasy

DESCRIPTION:
    This script performs BACKAZIMUTH estimation for seismic events. It loads
    distance/timing results from a pickle file created by sparse_data_distance_timing.py
    and performs polarization analysis to extract backazimuth PDFs.
    
    WORKFLOW:
    1. Load comprehensive results pickle file (from sparse_data_distance_timing.py)
    2. Load raw waveform data  
    3. Run polarization analysis on P-wave and P-from-S phases
    4. Extract backazimuth PDFs from polarization analysis
    5. Optionally combine with Rayleigh wave backazimuth (if available)
    6. Export backazimuth results and plots
    
    KEY FEATURES:
    - Loads distance/timing estimates from pickle file
    - Automatic P-wave polarization analysis → backazimuth PDF
    - P-from-S polarization analysis → backazimuth PDF  
    - Rayleigh wave backazimuth from Stockwell analysis (if available)
    - PDF combination for improved backazimuth estimates
    - Comprehensive backazimuth PDF export
    
    NOTE: Run sparse_data_distance_timing.py first to generate the required pickle file
    
VERSION HISTORY:
    v4.1: Separated backazimuth analysis from distance/timing (this file = BAZ only)
    v4.0: Added automatic backazimuth extraction, advanced Stockwell analysis
    v3.0: Two-stage interactive picking workflow
    v2.0: Grid search and PDF-based location estimation
    v1.0: Basic Rayleigh wave group velocity analysis
    
USAGE:
    python backazimuth_analysis.py --pickle path/to/comprehensive_results.pkl --par_file par_files/PAR_FILE_Peru.py
"""

import argparse
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from obspy import read
from obspy.core import UTCDateTime

# Add parent directory for imports
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)

# Import backazimuth analysis functions
from sparse_data_analysis.backazimuth import run_polarization_baz_analysis

# Import common utilities
UTILS_DIR = os.path.join(PARENT_DIR, 'utils')
sys.path.insert(0, UTILS_DIR)
import common_utils

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 7

def load_pickle_file(pickle_path):
    """
    Load comprehensive results from pickle file.
    
    Parameters:
        pickle_path: Path to pickle file created by sparse_data_distance_timing.py
    
    Returns:
        dict: Comprehensive results dictionary
    """
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
    
    print(f"Loading comprehensive results from: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        results = pickle.load(f)
    
    print("✓ Pickle file loaded successfully\n")
    return results

def load_par_file(par_file_path):
    """Load parameter file dynamically."""
    import importlib.util
    
    if not os.path.exists(par_file_path):
        raise FileNotFoundError(f"Parameter file not found: {par_file_path}")
    
    print(f"Loading parameters from: {par_file_path}")
    
    spec = importlib.util.spec_from_file_location("par_config", par_file_path)
    par_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(par_module)
    print("✓ Parameters loaded successfully\n")
    return par_module

def main():
    """Main backazimuth analysis workflow."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Backazimuth Analysis - Extract BAZ from polarization analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run BAZ analysis on Peru event
  python backazimuth_analysis.py --pickle Results_*/Peru_comprehensive_results.pkl --par_file par_files/PAR_FILE_Peru.py
        '''
    )
    
    parser.add_argument('--pickle', type=str, required=True,
                       help='Path to comprehensive results pickle file')
    parser.add_argument('--par_file', type=str, required=True,
                       help='Path to parameter file (e.g., par_files/PAR_FILE_Peru.py)')
    
    args = parser.parse_args()
    
    # Load pickle file
    try:
        results = load_pickle_file(args.pickle)
    except Exception as e:
        print(f"ERROR: Failed to load pickle file: {e}")
        return
    
    # Load parameter file
    try:
        par = load_par_file(args.par_file)
    except Exception as e:
        print(f"ERROR: Failed to load parameter file: {e}")
        return
    
    # Extract metadata
    metadata = results['metadata']
    station_name = metadata['station']
    event_name = metadata['event_name']
    p_arrival_time = metadata['p_arrival_time_utc']
    event_time = metadata['event_time_utc']
    output_dir = metadata['output_dir']
    
    print(f"\n{'='*80}")
    print(f"BACKAZIMUTH ANALYSIS")
    print(f"{'='*80}")
    print(f"Station: {station_name}")
    print(f"Event: {event_name}")
    print(f"P-arrival: {p_arrival_time}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    # Load raw waveform data
    print("Loading raw waveform data...")
    data_dir = par.DATA_DIR
    files = par.FILES
    
    from obspy import Stream
    st_raw = Stream()
    for file in files:
        filepath = os.path.join(data_dir, file)
        st_raw += read(filepath)
    
    print(f"✓ Loaded {len(st_raw)} traces\n")
    
    # Extract picks from results
    stage2_picks = results.get('picks_utc', {})
    
    # Extract Stockwell results if available
    stockwell_data = results.get('stockwell', {})
    stockwell_results = None
    if stockwell_data.get('distance_deg') is not None:
        stockwell_results = {
            'distance_deg': stockwell_data['distance_deg'],
            'baz_deg': stockwell_data.get('baz_deg'),
            'baz_std': stockwell_data.get('baz_std'),
            'baz_posterior': stockwell_data.get('baz_posterior'),
            'baz_azimuth_range': stockwell_data.get('baz_azimuth_range'),
            'baz_reliability': stockwell_data.get('baz_reliability'),
            'detected_orbits': stockwell_data.get('detected_orbits')
        }
    
    # Run polarization BAZ analysis
    print(f"\n{'='*80}")
    print("RUNNING POLARIZATION BACKAZIMUTH ANALYSIS")
    print(f"{'='*80}\n")
    
    polarization_params = par.POLARIZATION_PARAMS
    
    baz_result = run_polarization_baz_analysis(
        st_raw=st_raw,
        stage2_picks=stage2_picks,
        EVENT_TIME=event_time,
        OUTPUT_DIR=output_dir,
        EVENT_NAME=event_name,
        stockwell_results=stockwell_results,
        POLARIZATION_PARAMS=polarization_params
    )
    
    # Print final results
    if baz_result:
        print(f"\n{'='*80}")
        print("BACKAZIMUTH ANALYSIS COMPLETE")
        print(f"{'='*80}")
        
        if baz_result.get('p_baz') is not None:
            print(f"P-wave BAZ: {baz_result['p_baz']:.1f}° "
                  f"[{baz_result['p_baz_error'][0]:.1f}°, {baz_result['p_baz_error'][1]:.1f}°]")
        
        if baz_result.get('pfroms_baz') is not None:
            print(f"P-from-S BAZ: {baz_result['pfroms_baz']:.1f}° "
                  f"[{baz_result['pfroms_error'][0]:.1f}°, {baz_result['pfroms_error'][1]:.1f}°]")
        
        if baz_result.get('combined_baz') is not None:
            print(f"Combined BAZ: {baz_result['combined_baz']:.1f}° "
                  f"(±{baz_result['combined_baz_std']:.1f}°)")
        
        print(f"\nResults saved to: {output_dir}")
        print(f"{'='*80}\n")
    else:
        print("\nWARNING: Backazimuth analysis returned no results.\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
