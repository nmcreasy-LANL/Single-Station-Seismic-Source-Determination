#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P-wave Polarization Analysis integrated with Interactive Phase Picker
Uses picked P and S wave times from rayleighwave_picker_v3.py

Integrates:
- polarisation_calculation.py for polarization computation
- polarisation_plot.py for visualization (adapted for picker integration)

@author: nmcreasy
Created: 2026-06-15
"""

import sys
import os
import numpy as np
from obspy import Stream, UTCDateTime
from obspy.core import UTCDateTime as utct

# Add Polarization_Package to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
POLARIZATION_DIR = os.path.join(PARENT_DIR, 'Polarization_Package')
sys.path.insert(0, POLARIZATION_DIR)

# Import polarization functions with error handling
try:
    import polarisation_plot as ppl
except ImportError as e:
    raise ImportError(
        "\n" + "="*80 + "\n"
        "ERROR: Polarization package not found!\n\n"
        "The polarization analysis package must be installed separately.\n"
        "Please follow the installation instructions in README_polarization_integration.md\n\n"
        "Quick Installation:\n"
        "  cd " + SCRIPT_DIR + "\n"
        "  git clone https://github.com/gzenhaeusern/polarisation-package.git Polarization_Package\n\n"
        "GitHub: https://github.com/gzenhaeusern/polarisation-package\n"
        "="*80 + "\n"
    ) from e


def validate_picks(picks_dict, required_phases=['P', 'S']):
    """
    Validate that required picks exist in the picks dictionary.
    
    Parameters:
        picks_dict: Dictionary of picks from interactive picker
        required_phases: List of required phase names
    
    Returns:
        tuple: (is_valid, missing_phases)
    """
    missing_phases = []
    
    for phase in required_phases:
        if phase not in picks_dict:
            missing_phases.append(phase)
        elif not picks_dict[phase]:  # Empty list
            missing_phases.append(phase)
        elif len(picks_dict[phase]) == 0:
            missing_phases.append(phase)
    
    is_valid = len(missing_phases) == 0
    return is_valid, missing_phases


def calculate_time_windows(p_pick_time, s_pick_time, t_window_P, t_window_S, 
                           noise_duration, noise_gap):
    """
    Calculate time windows for P, S, and noise based on picked times.
    
    Parameters:
        p_pick_time: P pick time in seconds relative to event
        s_pick_time: S pick time in seconds relative to event
        t_window_P: [before, after] in seconds around P pick
        t_window_S: [before, after] in seconds around S pick
        noise_duration: Duration of noise window in seconds
        noise_gap: Gap between noise end and P pick in seconds
    
    Returns:
        tuple: (t_pick_P, t_pick_S, timing_noise)
            - t_pick_P: [start_offset, end_offset] relative to P pick
            - t_pick_S: [start_offset, end_offset] relative to S pick
            - timing_noise: [start_time, end_time] as UTCDateTime objects
    """
    # P and S windows are already provided as offsets
    t_pick_P = t_window_P
    t_pick_S = t_window_S
    
    # Calculate noise window times
    # Noise ends at (P_pick - noise_gap), and starts (noise_duration) seconds before that
    noise_end_time = p_pick_time - noise_gap
    noise_start_time = noise_end_time - noise_duration
    
    timing_noise = [noise_start_time, noise_end_time]
    
    return t_pick_P, t_pick_S, timing_noise


def analyze_polarization_from_picks(
    st,
    event_time,
    picks_dict,
    t_window_P=[-10, 10],
    t_window_S=[-5, 20],
    noise_duration=120,
    noise_gap=10,
    fmin=0.1,
    fmax=10.0,
    f_band_density=(0.3, 1.0),
    kind='cwt',
    w0=8,
    nf=100,
    winlen_sec=10.,
    overlap=0.5,
    dop_winlen=10,
    dop_specwidth=1.1,
    vmin=-180,
    vmax=-140,
    fname='Polarization_from_Picker',
    differentiate=False,
    detick_1Hz=False,
    zoom=False,
    rotation='ZNE',
    BAZ_fixed=None,
    inc_fixed=None,
    alpha_inc=None,
    alpha_elli=None,
    alpha_azi=None
):
    """
    Compute P-wave polarization analysis using picked P and S times
    from the interactive phase picker.
    
    Parameters:
        st: ObsPy Stream (3 components: Z, N, E or E, Z, N)
        event_time: UTCDateTime of event origin (or P-arrival time if unknown)
        picks_dict: Dictionary of picks from picker {'P': [times...], 'S': [times...]}
                   Times can be EITHER:
                   - UTCDateTime objects (recommended for correct noise window)
                   - Seconds relative to event_time (legacy support)
        
        Time Window Parameters:
        t_window_P: [before, after] time window around P pick in seconds (default: [-10, 10])
        t_window_S: [before, after] time window around S pick in seconds (default: [-5, 20])
        noise_duration: Duration of pre-event noise window in seconds (default: 120)
        noise_gap: Gap between noise window end and P pick in seconds (default: 10)
        
        Frequency Parameters:
        fmin: Minimum frequency for analysis in Hz (default: 0.1)
        fmax: Maximum frequency for analysis in Hz (default: 10.0)
        f_band_density: Frequency band for BAZ estimation in Hz (default: (0.3, 1.0))
        
        Time-Frequency Analysis Parameters:
        kind: 'cwt' or 'spec' for time-frequency analysis (default: 'cwt')
        w0: CWT parameter for time-freq resolution tradeoff (default: 8)
        nf: Number of frequencies for CWT (default: 100)
        winlen_sec: Window length for spectrograms in seconds (default: 10.)
        overlap: Overlap fraction for spectrograms (default: 0.5)
        
        Polarization Parameters:
        dop_winlen: Window length for degree of polarization (default: 10)
        dop_specwidth: Spectral width for degree of polarization (default: 1.1)
        
        Plotting Parameters:
        vmin: Minimum amplitude for plots in dB (default: -180)
        vmax: Maximum amplitude for plots in dB (default: -140)
        fname: Output filename prefix (default: 'Polarization_from_Picker')
        zoom: Zoom into P/S windows on plots (default: False)
        
        Data Processing Parameters:
        differentiate: Set True if using displacement data (default: False)
        detick_1Hz: Remove 1Hz tick noise (for Mars InSight data) (default: False)
        rotation: Coordinate system 'ZNE', 'RT', or 'LQT' (default: 'ZNE')
        
        Optional Manual Parameters:
        BAZ_fixed: Manual back azimuth in degrees for comparison (default: None)
        inc_fixed: Manual inclination in degrees for comparison (default: None)
        alpha_inc: Filtering factor for inclination (default: None)
        alpha_elli: Filtering factor for ellipticity (default: None)
        alpha_azi: Filtering factor for azimuth (default: None)
    
    Returns:
        tuple: (BAZ_estimated, inc_estimated, error_range)
            - BAZ_estimated: Estimated back azimuth in degrees
            - inc_estimated: Estimated inclination in degrees
            - error_range: [lower_bound, upper_bound] uncertainty range in degrees
    """
    
    print("\n" + "="*80)
    print("P-WAVE POLARIZATION ANALYSIS FROM INTERACTIVE PICKS")
    print("="*80)
    
    # Validate picks
    is_valid, missing = validate_picks(picks_dict, required_phases=['P', 'S'])
    if not is_valid:
        error_msg = f"ERROR: Missing required picks: {', '.join(missing)}"
        print(error_msg)
        print("Polarization analysis requires both P and S picks.")
        print("="*80 + "\n")
        return None, None, None
    
    # Extract first picks of P and S
    # Picks can be either seconds relative to event_time OR UTC timestamps
    p_pick = picks_dict['P'][0]
    s_pick = picks_dict['S'][0]
    
    # Check if picks are UTC timestamps or seconds relative to event
    if isinstance(p_pick, (UTCDateTime, utct)):
        # Picks are already in UTC
        timing_P = p_pick
        timing_S = s_pick
        p_pick_time = p_pick - event_time  # Calculate relative time for display
        s_pick_time = s_pick - event_time
        print(f"Using picks (UTC format):")
        print(f"  P pick: {timing_P} = {p_pick_time:.2f}s after event ({p_pick_time/60:.2f} min)")
        print(f"  S pick: {timing_S} = {s_pick_time:.2f}s after event ({s_pick_time/60:.2f} min)")
    else:
        # Picks are in seconds relative to event
        p_pick_time = p_pick
        s_pick_time = s_pick
        print(f"Using picks (seconds relative to event):")
        print(f"  P pick: {p_pick_time:.2f}s ({p_pick_time/60:.2f} min after event)")
        print(f"  S pick: {s_pick_time:.2f}s ({s_pick_time/60:.2f} min after event)")
        
        # Convert to absolute times
        timing_P = utct(event_time) + p_pick_time
        timing_S = utct(event_time) + s_pick_time
    
    # Calculate time windows (offsets are always relative to picks)
    t_pick_P = t_window_P
    t_pick_S = t_window_S
    
    # Calculate noise window directly from UTC P pick
    timing_noise = [timing_P - noise_duration - noise_gap, 
                   timing_P - noise_gap]
    
    print(f"\nTime windows:")
    print(f"  P window: [{t_pick_P[0]}, {t_pick_P[1]}]s relative to P pick")
    print(f"  S window: [{t_pick_S[0]}, {t_pick_S[1]}]s relative to S pick")
    print(f"  Noise window: {noise_duration}s ending {noise_gap}s before P pick")
    
    print(f"\nAbsolute times:")
    print(f"  P pick: {timing_P}")
    print(f"  S pick: {timing_S}")
    print(f"  Noise: {timing_noise[0]} to {timing_noise[1]}")
    
    # Set event start/end times for trimming
    # Use noise start as event start, and give plenty of time after S pick
    tstart = timing_noise[0] - 30  # Add 30s buffer before noise
    tend = timing_S + max(60, t_pick_S[1] + 60)  # At least 60s after S window ends
    
    print(f"\nData window:")
    print(f"  Start: {tstart}")
    print(f"  End: {tend}")
    
    print(f"\nPolarization parameters:")
    print(f"  Frequency range: {fmin}-{fmax} Hz")
    print(f"  BAZ estimation band: {f_band_density[0]}-{f_band_density[1]} Hz")
    print(f"  Method: {kind.upper()}")
    if kind == 'cwt':
        print(f"  CWT w0: {w0}, nf: {nf}")
    
    print(f"\nRunning polarization analysis...")
    print("="*80 + "\n")
    
    # Call the main polarization plotting function from polarisation_plot.py
    # This function will compute polarization and generate the full plot
    try:
        ppl.plot_polarization_event_noise(
            st,
            t_pick_P=t_pick_P,
            t_pick_S=t_pick_S,
            timing_P=timing_P,
            timing_S=timing_S,
            timing_noise=timing_noise,
            phase_P='P',
            phase_S='S',
            delta_P='',  # No uncertainty from catalog
            delta_S='',
            rotation=rotation,
            BAZ=None,  # No "true" BAZ for comparison
            BAZ_fixed=BAZ_fixed,
            inc_fixed=inc_fixed,
            kind=kind,
            fmin=fmin,
            fmax=fmax,
            winlen_sec=winlen_sec,
            overlap=overlap,
            tstart=tstart,
            tend=tend,
            vmin=vmin,
            vmax=vmax,
            log=True,
            fname=fname,
            path='.',
            dop_winlen=dop_winlen,
            dop_specwidth=dop_specwidth,
            nf=nf,
            w0=w0,
            alpha_inc=alpha_inc,
            alpha_elli=alpha_elli,
            alpha_azi=alpha_azi,
            f_band_density=f_band_density,
            zoom=zoom,
            differentiate=differentiate,
            detick_1Hz=detick_1Hz
        )
        
        print("\n" + "="*80)
        print("POLARIZATION ANALYSIS COMPLETE")
        print("="*80)
        print(f"Plots saved to: ./Plots/{fname}_joined.png")
        print(f"Data saved to: {fname}out.pkl")
        
        # Load the results from the saved pickle file
        import pickle
        try:
            with open(f'{fname}out.pkl', 'rb') as f:
                kde_dataframe_S = pickle.load(f)
            print("\nResults have been saved and can be loaded for further analysis.")
        except:
            print("\nNote: Results pickle file may not have been created.")
        
        print("\nTo extract BAZ and inclination estimates:")
        print("  - Check the plot in ./Plots/ directory")
        print("  - Red diamond marker shows estimated BAZ on density plot")
        print("  - Blue dot shows estimated P-wave vector on stereo plots")
        print("="*80 + "\n")
        
        # Note: The polarisation_plot function doesn't return values directly
        # BAZ and inclination are computed internally and shown in plots
        # To get numerical values, user would need to examine the plot or 
        # we'd need to modify polarisation_plot.py
        
        return True, fname, None
        
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR IN POLARIZATION ANALYSIS")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("="*80 + "\n")
        return None, None, None


def main():
    """
    Example usage of polarization analysis with interactive picker.
    """
    print("="*80)
    print("EXAMPLE: Polarization Analysis from Interactive Picks")
    print("="*80)
    print("\nThis script is designed to be imported and used with rayleighwave_picker_v3.py")
    print("\nExample usage:")
    print("-"*80)
    print("""
from polarization_picker_analysis import analyze_polarization_from_picks
from obspy import read, UTCDateTime

# Load your data
st = read('path/to/your/data/*.SAC')
event_time = UTCDateTime('2015-06-29T09:09:21')

# After running interactive picker to get picks:
picks = {
    'P': [720.5],   # seconds after event
    'S': [1300.2],  # seconds after event
}

# Run polarization analysis
result = analyze_polarization_from_picks(
    st=st,
    event_time=event_time,
    picks_dict=picks,
    t_window_P=[-10, 10],
    t_window_S=[-5, 20],
    fname='MyEvent_Polarization'
)

print(f"Analysis complete! Check ./Plots/ directory for results.")
    """)
    print("-"*80)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
