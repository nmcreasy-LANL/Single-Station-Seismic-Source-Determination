#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAR_FILE_bosedata.py - Parameter file for bosedata 2015 event analysis

This file contains all configuration parameters for analyzing the bosedata event.
To run analysis with this parameter file:
    python sparse_data_location_analysis.py --par_file par_files/PAR_FILE_bosedata.py

@author: nmcreasy
@date: Wed Jul 22 10:17:46 2026
"""

import os
import numpy as np

#═════════════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION - MODIFY THESE PARAMETERS FOR YOUR ANALYSIS
#═════════════════════════════════════════════════════════════════════════════════

#─────────────────────────────────────────────────────────────────────────────────
# A. FILE AND DATA CONFIGURATION
#─────────────────────────────────────────────────────────────────────────────────

# File paths - automatically constructed relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data', 'bosedata')

# Data files to process (must be in DATA_DIR)
# Expected format: 3-component SAC files (E, Z, N)
FILES = [
    'II.BFO.00.BHE.M.2015.178.153452.SAC',  # East component
    'II.BFO.00.BHZ.M.2015.178.153452.SAC',  # Vertical component
    'II.BFO.00.BHN.M.2015.178.153452.SAC'   # North component
]

#─────────────────────────────────────────────────────────────────────────────────
# B. EVENT PARAMETERS (FOR VALIDATION MODE)
#─────────────────────────────────────────────────────────────────────────────────

# EVENT_TIME: Known origin time for validation/testing
#   - Set to ISO8601 string if known (enables validation mode)
#     Format: "YYYY-MM-DDTHH:MM:SS" or "YYYY-MM-DDTHH:MM:SS.ffffff"
#   - Set to None for blind analysis (real-world scenario)
# 
# VALIDATION MODE (EVENT_TIME is set):
#   - Compares estimates to true values
#   - Useful for testing algorithm performance
#   - Times displayed relative to event origin
#
# BLIND MODE (EVENT_TIME is None):
#   - No prior knowledge of event location/time
#   - Times referenced to P-wave arrival (t=0 at P)
#   - Real-world operational scenario
#
# NOTE: This is stored as a string to avoid requiring obspy in all environments.
#       Scripts that need UTCDateTime will convert it automatically.

EVENT_TIME = "2015-06-27T15:34:02"  # Set to None for blind analysis
USE_KNOWN_ORIGIN = (EVENT_TIME is not None)  # Auto-computed validation flag

# True event parameters (for validation comparison only)
# Only used when EVENT_TIME is set to compare against estimates
SOURCE_DEPTH_KM = 28      # True source depth (estimated) in kilometers
DISTANCE_DEG = 28.1       # True epicentral (from SAC) distance in degrees

# True event geographic location (for validation comparison only)
# Set to None for blind analysis where true location is unknown
TRUE_EVENT_LAT = 28.83  # True event latitude - update if known (degrees)
TRUE_EVENT_LON = 34.62 # True event longitude - update if known (degrees)

# Time reference system (DO NOT MODIFY - set automatically)
# P_ARRIVAL_TIME: Set automatically from first P-wave pick
# This becomes the t=0 reference for all subsequent time calculations
P_ARRIVAL_TIME = None  # Auto-set during Stage 1 picking

# Output directory (DO NOT MODIFY - set automatically in main())
# Directory name format: Results_{STATION}_{YYYYMMDD_HHMMSS}
STATION_NAME = None  # Auto-set from data
EVENT_NAME = None    # Auto-set from waveform start time
OUTPUT_DIR = None    # Auto-set in main() function

#─────────────────────────────────────────────────────────────────────────────────
# C. TWO-STAGE INTERACTIVE PICKING CONFIGURATION
#─────────────────────────────────────────────────────────────────────────────────

# Enable interactive picking (recommended) or use legacy STA/LTA
USE_INTERACTIVE_PICKING = True  # True = interactive GUI, False = STA/LTA (not fully implemented)

# STAGE 1: Initial coarse location estimate using P and S only
# Purpose: Quick first estimate to narrow down search space
STAGE1_PHASES = ["P", "S"]              # Phase names to pick
STAGE1_DISTANCE_RANGE = [10, 50]       # Broad initial distance guess (degrees)
STAGE1_DEPTH_RANGE = [0, 200]          # Broad initial depth guess (km)

# STAGE 2: Refined location with additional phases
# Purpose: High-precision estimate using multiple phases
STAGE2_PHASES = ["P", "PP", "S", "PS"]  # Additional phases improve accuracy
                                         # Can add: PKP, SKS, PcP, ScS, Pdiff, etc.
STAGE2_DISTANCE_BUFFER = 10             # ± degrees from Stage 1 estimate
STAGE2_DEPTH_BUFFER = 50                # ± km from Stage 1 estimate

# Interactive picker filter band (applied to waveforms during picking)
# Bandpass filter helps visualize target phases
PICKER_PERIOD_HIGH = 50  # High period cutoff (seconds) - low frequency
PICKER_PERIOD_LOW = 25   # Low period cutoff (seconds) - high frequency
                          # Effective band: 0.02-0.04 Hz (25-50s period)

#─────────────────────────────────────────────────────────────────────────────────
# D. GRID SEARCH CONFIGURATION (BODY WAVE LOCATION)
#─────────────────────────────────────────────────────────────────────────────────

# Enable grid search for distance and depth estimation
ENABLE_GRID_SEARCH = True

# STAGE 1 GRID: Broad grid search after initial P&S picks
# Purpose: Coarse location estimate across full teleseismic range
# Search strategy: Wide coverage with coarse spacing
STAGE1_GRID_DEPTHS = np.arange(0, 750, 50)     # 0-700 km, 50 km spacing (15 points)
STAGE1_GRID_DISTANCES = np.arange(10,50, 3)  # 71-98°, 3° spacing (10 points)
# Total grid points per model: 15 × 10 = 150 points

# STAGE 2 GRID: Refined grid search centered on Stage 1 estimate
# Purpose: High-resolution refinement around Stage 1 result
# Search strategy: Narrow coverage with fine spacing
STAGE2_DISTANCE_BUFFER = 10      # ± degrees from Stage 1 estimate
STAGE2_DEPTH_BUFFER = 50         # ± km from Stage 1 estimate
STAGE2_DISTANCE_SPACING = 1      # Fine distance spacing (degrees)
STAGE2_DEPTH_SPACING = 5         # Fine depth spacing (km)
# Dynamic grid size example: ±10° × ±50 km = ~21 × 11 = 231 points per model

# Earth models to test (grid search runs for each model)
# Models: PREM (oceanic), IASP91 (global average), AK135 (global reference)
GRID_SEARCH_MODELS = ["prem", "iasp91", "ak135"]

# Body wave phases to use in grid search
# Note: P is always required as the reference phase
# Common phases: P, PP (surface reflection), S, PS, PKP, SKS, etc.
# See TauP documentation for available phase names
BODY_WAVE_PHASES = ['P', 'PP', 'S', 'PS']

# Pick uncertainties for each phase (seconds)
# Used to weight phase arrival observations in grid search
# Larger values = less confidence in pick timing
# These must correspond to the phases listed in BODY_WAVE_PHASES
PICK_UNCERTAINTIES = {
    'P': 10.0,    # P-wave uncertainty (±10s typical)
    'PP': 10.0,   # PP (surface reflection) uncertainty
    'S': 10.0,    # S-wave uncertainty
    'PS': 10.0    # PS (P-to-S conversion) uncertainty
}

# Legacy variables for backward compatibility (auto-generated from PICK_UNCERTAINTIES)
PICK_UNCERTAINTY_P = PICK_UNCERTAINTIES.get('P', 10.0)
PICK_UNCERTAINTY_PP = PICK_UNCERTAINTIES.get('PP', 10.0)
PICK_UNCERTAINTY_S = PICK_UNCERTAINTIES.get('S', 10.0)
PICK_UNCERTAINTY_PS = PICK_UNCERTAINTIES.get('PS', 10.0)

#─────────────────────────────────────────────────────────────────────────────────
# E. POLARIZATION ANALYSIS CONFIGURATION (BACKAZIMUTH EXTRACTION)
#─────────────────────────────────────────────────────────────────────────────────

# Enable automatic backazimuth extraction from polarization analysis
# After Stage 2, automatically analyzes P and S waveform polarization
ENABLE_POLARIZATION_ANALYSIS = True

# Polarization analysis parameters (passed to polarization_picker_analysis.py)
# These control the time-frequency analysis for backazimuth estimation
POLARIZATION_PARAMS = {
    # Time windows around picks (seconds)
    't_window_P': [-20, 100],        # [before, after] P pick for analysis window
    't_window_S': [-20, 100],        # [before, after] S pick for analysis window
    
    # Noise estimation
    'noise_duration': 120,           # Pre-P noise window duration (seconds)
    'noise_gap': 10,                 # Gap before P to avoid coda (seconds)
    
    # Frequency range for analysis
    'fmin': 1/50,                    # Minimum frequency (Hz) = 50s period
    'fmax': 1/10,                    # Maximum frequency (Hz) = 10s period
    'f_band_density': (1/50, 1/20),  # Frequency band for BAZ estimation (Hz)
    
    # Time-frequency decomposition method
    'kind': 'cwt',                   # 'cwt' (Continuous Wavelet) or 'spec' (Spectrogram)
    'w0': 8,                         # CWT parameter (wavelet width)
    'nf': 100,                       # Number of frequencies
    'winlen_sec': 60.,               # Window length (seconds)
    'overlap': 0.5,                  # Window overlap fraction (0-1)
    
    # Polarization analysis parameters
    'dop_winlen': 10,                # Degree of polarization window
    'dop_specwidth': 1.1,            # Spectral width parameter
    
    # Plotting parameters
    'vmin': -180,                    # dB minimum for plots
    'vmax': -140,                    # dB maximum for plots
    'differentiate': False,          # True if using displacement data (vs velocity)
    'zoom': False                    # Zoom into P/S windows on plots
}

#─────────────────────────────────────────────────────────────────────────────────
# F. RAYLEIGH WAVE CONFIGURATION (GROUP VELOCITY ANALYSIS)
#─────────────────────────────────────────────────────────────────────────────────

# Frequency bands for Rayleigh wave analysis
# Multiple overlapping bands improve distance/timing estimates

# Automatic band generation (recommended)
USE_AUTO_BANDS = True       # True = auto-generate, False = manual specification
NUM_BANDS = 5               # Number of overlapping frequency bands
PERIOD_MIN = 80            # Minimum period (seconds)
PERIOD_MAX = 160            # Maximum period (seconds)
OVERLAP_PERCENT = 85        # Band overlap percentage (high overlap = more data)
                            # Example: 90% overlap means adjacent bands share 90% of frequency content

# Manual band specification (only used if USE_AUTO_BANDS = False)
# Uncomment and modify if you need custom bands:
# BAND_LOW = np.array([120, 135, 150, 165, 180, 195])   # Low period boundaries (s)
# BAND_HIGH = np.array([140, 155, 170, 185, 200, 215])  # High period boundaries (s)
