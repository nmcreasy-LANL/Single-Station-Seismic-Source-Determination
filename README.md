# Seismic Event Location Analysis - Quick Start Guide

**Single-station seismic event location using sparse data analysis**

**Author:** nmcreasy  
**Institution:** Los Alamos National Laboratory  
**Last Updated:** 2026-07-24

---

## Overview

This package provides tools for locating seismic events using data from a single station by analyzing:
- **Distance**: Body wave travel times + Rayleigh wave dispersion
- **Backazimuth**: Polarization analysis + Rayleigh wave particle motion
- **Uncertainty**: Monte Carlo sampling and PDF combination

The workflow is split into two main scripts:
1. **`sparse_data_distance_timing.py`** - Distance and timing estimation
2. **`backazimuth_analysis.py`** - Backazimuth extraction from results

---

## Quick Start

### Installation

Create conda environment with required packages:

```bash
conda create -n seismic_analysis python=3.9
conda activate seismic_analysis
conda install -c conda-forge obspy numpy matplotlib scipy pyproj pandas
```

### Working Directory

**All commands should be run from the Single_Station_Seismic_Source_Determination directory:**

```bash
cd Single_Station_Seismic_Source_Determination/
```

### Basic Workflow

```bash
# Step 1: Distance/timing analysis
python sparse_data_analysis/sparse_data_distance_timing.py --par_file par_files/PAR_FILE_Peru.py

# Note the Results directory name created (e.g., Results_BFO_20150629_082243/)

# Step 2: Backazimuth analysis (REQUIRES --pickle from Step 1)
python sparse_data_analysis/backazimuth_analysis.py \
    --pickle Results_YYYYMMDD_HHMMSS/YYYYMMDD_HHMMSS_comprehensive_results.pkl \
    --par_file par_files/PAR_FILE_Peru.py

# Or use wildcard to find the file automatically:
python sparse_data_analysis/backazimuth_analysis.py \
    --pickle Results_*/20*_comprehensive_results.pkl \
    --par_file par_files/PAR_FILE_Peru.py

# Step 3: Monte Carlo location sampling
python sparse_data_analysis/monte_carlo_sampler.py --results Results_YYYYMMDD_HHMMSS/ --samples 300

# Step 4: Create location map (requires pygmt environment)
conda activate pygmt_env
python sparse_data_analysis/plot_location_map_pygmt.py --results Results_YYYYMMDD_HHMMSS/ --mode density
```

---

## Configuration

All analysis parameters are specified in **parameter files** located in `par_files/`:

### Creating a Parameter File

1. Copy an existing parameter file:
   ```bash
   cp par_files/PAR_FILE_Peru.py par_files/PAR_FILE_MyEvent.py
   ```

2. Edit the key parameters:
   ```python
   # Event identification
   EVENT_NAME = "MyEvent_2026"
   
   # SAC file paths
   FILES = ['data/MyEvent/station.BHE.SAC',
            'data/MyEvent/station.BHN.SAC', 
            'data/MyEvent/station.BHZ.SAC']
   
   # Event timing (if known, or set to None for blind analysis)
   EVENT_TIME = UTCDateTime("2026-01-15T12:34:56")
   
   # Search parameters
   STAGE1_DISTANCE_RANGE = (30, 180)  # degrees
   STAGE1_DEPTH_RANGE = (0, 700)       # km
   ```

3. Run with your parameter file:
   ```bash
   python sparse_data_analysis/sparse_data_distance_timing.py --par_file par_files/PAR_FILE_MyEvent.py
   ```

See `par_files/DATA_SETUP_GUIDE.md` for complete parameter documentation.

---

## What Each Script Does

### 1. `sparse_data_distance_timing.py`

**Purpose:** Estimates event distance, depth, and origin time

**Process:**
- **Stage 1**: Pick P & S phases → Initial coarse distance/depth estimate
- **Stage 2**: Pick additional phases (PP, PS) → Refined estimate
- **Optional**: Rayleigh wave analysis → Independent distance constraint
- **Optional**: Stockwell transform → Advanced orbit detection

**Output:**
- Distance PDF pickle file (for Monte Carlo sampling)
- Comprehensive plots of all PDFs
- Analysis log with all results

**Runtime:** ~10-20 minutes (interactive picking + computation)

### 2. `backazimuth_analysis.py`

**Purpose:** Extracts backazimuth from polarization analysis

**Process:**
- Loads results from distance/timing analysis
- Runs polarization analysis on P-wave and/or S-wave windows
- Extracts backazimuth PDF with uncertainties
- Optionally combines Rayleigh wave backazimuth

**Output:**
- Backazimuth PDF pickle file (for Monte Carlo sampling)
- BAZ probability plots
- Polarization diagnostics

**Runtime:** ~5-10 minutes

### 3. `monte_carlo_sampler.py`

**Purpose:** Converts distance & BAZ PDFs to geographic locations

**Process:**
- Samples from distance PDF (weighted random)
- Samples from BAZ PDF (weighted random)
- Converts (distance, BAZ) → (latitude, longitude) via geodesics
- Creates probability distributions of event location

**Output:**
- `mc_samples.csv` with lat/lon samples
- Verification plots showing sampling quality
- Lat/lon PDF plots

**Runtime:** ~1-2 minutes

### 4. `plot_location_map_pygmt.py`

**Purpose:** Creates publication-quality location maps

**Process:**
- Loads Monte Carlo samples
- Creates density map or contour plot
- Shows station, estimated location, and uncertainty cloud

**Output:**
- High-resolution PNG map (300 DPI)

**Runtime:** ~2-5 minutes

**Note:** Requires separate `pygmt_env` conda environment:
```bash
conda create -n pygmt_env python=3.9
conda activate pygmt_env
conda install -c conda-forge pygmt numpy pandas pyproj obspy
```

---

## Output Structure

After running the full workflow:

```
Results_YYYYMMDD_HHMMSS/
├── YYYYMMDD_HHMMSS_distance_pdf.pkl          # Distance PDF (Step 1)
├── YYYYMMDD_HHMMSS_Polarizationout.pkl       # BAZ PDF (Step 2)
├── mc_samples.csv                             # Lat/lon samples (Step 3)
├── location_map_density.png                   # Final map (Step 4)
├── bodywave_distance_pdfstage_1.png           # Stage 1 distance
├── bodywave_distance_pdfstage_2.png           # Stage 2 distance
├── combined_distance_pdf_stage2.png           # Combined distance
├── YYYYMMDD_HHMMSS_BAZ_P.png                  # P-wave BAZ
├── mc_distribution_verification.png           # MC verification
└── analysis_log_YYYYMMDD_HHMMSS.txt          # Complete log
```

---

## Additional Tools

### Magnitude Calculation

```bash
python sparse_data_analysis/calculate_magnitude.py \
    --distance-pdf Results_XXX/XXX_distance_pdf.pkl \
    --sac-files data/event/*.SAC \
    --output Results_XXX/
```

### Multi-Station Combination

```bash
python sparse_data_analysis/combine_multistation_pdfs.py \
    --station Results_Station1/mc_latlon_pdf.pkl \
    --station Results_Station2/mc_latlon_pdf.pkl \
    --station Results_Station3/mc_latlon_pdf.pkl \
    --output combined_results/
```

---

## Troubleshooting

### "Could not find parameter file"
```bash
# Use absolute path or run from FullMethod_KnownLocation/ directory with proper path
python sparse_data_analysis/sparse_data_distance_timing.py --par_file par_files/PAR_FILE.py
```

### "ModuleNotFoundError: No module named 'obspy'"
```bash
# Make sure correct conda environment is activated
conda activate seismic_analysis
```

### "Could not find distance PDF pickle file"
```bash
# Make sure Step 1 (sparse_data_distance_timing.py) completed successfully
# Check Results directory for *_distance_pdf.pkl file
```

### "No backazimuth PDF found"
```bash
# Make sure Step 2 (backazimuth_analysis.py) completed successfully
# Check Results directory for *_Polarizationout.pkl file
```

### "backazimuth_analysis.py: error: the following arguments are required: --pickle"

**This is the correct behavior!** The `--pickle` argument is required.

**Solution:**
```bash
# Step 2 requires the pickle file created by Step 1
# Find the pickle file in your Results directory:
ls Results_*/20*_comprehensive_results.pkl

# Then run with the correct path:
python sparse_data_analysis/backazimuth_analysis.py \
    --pickle Results_BFO_20150629_082243/20150629_082243_comprehensive_results.pkl \
    --par_file par_files/PAR_FILE_Peru.py
```

**Why?** Version 4.1 split the analysis into two scripts:
- Step 1 (`sparse_data_distance_timing.py`) generates the comprehensive_results.pkl
- Step 2 (`backazimuth_analysis.py`) loads that pickle and adds BAZ analysis

This allows you to rerun BAZ analysis with different parameters without redoing distance/timing.

---

## Recent Fixes

### Backazimuth Pickle Generation (Fixed 2026-08-25)

**Issue:** Running `backazimuth_analysis.py` failed with:
```
NameError: name 'ENABLE_POLARIZATION_ANALYSIS' is not defined
```

**Fix Applied:** Modified `sparse_data_analysis/backazimuth.py` to handle missing global variable gracefully.

**Status:** ✅ Fixed - `backazimuth_analysis.py` now works correctly in standalone mode.

---

## Documentation

For detailed documentation see:
- **`DOCUMENTATION.md`** - Comprehensive reference with theory and advanced topics
- **`par_files/DATA_SETUP_GUIDE.md`** - Parameter file setup guide
- **`REFACTORING_ROADMAP.md`** - Code architecture and development notes
- **`archive/`** - Legacy specialized documentation files

---

## Typical Analysis Time

- **Distance/Timing Analysis** (Step 1): 10-20 minutes
- **Backazimuth Analysis** (Step 2): 5-10 minutes
- **Monte Carlo Sampling** (Step 3): 1-2 minutes
- **Map Visualization** (Step 4): 2-5 minutes

**Total:** ~20-40 minutes per event

---

## Support

For questions or issues, contact: nmcreasy

**Version:** 4.1  
**Compatible with:** Python 3.8+, ObsPy 1.2+
