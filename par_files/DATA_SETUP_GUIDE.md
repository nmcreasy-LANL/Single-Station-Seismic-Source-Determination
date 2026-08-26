# Data Setup Guide for Sparse Data Analysis

## Overview

This guide explains how to set up seismic data files for use with the sparse data location analysis tools.

## Problem: Missing Data Files Error

If you encounter an error like:
```
ERROR: The following data files are missing:
  - /path/to/par_files/data/Peru/II.BFO.00.BHE.M.2015.180.082243.SAC
  - /path/to/par_files/data/Peru/II.BFO.00.BHZ.M.2015.180.082243.SAC
  - /path/to/par_files/data/Peru/II.BFO.00.BHN.M.2015.180.082243.SAC

Please ensure data files are in: /path/to/par_files/data/Peru
ERROR: Data file validation failed. Exiting.
ERROR: Setup failed. Exiting.
```

This means the required SAC data files are not present in the expected directory.

## Solution: Setting Up Your Data Files

### Option 1: Place Data Files in the Expected Directory (Recommended)

1. **Locate the data directory** specified in your parameter file:
   - For `PAR_FILE_Peru.py`: `FullMethod_KnownLocation/par_files/data/Peru/`
   - The directory structure is now created for you with a README

2. **Obtain the SAC data files**:
   - Download from IRIS DMC: https://ds.iris.edu/wilber3/
   - Download from GEOFON: https://geofon.gfz-potsdam.de/
   - Use your local seismic data archive

3. **Copy the SAC files** to the data directory:
   ```bash
   cp /path/to/your/sac/files/*.SAC FullMethod_KnownLocation/par_files/data/Peru/
   ```

4. **Verify files are in place**:
   ```bash
   ls -la FullMethod_KnownLocation/par_files/data/Peru/*.SAC
   ```

### Option 2: Modify the Parameter File to Point to Your Data

If your SAC files are located elsewhere and you don't want to move them:

1. **Open your parameter file** (e.g., `PAR_FILE_Peru.py`)

2. **Modify the `DATA_DIR` variable** to point to your data location:
   ```python
   # Original (relative path):
   DATA_DIR = os.path.join(SCRIPT_DIR, 'data', 'Peru')
   
   # Modified (absolute path to your data):
   DATA_DIR = '/path/to/your/seismic/data/Peru2015'
   ```

3. **Save the file** and run the analysis again

## Required Data Format

The analysis requires **three-component SAC files** with the following:

### File Requirements:
- **Format**: SAC (Seismic Analysis Code) binary format
- **Components**: Three files (East, North, Vertical OR E, N, Z)
- **Headers**: Must include proper SAC headers with:
  - Station information (network, station, location, channel)
  - Event information (origin time, location)
  - Distance (GCARC header)
  - Backazimuth (BAZ header) - optional but recommended
  - Component orientation

### Example File Names:
```
II.BFO.00.BHE.M.2015.180.082243.SAC  # East component
II.BFO.00.BHN.M.2015.180.082243.SAC  # North component  
II.BFO.00.BHZ.M.2015.180.082243.SAC  # Vertical component
```

**Format**: `{network}.{station}.{location}.{channel}.{quality}.{year}.{julday}.{time}.SAC`

## Creating Your Own Parameter File

To analyze a different event:

1. **Copy an existing parameter file**:
   ```bash
   cp PAR_FILE_Peru.py PAR_FILE_MyEvent.py
   ```

2. **Create a data directory** for your event:
   ```bash
   mkdir -p data/MyEvent
   ```

3. **Edit the new parameter file**:
   - Update `DATA_DIR` to point to your new data directory
   - Update `FILES` list with your SAC file names
   - Update event parameters (EVENT_TIME, SOURCE_DEPTH_KM, DISTANCE_DEG)
   - Adjust analysis parameters as needed

4. **Place your SAC files** in the data directory

5. **Run the analysis**:
   ```bash
   cd FullMethod_KnownLocation/sparse_data_analysis
   python sparse_data_location_analysis.py --par_file ../par_files/PAR_FILE_MyEvent.py
   ```

## Directory Structure

The expected directory structure after setup:

```
FullMethod_KnownLocation/
├── par_files/
│   ├── PAR_FILE_Peru.py           # Parameter file
│   ├── DATA_SETUP_GUIDE.md        # This file
│   └── data/                      # Data directory
│       └── Peru/                  # Event-specific directory
│           ├── README.md          # Data directory documentation
│           ├── .gitkeep           # Keeps directory in git
│           ├── *.SAC              # Your SAC files (not tracked by git)
│           └── ...
└── sparse_data_analysis/          # Analysis scripts
    └── sparse_data_location_analysis.py
```

## Notes

- **SAC files are not tracked by git** (they're in .gitignore) because they are large binary files
- The directory structure (with README and .gitkeep) IS tracked by git
- Each user must obtain and place their own SAC data files
- Data files can be hundreds of MB, so storing them in git is not recommended

## Troubleshooting

### "File not found" error
- Check that the file names in `FILES` list match exactly (case-sensitive)
- Verify files are in the correct directory
- Check file permissions (should be readable)

### "No SAC header" error
- Files must be in SAC format (not miniSEED or other formats)
- Convert to SAC using ObsPy if needed:
  ```python
  from obspy import read
  st = read('myfile.mseed')
  st.write('myfile.SAC', format='SAC')
  ```

### "Missing SAC header fields" error
- Ensure SAC headers are properly set (GCARC, BAZ, event time, etc.)
- Use `obspy.io.sac` to add/modify headers if needed

## Additional Resources

- ObsPy documentation: https://docs.obspy.org/
- SAC format specification: https://ds.iris.edu/files/sac-manual/
- IRIS DMC data access: https://ds.iris.edu/ds/nodes/dmc/
- Parameter file documentation: See comments in `PAR_FILE_Peru.py`
