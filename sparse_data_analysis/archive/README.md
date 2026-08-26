# Archive Directory

This directory contains archived versions of scripts that have been replaced by newer, improved implementations.

## Archived Files

### sparse_data_location_analysis_v4.0.py
**Date Archived:** July 23, 2026  
**Lines:** 2541  
**Reason:** Replaced by modular two-script workflow

**Why Archived:**
The monolithic v4.0 script combined distance/timing and backazimuth analysis in a single file. This has been replaced by a more flexible two-script approach:

1. **sparse_data_distance_timing.py** - Distance and timing estimation
2. **backazimuth_analysis.py** - Backazimuth extraction from pickle results

**Benefits of New Approach:**
- ✅ Distance/timing analysis can run independently
- ✅ BAZ analysis can re-run on existing results without re-picking phases
- ✅ Cleaner code with clear separation of concerns
- ✅ Easier to maintain and test
- ✅ More flexible workflow

**Migration Guide:**
Instead of running the old monolithic script:
```bash
# OLD WAY (v4.0 - archived)
python sparse_data_location_analysis.py --par_file par_files/PAR_FILE_Peru.py
```

Use the new two-script workflow:
```bash
# NEW WAY (v4.1)
# Step 1: Distance/timing analysis
python sparse_data_distance_timing.py --par_file par_files/PAR_FILE_Peru.py

# Step 2: Backazimuth analysis (loads results from step 1)
python backazimuth_analysis.py \
  --pickle Results_*/Peru_comprehensive_results.pkl \
  --par_file par_files/PAR_FILE_Peru.py
```

**Results:**
The new workflow produces identical numerical results to v4.0, but with improved flexibility and maintainability.

**Reference:**
For complete details, see: `FullMethod_KnownLocation/docs/REFACTORING_ROADMAP.md`

---

## Notes

- Archived files are preserved for reference only
- Do not use archived files for new analysis
- If you need the old workflow for comparison, use these archived versions
- All new work should use the modular v4.1 scripts

---

**Last Updated:** July 23, 2026
