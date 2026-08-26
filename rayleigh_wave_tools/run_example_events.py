#!/usr/bin/env python3
"""Batch example driver for Rayleigh BAZ and orbit detection.

The script defines an explicit list of event runs, processes each event in its
own output directory, and writes cross-event summary figures and tables for
back-azimuth and epicentral distance.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Mapping

import matplotlib
matplotlib.use("Agg")
import numpy as np
from obspy import Stream, UTCDateTime, read

from .rayleigh_backazimuth import estimate_rayleigh_baz_posterior
from .rayleigh_filter import (
    add_particleman_nip_energy_masks,
    simple_stockwell_filter_rayleigh,
)
from .rayleigh_orbit_detector import detect_rayleigh_orbits_from_stockwell
from .plotting import (
    plot_batch_baz_linear,
    plot_batch_baz_radial,
    plot_batch_distance_linear,
    plot_rayleigh_filter_diagnostics,
    plot_truth_vs_map_error_summary,
)

SAC_UNSET = -12345.0

EVENT_RUNS = [
    {
        "name": "2015_180_082243",
        "files": [
            "II.BFO.00.BHE.M.2015.180.082243.SAC",
            "II.BFO.00.BHZ.M.2015.180.082243.SAC",
            "II.BFO.00.BHN.M.2015.180.082243.SAC",
        ],
    },
    {
        "name": "2015_180_091243",
        "files": [
            "II.BFO.00.BHE.M.2015.180.091243.SAC",
            "II.BFO.00.BHZ.M.2015.180.091243.SAC",
            "II.BFO.00.BHN.M.2015.180.091243.SAC",
        ],
    },
    {
        "name": "2015_178_153452",
        "files": [
            "II.BFO.00.BHE.M.2015.178.153452.SAC",
            "II.BFO.00.BHZ.M.2015.178.153452.SAC",
            "II.BFO.00.BHN.M.2015.178.153452.SAC",
        ],
    },
]


def _is_set(value):
    if value is None:
        return False
    try:
        value = float(value)
    except Exception:
        return False
    return np.isfinite(value) and abs(value - SAC_UNSET) > 1e-3


def _sac_reference_time(sac):
    return UTCDateTime(
        year=int(sac.nzyear),
        julday=int(sac.nzjday),
        hour=int(sac.nzhour),
        minute=int(sac.nzmin),
        second=int(sac.nzsec),
        microsecond=int(sac.nzmsec) * 1000,
    )


def timing_from_sac_origin(trace):
    if not hasattr(trace.stats, "sac"):
        raise ValueError("Trace has no SAC header. Cannot derive origin time.")
    sac = trace.stats.sac
    if not _is_set(getattr(sac, "o", None)):
        raise ValueError("SAC header does not contain a valid origin offset 'o'.")
    b = float(sac.b) if _is_set(getattr(sac, "b", None)) else 0.0
    o = float(sac.o)
    ref_time = _sac_reference_time(sac)
    event_time = ref_time + o
    event_offset_sec = float(trace.stats.starttime - event_time)
    return {
        "sac_reference_time": ref_time,
        "trace_start_time": trace.stats.starttime,
        "event_time": event_time,
        "b": b,
        "o": o,
        "event_offset_sec": event_offset_sec,
    }


def get_sac_distance_deg(trace, fallback_distance_deg):
    sac = getattr(trace.stats, "sac", None)
    if sac is not None and _is_set(getattr(sac, "gcarc", None)):
        return float(sac.gcarc)
    return float(fallback_distance_deg)


def get_sac_baz_deg(trace):
    sac = getattr(trace.stats, "sac", None)
    if sac is not None and _is_set(getattr(sac, "baz", None)):
        return float(sac.baz)
    return None


def stockwell_time_axis_seconds(result_dict, reference_trace):
    S_R = result_dict.get("S_R")
    if S_R is not None and np.ndim(S_R) == 2:
        n_times = int(S_R.shape[1])
    elif "rayleigh_envelope" in result_dict:
        n_times = len(result_dict["rayleigh_envelope"])
    else:
        raise ValueError("Cannot determine Stockwell time dimension.")
    trace_duration_sec = (reference_trace.stats.npts - 1) * reference_trace.stats.delta
    for key in ("time_array", "time_seconds", "times_sec", "times", "t"):
        if key in result_dict:
            arr = np.asarray(result_dict[key], dtype=float).squeeze()
            if arr.ndim == 1 and arr.size == n_times:
                arr0 = arr - arr[0]
                span = float(arr0[-1] - arr0[0])
                if np.isfinite(span) and span > 0:
                    if abs(span - trace_duration_sec) / trace_duration_sec <= 0.05:
                        return arr0
                    if abs(span * 60.0 - trace_duration_sec) / trace_duration_sec <= 0.05:
                        return arr0 * 60.0
    reported_dt = None
    for key in ("output_dt", "target_dt", "dt_out", "dt"):
        if key in result_dict:
            try:
                reported_dt = float(result_dict[key])
            except Exception:
                pass
            if reported_dt is not None and np.isfinite(reported_dt) and reported_dt > 0:
                break
    if reported_dt is None:
        reported_dt = float(reference_trace.stats.delta)
    reported_duration = (n_times - 1) * reported_dt
    if abs(reported_duration - trace_duration_sec) / trace_duration_sec > 0.05:
        reported_dt = trace_duration_sec / (n_times - 1)
    return np.arange(n_times, dtype=float) * reported_dt


def print_time_sanity_checks(time_array, event_offset_sec, distance_deg):
    time_min = (time_array + event_offset_sec) / 60.0
    print("\nTIME WINDOW CHECK")
    print(f"  Detector window: {time_min[0]:.2f} to {time_min[-1]:.2f} min after origin")
    print(f"  Output duration: {(time_array[-1] - time_array[0]) / 60.0:.2f} min")
    R_earth_km = 6371.0
    U100_km_s = 3.77
    for orbit, path_deg in {"R1": distance_deg, "R2": 360.0 - distance_deg, "R3": 360.0 + distance_deg}.items():
        arrival_min = (path_deg * np.pi / 180.0 * R_earth_km / U100_km_s) / 60.0
        marker = "inside" if time_min[0] <= arrival_min <= time_min[-1] else "outside"
        print(f"    {orbit}: {arrival_min:.1f} min ({marker} window)")


def print_detection_summary(detected_orbits, catalog_distance, output_dir):
    metadata = detected_orbits.get("metadata", {})
    n_detected = metadata.get("n_orbits_detected", 0)
    print("\n" + "=" * 80)
    print("DETECTION RESULTS")
    print("=" * 80)
    print(f"\nDetected {n_detected}/3 orbits")
    if metadata.get("method") == "joint" and "shared_distance_deg" in metadata:
        shared_dist = metadata["shared_distance_deg"]
        error = abs(shared_dist - catalog_distance)
        print(f"\nJoint fit distance: {shared_dist:.1f} deg")
        print(f"   SAC/catalog distance: {catalog_distance:.1f} deg")
        print(f"   Error: {error:.1f} deg ({error / catalog_distance * 100:.1f}%)")
    for orbit in ["R1", "R2", "R3"]:
        if orbit in detected_orbits:
            data = detected_orbits[orbit]
            comp = data.get("confidence_components", {})
            print(f"\n{orbit} DETECTED:")
            print(f"   Arrival: {data['dispersion']['peak_time']:.1f} min")
            print(f"   Distance: {data['distance_deg']:.1f} deg")
            print(f"   Confidence: {data['confidence']:.3f}")
            print(f"   Shape correlation: {comp.get('shape_correlation', np.nan):.3f}")
            print(f"   SNR: {comp.get('snr', np.nan):.1f}")
            print(f"   RMS: {comp.get('rms_residual_sec', np.nan):.1f} s")
        else:
            print(f"\n{orbit} NOT DETECTED")
    print(f"\nDiagnostics: {output_dir}")
    print("=" * 80 + "\n")


def circular_difference_deg(estimate_deg, truth_deg):
    if estimate_deg is None or truth_deg is None:
        return np.nan
    return ((float(estimate_deg) - float(truth_deg) + 180.0) % 360.0) - 180.0


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, UTCDateTime):
        return value.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return value


def _format_utc_label(dt):
    if isinstance(dt, UTCDateTime):
        return dt.strftime("%Y-%m-%dT%H_%M_%S.%fZ")
    return str(dt).replace(":", "_")


def _event_label(spec, trZ, timing):
    if spec.get("label"):
        return spec["label"]
    stats = trZ.stats
    net = getattr(stats, "network", "")
    sta = getattr(stats, "station", "")
    loc = getattr(stats, "location", "") or "--"
    chan = getattr(stats, "channel", "BH?")
    band = chan[:2] if len(chan) >= 2 else chan
    return f"{net}.{sta}.{loc}.{band}.{_format_utc_label(timing['event_time'])}"


def _coerce_1d_array(value):
    try:
        arr = np.asarray(value, dtype=float).squeeze()
    except Exception:
        return None
    if arr.ndim == 1 and arr.size > 1 and np.all(np.isfinite(arr)):
        return arr
    return None


def _find_1d_array(obj, keys):
    if not isinstance(obj, Mapping):
        return None
    for key in keys:
        if key in obj:
            arr = _coerce_1d_array(obj[key])
            if arr is not None:
                return arr
    for key in ("posterior", "baz_posterior", "distance_posterior", "metadata", 
                "dispersion", "R1", "R2", "R3"):
        child = obj.get(key)
        if isinstance(child, Mapping):
            arr = _find_1d_array(child, keys)
            if arr is not None:
                return arr
    return None


def _normalize_pdf_for_plot(pdf):
    pdf = np.asarray(pdf, dtype=float).copy()
    pdf[~np.isfinite(pdf)] = 0.0
    pdf[pdf < 0.0] = 0.0
    max_pdf = float(np.max(pdf)) if pdf.size else 0.0
    if max_pdf <= 0.0:
        return np.zeros_like(pdf)
    return pdf / max_pdf


def angular_posterior_curve(posterior, map_baz_deg, fallback_std_deg=10.0):
    angle_keys = (
        "baz_grid_deg",
        "angle_grid_deg",
        "angles_deg",
        "theta_deg",
        "azimuth_grid_deg",
        "grid_deg",
        "baz_grid",
        "theta",
    )
    pdf_keys = (
        "posterior_pdf",
        "baz_posterior_pdf",
        "baz_pdf",
        "probability",
        "pdf",
        "posterior",
        "p_baz",
    )
    grid = _find_1d_array(posterior, angle_keys)
    pdf = _find_1d_array(posterior, pdf_keys)
    if grid is not None and pdf is not None and grid.size == pdf.size:
        if np.nanmax(np.abs(grid)) <= 2.0 * np.pi + 0.1:
            grid = np.rad2deg(grid)
        grid = np.mod(grid, 360.0)
        order = np.argsort(grid)
        return grid[order], _normalize_pdf_for_plot(pdf[order])

    sigma = fallback_std_deg
    try:
        sigma = float(posterior.get("posterior_std_deg", sigma))
    except Exception:
        pass
    sigma = max(float(sigma), 1.0)
    center = float(map_baz_deg) % 360.0
    grid = np.linspace(0.0, 360.0, 721)
    delta = ((grid - center + 180.0) % 360.0) - 180.0
    pdf = np.exp(-0.5 * (delta / sigma) ** 2)
    return grid, _normalize_pdf_for_plot(pdf)


def _metadata_distance_map(detected_orbits):
    metadata = detected_orbits.get("metadata", {}) if isinstance(detected_orbits, Mapping) else {}
    for key in ("map_distance_deg", "shared_distance_deg", "best_distance_deg", "distance_deg"):
        if key in metadata:
            try:
                return float(metadata[key])
            except Exception:
                pass
    distances = []
    weights = []
    for orbit in ("R1", "R2", "R3"):
        if orbit in detected_orbits and "distance_deg" in detected_orbits[orbit]:
            distances.append(float(detected_orbits[orbit]["distance_deg"]))
            weights.append(float(detected_orbits[orbit].get("confidence", 1.0)))
    if distances:
        weights = np.asarray(weights, dtype=float)
        if np.sum(weights) <= 0.0:
            weights[:] = 1.0
        return float(np.average(np.asarray(distances, dtype=float), weights=weights))
    return np.nan


def distance_posterior_curve(detected_orbits, catalog_distance_deg, map_distance_deg):
    grid_keys = (
        "distance_grid_deg",
        "dist_grid_deg",
        "distance_degrees",
        "distance_grid",
        "grid_deg",
    )
    pdf_keys = (
        "distance_posterior_pdf",
        "distance_pdf",
        "posterior_pdf",
        "pdf",
        "posterior",
        "probability",
    )
    metadata = detected_orbits.get("metadata", {}) if isinstance(detected_orbits, Mapping) else {}
    grid = _find_1d_array(metadata, grid_keys)
    pdf = _find_1d_array(metadata, pdf_keys)
    if grid is None or pdf is None:
        grid = _find_1d_array(detected_orbits, grid_keys)
        pdf = _find_1d_array(detected_orbits, pdf_keys)
    if grid is not None and pdf is not None and grid.size == pdf.size:
        order = np.argsort(grid)
        return grid[order], _normalize_pdf_for_plot(pdf[order])

    center = float(map_distance_deg) if np.isfinite(map_distance_deg) else float(catalog_distance_deg)
    orbit_distances = []
    orbit_weights = []
    for orbit in ("R1", "R2", "R3"):
        if orbit in detected_orbits and "distance_deg" in detected_orbits[orbit]:
            orbit_distances.append(float(detected_orbits[orbit]["distance_deg"]))
            orbit_weights.append(float(detected_orbits[orbit].get("confidence", 1.0)))
    if len(orbit_distances) > 1:
        arr = np.asarray(orbit_distances, dtype=float)
        weights = np.asarray(orbit_weights, dtype=float)
        if np.sum(weights) <= 0.0:
            weights[:] = 1.0
        mean = float(np.average(arr, weights=weights))
        variance = float(np.average((arr - mean) ** 2, weights=weights))
        sigma = max(np.sqrt(variance), 2.0)
    else:
        sigma = 5.0
    try:
        sigma = float(metadata.get("distance_std_deg", sigma))
    except Exception:
        pass
    sigma = max(float(sigma), 1.0)
    lo = max(0.0, min(float(catalog_distance_deg), center) - 5.0 * sigma - 10.0)
    hi = min(180.0, max(float(catalog_distance_deg), center) + 5.0 * sigma + 10.0)
    if hi <= lo:
        lo, hi = max(0.0, center - 30.0), min(180.0, center + 30.0)
    grid = np.linspace(lo, hi, 721)
    pdf = np.exp(-0.5 * ((grid - center) / sigma) ** 2)
    return grid, _normalize_pdf_for_plot(pdf)


def write_summary_csv(rows, output_path):
    fieldnames = [
        "event_name",
        "event_label",
        "output_dir",
        "files",
        "catalog_baz_deg",
        "map_baz_deg",
        "baz_error_deg",
        "baz_posterior_std_deg",
        "catalog_distance_deg",
        "map_distance_deg",
        "distance_error_deg",
        "n_orbits_detected",
        "R1_distance_deg",
        "R1_confidence",
        "R2_distance_deg",
        "R2_confidence",
        "R3_distance_deg",
        "R3_confidence",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary_json(rows, output_path):
    compact_rows = []
    drop = {"baz_grid_deg", "baz_pdf", "distance_grid_deg", "distance_pdf"}
    for row in rows:
        compact_rows.append({k: v for k, v in row.items() if k not in drop})
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(compact_rows), f, indent=2)


def build_summary_row(spec, output_dir, trZ, timing, posterior, detected_orbits, catalog_distance, header_baz):
    event_label = _event_label(spec, trZ, timing)
    map_baz = float(posterior["best_estimate_deg"])
    baz_std = posterior.get("posterior_std_deg", np.nan)
    try:
        baz_std = float(baz_std)
    except Exception:
        baz_std = np.nan
    baz_grid, baz_pdf = angular_posterior_curve(posterior, map_baz, fallback_std_deg=10.0)

    map_distance = _metadata_distance_map(detected_orbits)
    distance_grid, distance_pdf = distance_posterior_curve(detected_orbits, catalog_distance, map_distance)
    metadata = detected_orbits.get("metadata", {})

    row = {
        "event_name": spec["name"],
        "event_label": event_label,
        "output_dir": str(output_dir),
        "files": ";".join(spec["files"]),
        "catalog_baz_deg": float(header_baz) if header_baz is not None else None,
        "map_baz_deg": map_baz,
        "baz_error_deg": circular_difference_deg(map_baz, header_baz) if header_baz is not None else np.nan,
        "baz_posterior_std_deg": baz_std,
        "catalog_distance_deg": float(catalog_distance),
        "map_distance_deg": map_distance,
        "distance_error_deg": float(map_distance - catalog_distance) if np.isfinite(map_distance) else np.nan,
        "n_orbits_detected": int(metadata.get("n_orbits_detected", 0)),
        "baz_grid_deg": baz_grid,
        "baz_pdf": baz_pdf,
        "distance_grid_deg": distance_grid,
        "distance_pdf": distance_pdf,
    }
    for orbit in ("R1", "R2", "R3"):
        orbit_data = detected_orbits.get(orbit, {})
        row[f"{orbit}_distance_deg"] = orbit_data.get("distance_deg", "")
        row[f"{orbit}_confidence"] = orbit_data.get("confidence", "")
    return row


def run_rayleigh_event(spec, data_dir="./data/combined", output_dir="./orbit_detection_output"):
    files = spec["files"]
    st = Stream()
    for filename in files:
        st += read(os.path.join(data_dir, filename))
    trZ = st.select(channel="*Z")[0]
    trN = st.select(channel="*N")[0]
    trE = st.select(channel="*E")[0]
    dataN, dataE, dataZ = trN.data, trE.data, trZ.data
    fs = trZ.stats.sampling_rate

    timing = timing_from_sac_origin(trZ)
    catalog_distance = get_sac_distance_deg(trZ, fallback_distance_deg=97.3)
    header_baz = get_sac_baz_deg(trZ)

    print("\n" + "=" * 80)
    print(f"RAYLEIGH ORBIT DETECTION - {spec['name']}")
    print("=" * 80)
    print(f"\nLoaded data: {len(dataZ)} samples @ {fs} Hz")
    print(f"  Duration from samples: {(len(dataZ) - 1) / fs / 60.0:.2f} minutes")
    print(f"  Event origin time: {timing['event_time']}")
    print(f"  Detector offset: {timing['event_offset_sec']:.6f} s")
    print(f"  SAC gcarc distance: {catalog_distance:.3f} deg")
    if header_baz is not None:
        print(f"  SAC back-azimuth: {header_baz:.3f} deg")

    print("\n" + "=" * 80)
    print("ESTIMATING BACK-AZIMUTH")
    print("=" * 80)
    posterior = estimate_rayleigh_baz_posterior(
        north=dataN,
        east=dataE,
        vertical=dataZ,
        fs=fs,
        target_dt=.2,
        fmin_hz=0.01,
        fmax_hz=3,
    )
    baz_deg = float(posterior["best_estimate_deg"])
    print(f"\nEstimated BAZ: {baz_deg:.2f} deg (+/- {posterior.get('posterior_std_deg', 0):.1f} deg)")
    if header_baz is not None:
        print(f"Header BAZ:    {header_baz:.2f} deg")

    print("\n" + "=" * 80)
    print("APPLYING PARTICLEMAN-STYLE NIP + ENERGY FILTER")
    print("=" * 80)
    fmin_hz = 1.0 / 150.0
    fmax_hz = 1.0 / 50.0
    result_dict = simple_stockwell_filter_rayleigh(
        dataN,
        dataE,
        dataZ,
        fs,
        baz_deg=baz_deg,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
        sense="retro",
        quadrature_min=0.0,
        hard_mask=False,
        amp_min=0.0,
        target_dt=0.5,
        zero_transverse=False,
        eps=1e-30,
    )

    add_particleman_nip_energy_masks(
        result_dict,
        threshold=0.8,
        width=0.05,
        eps=0.04,
        energy_floor=1.e-19,
        energy_width=0.02,
        energy_floor_mode="per_frequency",
        transverse_max=None,
        polarization_mode="elliptic",
    )
    print(f"\nStockwell transform complete: {len(result_dict['rayleigh_envelope'])} envelope samples")

    time_array = stockwell_time_axis_seconds(result_dict, trZ)
    event_offset_sec = timing["event_offset_sec"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_rayleigh_filter_diagnostics(
        result_dict,
        output_dir=str(output_dir),
        prefix="rayleigh_filter_pre_detection",
        event_offset_sec=event_offset_sec,
    )

    print("\n" + "=" * 80)
    print("DETECTING R1/R2/R3 ORBITS")
    print("=" * 80)
    print_time_sanity_checks(time_array, event_offset_sec, catalog_distance)

    detected_orbits = detect_rayleigh_orbits_from_stockwell(
        result_dict=result_dict,
        time_array=time_array,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
        distance_range_deg=(1., 175),
        event_offset_sec=event_offset_sec,
        use_joint_fitting=True,
        min_confidence=0.20,
        expected_distance_deg=catalog_distance,
        velocity_uncertainty=0.25,
        distance_uncertainty_deg=20.0,
        distance_prior_sigma_deg=10.0,
        distance_grid_step_deg=0.05,
        output_dir=str(output_dir),
        verbose=True,
    )

    plot_rayleigh_filter_diagnostics(
        result_dict,
        output_dir=str(output_dir),
        prefix="rayleigh_filter_post_detection",
        event_offset_sec=event_offset_sec,
        detected_orbits=detected_orbits,
    )

    print_detection_summary(detected_orbits, catalog_distance, output_dir)

    row = build_summary_row(
        spec=spec,
        output_dir=output_dir,
        trZ=trZ,
        timing=timing,
        posterior=posterior,
        detected_orbits=detected_orbits,
        catalog_distance=catalog_distance,
        header_baz=header_baz,
    )
    with open(output_dir / "event_summary.json", "w", encoding="utf-8") as f:
        event_summary = {k: v for k, v in row.items() if k not in {"baz_grid_deg", "baz_pdf", "distance_grid_deg", "distance_pdf"}}
        event_summary["detected_orbits"] = detected_orbits
        json.dump(_jsonable(event_summary), f, indent=2)
    return row


def run_batch_examples(
    data_dir="./data/combined",
    output_root="./orbit_detection_output",
    event_names=None,
    make_summary_plots=True,
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    specs = EVENT_RUNS
    if event_names:
        wanted = set(event_names)
        specs = [spec for spec in EVENT_RUNS if spec["name"] in wanted]
        missing = wanted - {spec["name"] for spec in specs}
        if missing:
            raise ValueError(f"Unknown event name(s): {sorted(missing)}")

    rows = []
    for spec in specs:
        event_output_dir = output_root / spec["name"]
        rows.append(run_rayleigh_event(spec, data_dir=data_dir, output_dir=event_output_dir))

    write_summary_csv(rows, output_root / "batch_summary.csv")
    write_summary_json(rows, output_root / "batch_summary.json")

    if make_summary_plots and rows:
        plot_batch_baz_linear(rows, output_root / "batch_baz_posterior_summary_linear.png")
        plot_batch_baz_radial(rows, output_root / "batch_baz_posterior_summary_radial.png")
        plot_batch_distance_linear(rows, output_root / "batch_distance_posterior_summary_linear.png")
        plot_truth_vs_map_error_summary(rows, output_root / "batch_truth_vs_map_errors.png")

    print("\n" + "=" * 80)
    print("BATCH COMPLETE")
    print("=" * 80)
    print(f"Output root: {output_root}")
    print(f"Summary CSV: {output_root / 'batch_summary.csv'}")
    print(f"Summary JSON: {output_root / 'batch_summary.json'}")
    if make_summary_plots:
        print(f"BAZ linear summary: {output_root / 'batch_baz_posterior_summary_linear.png'}")
        print(f"BAZ radial summary: {output_root / 'batch_baz_posterior_summary_radial.png'}")
        print(f"Distance summary: {output_root / 'batch_distance_posterior_summary_linear.png'}")
        print(f"Error summary: {output_root / 'batch_truth_vs_map_errors.png'}")
    print("=" * 80 + "\n")
    return rows



def main():
    data_dir="./data/combined"
    output_dir = "./orbit_detection_output"
    run_batch_examples(
        data_dir=data_dir,
        output_root=output_dir
    )


if __name__ == "__main__":
    main()
