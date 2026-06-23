import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences, peak_widths


def _safe_mean(x):
    x = np.asarray(x, dtype=float)
    return np.nan if x.size == 0 else float(np.nanmean(x))


def _safe_median(x):
    x = np.asarray(x, dtype=float)
    return np.nan if x.size == 0 else float(np.nanmedian(x))


def _safe_std(x):
    x = np.asarray(x, dtype=float)
    return np.nan if x.size == 0 else float(np.nanstd(x, ddof=0))


def _contiguous_regions(mask):
    """Return start/end frame indices for True regions in a 1D Boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    regions = []
    in_region = False
    start = None
    for idx, value in enumerate(mask):
        if value and not in_region:
            in_region = True
            start = idx
        elif not value and in_region:
            regions.append((start, idx - 1))
            in_region = False
    if in_region:
        regions.append((start, len(mask) - 1))
    return regions


def analyze_bmps_burst_duration(
    data,
    frame_interval_s,
    prominence=80,
    min_peak_count=3,
    max_peak_count=100,
    peak_width_rel_height=1.0,
    half_width_rel_height=0.5,
    wlen=None,
    min_peak_distance_frames=None,
    min_peak_width_frames=None,
    organoid_id="Organoid_01",
    condition="Condition_01",
    network_participation_threshold=0.20,
    min_network_burst_duration_s=0.0,
):
    """
    BMPS-style burst-duration analysis for the Calcium_Frequency workflow.

    Parameters
    ----------
    data : ndarray, shape (n_rois, n_frames)
        ROI x time fluorescence traces. This is designed to run after your existing
        extraction and polynomial correction steps:
            data, mask, morph, size = extract_data_wloc(...)
            data = poly_corr(data, 6)
    frame_interval_s : float
        Seconds per image frame. Use 5.0 if your acquisition interval is 5 seconds.
    prominence : float
        Prominence threshold passed to scipy.signal.find_peaks.
    min_peak_count : int
        Minimum number of peaks required for ROI to pass the original-style
        spiking/activity filter. Uses >= min_peak_count.
    max_peak_count : int
        Maximum number of peaks allowed for ROI to pass the activity filter.
    peak_width_rel_height : float
        rel_height for scipy.signal.peak_widths. The bMPS notebook used rel_height=1
        and called the resulting full-prominence width burst duration.
    half_width_rel_height : float
        rel_height for half-width calculation. Usually 0.5.
    wlen : int or None
        Window length used by scipy.signal.peak_prominences. None uses full trace.
    min_peak_distance_frames : int or None
        Optional minimum distance between detected peaks.
    min_peak_width_frames : int or None
        Optional minimum width for detected peaks.
    organoid_id, condition : str
        Metadata added to output tables.
    network_participation_threshold : float
        Fraction of ROIs that must be active at a frame to call a network burst.
    min_network_burst_duration_s : float
        Minimum network-burst duration in seconds.

    Returns
    -------
    dict with:
        event_table : one row per detected peak/calcium transient
        roi_summary : one row per ROI
        organoid_summary : one-row summary for this imaging file/organoid
        network_bursts : one row per network burst
        binary_activity : ROI x frame Boolean matrix using full-width event windows
        population_activity : fraction of ROIs active at each frame
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be a 2D array with shape (n_rois, n_frames)")
    if frame_interval_s <= 0:
        raise ValueError("frame_interval_s must be positive")

    n_rois, n_frames = data.shape
    recording_duration_s = n_frames * frame_interval_s
    recording_duration_min = recording_duration_s / 60.0 if recording_duration_s > 0 else np.nan

    event_rows = []
    roi_rows = []
    binary_activity = np.zeros((n_rois, n_frames), dtype=bool)

    find_peak_kwargs = {"prominence": prominence}
    if min_peak_distance_frames is not None:
        find_peak_kwargs["distance"] = int(min_peak_distance_frames)
    if min_peak_width_frames is not None:
        find_peak_kwargs["width"] = int(min_peak_width_frames)

    for roi in range(n_rois):
        trace = np.asarray(data[roi, :], dtype=float)
        valid = np.isfinite(trace)

        if valid.sum() < 3:
            peaks = np.array([], dtype=int)
            full_widths_frames = np.array([], dtype=float)
            half_widths_frames = np.array([], dtype=float)
            prominences = np.array([], dtype=float)
            left_ips_full = np.array([], dtype=float)
            right_ips_full = np.array([], dtype=float)
        else:
            clean_trace = trace.copy()
            if not valid.all():
                # Replace sparse NaNs by the median so scipy can still run.
                clean_trace[~valid] = np.nanmedian(clean_trace[valid])

            peaks, peak_props = find_peaks(clean_trace, **find_peak_kwargs)

            if len(peaks) > 0:
                prom_data = peak_prominences(clean_trace, peaks, wlen=wlen)
                prominences = prom_data[0]

                res_full = peak_widths(
                    clean_trace,
                    peaks,
                    rel_height=peak_width_rel_height,
                    prominence_data=prom_data,
                )
                full_widths_frames = res_full[0]
                left_ips_full = res_full[2]
                right_ips_full = res_full[3]

                res_half = peak_widths(
                    clean_trace,
                    peaks,
                    rel_height=half_width_rel_height,
                    prominence_data=prom_data,
                )
                half_widths_frames = res_half[0]
            else:
                full_widths_frames = np.array([], dtype=float)
                half_widths_frames = np.array([], dtype=float)
                prominences = np.array([], dtype=float)
                left_ips_full = np.array([], dtype=float)
                right_ips_full = np.array([], dtype=float)

        n_peaks = int(len(peaks))
        passes_peak_count_filter = (n_peaks >= min_peak_count) and (n_peaks <= max_peak_count)
        is_active = n_peaks > 0

        for event_id, peak_frame in enumerate(peaks):
            onset_frame_float = float(left_ips_full[event_id])
            end_frame_float = float(right_ips_full[event_id])
            onset_frame = max(0, int(np.floor(onset_frame_float)))
            end_frame = min(n_frames - 1, int(np.ceil(end_frame_float)))
            if end_frame >= onset_frame:
                binary_activity[roi, onset_frame:end_frame + 1] = True

            event_rows.append({
                "organoid_id": organoid_id,
                "condition": condition,
                "roi": roi,
                "event_id": event_id,
                "peak_frame": int(peak_frame),
                "peak_time_s": float(peak_frame * frame_interval_s),
                "peak_prominence": float(prominences[event_id]),
                "burst_duration_frames_bmps_full_width": float(full_widths_frames[event_id]),
                "burst_duration_s_bmps_full_width": float(full_widths_frames[event_id] * frame_interval_s),
                "half_width_frames": float(half_widths_frames[event_id]),
                "half_width_s": float(half_widths_frames[event_id] * frame_interval_s),
                "onset_frame_interp": onset_frame_float,
                "end_frame_interp": end_frame_float,
                "onset_time_s_interp": onset_frame_float * frame_interval_s,
                "end_time_s_interp": end_frame_float * frame_interval_s,
            })

        roi_rows.append({
            "organoid_id": organoid_id,
            "condition": condition,
            "roi": roi,
            "is_active": bool(is_active),
            "passes_peak_count_filter": bool(passes_peak_count_filter),
            "n_peaks": n_peaks,
            "event_frequency_per_min": n_peaks / recording_duration_min if recording_duration_min and recording_duration_min > 0 else np.nan,
            "mean_burst_duration_s_bmps_full_width": _safe_mean(full_widths_frames * frame_interval_s),
            "median_burst_duration_s_bmps_full_width": _safe_median(full_widths_frames * frame_interval_s),
            "std_burst_duration_s_bmps_full_width": _safe_std(full_widths_frames * frame_interval_s),
            "mean_half_width_s": _safe_mean(half_widths_frames * frame_interval_s),
            "median_half_width_s": _safe_median(half_widths_frames * frame_interval_s),
            "std_half_width_s": _safe_std(half_widths_frames * frame_interval_s),
            "mean_peak_prominence": _safe_mean(prominences),
            "total_normalized_activation_time_bmps": float(np.nansum(full_widths_frames) / n_frames) if n_frames > 0 else np.nan,
        })

    event_table = pd.DataFrame(event_rows)
    roi_summary = pd.DataFrame(roi_rows)

    # Network-level bursts from the ROI activity windows.
    population_activity = binary_activity.sum(axis=0) / n_rois if n_rois > 0 else np.array([])
    burst_mask = population_activity >= network_participation_threshold
    min_network_burst_frames = int(np.ceil(min_network_burst_duration_s / frame_interval_s)) if frame_interval_s > 0 else 0
    network_rows = []
    for network_event_id, (start, end) in enumerate(_contiguous_regions(burst_mask)):
        duration_frames = end - start + 1
        if duration_frames < max(1, min_network_burst_frames):
            continue
        network_rows.append({
            "organoid_id": organoid_id,
            "condition": condition,
            "network_event_id": network_event_id,
            "start_frame": int(start),
            "end_frame": int(end),
            "start_time_s": float(start * frame_interval_s),
            "end_time_s": float(end * frame_interval_s),
            "network_burst_duration_s": float(duration_frames * frame_interval_s),
            "max_population_participation": float(np.nanmax(population_activity[start:end + 1])),
            "mean_population_participation": float(np.nanmean(population_activity[start:end + 1])),
        })
    network_bursts = pd.DataFrame(network_rows)

    active_indices = roi_summary.index[roi_summary["is_active"]].to_numpy()
    if len(active_indices) >= 2:
        corr_matrix = np.corrcoef(data[active_indices, :])
        upper = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        synchrony_index = float(np.nanmean(upper))
    else:
        synchrony_index = np.nan

    organoid_summary = pd.DataFrame([{
        "organoid_id": organoid_id,
        "condition": condition,
        "n_rois": n_rois,
        "n_frames": n_frames,
        "frame_interval_s": frame_interval_s,
        "recording_duration_s": recording_duration_s,
        "active_cell_fraction": float(roi_summary["is_active"].mean()) if n_rois > 0 else np.nan,
        "passes_peak_count_filter_fraction": float(roi_summary["passes_peak_count_filter"].mean()) if n_rois > 0 else np.nan,
        "median_event_frequency_per_min": float(roi_summary["event_frequency_per_min"].median()),
        "mean_burst_duration_s_bmps_full_width": float(roi_summary["mean_burst_duration_s_bmps_full_width"].mean()),
        "median_burst_duration_s_bmps_full_width": float(roi_summary["median_burst_duration_s_bmps_full_width"].median()),
        "median_half_width_s": float(roi_summary["median_half_width_s"].median()),
        "network_burst_count": int(len(network_bursts)),
        "network_burst_frequency_per_min": float(len(network_bursts) / recording_duration_min) if recording_duration_min and recording_duration_min > 0 else np.nan,
        "mean_network_burst_duration_s": float(network_bursts["network_burst_duration_s"].mean()) if len(network_bursts) else np.nan,
        "median_network_burst_duration_s": float(network_bursts["network_burst_duration_s"].median()) if len(network_bursts) else np.nan,
        "synchrony_index_raw_trace_corr_active_rois": synchrony_index,
    }])

    return {
        "event_table": event_table,
        "roi_summary": roi_summary,
        "organoid_summary": organoid_summary,
        "network_bursts": network_bursts,
        "binary_activity": binary_activity,
        "population_activity": population_activity,
    }
