"""
PancrAI — DICOM Reader Utility
Helpers for reading and processing DICOM files.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List


def read_dicom_series(directory: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read an entire DICOM series from a directory.

    Sorts slices by ImagePositionPatient (Z-axis) if available,
    otherwise by filename.

    Args:
        directory: Path to directory containing .dcm files.

    Returns:
        Tuple of (volume_array, metadata_dict)
        volume_array shape: (num_slices, H, W) float32 in HU
    """
    import pydicom

    dcm_files = sorted(Path(directory).glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found in {directory}")

    datasets = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f))
            datasets.append(ds)
        except Exception as e:
            print(f"[DICOM] Skipping {f}: {e}")

    if not datasets:
        raise ValueError("No readable DICOM files found.")

    # Sort by ImagePositionPatient Z if available
    try:
        datasets.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    except Exception:
        pass

    # Extract pixel data and convert to HU
    slices = []
    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)
        if hasattr(ds, "RescaleSlope"):
            arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        slices.append(arr)

    volume = np.stack(slices, axis=0)

    # Extract metadata from first slice
    meta = _extract_metadata(datasets[0])
    meta["num_slices"] = len(slices)
    meta["volume_shape"] = volume.shape

    return volume, meta


def read_single_dicom(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read a single DICOM file.

    Returns pixel array in HU (float32) and metadata dict.
    """
    import pydicom
    ds = pydicom.dcmread(filepath)
    arr = ds.pixel_array.astype(np.float32)
    if hasattr(ds, "RescaleSlope"):
        arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    return arr, _extract_metadata(ds)


def _extract_metadata(ds) -> Dict[str, Any]:
    """Extract clinically relevant metadata from DICOM dataset."""
    meta = {}
    fields = {
        "PatientID": "patient_id",
        "PatientName": "patient_name",
        "PatientAge": "patient_age",
        "PatientSex": "patient_sex",
        "StudyDate": "study_date",
        "Modality": "modality",
        "SliceThickness": "slice_thickness",
        "PixelSpacing": "pixel_spacing",
        "Rows": "rows",
        "Columns": "columns",
        "StudyDescription": "study_description",
        "SeriesDescription": "series_description",
        "KVP": "kvp",
        "ExposureTime": "exposure_time",
    }
    for dicom_key, our_key in fields.items():
        val = getattr(ds, dicom_key, None)
        if val is not None:
            meta[our_key] = str(val)
    return meta


def apply_window(arr: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Apply CT windowing to HU array.

    Common windows:
    - Soft tissue: center=40, width=400
    - Lung: center=-600, width=1500
    - Bone: center=400, width=1800
    - Brain: center=40, width=80
    """
    lower = center - width / 2
    upper = center + width / 2
    windowed = np.clip(arr, lower, upper)
    normalized = (windowed - lower) / (upper - lower)
    return normalized.astype(np.float32)


def get_middle_slice(volume: np.ndarray,
                     axis: int = 0) -> np.ndarray:
    """Extract the middle slice along the given axis."""
    mid = volume.shape[axis] // 2
    if axis == 0:
        return volume[mid]
    elif axis == 1:
        return volume[:, mid, :]
    else:
        return volume[:, :, mid]
