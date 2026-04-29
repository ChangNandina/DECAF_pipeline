import os
import gzip
import struct
import numpy as np


def _get_dtype_info(datatype):
    return {
        2: (np.uint8, 1), 4: (np.int16, 2), 8: (np.int32, 4),
        16: (np.float32, 4), 64: (np.float64, 8),
        512: (np.uint16, 2), 768: (np.uint32, 4),
    }.get(datatype, (np.float32, 4))


def _read_nifti(filepath):
    with gzip.open(filepath, 'rb') as f:
        header_data    = f.read(348)
        extension_bytes = f.read(4)
        data_bytes     = f.read()

    datatype = struct.unpack('<h', header_data[70:72])[0]
    pixdim   = struct.unpack('<8f', header_data[76:108])
    dtype, bpv = _get_dtype_info(datatype)

    actual_dim = round((len(data_bytes) // bpv) ** (1/3))
    n = actual_dim
    data = np.frombuffer(data_bytes, dtype=dtype)[:n*n*n].reshape((n, n, n))

    print(f"  {os.path.basename(filepath)}: shape={data.shape}  "
          f"range=[{data.min():.3f}, {data.max():.3f}]")
    return header_data, extension_bytes, data, pixdim, dtype


def _write_nifti(filepath, header, extension_bytes, data, dtype):
    with gzip.open(filepath, 'wb', compresslevel=6) as f:
        f.write(header)
        f.write(extension_bytes)
        f.write(data.astype(dtype).tobytes())
    print(f"  Saved: {os.path.basename(filepath)}")


def save_csf_mask(skull_less_brain_path, csf_seg_path, output_dir,
                  csf_threshold=0.6, brain_threshold_percent=0.1):
    """Create CSF mask from FSL outputs.

    Args:
        skull_less_brain_path: skull-stripped brain NIfTI (from FSL BET)
        csf_seg_path:          CSF partial-volume map NIfTI (FSL FAST pve_0)
        output_dir:            directory for output masks
        csf_threshold:         CSF probability threshold (default 0.6)
        brain_threshold_percent: brain mask threshold as fraction of max (default 0.1)
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"save_csf_mask  |  csf_thr={csf_threshold}  brain_thr={brain_threshold_percent}")
    print(f"  skull_less_brain : {skull_less_brain_path}")
    print(f"  csf_seg          : {csf_seg_path}")
    print(f"  output_dir       : {output_dir}")
    print("=" * 60)

    brain_header, brain_ext, brain_data, _, brain_dtype = _read_nifti(skull_less_brain_path)
    csf_header,   csf_ext,   csf_data,   _, csf_dtype   = _read_nifti(csf_seg_path)

    if brain_data.shape != csf_data.shape:
        raise ValueError(
            f"Shape mismatch: brain={brain_data.shape}  csf={csf_data.shape}"
        )

    # Brain mask (10% of max)
    thr_val = brain_data.max() * brain_threshold_percent
    brain_mask = (brain_data >= thr_val).astype(np.uint8)
    print(f"\n  Brain mask voxels : {brain_mask.sum()}")
    _write_nifti(
        os.path.join(output_dir, "brain_mask_10pct.nii.gz"),
        brain_header, brain_ext, brain_mask, brain_dtype,
    )

    # CSF mask + intersection
    csf_mask  = (csf_data >= csf_threshold).astype(np.uint8)
    inter_mask = csf_mask & brain_mask
    print(f"  CSF mask voxels   : {csf_mask.sum()}")
    print(f"  Intersection      : {inter_mask.sum()}")

    _write_nifti(
        os.path.join(output_dir, f"csf_mask_thr{csf_threshold:.1f}.nii.gz"),
        csf_header, csf_ext, csf_mask, csf_dtype,
    )
    _write_nifti(
        os.path.join(output_dir, f"csf_brain_inter_thr{csf_threshold:.1f}.nii.gz"),
        csf_header, csf_ext, inter_mask, csf_dtype,
    )

    print("\nDone.")
