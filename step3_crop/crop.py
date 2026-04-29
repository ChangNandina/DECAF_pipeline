import os
import json
import time
import nibabel as nib
import numpy as np


def _compute_center_crop(original_shape, cube_size):
    center = np.array(original_shape) // 2
    half   = cube_size // 2

    origin = np.zeros(3, dtype=int)
    for d in range(3):
        lo = center[d] - half
        hi = lo + cube_size
        if lo < 0:
            lo, hi = 0, cube_size
        if hi > original_shape[d]:
            hi, lo = original_shape[d], original_shape[d] - cube_size
        if lo < 0:
            lo, hi = 0, original_shape[d]
        origin[d] = lo

    crop_slices, actual_size = [], []
    for d in range(3):
        lo = int(origin[d])
        hi = min(lo + cube_size, original_shape[d])
        crop_slices.append(slice(lo, hi))
        actual_size.append(hi - lo)

    return {
        'crop_slices':    tuple(crop_slices),
        'crop_origin':    tuple(int(x) for x in origin),
        'crop_size':      tuple(actual_size),
        'cube_size':      cube_size,
        'original_shape': tuple(int(x) for x in original_shape),
        'center':         tuple(int(x) for x in center),
    }


def _crop_and_save(input_path, output_path, crop_info, affine_full):
    img       = nib.load(input_path)
    data_full = img.get_fdata()
    sl        = crop_info['crop_slices']
    data_crop = data_full[sl[0], sl[1], sl[2]].copy()

    crop_origin = np.array(crop_info['crop_origin'], dtype=float)
    crop_affine = affine_full.copy()
    crop_affine[:3, 3] = affine_full[:3, 3] + affine_full[:3, :3] @ crop_origin

    nib.save(nib.Nifti1Image(data_crop, crop_affine), output_path)
    return data_crop.shape


def crop_phases(nifti_dir, output_dir, file_prefix="", n_phases=25, cube_size=480):
    """Crop all phases to a cube centered at the volume center.

    Args:
        nifti_dir:   directory containing input NIfTI files
        output_dir:  directory for cropped output
        file_prefix: "" for scan1 (phase{p}.nii.gz), "r_" for scan2 (r_phase{p}.nii.gz)
        n_phases:    number of cardiac phases
        cube_size:   side length of the cubic crop in voxels
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Crop to {cube_size}³  |  prefix='{file_prefix}'")
    print(f"  Input : {nifti_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    ref_path = os.path.join(nifti_dir, f"{file_prefix}phase1.nii.gz")
    ref_img  = nib.load(ref_path)
    original_shape = ref_img.shape[:3]
    affine_full    = ref_img.affine
    print(f"  Original shape : {original_shape}")

    crop_info = _compute_center_crop(original_shape, cube_size)
    print(f"  Volume center  : {crop_info['center']}")
    print(f"  Crop origin    : {crop_info['crop_origin']}")
    print(f"  Crop size      : {crop_info['crop_size']}")

    info_to_save = {k: crop_info[k] for k in
                    ('original_shape', 'crop_origin', 'crop_size', 'cube_size', 'center')}
    info_to_save['note'] = (
        'orig_coord = crop_coord + crop_origin; '
        'crop_coord = orig_coord - crop_origin'
    )
    with open(os.path.join(output_dir, "crop_info.json"), 'w') as f:
        json.dump(info_to_save, f, indent=2)

    print(f"\nCropping {n_phases} phases...")
    for p in range(1, n_phases + 1):
        in_path = os.path.join(nifti_dir, f"{file_prefix}phase{p}.nii.gz")
        if not os.path.exists(in_path):
            print(f"  Phase {p:2d}: NOT FOUND, skipping")
            continue

        out_path = os.path.join(output_dir, f"phase{p}.nii.gz")
        t0       = time.time()
        shape    = _crop_and_save(in_path, out_path, crop_info, affine_full)
        size_mb  = os.path.getsize(out_path) / 1024 / 1024
        print(f"  Phase {p:2d}: {shape}  {size_mb:.1f} MB  ({time.time()-t0:.1f}s)")

    print(f"\nDone  ->  {output_dir}")
