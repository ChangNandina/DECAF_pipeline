import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom


def downsample_phase1(nifti_dir, output_dir, file_prefix="r_"):
    """Downsample phase1 by 0.5x in each axis (960->480, ~0.21mm->~0.42mm).

    Only phase1 is downsampled — intended for FSL segmentation input.

    Args:
        nifti_dir:   directory containing the 960^3 NIfTI (registered scan2)
        output_dir:  directory to save the downsampled phase1
        file_prefix: "r_" for registered scan2 (default)
    """
    os.makedirs(output_dir, exist_ok=True)

    in_path  = os.path.join(nifti_dir,  f"{file_prefix}phase1.nii.gz")
    out_path = os.path.join(output_dir, f"{file_prefix}phase1.nii.gz")

    print(f"Downsample 960->480  |  prefix='{file_prefix}'")
    print(f"  Input : {in_path}")
    print(f"  Output: {out_path}")

    img    = nib.load(in_path)
    data   = img.get_fdata(dtype=np.float32)
    affine = img.affine.copy()
    header = img.header.copy()

    print(f"  Input  shape : {data.shape}")
    print(f"  Input  voxel : {np.abs(np.diag(affine))[:3]} mm")

    target_shape = tuple(s // 2 for s in data.shape)
    zoom_factors = tuple(t / s for t, s in zip(target_shape, data.shape))

    print(f"  Zoom factors : {zoom_factors}")
    print(f"  Target shape : {target_shape}")
    print("  Resampling (order=3 spline)...")

    data_ds = zoom(data, zoom_factors, order=3, prefilter=True)

    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, :3] / np.array(zoom_factors)
    old_vox = np.abs(np.diag(affine)[:3])
    new_vox = old_vox / np.array(zoom_factors)
    shift   = (new_vox - old_vox) / 2.0
    for i in range(3):
        new_affine[:3, 3] += affine[:3, i] / np.linalg.norm(affine[:3, i]) * shift[i]

    print(f"  Output voxel : {np.abs(np.diag(new_affine))[:3]} mm")

    out_img = nib.Nifti1Image(data_ds, new_affine, header)
    out_img.header.set_data_shape(data_ds.shape)
    out_img.header.set_zooms(np.abs(np.diag(new_affine))[:3])
    nib.save(out_img, out_path)

    print(f"  Saved: {out_path}")
    print("Done.")
