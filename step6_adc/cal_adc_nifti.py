import os
import numpy as np
import nibabel as nib

from ._utils import (save_slice_image, load_csf_mask, save_mat,
                     compute_correction_constants, apply_adc_correction)


def _load_nifti(filepath):
    img  = nib.load(filepath)
    data = np.asarray(img.dataobj, dtype=np.float32)
    data = np.transpose(data, (1, 0, 2))  # MATLAB permute([2,1,3])
    print(f"  Loaded: {os.path.basename(filepath)}  shape={data.shape}")
    return data


def calc_adc_nifti(b0_path, b1_folder, csf_mask_path, output_folder,
                   b1_filename_template="r_phase{phase}.nii.gz",
                   phase_count=25,
                   d=22, TEp=52, TR=240, segment=32, TE=2.79, fs=16,
                   alpha_deg=45, b_val=40):
    """Calculate ADC from NIfTI inputs (scan2, after registration).

    Args:
        b0_path:               path to b=0 NIfTI file (e.g. r_b0.nii.gz)
        b1_folder:             directory containing b=40 phase NIfTI files
        csf_mask_path:         .mat file with CSF mask
        output_folder:         directory for .mat ADC outputs
        b1_filename_template:  filename template, e.g. "r_phase{phase}.nii.gz"
        phase_count:           number of cardiac phases
        d, TEp, TR, segment, TE, fs, alpha_deg, b_val: scan parameters
    """
    qc_dir = os.path.join(output_folder, "qc_images")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(qc_dir, exist_ok=True)

    print("calc_adc_nifti")
    print(f"  b0_path   : {b0_path}")
    print(f"  b1_folder : {b1_folder}")
    print(f"  template  : {b1_filename_template}")
    print(f"  output    : {output_folder}")
    print("=" * 60)

    csf_mask = load_csf_mask(csf_mask_path)
    csf_mask = np.transpose(csf_mask, (1, 0, 2))  # match MATLAB permute([2,1,3])

    print("\nLoading b=0...")
    b0 = _load_nifti(b0_path)
    save_slice_image(b0, os.path.join(qc_dir, 'b0_image.png'), 'b=0 Image')

    constants = compute_correction_constants(d, TEp, TR, segment, TE, fs, alpha_deg, b_val)

    for phase in range(1, phase_count + 1):
        print(f"\n--- Phase {phase}/{phase_count} ---")
        b1_path = os.path.join(b1_folder, b1_filename_template.format(phase=phase))
        if not os.path.exists(b1_path):
            print(f"  SKIPPED (not found: {b1_path})")
            continue

        b1 = _load_nifti(b1_path)
        if b1.shape != b0.shape:
            raise ValueError(f"Phase {phase}: shape mismatch b1={b1.shape} b0={b0.shape}")

        epsilon = 1e-10
        adc = -np.log(np.maximum(b1, epsilon) / np.maximum(b0, epsilon)) / b_val

        save_mat(os.path.join(output_folder, f'adc_phase{phase:02d}_single.mat'),
                 {'adcImage': adc})

        adc_corr = apply_adc_correction(adc, constants)
        print(f"  ADC corrected range: [{adc_corr.min():.6f}, {adc_corr.max():.6f}]")

        save_mat(os.path.join(output_folder, f'adc_corrected_phase{phase:02d}_single.mat'),
                 {'adcCorrected': adc_corr})

        adc_csf = adc_corr * csf_mask.astype(np.float32)
        save_slice_image(adc_csf,
                         os.path.join(qc_dir, f'adc_corrected_csf_phase{phase:02d}.png'),
                         f'ADC Corrected + CSF - Phase {phase}')
        save_mat(os.path.join(output_folder, f'adc_corrected_csf_phase{phase:02d}_single.mat'),
                 {'adcCorrectedCSF': adc_csf})

        valid = adc > 0
        print(f"  ADC mean={adc[valid].mean():.6f}  "
              f"median={np.median(adc[valid]):.6f}  "
              f"std={adc[valid].std():.6f}")

        del b1, adc, adc_corr, adc_csf

    print(f"\nDone  ->  {output_folder}")
