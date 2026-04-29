import os
import numpy as np
import pydicom

from ._utils import (save_slice_image, load_csf_mask, save_mat,
                     compute_correction_constants, apply_adc_correction)


def _load_dicom_volume(folder_path, image_size):
    files = sorted(
        f for f in os.listdir(folder_path)
        if f.startswith('img_') and f.endswith('.dcm')
    )
    if not files:
        raise ValueError(f"No DICOM files in {folder_path}")

    volume = np.zeros(image_size, dtype=np.float32)
    for fname in files:
        sl = int(fname.replace('img_', '').replace('.dcm', ''))
        if sl < image_size[2]:
            dcm = pydicom.dcmread(os.path.join(folder_path, fname))
            volume[:, :, sl] = dcm.pixel_array.astype(np.float32)

    print(f"  Loaded {len(files)} slices from {os.path.basename(folder_path)}")
    return volume


def calc_adc_dicom(b0_folder, b1_base_path, csf_mask_path, output_folder,
                   phase_count=25, image_size=(960, 960, 960),
                   d=22, TEp=52, TR=240, segment=32, TE=2.79, fs=16,
                   alpha_deg=45, b_val=40):
    """Calculate ADC from DICOM inputs (scan1).

    Args:
        b0_folder:     path to b=0 DICOM folder
        b1_base_path:  root path; phase folders are at {b1_base_path}/dicom_bbcine_phase{p}
        csf_mask_path: .mat file with CSF mask
        output_folder: directory for .mat ADC outputs
        phase_count:   number of cardiac phases
        image_size:    expected 3D volume shape
        d, TEp, TR, segment, TE, fs, alpha_deg, b_val: scan parameters
    """
    qc_dir = os.path.join(output_folder, "qc_images")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(qc_dir, exist_ok=True)

    print("calc_adc_dicom")
    print(f"  b0_folder    : {b0_folder}")
    print(f"  b1_base_path : {b1_base_path}")
    print(f"  output       : {output_folder}")
    print("=" * 60)

    csf_mask = load_csf_mask(csf_mask_path)
    csf_mask = np.transpose(csf_mask, (1, 0, 2))  # match MATLAB permute([2,1,3])

    print("\nLoading b=0...")
    b0 = _load_dicom_volume(b0_folder, image_size)
    save_slice_image(b0, os.path.join(qc_dir, 'b0_image.png'), 'b=0 Image')

    constants = compute_correction_constants(d, TEp, TR, segment, TE, fs, alpha_deg, b_val)

    for phase in range(1, phase_count + 1):
        print(f"\n--- Phase {phase}/{phase_count} ---")
        phase_folder = os.path.join(b1_base_path, f'dicom_bbcine_phase{phase}')
        if not os.path.exists(phase_folder):
            print(f"  SKIPPED (not found)")
            continue

        b1 = _load_dicom_volume(phase_folder, image_size)

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
