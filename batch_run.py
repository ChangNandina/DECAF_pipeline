#!/usr/bin/env python3
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from step1_dicom_to_nifti.convert import convert_dicom_to_nifti
from step2_registration.register import register_interscan
from step3_crop.crop import crop_phases
from step4_downsample.downsample import downsample_phase1
from step5_csf_mask.save_csfmask import save_csf_mask
from step6_adc.cal_adc_dicom import calc_adc_dicom
from step6_adc.cal_adc_nifti import calc_adc_nifti


def run_case(case):
    subject_id  = case['subject_id']
    scan1_dicom = case['scan1_dicom_dir']
    scan2_dicom = case['scan2_dicom_dir']
    base_out    = case['output_base_dir']
    scan_type   = case.get('scan_type', 'b40')
    n_phases    = int(case.get('n_phases', 25))

    # FSL inputs (filled in after FSL BET + FAST on downsampled phase1)
    scan1_skull_stripped = case.get('scan1_skull_stripped', '')
    scan1_csf_seg        = case.get('scan1_csf_seg', '')
    scan2_skull_stripped = case.get('scan2_skull_stripped', '')
    scan2_csf_seg        = case.get('scan2_csf_seg', '')

    # .mat CSF masks (created manually from interseg NIfTI after save_csf_mask)
    scan1_csf_mat = case.get('scan1_csf_mat', '')
    scan2_csf_mat = case.get('scan2_csf_mat', '')

    print(f"\n{'#' * 70}")
    print(f"# Subject: {subject_id}")
    print(f"{'#' * 70}")

    scan1_nifti  = os.path.join(base_out, "scan1_nifti")
    scan2_nifti  = os.path.join(base_out, "scan2_nifti")
    scan1_crop   = os.path.join(base_out, "scan1_crop")
    scan2_crop   = os.path.join(base_out, "scan2_crop")
    scan2_ds480  = os.path.join(base_out, "scan2_downsample480")
    scan1_csf    = os.path.join(base_out, "scan1_csf_mask")
    scan2_csf    = os.path.join(base_out, "scan2_csf_mask")
    scan1_adc    = os.path.join(base_out, "scan1_adc")
    scan2_adc    = os.path.join(base_out, "scan2_adc")

    # ── Steps 1-4: image preparation ─────────────────────────────────────
    print(f"\n[Step 1] scan1  DICOM -> NIfTI")
    convert_dicom_to_nifti(scan1_dicom, scan1_nifti, scan_type, n_phases)

    print(f"\n[Step 1] scan2  DICOM -> NIfTI")
    convert_dicom_to_nifti(scan2_dicom, scan2_nifti, scan_type, n_phases)

    print(f"\n[Step 2] Registration  scan2 -> scan1")
    scan2_reg = register_interscan(ref_dir=scan1_nifti, src_dir=scan2_nifti)

    print(f"\n[Step 3] Crop  scan1")
    crop_phases(scan1_nifti, scan1_crop, file_prefix="", n_phases=n_phases)

    print(f"\n[Step 3] Crop  scan2 (registered)")
    crop_phases(scan2_reg, scan2_crop, file_prefix="r_", n_phases=n_phases)

    print(f"\n[Step 4] Downsample scan2 phase1  960->480  (for FSL)")
    downsample_phase1(scan2_reg, scan2_ds480, file_prefix="r_")

    # ── Step 5: CSF mask (run after FSL BET + FAST) ───────────────────────
    # NOTE: after this step, manually convert the interseg NIfTI to .mat
    #       and fill scan1_csf_mat / scan2_csf_mat in cases.csv before
    #       running Step 6.
    if scan1_skull_stripped and scan1_csf_seg:
        print(f"\n[Step 5] CSF mask  scan1")
        save_csf_mask(scan1_skull_stripped, scan1_csf_seg, scan1_csf)

    if scan2_skull_stripped and scan2_csf_seg:
        print(f"\n[Step 5] CSF mask  scan2")
        save_csf_mask(scan2_skull_stripped, scan2_csf_seg, scan2_csf)

    # ── Step 6: ADC calculation ───────────────────────────────────────────
    if scan1_csf_mat:
        print(f"\n[Step 6] ADC  scan1  (from DICOM)")
        calc_adc_dicom(
            b0_folder    = os.path.join(scan1_dicom, 'dicom_bbcine_combined'),
            b1_base_path = scan1_dicom,
            csf_mask_path= scan1_csf_mat,
            output_folder= scan1_adc,
            phase_count  = n_phases,
        )

    if scan2_csf_mat:
        print(f"\n[Step 6] ADC  scan2  (from NIfTI)")
        calc_adc_nifti(
            b0_path      = os.path.join(scan2_reg, "r_b0.nii.gz"),
            b1_folder    = scan2_reg,
            csf_mask_path= scan2_csf_mat,
            output_folder= scan2_adc,
            phase_count  = n_phases,
        )

    print(f"\n[Done]  {subject_id}  ->  {base_out}")


def main():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.csv")
    with open(csv_path, newline='') as f:
        for case in csv.DictReader(f):
            run_case(case)


if __name__ == '__main__':
    main()
