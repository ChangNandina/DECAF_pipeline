#!/usr/bin/env python3
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from step1_dicom_to_nifti.convert import convert_dicom_to_nifti
from step2_registration.register import register_interscan
from step3_crop.crop import crop_phases
from step4_downsample.downsample import downsample_phase1


def run_case(case):
    subject_id  = case['subject_id']
    scan1_dicom = case['scan1_dicom_dir']
    scan2_dicom = case['scan2_dicom_dir']
    base_out    = case['output_base_dir']
    scan_type   = case.get('scan_type', 'b40')
    n_phases    = int(case.get('n_phases', 25))

    print(f"\n{'#' * 70}")
    print(f"# Subject: {subject_id}")
    print(f"{'#' * 70}")

    scan1_nifti    = os.path.join(base_out, "scan1_nifti")
    scan2_nifti    = os.path.join(base_out, "scan2_nifti")
    scan1_crop     = os.path.join(base_out, "scan1_crop")
    scan2_crop     = os.path.join(base_out, "scan2_crop")
    scan2_ds480    = os.path.join(base_out, "scan2_downsample480")

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

    print(f"\n[Done]  {subject_id}  ->  {base_out}")


def main():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.csv")
    with open(csv_path, newline='') as f:
        for case in csv.DictReader(f):
            run_case(case)


if __name__ == '__main__':
    main()
