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
from step7_vessel_parta.parta import run_parta
from step8_vessel_partb.partb import run_partb
from step9_graph.build_graph import build_vessel_graph
from step10_pi.compute_pi import compute_pulsatility
from step11_paravascular.csf_paravascular import compute_paravascular_adc


def _parse_skip_segments(s):
    """Parse '9,13' → [9, 13], or '' → []"""
    if not s or not s.strip():
        return []
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def run_case(case):
    subject_id     = case['subject_id']
    scan1_b40_dicom = case['scan1_b40_dicom_dir']
    scan1_b0_dicom  = case.get('scan1_b0_dicom_dir', '')
    scan2_b40_dicom = case['scan2_b40_dicom_dir']
    scan2_b0_dicom  = case.get('scan2_b0_dicom_dir', '')
    base_out        = case['output_base_dir']
    n_phases        = int(case.get('n_phases', 25))

    # FSL + ADC inputs (fill after FSL BET/FAST)
    scan1_skull_stripped = case.get('scan1_skull_stripped', '')
    scan1_csf_seg        = case.get('scan1_csf_seg', '')
    scan2_skull_stripped = case.get('scan2_skull_stripped', '')
    scan2_csf_seg        = case.get('scan2_csf_seg', '')
    scan1_csf_mat        = case.get('scan1_csf_mat', '')
    scan2_csf_mat        = case.get('scan2_csf_mat', '')

    # Vessel inputs (fill after ITK-SNAP annotation + Part A review)
    scan1_vessel_mask   = case.get('scan1_vessel_mask', '')
    scan2_vessel_mask   = case.get('scan2_vessel_mask', '')
    scan1_skip_segments = _parse_skip_segments(case.get('scan1_skip_segments', ''))
    scan2_skip_segments = _parse_skip_segments(case.get('scan2_skip_segments', ''))

    print(f"\n{'#' * 70}")
    print(f"# Subject: {subject_id}")
    print(f"{'#' * 70}")

    # ── Output paths ─────────────────────────────────────────────
    scan1_nifti  = os.path.join(base_out, "scan1_nifti")
    scan2_nifti  = os.path.join(base_out, "scan2_nifti")
    scan1_crop   = os.path.join(base_out, "scan1_crop")
    scan2_crop   = os.path.join(base_out, "scan2_crop")
    scan2_ds480  = os.path.join(base_out, "scan2_downsample480")
    scan1_csf    = os.path.join(base_out, "scan1_csf_mask")
    scan2_csf    = os.path.join(base_out, "scan2_csf_mask")
    scan1_adc    = os.path.join(base_out, "scan1_adc")
    scan2_adc    = os.path.join(base_out, "scan2_adc")

    # Vessel paths
    scan1_vessel_out = os.path.join(base_out, "scan1_vessel")
    scan2_vessel_out = os.path.join(base_out, "scan2_vessel")
    scan1_graph      = os.path.join(scan1_vessel_out, "graph")
    scan2_graph      = os.path.join(scan2_vessel_out, "graph")
    scan1_pi         = os.path.join(scan1_vessel_out, "pi")
    scan2_pi         = os.path.join(scan2_vessel_out, "pi")
    scan1_paravasc   = os.path.join(scan1_vessel_out, "paravascular")
    scan2_paravasc   = os.path.join(scan2_vessel_out, "paravascular")

    # ── Steps 1-4: image preparation ─────────────────────────────
    # scan1: convert b40 phases + b0 (b0 not used downstream but kept for consistency)
    print(f"\n[Step 1] scan1  DICOM -> NIfTI")
    convert_dicom_to_nifti(scan1_b40_dicom, scan1_nifti, n_phases=n_phases,
                           b0_dicom_dir=scan1_b0_dicom or None)

    # scan2: convert b40 phases + b0 (b0 needed for registration -> r_b0.nii.gz -> ADC)
    print(f"\n[Step 1] scan2  DICOM -> NIfTI")
    convert_dicom_to_nifti(scan2_b40_dicom, scan2_nifti, n_phases=n_phases,
                           b0_dicom_dir=scan2_b0_dicom or None)

    print(f"\n[Step 2] Registration  scan2 -> scan1")
    scan2_reg = register_interscan(ref_dir=scan1_nifti, src_dir=scan2_nifti)

    print(f"\n[Step 3] Crop  scan1")
    crop_phases(scan1_nifti, scan1_crop, file_prefix="", n_phases=n_phases)

    print(f"\n[Step 3] Crop  scan2 (registered)")
    crop_phases(scan2_reg, scan2_crop, file_prefix="r_", n_phases=n_phases)

    print(f"\n[Step 4] Downsample scan2 phase1  960->480  (for FSL)")
    downsample_phase1(scan2_reg, scan2_ds480, file_prefix="r_")

    # ── Step 5: CSF mask (run after FSL BET + FAST) ───────────────
    # NOTE: after this step, manually convert the interseg NIfTI to .mat
    #       (scan1_csf_mat / scan2_csf_mat in cases.csv) before Step 6.
    if scan1_skull_stripped and scan1_csf_seg:
        print(f"\n[Step 5] CSF mask  scan1")
        save_csf_mask(scan1_skull_stripped, scan1_csf_seg, scan1_csf)

    if scan2_skull_stripped and scan2_csf_seg:
        print(f"\n[Step 5] CSF mask  scan2")
        save_csf_mask(scan2_skull_stripped, scan2_csf_seg, scan2_csf)

    # ── Step 6: ADC calculation ────────────────────────────────────
    if scan1_csf_mat:
        print(f"\n[Step 6] ADC  scan1  (from DICOM)")
        calc_adc_dicom(
            b0_folder    =os.path.join(scan1_b0_dicom, 'dicom_bbcine_combined'),
            b1_base_path =scan1_b40_dicom,
            csf_mask_path=scan1_csf_mat,
            output_folder=scan1_adc,
            phase_count  =n_phases,
        )

    if scan2_csf_mat:
        print(f"\n[Step 6] ADC  scan2  (from NIfTI)")
        calc_adc_nifti(
            b0_path      =os.path.join(scan2_reg, "r_b0.nii.gz"),
            b1_folder    =scan2_reg,
            csf_mask_path=scan2_csf_mat,
            output_folder=scan2_adc,
            phase_count  =n_phases,
        )

    # ── Steps 7-11: vessel pipeline ───────────────────────────────
    # MANUAL CHECKPOINT between Part A and Part B:
    #   1. Run Part A for all subjects first
    #   2. Open ITK-SNAP, review p01_binary.nii.gz + p01_multilabel.nii.gz
    #   3. Fill scan1_skip_segments / scan2_skip_segments in cases.csv
    #   4. Then run Part B

    if scan1_vessel_mask:
        print(f"\n[Step 7] Vessel Part A  scan1  (requires nninteractive server at localhost:8911)")
        run_parta(nifti_dir=scan1_crop, mask_path=scan1_vessel_mask,
                  output_dir=scan1_vessel_out, skip_segments=scan1_skip_segments)

    if scan2_vessel_mask:
        print(f"\n[Step 7] Vessel Part A  scan2  (requires nninteractive server at localhost:8911)")
        run_parta(nifti_dir=scan2_crop, mask_path=scan2_vessel_mask,
                  output_dir=scan2_vessel_out, skip_segments=scan2_skip_segments,
                  file_prefix="")

    if scan1_vessel_mask and os.path.exists(os.path.join(scan1_vessel_out, "segments.json")):
        print(f"\n[Step 8] Vessel Part B  scan1")
        run_partb(nifti_dir=scan1_crop, mask_path=scan1_vessel_mask,
                  output_dir=scan1_vessel_out, skip_segments=scan1_skip_segments,
                  n_phases=n_phases)

    if scan2_vessel_mask and os.path.exists(os.path.join(scan2_vessel_out, "segments.json")):
        print(f"\n[Step 8] Vessel Part B  scan2")
        run_partb(nifti_dir=scan2_crop, mask_path=scan2_vessel_mask,
                  output_dir=scan2_vessel_out, skip_segments=scan2_skip_segments,
                  n_phases=n_phases)

    if os.path.exists(os.path.join(scan1_vessel_out, "p25_binary.nii.gz")):
        print(f"\n[Step 9]  Graph  scan1")
        build_vessel_graph(planb_output_dir=scan1_vessel_out,
                           planb_data_dir=scan1_crop,
                           out_dir=scan1_graph,
                           skip_segments=scan1_skip_segments, n_phases=n_phases)

        print(f"\n[Step 10] PI  scan1")
        compute_pulsatility(step1_dir=scan1_graph,
                            data_dir=scan1_vessel_out,
                            out_dir=scan1_pi, n_phases=n_phases)

        print(f"\n[Step 11] Paravascular ADC  scan1")
        compute_paravascular_adc(step2_dir=scan1_pi,
                                 crop_dir=scan1_crop,
                                 adc_folder=scan1_adc,
                                 vessel_dir=scan1_vessel_out,
                                 out_dir=scan1_paravasc, n_phases=n_phases)

    if os.path.exists(os.path.join(scan2_vessel_out, "p25_binary.nii.gz")):
        print(f"\n[Step 9]  Graph  scan2")
        build_vessel_graph(planb_output_dir=scan2_vessel_out,
                           planb_data_dir=scan2_crop,
                           out_dir=scan2_graph,
                           skip_segments=scan2_skip_segments, n_phases=n_phases)

        print(f"\n[Step 10] PI  scan2")
        compute_pulsatility(step1_dir=scan2_graph,
                            data_dir=scan2_vessel_out,
                            out_dir=scan2_pi, n_phases=n_phases)

        print(f"\n[Step 11] Paravascular ADC  scan2")
        compute_paravascular_adc(step2_dir=scan2_pi,
                                 crop_dir=scan2_crop,
                                 adc_folder=scan2_adc,
                                 vessel_dir=scan2_vessel_out,
                                 out_dir=scan2_paravasc, n_phases=n_phases)

    print(f"\n[Done]  {subject_id}  ->  {base_out}")


def main():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.csv")
    with open(csv_path, newline='') as f:
        for case in csv.DictReader(f):
            run_case(case)


if __name__ == '__main__':
    main()
