import os
import glob
import SimpleITK as sitk


def _convert_folder(dicom_dir, out_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, out_path)
    return os.path.getsize(out_path) / 1024 / 1024


def convert_dicom_to_nifti(b40_dicom_dir, output_dir, n_phases=25, b0_dicom_dir=None):
    """Convert b40 phase DICOMs (and optionally b0) to NIfTI.

    Args:
        b40_dicom_dir: folder containing dicom_bbcine_phase1/ ... dicom_bbcine_phase{n}/
        output_dir:    where to write phase1.nii.gz ... phase{n}.nii.gz (and b0.nii.gz)
        n_phases:      number of cardiac phases
        b0_dicom_dir:  if given, folder containing dicom_bbcine_combined/ -> written as b0.nii.gz
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Converting {n_phases} b40 phases")
    print(f"  b40 source : {b40_dicom_dir}")
    if b0_dicom_dir:
        print(f"  b0  source : {b0_dicom_dir}")
    print(f"  Output     : {output_dir}")
    print("=" * 60)

    for phase in range(1, n_phases + 1):
        dicom_dir = os.path.join(b40_dicom_dir, f"dicom_bbcine_phase{phase}")
        if not os.path.isdir(dicom_dir):
            print(f"  Phase {phase:2d}: SKIPPED (not found)")
            continue

        n_files = len(glob.glob(os.path.join(dicom_dir, "*.dcm")))
        print(f"  Phase {phase:2d}: {n_files} DICOM files ... ", end="", flush=True)
        try:
            out_path = os.path.join(output_dir, f"phase{phase}.nii.gz")
            mb = _convert_folder(dicom_dir, out_path)
            print(f"OK ({mb:.1f} MB)")
        except Exception as e:
            print(f"ERROR: {e}")

    if b0_dicom_dir:
        b0_combined = os.path.join(b0_dicom_dir, "dicom_bbcine_combined")
        if os.path.isdir(b0_combined):
            n_files = len(glob.glob(os.path.join(b0_combined, "*.dcm")))
            print(f"  b0         : {n_files} DICOM files ... ", end="", flush=True)
            try:
                out_path = os.path.join(output_dir, "b0.nii.gz")
                mb = _convert_folder(b0_combined, out_path)
                print(f"OK ({mb:.1f} MB)")
            except Exception as e:
                print(f"ERROR: {e}")
        else:
            print(f"  b0: SKIPPED ({b0_combined} not found)")

    print("\nDone!")
