import os
import glob
import SimpleITK as sitk


def get_dicom_dir(base_dir, phase, scan_type):
    if scan_type == "b0":
        return os.path.join(base_dir, "dicom_bbcine_combined")
    return os.path.join(base_dir, f"dicom_bbcine_phase{phase}")


def convert_dicom_to_nifti(dicom_base_dir, output_dir, scan_type="b40", n_phases=25):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Converting {n_phases} phases  |  scan_type={scan_type}")
    print(f"  Source : {dicom_base_dir}")
    print(f"  Output : {output_dir}")
    print("=" * 60)

    for phase in range(1, n_phases + 1):
        dicom_dir = get_dicom_dir(dicom_base_dir, phase, scan_type)

        if not os.path.isdir(dicom_dir):
            print(f"  Phase {phase:2d}: SKIPPED")
            continue

        n_files = len(glob.glob(os.path.join(dicom_dir, "*.dcm")))
        print(f"  Phase {phase:2d}: {n_files} DICOM files ... ", end="", flush=True)

        try:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()

            out_path = os.path.join(output_dir, f"phase{phase}.nii.gz")
            sitk.WriteImage(image, out_path)

            size_mb = os.path.getsize(out_path) / 1024 / 1024
            print(f"OK ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"ERROR: {e}")

    print("\nDone!")
