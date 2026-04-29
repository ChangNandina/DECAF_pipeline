import os
import glob
import re
import SimpleITK as sitk


def _natural_key(path):
    m = re.search(r"(\d+)\.nii\.gz$", os.path.basename(path))
    return int(m.group(1)) if m else path


def _read(path):
    return sitk.ReadImage(path)


def _rigid_register(fixed, moving):
    fixed_f  = sitk.Cast(fixed,  sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)

    initial = sitk.CenteredTransformInitializer(
        fixed_f, moving_f,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1e-4, numberOfIterations=300,
        convergenceWindowSize=15,
        estimateLearningRate=reg.EachIteration,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial)
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return reg.Execute(fixed_f, moving_f)


def _apply(fixed, moving, transform):
    return sitk.Resample(
        moving, fixed, transform,
        sitk.sitkBSpline, 0.0, moving.GetPixelID(),
    )


def _apply_to_files(file_list, fixed, transform, out_dir, label):
    print(f"  Applying transform to {len(file_list)} {label} file(s)...")
    for in_path in file_list:
        fname    = os.path.basename(in_path)
        out_path = os.path.join(out_dir, "r_" + fname)
        sitk.WriteImage(_apply(fixed, _read(in_path), transform), out_path)
        print(f"    {fname} -> r_{fname}")


def register_interscan(ref_dir, src_dir):
    """Register src_dir scan to ref_dir scan (rigid, phase1-derived transform).

    Returns out_dir path (src_dir + '_sitk_rigid').
    """
    out_dir = src_dir + "_sitk_rigid"
    os.makedirs(out_dir, exist_ok=True)

    ref_path       = os.path.join(ref_dir, "phase1.nii.gz")
    mov_path       = os.path.join(src_dir, "phase1.nii.gz")
    transform_path = os.path.join(out_dir, "phase1_to_ref.tfm")

    print(f"[register_interscan]")
    print(f"  ref : {ref_path}")
    print(f"  src : {mov_path}")

    fixed    = _read(ref_path)
    moving   = _read(mov_path)
    transform = _rigid_register(fixed, moving)
    sitk.WriteTransform(transform, transform_path)
    print(f"  Transform saved: {transform_path}")

    phase_files = sorted(
        glob.glob(os.path.join(src_dir, "phase*.nii.gz")), key=_natural_key
    )
    _apply_to_files(phase_files, fixed, transform, out_dir, "phase")

    b0_files = sorted(glob.glob(os.path.join(src_dir, "b0.nii.gz")))
    if b0_files:
        _apply_to_files(b0_files, fixed, transform, out_dir, "b0")

    print(f"  Output: {out_dir}")
    return out_dir
