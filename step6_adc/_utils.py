import os
import numpy as np
import hdf5storage
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_slice_image(data, filepath, title="", slice_idx=None):
    if slice_idx is None:
        slice_idx = data.shape[2] // 2
    plt.figure(figsize=(10, 10))
    plt.imshow(data[:, :, slice_idx], cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  QC image: {os.path.basename(filepath)}")


def load_csf_mask(filepath):
    """Load CSF mask from .mat file (supports v7.3 HDF5 and legacy formats)."""
    print(f"  Loading CSF mask: {os.path.basename(filepath)}")
    try:
        import h5py
        with h5py.File(filepath, 'r') as f:
            keys = [k for k in f.keys() if not k.startswith('#')]
            key  = 'data' if 'data' in f else ('csfMask' if 'csfMask' in f else keys[0])
            mask = np.array(f[key])
            if mask.ndim == 3:
                mask = np.transpose(mask, (2, 1, 0))  # MATLAB col-major → row-major
    except (ImportError, OSError):
        mat  = loadmat(filepath)
        keys = [k for k in mat.keys() if not k.startswith('__')]
        key  = 'data' if 'data' in mat else ('csfMask' if 'csfMask' in mat else keys[0])
        mask = mat[key]

    print(f"    shape={mask.shape}  range=[{mask.min()}, {mask.max()}]  "
          f"nonzero={np.count_nonzero(mask)}")
    return mask


def save_mat(filepath, data_dict):
    """Save as MATLAB v7.3 (HDF5), overwriting if exists."""
    if os.path.exists(filepath):
        os.remove(filepath)
    hdf5storage.savemat(filepath, data_dict, format='7.3', oned_as='column',
                        store_python_metadata=False)
    print(f"  Saved: {os.path.basename(filepath)}")


def compute_correction_constants(d=22, TEp=52, TR=240, segment=32,
                                  TE=2.79, fs=16, alpha_deg=45, b_val=40):
    """Pre-compute ADC correction constants from scan parameters."""
    echo_spacing = 2 * TE
    trc   = TR - (4 + segment) * echo_spacing - fs - TE
    alpha = np.deg2rad(alpha_deg)

    T1, T2 = 4000.0, 2000.0
    E1  = np.exp(-echo_spacing / T1)
    E2  = np.exp(-echo_spacing / T2)
    Oz  = ((1 - E1) * (E2 + np.cos(alpha))) / \
          ((1 - E1 * np.cos(alpha)) - (E1 - np.cos(alpha)) * E2)
    Op  = np.exp(-TEp / T2) * np.exp(-d / T1)
    OB  = Op * Oz * np.exp(-trc / T1)
    Orc = 1 - np.exp(-trc / T1)
    Od  = 1 - np.exp(-trc / T1)
    OA  = Orc + Od * Oz * np.exp(-trc / T1)

    return dict(OA=OA, OB=OB, Od=Od, Op=Op, b=b_val)


def apply_adc_correction(adc_image, constants):
    """Apply ADC correction formula (vectorized)."""
    OA, OB, Od, Op, b = (constants[k] for k in ('OA', 'OB', 'Od', 'Op', 'b'))

    adc_corrected = np.zeros_like(adc_image)
    valid = (adc_image > 0) & np.isfinite(adc_image)

    if valid.any():
        af  = adc_image[valid]
        E   = np.exp(af * b)
        Ok  = (((Od + (OA * Op / (1 - OB))) / E) - Od) / (OA * Op)
        G   = np.log(1 + Ok * OB) - np.log(Ok)
        adc_corrected[valid] = G / b

    adc_corrected[~np.isfinite(adc_corrected)] = 0
    adc_corrected[adc_corrected < 0] = 0
    return adc_corrected
