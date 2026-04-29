import os
import json
import pickle
import time
import numpy as np
import nibabel as nib
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.ndimage import map_coordinates, distance_transform_edt


# ── Geometry helpers ──────────────────────────────────────────────

def _get_plane_basis(tang_unit):
    t   = tang_unit / np.linalg.norm(tang_unit)
    ref = np.array([1., 0., 0.])
    if abs(np.dot(t, ref)) > 0.9:
        ref = np.array([0., 1., 0.])
    e1 = np.cross(t, ref);  e1 /= np.linalg.norm(e1)
    e2 = np.cross(t, e1);   e2 /= np.linalg.norm(e2)
    return e1, e2


def _sample_plane(vol, center_mm, e1, e2, inv_affine, half_mm, resolution):
    n        = int(2 * half_mm / resolution) + 1
    lin      = np.linspace(-half_mm, half_mm, n)
    g1, g2   = np.meshgrid(lin, lin, indexing='xy')
    pts_mm   = (center_mm[None,None,:]
                + g1[...,None]*e1[None,None,:]
                + g2[...,None]*e2[None,None,:])
    pts_flat = pts_mm.reshape(-1, 3)
    pts_vox  = (inv_affine @ np.hstack([pts_flat,
                np.ones((len(pts_flat),1))]).T).T[:,:3].reshape(n,n,3)
    patch = map_coordinates(vol, [pts_vox[...,i].ravel() for i in range(3)],
                            order=1, mode='constant', cval=0.0).reshape(n,n)
    return patch, lin


def _dist_from_vessel(vessel_patch):
    return distance_transform_edt(vessel_patch == 0)


def _extract_csf_adc_at_point(pos_mm, tang_unit,
                                adc_vol,    adc_inv_affine,
                                vessel_vol, vessel_inv_affine,
                                slab_half_mm, perivas_dist_mm,
                                grid_res_mm, grid_half_mm):
    e1, e2 = _get_plane_basis(tang_unit)
    vessel_patch, lin = _sample_plane(vessel_vol, pos_mm, e1, e2,
                                      vessel_inv_affine, grid_half_mm, grid_res_mm)
    vessel_bin = vessel_patch > 0.5
    if not np.any(vessel_bin):
        return np.nan, 0

    dist_mm_grid = _dist_from_vessel(vessel_bin) * grid_res_mm
    shell_2d     = (vessel_bin == 0) & (dist_mm_grid <= perivas_dist_mm)
    if not np.any(shell_2d):
        return np.nan, 0

    shell_idx    = np.argwhere(shell_2d)
    d1_vals      = lin[shell_idx[:, 0]]
    d2_vals      = lin[shell_idx[:, 1]]
    shell_pts_mm = pos_mm[None,:] + d1_vals[:,None]*e1[None,:] + d2_vals[:,None]*e2[None,:]

    ones      = np.ones((len(shell_pts_mm), 1))
    shell_vox = (adc_inv_affine @ np.hstack([shell_pts_mm, ones]).T).T[:, :3]

    shape = adc_vol.shape
    in_bounds = ((shell_vox[:,0] >= 0) & (shell_vox[:,0] < shape[0]) &
                 (shell_vox[:,1] >= 0) & (shell_vox[:,1] < shape[1]) &
                 (shell_vox[:,2] >= 0) & (shell_vox[:,2] < shape[2]))
    if not np.any(in_bounds):
        return np.nan, 0

    adc_vals = map_coordinates(adc_vol, [shell_vox[in_bounds, i] for i in range(3)],
                               order=1, mode='constant', cval=0.0)
    valid = adc_vals > 0
    return (float(np.nanmean(adc_vals[valid])), int(valid.sum())) if valid.any() else (np.nan, 0)


def _load_crop_info(crop_dir):
    with open(os.path.join(crop_dir, 'crop_info.json')) as f:
        info = json.load(f)
    return info['crop_origin'], info['crop_size']


def _load_adc_phase_cropped(adc_folder, phase, crop_origin, crop_size,
                              adc_permute=(1,2,0)):
    import h5py
    fname = os.path.join(adc_folder, f'adc_corrected_csf_phase{phase:02d}_single.mat')
    with h5py.File(fname, 'r') as f:
        adc_raw = f['adcCorrectedCSF'][()].astype(np.float32)
    adc = np.transpose(adc_raw, adc_permute)
    o, s = crop_origin, crop_size
    return adc[o[0]:o[0]+s[0], o[1]:o[1]+s[1], o[2]:o[2]+s[2]].copy()


# ── Main function ─────────────────────────────────────────────────

def compute_paravascular_adc(step2_dir, crop_dir, adc_folder, vessel_dir, out_dir,
                              n_phases=25, slab_half_mm=0.5, perivas_dist_mm=3.0,
                              grid_res_mm=0.2, grid_half_mm=7.0,
                              adc_permute=(1,2,0)):
    """Compute paravascular CSF ADC along vessel centerlines.

    Args:
        step2_dir:       directory with pi_results.pkl (from step10)
        crop_dir:        directory with crop_info.json and vessel mask
        adc_folder:      directory with adc_corrected_csf_phase*.mat files (from step6)
        vessel_dir:      directory with p01-p25_multilabel.nii.gz (Part B output)
        out_dir:         output directory for paravascular_adc.mat and visualizations
        n_phases:        number of cardiac phases
        slab_half_mm:    half-thickness of cross-section slab (mm)
        perivas_dist_mm: paravascular shell thickness (mm from vessel surface)
        grid_res_mm:     2D grid resolution for distance transform (mm)
        grid_half_mm:    half-width of 2D grid (mm)
        adc_permute:     permutation to apply after loading ADC .mat (default: (1,2,0))
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    vis_dir = os.path.join(out_dir, "adc_xsec_vis")
    Path(vis_dir).mkdir(parents=True, exist_ok=True)
    t_global = time.time()

    print("=" * 60)
    print("Step 11: Paravascular CSF ADC")
    print(f"  step2_dir : {step2_dir}")
    print(f"  adc_folder: {adc_folder}")
    print(f"  vessel_dir: {vessel_dir}")
    print("=" * 60)

    with open(os.path.join(step2_dir, 'pi_results.pkl'), 'rb') as f:
        step2 = pickle.load(f)
    pi_all        = step2['pi']
    seg_label_map = step2['seg_label_map']
    print(f"  {len(pi_all)} segments")

    crop_origin, crop_size = _load_crop_info(crop_dir)
    crop_nib    = nib.load(os.path.join(crop_dir, next(
        f for f in os.listdir(crop_dir) if f.endswith('.nii.gz') and 'phase' in f)))
    crop_affine    = crop_nib.affine
    adc_inv_affine = np.linalg.inv(crop_affine)
    adc_voxel_size = np.abs(np.diag(crop_affine)[:3])
    print(f"  Crop affine voxel size: {adc_voxel_size} mm")

    all_seg_ids  = []
    all_pos_mm   = []
    all_tang     = []
    all_area_mat = []
    all_pi_val   = []

    for seg_id, r in sorted(pi_all.items()):
        if r is None:
            continue
        n = r['n_samples']
        for si in range(n):
            all_seg_ids.append(seg_id)
            all_pos_mm.append(r['recen_pts_mm'][si])
            all_tang.append(r['tangents'][si])
            all_area_mat.append(r['area_mat'][:, si])
            all_pi_val.append(r['pi_per_s'][si])

    N_total  = len(all_pos_mm)
    adc_out  = np.full((N_total, n_phases), np.nan, dtype=np.float32)
    nvox_out = np.zeros((N_total, n_phases), dtype=np.int32)
    area_out = np.array(all_area_mat, dtype=np.float32)
    pi_out   = np.array(all_pi_val,   dtype=np.float32)
    cl_mm_out = np.array(all_pos_mm,  dtype=np.float32)
    print(f"  Total sample points: {N_total}")

    for ph in range(1, n_phases + 1):
        t_ph = time.time()
        print(f"\nPhase {ph:2d}/{n_phases} ...")

        adc_vol = _load_adc_phase_cropped(adc_folder, ph, crop_origin, crop_size, adc_permute)
        print(f"  ADC shape: {adc_vol.shape}  nonzero: {(adc_vol>0).sum():,}")

        vessel_nib        = nib.load(os.path.join(vessel_dir, f'p{ph:02d}_multilabel.nii.gz'))
        vessel_multilabel = vessel_nib.get_fdata().astype(np.uint8)
        vessel_inv_affine = np.linalg.inv(vessel_nib.affine)

        n_found = 0
        for i in range(N_total):
            seg_label = seg_label_map.get(all_seg_ids[i])
            if seg_label is None:
                continue
            vessel_seg_f = (vessel_multilabel == seg_label).astype(np.float32)
            adc_mean, n_vox = _extract_csf_adc_at_point(
                pos_mm=all_pos_mm[i], tang_unit=all_tang[i],
                adc_vol=adc_vol, adc_inv_affine=adc_inv_affine,
                vessel_vol=vessel_seg_f, vessel_inv_affine=vessel_inv_affine,
                slab_half_mm=slab_half_mm, perivas_dist_mm=perivas_dist_mm,
                grid_res_mm=grid_res_mm, grid_half_mm=grid_half_mm)
            adc_out[i, ph-1]  = adc_mean
            nvox_out[i, ph-1] = n_vox
            if not np.isnan(adc_mean):
                n_found += 1

        del adc_vol, vessel_multilabel
        print(f"  valid: {n_found}/{N_total} ({100*n_found/N_total:.1f}%)  "
              f"({time.time()-t_ph:.1f}s)")

    valid_adc = adc_out[~np.isnan(adc_out)]
    print(f"\nADC mean={valid_adc.mean():.6f}  "
          f"range=[{valid_adc.min():.6f}, {valid_adc.max():.6f}]  "
          f"valid={len(valid_adc)}/{adc_out.size}")

    out_path = os.path.join(out_dir, 'paravascular_adc.mat')
    sio.savemat(out_path, {
        'cl_mm':    cl_mm_out, 'area_all': area_out,
        'pi_val':   pi_out,    'adc_all':  adc_out,
        'adc_nvox': nvox_out,  'seg_ids':  np.array(all_seg_ids, dtype=object),
        'affine':   crop_affine,
        'config': {'slab_half_mm': slab_half_mm, 'perivas_dist_mm': perivas_dist_mm,
                   'grid_res_mm': grid_res_mm},
    })
    print(f"\nSaved → {out_path}")
    print(f"Total time: {time.time()-t_global:.1f}s")
    print(f"Done  ->  {out_dir}")
