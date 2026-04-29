import os
import pickle
import numpy as np
import nibabel as nib
from pathlib import Path
import time
from scipy import ndimage
from scipy.ndimage import map_coordinates, gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ── Utilities ─────────────────────────────────────────────────────

def _to_py_tuple(vox):
    return tuple(int(x) for x in vox)


def _path_to_mm(path, affine):
    vox  = np.array(path, dtype=float)
    ones = np.ones((len(vox), 1))
    return (affine @ np.hstack([vox, ones]).T).T[:, :3]


def _arc_cumlen(path_mm):
    segs = np.linalg.norm(np.diff(path_mm, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segs)])


def _interp_at_arc(path_mm, cumlen, s):
    s   = float(np.clip(s, cumlen[0], cumlen[-1]))
    idx = int(np.clip(np.searchsorted(cumlen, s, side='right') - 1,
                      0, len(path_mm) - 2))
    seg_len = cumlen[idx + 1] - cumlen[idx]
    if seg_len < 1e-9:
        for k in range(len(path_mm) - 1):
            if (cumlen[k + 1] - cumlen[k]) >= 1e-9:
                d = path_mm[k + 1] - path_mm[k]
                return path_mm[idx].copy(), d / np.linalg.norm(d)
        return path_mm[idx].copy(), np.array([1., 0., 0.])
    t    = (s - cumlen[idx]) / seg_len
    d    = path_mm[idx + 1] - path_mm[idx]
    pos  = path_mm[idx] + t * d
    tang = d / np.linalg.norm(d)
    return pos, tang


def _smooth_path_mm(path_mm, sigma_mm=1.0):
    cumlen  = _arc_cumlen(path_mm)
    total   = cumlen[-1]
    if total < 1e-6 or len(path_mm) < 5:
        return path_mm.copy()
    avg_step = total / (len(path_mm) - 1)
    sigma_s  = max(sigma_mm / max(avg_step, 1e-9), 0.5)
    smoothed = np.empty_like(path_mm)
    for ax in range(3):
        smoothed[:, ax] = gaussian_filter1d(path_mm[:, ax], sigma=sigma_s)
    smoothed[0]  = path_mm[0]
    smoothed[-1] = path_mm[-1]
    return smoothed


def _smooth_1d(arr, window):
    if window <= 1:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    n, half = len(arr), window // 2
    out = np.full(n, np.nan)
    for i in range(n):
        chunk = arr[max(0, i - half):min(n, i + half + 1)]
        valid = chunk[~np.isnan(chunk)]
        if len(valid):
            out[i] = np.mean(valid)
    return out


def _find_segment_label(ref_path, multilabel_p1):
    votes = {}
    for vox in ref_path:
        z, y, x = int(round(vox[0])), int(round(vox[1])), int(round(vox[2]))
        if (0 <= z < multilabel_p1.shape[0] and
                0 <= y < multilabel_p1.shape[1] and
                0 <= x < multilabel_p1.shape[2]):
            lab = int(multilabel_p1[z, y, x])
            if lab > 0:
                votes[lab] = votes.get(lab, 0) + 1
    return max(votes, key=votes.get) if votes else None


def _recenter_and_area(pos_mm, tang_unit, multilabel, seg_label,
                        inv_affine, affine, voxel_size,
                        rc_radius_mm, rc_slab_mm, area_r_mm, area_slab_half_mm):
    max_r   = max(rc_radius_mm, area_r_mm)
    pos_vox = (inv_affine @ np.append(pos_mm, 1.0))[:3]
    r_vox   = int(np.ceil(max_r / np.min(voxel_size))) + 2
    shape   = multilabel.shape
    iz, iy, ix = int(round(pos_vox[0])), int(round(pos_vox[1])), int(round(pos_vox[2]))

    zs = slice(max(0, iz - r_vox), min(shape[0], iz + r_vox + 1))
    ys = slice(max(0, iy - r_vox), min(shape[1], iy + r_vox + 1))
    xs = slice(max(0, ix - r_vox), min(shape[2], ix + r_vox + 1))

    local = (multilabel[zs, ys, xs] == seg_label)
    if not local.sum():
        return pos_mm.copy(), 0.0

    origin  = np.array([zs.start, ys.start, xs.start], dtype=float)
    lv      = np.argwhere(local).astype(float)
    gv      = lv + origin
    pts_mm  = (affine @ np.hstack([gv, np.ones((len(gv), 1))]).T).T[:, :3]
    offsets = pts_mm - pos_mm
    along   = offsets @ tang_unit

    recentered_pos = pos_mm.copy()
    rc_in_slab = np.abs(along) <= rc_slab_mm
    if np.any(rc_in_slab):
        perp      = offsets[rc_in_slab] - along[rc_in_slab, np.newaxis] * tang_unit
        perp_dist = np.linalg.norm(perp, axis=1)
        in_r      = perp_dist <= rc_radius_mm
        if np.any(in_r):
            recentered_pos = pos_mm + perp[in_r].mean(axis=0)

    offsets2 = pts_mm - recentered_pos
    along2   = offsets2 @ tang_unit
    area_in_slab = np.abs(along2) <= area_slab_half_mm
    if not np.any(area_in_slab):
        return recentered_pos, 0.0

    perp2 = offsets2[area_in_slab] - along2[area_in_slab, np.newaxis] * tang_unit
    count = int((np.sum(perp2 ** 2, axis=1) <= area_r_mm ** 2).sum())
    area  = count * float(np.prod(voxel_size)) / (2.0 * area_slab_half_mm)
    return recentered_pos, area


def _extract_ref_segments(ref_graph):
    nids = {_to_py_tuple(k): v for k, v in ref_graph['node_ids'].items()}
    segments = {}
    for _key, edge in ref_graph['edges'].items():
        path = edge['path']
        if len(path) < 2:
            continue
        a, b = _to_py_tuple(path[0]), _to_py_tuple(path[-1])
        a_id, b_id = nids.get(a, str(a)), nids.get(b, str(b))
        a_bif, b_bif = a_id.startswith('BIF'), b_id.startswith('BIF')

        if a_bif and b_bif:
            if a_id <= b_id:
                na_id, nb_id, path_ordered = a_id, b_id, list(path)
            else:
                na_id, nb_id, path_ordered = b_id, a_id, list(reversed(path))
            seg_id   = f"{na_id}—{nb_id}"
            seg_type = 'bif-bif'
        elif a_bif:
            na_id, nb_id, path_ordered = a_id, b_id, list(path)
            seg_id = f"{na_id}→{nb_id}"; seg_type = 'bif-ep'
        elif b_bif:
            na_id, nb_id, path_ordered = b_id, a_id, list(reversed(path))
            seg_id = f"{na_id}→{nb_id}"; seg_type = 'bif-ep'
        else:
            if a_id <= b_id:
                na_id, nb_id, path_ordered = a_id, b_id, list(path)
            else:
                na_id, nb_id, path_ordered = b_id, a_id, list(reversed(path))
            seg_id = f"{na_id}—{nb_id}"; seg_type = 'ep-ep'

        if seg_id not in segments:
            segments[seg_id] = {'node_a': na_id, 'node_b': nb_id, 'seg_type': seg_type,
                                 'length_mm': edge['length_mm'], 'ref_path': path_ordered}
    return segments


# ── Main function ─────────────────────────────────────────────────

def compute_pulsatility(step1_dir, data_dir, out_dir, n_phases=25,
                         sample_spacing_mm=0.5, min_samples=5, max_samples=150,
                         min_seg_length_mm=3.0, s_skip_mm=2.0,
                         slab_half_mm=0.5, r_max_mm=20.0,
                         recenter_radius_mm=2.5, recenter_slab_mm=1.0,
                         smooth_window=15, path_smooth_mm=1.0,
                         xsec_n_slices=9, xsec_radius_mm=5.0,
                         xsec_resolution=0.2, xsec_phase=1):
    """Compute cross-sectional area and pulsatility index for each vessel segment.

    Args:
        step1_dir: directory with reference_graph.pkl (from step9)
        data_dir:  directory with p01-p25_multilabel.nii.gz (Part B output)
        out_dir:   output directory for pi_results.pkl and cross-section images
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    xsec_out_dir = os.path.join(out_dir, "cross_sections")
    Path(xsec_out_dir).mkdir(parents=True, exist_ok=True)
    t_global = time.time()

    print("=" * 60)
    print("Step 10: Cross-section area + Pulsatility Index")
    print(f"  step1_dir: {step1_dir}")
    print(f"  data_dir : {data_dir}")
    print("=" * 60)

    with open(os.path.join(step1_dir, "reference_graph.pkl"), "rb") as f:
        ref = pickle.load(f)
    ref_affine   = ref['affine']
    ref_segments = _extract_ref_segments(ref)

    skipped = {sid: s for sid, s in ref_segments.items() if s['length_mm'] < min_seg_length_mm}
    kept    = {sid: s for sid, s in ref_segments.items() if s['length_mm'] >= min_seg_length_mm}
    print(f"  {len(ref_segments)} segments: {len(kept)} kept, {len(skipped)} skipped (<{min_seg_length_mm}mm)")
    ref_segments = kept

    p1_ml = nib.load(os.path.join(data_dir, "p01_multilabel.nii.gz")).get_fdata().astype(np.uint8)
    seg_label_map = {}
    for seg_id, seg in sorted(ref_segments.items()):
        seg_label_map[seg_id] = _find_segment_label(seg['ref_path'], p1_ml)
        print(f"  {seg_id:30s} → label {seg_label_map[seg_id]}")
    del p1_ml

    print(f"\nLoading {n_phases} multilabel masks...")
    phase_data = {}
    for ph in range(1, n_phases + 1):
        img = nib.load(os.path.join(data_dir, f"p{ph:02d}_multilabel.nii.gz"))
        ml  = img.get_fdata().astype(np.uint8)
        aff = img.affine
        phase_data[ph] = {'multilabel': ml, 'affine': aff,
                          'inv_affine': np.linalg.inv(aff),
                          'voxel_size': np.abs(np.diag(aff)[:3])}

    all_pi = {}
    for seg_id, seg in sorted(ref_segments.items()):
        t_seg = time.time()
        ref_path_mm = _smooth_path_mm(_path_to_mm(seg['ref_path'], ref_affine), path_smooth_mm)
        cumlen      = _arc_cumlen(ref_path_mm)
        total_len   = cumlen[-1]
        seg_label   = seg_label_map.get(seg_id)

        s_start = min(s_skip_mm, total_len * 0.1)
        s_end   = max(total_len - s_skip_mm, total_len * 0.9)
        if s_end <= s_start + 1e-6:
            s_start, s_end = 0.0, total_len

        n_samples = max(min_samples, min(max_samples,
                        int(round((s_end - s_start) / sample_spacing_mm))))
        s_vals    = np.linspace(s_start, s_end, n_samples)
        spacing   = (s_end - s_start) / max(n_samples - 1, 1)

        ref_info = [_interp_at_arc(ref_path_mm, cumlen, s) for s in s_vals]
        orig_pts_mm  = np.array([pos  for pos, _ in ref_info])
        tangents_arr = np.array([tang for _, tang in ref_info])
        recen_pts    = np.zeros((n_samples, 3))

        print(f"\n{seg_id}  [{seg['seg_type']}  {total_len:.1f}mm  "
              f"label={seg_label}  {n_samples} samples]")

        area_raw_by_ph = {}
        shift_norms    = []
        for ph in range(1, n_phases + 1):
            pd    = phase_data[ph]
            areas = np.zeros(n_samples)
            for si, (ref_pos, ref_tang) in enumerate(ref_info):
                ml_ph   = pd['multilabel']
                lbl     = seg_label if seg_label is not None else None
                if lbl is None:
                    rc_pos, area = ref_pos.copy(), 0.0
                else:
                    rc_pos, area = _recenter_and_area(
                        ref_pos, ref_tang, ml_ph, lbl,
                        pd['inv_affine'], pd['affine'], pd['voxel_size'],
                        recenter_radius_mm, recenter_slab_mm, r_max_mm, slab_half_mm)
                areas[si] = area
                shift_norms.append(np.linalg.norm(rc_pos - ref_pos))
                if ph == xsec_phase:
                    recen_pts[si] = rc_pos
            area_raw_by_ph[ph] = areas

        sw         = max(3, int(round(smooth_window * n_samples / 50)))
        area_by_ph = {ph: _smooth_1d(area_raw_by_ph[ph], sw) for ph in range(1, n_phases + 1)}
        area_mat   = np.vstack([area_by_ph[ph] for ph in range(1, n_phases + 1)])
        mean_area  = np.nanmean(area_mat, axis=0)

        pi_per_s = np.full(n_samples, np.nan)
        valid_s  = mean_area > 1e-6
        if np.any(valid_s):
            pi_per_s[valid_s] = ((np.nanmax(area_mat[:, valid_s], axis=0) -
                                   np.nanmin(area_mat[:, valid_s], axis=0)) /
                                  mean_area[valid_s])

        pi_mean   = float(np.nanmean(pi_per_s))
        ph_means  = {ph: float(np.nanmean(area_by_ph[ph])) for ph in range(1, n_phases + 1)}
        print(f"  area [{min(ph_means.values()):.1f}, {max(ph_means.values()):.1f}] mm²  "
              f"PI={pi_mean:.4f}  shift mean={np.mean(shift_norms):.2f}mm  "
              f"({time.time()-t_seg:.1f}s)")

        all_pi[seg_id] = {
            'seg_id': seg_id, 'seg_type': seg['seg_type'], 'length_mm': total_len,
            'seg_label': seg_label, 'n_samples': n_samples, 'spacing_mm': spacing,
            's_vals': s_vals, 'area_raw_by_ph': area_raw_by_ph, 'area_by_ph': area_by_ph,
            'area_mat': area_mat, 'pi_per_s': pi_per_s, 'pi_mean': pi_mean,
            'smooth_window': sw, 'mean_shift_mm': np.mean(shift_norms),
            'max_shift_mm':  np.max(shift_norms),
            'orig_pts_mm': orig_pts_mm, 'tangents': tangents_arr, 'recen_pts_mm': recen_pts,
        }

    out_path = os.path.join(out_dir, "pi_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({'pi': all_pi, 'seg_label_map': seg_label_map,
                     'skipped': {sid: s['length_mm'] for sid, s in skipped.items()},
                     'config': {'sample_spacing_mm': sample_spacing_mm,
                                'slab_half_mm': slab_half_mm, 'r_max_mm': r_max_mm,
                                'recenter_radius_mm': recenter_radius_mm,
                                'smooth_window': smooth_window}}, f)
    print(f"\nSaved → {out_path}")
    print(f"Total time: {time.time()-t_global:.1f}s")
    print(f"Done  ->  {out_dir}")
