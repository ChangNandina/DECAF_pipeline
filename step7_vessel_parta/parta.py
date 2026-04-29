import requests
import json
import nibabel as nib
import numpy as np
import gzip
import base64
import os
import time
from multiprocessing import Pool
from skimage.morphology import skeletonize
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, binary_dilation, generate_binary_structure


# ── Geometry helpers ────────────────────────────────────────────

def _edt_one_segment(args):
    sid, coords, shape, bbox_pad = args
    valid = ((coords[:, 0] >= 0) & (coords[:, 0] < shape[0]) &
             (coords[:, 1] >= 0) & (coords[:, 1] < shape[1]) &
             (coords[:, 2] >= 0) & (coords[:, 2] < shape[2]))
    c = coords[valid]
    if len(c) == 0:
        return sid, np.full(shape, np.inf, dtype=np.float32)
    lo = np.maximum(c.min(axis=0) - bbox_pad, 0)
    hi = np.minimum(c.max(axis=0) + bbox_pad + 1, np.array(shape))
    marker = np.zeros(tuple(hi - lo), dtype=bool)
    c_local = c - lo
    marker[c_local[:, 0], c_local[:, 1], c_local[:, 2]] = True
    d_sub = distance_transform_edt(~marker).astype(np.float32)
    d_full = np.full(shape, np.inf, dtype=np.float32)
    d_full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = d_sub
    return sid, d_full


def compute_dist_and_voronoi_fast(shape, segments, margin=5, n_workers=16, bbox_pad=80):
    sids = sorted(segments.keys())
    work_items = [(sid, segments[sid], shape, bbox_pad) for sid in sids]
    dist_maps = {}
    with Pool(processes=n_workers) as pool:
        for i, (sid, d_full) in enumerate(pool.imap_unordered(_edt_one_segment, work_items)):
            dist_maps[sid] = d_full
            if (i + 1) % 10 == 0 or (i + 1) == len(sids):
                print(f"      EDT: {i+1}/{len(sids)} done")

    any_skeleton = np.zeros(shape, dtype=bool)
    for sid, coords in segments.items():
        valid = ((coords[:, 0] >= 0) & (coords[:, 0] < shape[0]) &
                 (coords[:, 1] >= 0) & (coords[:, 1] < shape[1]) &
                 (coords[:, 2] >= 0) & (coords[:, 2] < shape[2]))
        c = coords[valid]
        if len(c) > 0:
            any_skeleton[c[:, 0], c[:, 1], c[:, 2]] = True

    nearest_dist = distance_transform_edt(~any_skeleton).astype(np.float32)
    threshold = nearest_dist + margin
    soft_regions = {sid: dist_maps[sid] < threshold for sid in sids}
    return dist_maps, soft_regions


def farthest_point_sampling(points, n_samples):
    n = len(points)
    if n_samples >= n:
        return list(range(n))
    center = points.mean(axis=0)
    first = np.argmin(np.sum((points - center) ** 2, axis=1))
    selected = [first]
    min_dists = np.full(n, np.inf)
    for _ in range(n_samples - 1):
        last = points[selected[-1]]
        dists = np.sum((points - last) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        selected.append(np.argmax(min_dists))
    return selected


def find_branch_points(skeleton):
    struct = ndimage.generate_binary_structure(3, 3)
    neighbor_count = ndimage.convolve(
        (skeleton > 0).astype(np.int32), struct.astype(np.int32), mode='constant') - 1
    return (skeleton > 0) & (neighbor_count >= 3)


def prune_skeleton(skeleton, min_spur_length=10):
    skel = skeleton.copy().astype(np.uint8)
    struct = ndimage.generate_binary_structure(3, 3)
    changed = True
    while changed:
        changed = False
        neighbor_count = ndimage.convolve(
            (skel > 0).astype(np.int32), struct.astype(np.int32), mode='constant') - 1
        bp_mask  = (skel > 0) & (neighbor_count >= 3)
        tip_mask = (skel > 0) & (neighbor_count == 1)
        skel_no_bp = skel.copy()
        skel_no_bp[bp_mask] = 0
        labeled, n_labels = ndimage.label(skel_no_bp, structure=struct)
        for lab in range(1, n_labels + 1):
            comp_coords = np.argwhere(labeled == lab)
            if len(comp_coords) >= min_spur_length:
                continue
            if any(tip_mask[c[0], c[1], c[2]] for c in comp_coords):
                skel[comp_coords[:, 0], comp_coords[:, 1], comp_coords[:, 2]] = 0
                changed = True
    return skel


def extract_segments(mask_data, min_length=30, bp_dilation=1, min_spur_length=10):
    binary = (mask_data > 0).astype(np.uint8)
    print("  Skeletonizing...")
    skeleton = skeletonize(binary).astype(np.uint8)
    skeleton = prune_skeleton(skeleton, min_spur_length=min_spur_length)

    bp = find_branch_points(skeleton)
    struct = ndimage.generate_binary_structure(3, 3)
    bp_dilated = ndimage.binary_dilation(bp, struct, iterations=bp_dilation) if bp_dilation > 0 else bp

    skel_no_bp = skeleton.copy()
    skel_no_bp[bp_dilated] = 0
    labeled, n_labels = ndimage.label(skel_no_bp, structure=struct)

    segments = {}
    for i in range(1, n_labels + 1):
        coords = np.argwhere(labeled == i)
        if len(coords) >= min_length:
            segments[len(segments) + 1] = coords
    print(f"  Segments (>= {min_length} voxels): {len(segments)}")
    return segments, skeleton


def allocate_seed_points(segments, total_points, min_per_seg=3):
    lengths = {sid: len(c) for sid, c in segments.items()}
    total_length = sum(lengths.values())
    allocation = {sid: max(min_per_seg, round(total_points * lengths[sid] / total_length))
                  for sid in segments}
    while sum(allocation.values()) > total_points:
        longest = max(allocation, key=allocation.get)
        if allocation[longest] > min_per_seg:
            allocation[longest] -= 1
        else:
            break
    while sum(allocation.values()) < total_points:
        longest = max(lengths, key=lengths.get)
        allocation[longest] += 1
        if sum(allocation.values()) >= total_points:
            break
    return allocation


def decode_response(response_json, shape):
    result_gz  = base64.b64decode(response_json["result"])
    result_raw = gzip.decompress(result_gz)
    return np.frombuffer(result_raw, dtype=np.int8).reshape(shape)


def compute_dice(mask_a, mask_b):
    a, b = (mask_a > 0).astype(bool), (mask_b > 0).astype(bool)
    inter = (a & b).sum()
    total = a.sum() + b.sum()
    return 2.0 * inter / total if total > 0 else 0.0


def run_one_phase(phase, nifti_dir, segment_seeds, soft_regions, dist_maps,
                  mask_shape, output_dir, mask_data_for_dice=None,
                  server="http://localhost:8911", file_prefix=""):
    nifti_path = os.path.join(nifti_dir, f"{file_prefix}phase{phase}.nii.gz")
    img_nib    = nib.load(nifti_path)
    img_data   = img_nib.get_fdata().astype(np.float32)
    affine     = img_nib.affine
    voxel_vol  = np.abs(np.linalg.det(affine[:3, :3]))

    img_gz     = gzip.compress(img_data.tobytes(), compresslevel=4)
    dimensions = [int(d) for d in img_data.shape[::-1]]

    r          = requests.get(f"{server}/start_session")
    session_id = r.json()["session_id"]
    metadata   = json.dumps({"dimensions": dimensions})
    r = requests.post(f"{server}/upload_raw/{session_id}",
                      files={"file": ("image.raw.gz", img_gz, "application/octet-stream")},
                      data={"metadata": metadata})
    del img_gz
    if r.status_code != 200:
        print(f"  Upload FAILED: {r.text[:200]}")
        requests.get(f"{server}/end_session/{session_id}")
        return None, None

    clipped_masks = {}
    total_raw = total_clipped_away = 0

    for sid, seeds in segment_seeds.items():
        requests.get(f"{server}/reset_interactions/{session_id}")
        last_response = None
        for pt in seeds:
            x, y, z = int(pt[2]), int(pt[1]), int(pt[0])
            r = requests.get(f"{server}/process_point_interaction/{session_id}",
                             params={"x": x, "y": y, "z": z, "foreground": True})
            if r.status_code == 200:
                last_response = r.json()

        if last_response and last_response.get("status") == "success":
            raw_mask     = decode_response(last_response, mask_shape)
            raw_fg       = (raw_mask > 0)
            clipped      = raw_fg & soft_regions[sid]
            overflow     = int(raw_fg.sum()) - int(clipped.sum())
            clipped_masks[sid] = clipped
            total_raw        += int(raw_fg.sum())
            total_clipped_away += overflow
            print(f"    Seg {sid}: {raw_fg.sum()} raw -> {clipped.sum()} kept")
        else:
            print(f"    Seg {sid}: FAILED")
            clipped_masks[sid] = np.zeros(mask_shape, dtype=bool)

    requests.get(f"{server}/end_session/{session_id}")

    multilabel      = np.zeros(mask_shape, dtype=np.uint8)
    coverage_count  = np.zeros(mask_shape, dtype=np.int32)
    for sid, mask in clipped_masks.items():
        coverage_count[mask] += 1
    for sid, mask in clipped_masks.items():
        multilabel[mask & (coverage_count == 1)] = sid

    overlap_zone = coverage_count >= 2
    if int(overlap_zone.sum()) > 0:
        best_dist = np.full(mask_shape, np.inf, dtype=np.float32)
        for sid, mask in clipped_masks.items():
            contest = mask & overlap_zone
            closer  = dist_maps[sid] < best_dist
            update  = contest & closer
            multilabel[update] = sid
            best_dist[update]  = dist_maps[sid][update]

    binary_combined = (multilabel > 0)
    binary_voxels   = int(binary_combined.sum())
    binary_vol      = binary_voxels * voxel_vol
    print(f"  Merge: {binary_voxels} voxels, {binary_vol:.2f} mm³")

    nib.save(nib.Nifti1Image(binary_combined.astype(np.uint8), affine),
             os.path.join(output_dir, f"p{phase:02d}_binary.nii.gz"))
    nib.save(nib.Nifti1Image(multilabel, affine),
             os.path.join(output_dir, f"p{phase:02d}_multilabel.nii.gz"))

    if mask_data_for_dice is not None:
        dice = compute_dice(binary_combined, mask_data_for_dice)
        print(f"  Phase {phase} Dice vs GT: {dice:.4f}")

    phase_volumes = {sid: int((multilabel == sid).sum()) * voxel_vol
                     for sid in sorted(segment_seeds.keys())}
    clip_stat = {'raw': total_raw, 'clipped_away': total_clipped_away,
                 'final': binary_voxels}
    return phase_volumes, clip_stat


# ── Main function ───────────────────────────────────────────────

def run_parta(nifti_dir, mask_path, output_dir,
              skip_segments=None, file_prefix="",
              server="http://localhost:8911",
              total_points=100, min_branch_length=1, min_points_per_seg=1,
              bp_dilation=0, min_spur_length=3, soft_margin=5,
              n_workers=16, bbox_pad=80):
    """Plan B+ Part A: prepare skeleton, seeds, Voronoi, run Phase 1.

    Args:
        nifti_dir:    directory with cropped 480³ NIfTI phase files
        mask_path:    vessel mask NIfTI annotated in ITK-SNAP
        output_dir:   directory for all Part A outputs
        skip_segments: list of segment IDs to exclude (e.g. [9, 13])
        file_prefix:  prefix on NIfTI filenames (e.g. "" or "r_")
        server:       nninteractive server URL
    """
    if skip_segments is None:
        skip_segments = []
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Plan B+ Part A")
    print(f"  nifti_dir : {nifti_dir}")
    print(f"  mask_path : {mask_path}")
    print(f"  output_dir: {output_dir}")
    print("=" * 60)

    mask_nib   = nib.load(mask_path)
    mask_data  = mask_nib.get_fdata()
    mask_shape = mask_data.shape

    segments, skeleton = extract_segments(
        mask_data, min_length=min_branch_length,
        bp_dilation=bp_dilation, min_spur_length=min_spur_length)
    if not segments:
        raise RuntimeError("No segments found in mask!")

    seg_save = {str(sid): coords.tolist() for sid, coords in segments.items()}
    with open(os.path.join(output_dir, "segments.json"), 'w') as f:
        json.dump(seg_save, f)

    seeds_file = os.path.join(output_dir, "segment_seeds.json")
    if os.path.exists(seeds_file):
        print(f"  Loading existing seeds: {seeds_file}")
        with open(seeds_file, 'r') as f:
            seeds_data = json.load(f)
        segment_seeds = {int(k): np.array(v) for k, v in seeds_data['seeds'].items()}
    else:
        allocation    = allocate_seed_points(segments, total_points, min_points_per_seg)
        segment_seeds = {}
        for sid, coords in segments.items():
            idx = farthest_point_sampling(coords.astype(np.float64), allocation[sid])
            segment_seeds[sid] = coords[idx]
        seeds_data = {
            'total_points': total_points,
            'note': 'coords are [z, y, x]. Delete a segment key to skip it.',
            'seeds': {str(sid): pts.tolist() for sid, pts in segment_seeds.items()}
        }
        with open(seeds_file, 'w') as f:
            json.dump(seeds_data, f, indent=2)

    for sid in skip_segments:
        segment_seeds.pop(sid, None)
    print(f"  Active segments: {len(segment_seeds)}")

    dist_maps, soft_regions = compute_dist_and_voronoi_fast(
        mask_shape, segments, margin=soft_margin,
        n_workers=n_workers, bbox_pad=bbox_pad)

    np.savez_compressed(os.path.join(output_dir, "dist_maps.npz"),
                        **{str(sid): dm for sid, dm in dist_maps.items()})
    np.savez_compressed(os.path.join(output_dir, "soft_regions.npz"),
                        **{str(sid): sr for sid, sr in soft_regions.items()})

    print("\nRunning Phase 1 inference...")
    run_one_phase(1, nifti_dir, segment_seeds, soft_regions, dist_maps,
                  mask_shape, output_dir, mask_data_for_dice=mask_data,
                  server=server, file_prefix=file_prefix)

    print("\nPart A complete.")
    print(f"  Review in ITK-SNAP:")
    print(f"    {os.path.join(output_dir, 'p01_binary.nii.gz')}")
    print(f"    {os.path.join(output_dir, 'p01_multilabel.nii.gz')}")
    print(f"  Edit skip_segments in cases.csv if needed, then run Part B.")
